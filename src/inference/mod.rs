//! # Inference Engine Module
//! 
//! The inference module provides core functionality for managing and interacting with 
//! large language models (LLMs) in GGUF format. It handles model discovery, loading,
//! tracking, and text generation operations.
//!
//! ## Key Components
//!
//! - `InferenceEngine`: The central component that manages model state and operations
//! - `ModelEntry`: Represents a model in the registry (persistent storage format)
//! - `ModelDetails`: Enriched model information for API responses and display
//!
//! ## Architecture
//!
//! The inference engine maintains thread-safe access to shared state using `RwLock`
//! to allow concurrent reads while ensuring exclusive writes. It interacts with the
//! file system to scan for models and provides an API for the server to expose
//! model operations.

use std::error::Error;
use std::sync::RwLock;  // Add this for thread-safe state
use std::path::PathBuf;
use std::fs;
use tracing::{info, error, debug};
use crate::gguf::{GGUFReader, GGUFError, is_gguf_file};  // Add GGUF parser and standalone function
use crate::model::Model;
use comfy_table::{Table, Cell, ContentArrangement};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, serde::ts_seconds};
use std::thread;
use std::time::Duration;
use indicatif::{ProgressBar, ProgressStyle};
use colored::*;

/// Represents a model entry in the registry file.
///
/// This struct contains persistent metadata about available models
/// and is serialized to/from the model_registry.json file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelEntry {
    /// Filename of the model file (relative to models directory)
    pub filename: String,
    /// Short identifier/label for the model
    pub label: String,
    /// Human-readable name of the model
    pub name: String,
    /// Size category of the model (e.g., "7B", "13B")
    pub size: String,
    /// Architecture of the model (e.g., "LLaMA", "Mistral")
    pub architecture: String,
    /// Quantization format (e.g., "Q4_K_M", "Q5_K_M")
    pub quantization: String,
    /// Number of tensors in the model
    pub tensor_count: u64,
    /// When the model was added to the registry
    #[serde(with = "ts_seconds")]
    pub added_date: DateTime<Utc>,
    /// When the model was last used (optional)
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used: Option<DateTime<Utc>>,
}

/// Detailed model information for display and API responses.
///
/// This struct contains enriched information about a model,
/// including runtime-calculated fields like tensor count and
/// absolute directory paths.
#[derive(Serialize, Deserialize)]
pub struct ModelDetails {
    /// Position in the model list (optional)
    pub number: Option<usize>,
    /// Short identifier for the model
    pub label: String,
    /// Human-readable name
    pub name: String,
    /// Size category (e.g., "7B")
    pub size: String,
    /// Model architecture
    pub architecture: String,
    /// Quantization format
    pub quantization: String,
    /// When the model was added
    #[serde(with = "ts_seconds")]
    pub added_date: DateTime<Utc>,
    /// Number of tensors in the model
    pub tensor_count: u64,
    /// Name of the model file
    pub filename: String,
    /// Absolute path to the models directory
    pub directory: String,
}

/// The core inference engine that manages model state and operations.
///
/// The InferenceEngine maintains thread-safe access to model state using RwLocks,
/// allowing multiple readers but exclusive writers. It provides methods for
/// scanning, loading, and interacting with models.
pub struct InferenceEngine {
    /// Currently selected model label (if any)
    pub current_model: RwLock<Option<String>>,
    /// Currently loaded model data (if any)
    pub loaded_model: RwLock<Option<Model>>,
    /// Models participating in discussion mode
    pub discussion_models: RwLock<Vec<String>>,
    /// Whether the engine is in discussion mode
    pub in_discussion: RwLock<bool>,
    /// Directory where model files are stored
    pub models_dir: PathBuf,
    /// Index of available models (label, path)
    pub model_index: RwLock<Vec<(String, PathBuf)>>,
    /// Registry of all available models and their metadata
    pub registry: RwLock<HashMap<String, ModelEntry>>,
}

impl InferenceEngine {
    /// Creates a new inference engine with the specified models directory.
    ///
    /// # Arguments
    ///
    /// * `models_dir` - Path to the directory containing model files
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            current_model: RwLock::new(None),
            loaded_model: RwLock::new(None),
            discussion_models: RwLock::new(Vec::new()),
            in_discussion: RwLock::new(false),
            models_dir,
            model_index: RwLock::new(Vec::new()),
            registry: RwLock::new(HashMap::new()),
        }
    }

    /// Loads or creates the model registry file.
    ///
    /// The registry is a JSON file that tracks all available models and their metadata.
    ///
    /// # Returns
    ///
    /// A hashmap of model entries where the key is the filename
    pub fn load_or_create_registry(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let registry_path = self.models_dir.join("model_registry.json");
        let mut registry = self.registry.write().map_err(|e| e.to_string())?;
        
        if registry_path.exists() {
            let content = fs::read_to_string(&registry_path)?;
            *registry = serde_json::from_str(&content)?;
        }
        Ok(())
    }

    /// Saves the model registry to disk.
    ///
    /// # Arguments
    ///
    /// * `registry` - The model registry to save
    fn save_registry(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let registry_path = self.models_dir.join("model_registry.json");
        let registry = self.registry.read().map_err(|e| e.to_string())?;
        let content = serde_json::to_string_pretty(&*registry)?;
        fs::write(registry_path, content)?;
        Ok(())
    }

    /// Ensures the models directory exists, creating it if necessary.
    fn ensure_models_dir(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if !self.models_dir.exists() {
            fs::create_dir_all(&self.models_dir)?;
            info!("Created models directory: {}", self.models_dir.display());
        }
        Ok(())
    }

    /// Scans the models directory for new models and adds them to the registry.
    ///
    /// # Arguments
    ///
    /// * `registry` - The current model registry to update
    fn scan_new_models(&self, registry: &mut HashMap<String, ModelEntry>) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.ensure_models_dir()?;

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{prefix:.bold.dim} {spinner} {wide_msg}")
                .unwrap()
        );
        pb.enable_steady_tick(Duration::from_millis(120));
        pb.set_message("Checking models directory...");

        // Get set of existing files in registry
        let existing_files: std::collections::HashSet<String> = registry.keys().cloned().collect();
        
        // Get current files in directory
        let current_files: Vec<_> = fs::read_dir(&self.models_dir)?
            .filter_map(Result::ok)
            .filter(|entry| {
                let path = entry.path();
                path.is_file() && 
                !path.file_name().map_or(true, |name| name.to_string_lossy().starts_with(".")) &&
                path.extension().map_or(true, |ext| ext.to_string_lossy().to_lowercase() == "gguf")
            })
            .collect();

        let total = current_files.len();
        if total == 0 {
            pb.finish_with_message("No GGUF model files found in models directory");
            return Ok(());
        }

        // Remove entries for files that no longer exist
        registry.retain(|filename, _| {
            self.models_dir.join(filename).exists()
        });

        let mut new_models = 0;
        let mut failed_models = 0;

        for (i, entry) in current_files.iter().enumerate() {
            let filename = entry.file_name().to_string_lossy().to_string();
            
            // Skip if file is already in registry
            if existing_files.contains(&filename) {
                continue;
            }
            
            pb.set_message(format!("Checking file: {}... ({}/{})", filename, i + 1, total));
            
            // Use the standalone function first
            if !is_gguf_file(&entry.path()) {
                pb.set_message(format!("Skipping non-GGUF file: {} ({}/{})", filename, i + 1, total));
                info!("Skipping non-GGUF file: {}", filename);
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            
            // Now try to open and extract metadata
            match GGUFReader::new(&entry.path()) {
                Ok(reader) => {
                    // File is a valid GGUF file, proceed with metadata extraction
                    pb.set_message(format!("Reading metadata from: {}... ({}/{})", filename, i + 1, total));
                    
                    // Proceed with metadata extraction...
                    match (
                        reader.get_metadata_value("general.name"),
                        reader.get_metadata_value("general.basename"),
                        reader.get_metadata_value("general.size_label"),
                        reader.get_metadata_value("general.architecture"),
                        reader.get_metadata_value("general.quantization_version")
                    ) {
                        (Ok(name), Ok(label), Ok(size), Ok(arch), Ok(quant)) => {
                            // Update spinner to show this file completed successfully
                            pb.set_message(format!("Successfully processed: {} ({}/{})", filename, i + 1, total));
                            
                            // Create new model entry
                            let model_entry = ModelEntry {
                                filename: filename.clone(),
                                label: label.to_string(),
                                name: name.to_string(),
                                size: size.to_string(),
                                architecture: arch.to_string(),
                                quantization: quant.to_string(),
                                tensor_count: reader.tensor_count,
                                added_date: Utc::now(),
                                last_used: None,
                            };

                            // Add to registry
                            registry.insert(filename, model_entry);
                            
                            // Update counter
                            new_models += 1;
                        },
                        _ => {
                            // Update spinner to show this file failed
                            pb.set_message(format!("Metadata extraction failed: {} ({}/{})", filename, i + 1, total));
                            failed_models += 1;
                            error!("Failed to read metadata from: {}", filename);
                        }
                    }
                },
                Err(e) => {
                    // Check if this is due to invalid format or other error
                    if let Some(gguf_err) = e.downcast_ref::<GGUFError>() {
                        match gguf_err {
                            GGUFError::InvalidFormat(_) => {
                                // Not a GGUF file, skip silently
                                pb.set_message(format!("Skipping non-GGUF file: {} ({}/{})", filename, i + 1, total));
                                info!("Skipping non-GGUF file: {}", filename);
                                thread::sleep(Duration::from_millis(10));
                            },
                            _ => {
                                // GGUF file but open failed
                                pb.set_message(format!("Failed to process GGUF file: {} ({}/{})", filename, i + 1, total));
                                failed_models += 1;
                                println!("\nFailed to open model file {}: {}", filename, e);
                            }
                        }
                    } else {
                        // Other error
                        pb.set_message(format!("Error processing file: {} ({}/{})", filename, i + 1, total));
                        failed_models += 1;
                        println!("\nError with file {}: {}", filename, e);
                    }
                    continue;
                }
            }
            
            // Brief pause between files to make progress visible
            thread::sleep(Duration::from_millis(50));
        }

        let status = format!(
            "Scan complete. Found {} new model{}, {} failed",
            new_models,
            if new_models == 1 { "" } else { "s" },
            failed_models
        );
        pb.disable_steady_tick();
        pb.finish_with_message(status.clone());

        if new_models > 0 {
            self.save_registry()?;
        }
        Ok(())
    }

    /// Scans for new models and updates the registry.
    ///
    /// This is a convenience method that combines loading the registry
    /// and scanning for new models.
    pub fn scan_models(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Load existing registry first
        self.load_or_create_registry()?;
        
        // Get a mutable reference to the registry
        let mut registry = self.registry.write().map_err(|e| e.to_string())?;
        
        // Perform the scan
        info!("Scanning for new models...");
        self.scan_new_models(&mut registry)?;
        Ok(())
    }
}