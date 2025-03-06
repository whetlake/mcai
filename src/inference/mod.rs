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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, serde::ts_seconds};
use std::thread;
use std::time::Duration;
use indicatif::{ProgressBar, ProgressStyle};
use std::fmt;

// Add the model module
pub mod model;
pub mod inference;
use model::Model;
use inference::InferenceContext;

/// Represents a model entry in the registry file.
///
/// This struct contains persistent metadata about available models
/// and is serialized to/from the model_registry.json file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelEntry {
    /// Position in the model list (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number: Option<usize>,
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
    /// Inference context for the current model
    pub inference_context: RwLock<Option<InferenceContext>>,
    /// Directory where model files are stored
    pub models_dir: PathBuf,
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
            inference_context: RwLock::new(None),
            models_dir,
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
        
        // Assign model numbers based on added_date (newest first)
        let mut models: Vec<(String, ModelEntry)> = registry.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        // Sort by added_date in descending order (newest first)
        models.sort_by(|a, b| b.1.added_date.cmp(&a.1.added_date));
        
        // Assign numbers (1 to newest, 2 to second newest, etc.)
        for (i, (filename, model_entry)) in models.into_iter().enumerate() {
            let mut updated_model = model_entry;
            updated_model.number = Some(i + 1);
            registry.insert(filename, updated_model);
        }
        
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
                                number: None, // Will be assigned when registry is saved
                                filename: filename.clone(),
                                label: label.to_string(),
                                name: name.to_string(),
                                size: size.to_string(),
                                architecture: arch.to_string(),
                                quantization: quant.to_string(),
                                tensor_count: reader.tensor_count,
                                added_date: Utc::now(),
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
            // Serialize and save the registry directly while we have the write lock
            let registry_path = self.models_dir.join("model_registry.json");
            let content = serde_json::to_string_pretty(&*registry)?;
            fs::write(registry_path, content)?;
            println!("Registry saved with {} new models", new_models);
        }

        Ok(())
    }

    /// Scans for new models and updates the registry.
    ///
    /// This is a convenience method that combines loading the registry
    /// and scanning for new models in the folder.
    pub fn scan_models(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Load existing registry first
        self.load_or_create_registry()?;
        
        // Get a mutable reference to the registry
        let mut registry = self.registry.write().map_err(|e| e.to_string())?;
        
        // Perform the scan
        info!("Scanning for new models...");
        
        // We need to handle the scan_new_models differently to avoid deadlock
        // First ensure models directory exists
        self.ensure_models_dir()?;
        
        // Then perform the scan and get the result
        let scan_result = self.scan_new_models(&mut registry);
        
        // If new models were added, reassign model numbers based on added_date
        let mut models: Vec<(String, ModelEntry)> = registry.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        // Sort by added_date in descending order (newest first)
        models.sort_by(|a, b| b.1.added_date.cmp(&a.1.added_date));
        
        // Assign numbers (1 to newest, 2 to second newest, etc.)
        for (i, (filename, model_entry)) in models.into_iter().enumerate() {
            let mut updated_model = model_entry;
            updated_model.number = Some(i + 1);
            registry.insert(filename, updated_model);
        }
        
        // Drop the write lock by ending the scope
        drop(registry);
        
        // Now return the result
        scan_result
    }

    /// Attaches a model by its number.
    ///
    /// This method finds a model by its number, loads it, and sets it as the current model.
    ///
    /// # Arguments
    ///
    /// * `model_number` - The number of the model to attach
    ///
    /// # Returns
    ///
    /// A Result containing the attached model details or an error
    pub fn attach_model(&self, model_number: usize) -> Result<ModelDetails, Box<dyn Error + Send + Sync>> {
        // Get a read lock on the registry
        let registry = self.registry.read().map_err(|e| e.to_string())?;

        // Find the model with the given number
        let model_entry = registry.values()
            .find(|entry| entry.number == Some(model_number))
            .ok_or_else(|| format!("Model with number {} not found", model_number))?
            .clone();
        
        // Create the full path to the model file
        let model_path = self.models_dir.join(&model_entry.filename);

        // Load the model
        let model = Model::load(
            model_entry.label.clone(),
            model_entry.name.clone(),
            model_entry.size.clone(),
            model_entry.architecture.clone(),
            model_entry.quantization.clone(),
            model_path.clone(),
        )?;
        
        // Set the current model
        {
            let mut current_model = self.current_model.write().map_err(|e| e.to_string())?;
            *current_model = Some(model_entry.label.clone());
        }
        
        // Create a clone of the model for the inference context
        let model_clone = model.clone();
        
        // Set the loaded model
        {
            let mut loaded_model = self.loaded_model.write().map_err(|e| e.to_string())?;
            *loaded_model = Some(model);
        }
        
        // Create and set the inference context
        {
            let mut inference_context = self.inference_context.write().map_err(|e| e.to_string())?;
            *inference_context = Some(InferenceContext::new(model_clone, 2048)); // Default context size of 2048 tokens
        }
        
        // Return the model details
        Ok(ModelDetails {
            number: model_entry.number,
            label: model_entry.label,
            name: model_entry.name,
            size: model_entry.size,
            architecture: model_entry.architecture,
            quantization: model_entry.quantization,
            added_date: model_entry.added_date,
            tensor_count: model_entry.tensor_count,
            filename: model_entry.filename,
            directory: self.models_dir.to_string_lossy().to_string(),
        })
    }

    /// Checks if a model is currently attached.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether a model is attached
    pub fn is_model_attached(&self) -> bool {
        self.current_model.read().map(|m| m.is_some()).unwrap_or(false)
    }

    /// Generates text from the current model using the provided prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input text to generate a response for
    ///
    /// # Returns
    ///
    /// A Result containing the generated text or an error
    pub fn generate(&self, prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        // Check if a model is attached
        if !self.is_model_attached() {
            return Err("No model attached".into());
        }
        
        // Get a reference to the inference context
        let mut inference_context = self.inference_context.write().map_err(|e| e.to_string())?;
        
        // Check if we have an inference context
        let context = inference_context.as_mut()
            .ok_or_else(|| "No inference context available")?;
        
        // Process the input and generate a response
        context.process_input(prompt)
    }
}