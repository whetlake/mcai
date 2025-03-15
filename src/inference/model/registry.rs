use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;
use std::thread;
use chrono::Utc;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{info, error};
use crate::gguf::{GGUFReader, GGUFError, is_gguf_file};
use crate::inference::model::types::{ModelEntry, ModelDetails};

/// Manages the model registry, including scanning for models and loading registry data.
pub struct ModelRegistry {
    /// Directory where model files are stored
    pub models_dir: PathBuf,
    /// Registry of all available models and their metadata
    pub registry: RwLock<HashMap<String, ModelEntry>>,
}

impl ModelRegistry {
    /// Creates a new model registry for the specified models directory.
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
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
    /// A Result indicating success or failure
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
        models.sort_by(|a: &(String, ModelEntry), b| b.1.added_date.cmp(&a.1.added_date));
        
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
                                file_type: reader.file_type,
                                quantization_version: reader.quantization_version,
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

    /// Gets a model entry by number.
    ///
    /// # Arguments
    ///
    /// * `model_number` - The number of the model to get
    ///
    /// # Returns
    ///
    /// A Result containing the model entry or an error
    pub fn get_model_by_number(&self, model_number: usize) -> Result<ModelEntry, Box<dyn Error + Send + Sync>> {
        // Get a read lock on the registry
        let registry = self.registry.read().map_err(|e| e.to_string())?;

        // Find the model with the given number
        let model_entry = registry.values()
            .find(|entry| entry.number == Some(model_number))
            .ok_or_else(|| format!("Model with number {} not found", model_number))?
            .clone();
        
        Ok(model_entry)
    }

    /// Gets the full path to a model file.
    ///
    /// # Arguments
    ///
    /// * `filename` - The filename of the model
    ///
    /// # Returns
    ///
    /// The full path to the model file
    pub fn get_model_path(&self, filename: &str) -> PathBuf {
        self.models_dir.join(filename)
    }

    /// Saves the registry to disk.
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure
    pub fn save_registry(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let registry = self.registry.read().map_err(|e| e.to_string())?;
        let registry_path = self.models_dir.join("model_registry.json");
        let content = serde_json::to_string_pretty(&*registry)?;
        fs::write(registry_path, content)?;
        Ok(())
    }
} 