use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use crate::llm::session::InferenceContext;
use crate::llm::model::modelmetadata::ModelMetadata;
use crate::llm::model::ModelDetails;
use crate::llm::registry::ModelRegistry;
use crate::gguf::TensorInfo;
use llama_cpp::{LlamaModel, LlamaParams};
use std::sync::RwLock;
use crate::config::Settings;
use std::pin::Pin;
use futures::stream::Stream;
use uuid::Uuid;
use super::names;
use serde::{Serialize, Deserialize};

// Simplified struct for attached model info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachedModelInfo {
    pub uuid: String,
    pub user_label: Option<String>,
    pub name: String, // Model's actual name
    pub number: usize, // Original registry number
}

pub(super) struct ModelInstanceState {
    #[allow(dead_code)]
    uuid: String,
    metadata: Arc<ModelMetadata>,
    #[allow(dead_code)]
    model: Arc<LlamaModel>,
    context: InferenceContext,
    user_label: Option<String>,
    model_number: usize,
}

/// The core inference engine that manages model state and operations.
///
/// The InferenceEngine maintains thread-safe access to model state using RwLocks,
/// allowing multiple readers but exclusive writers. It provides methods for
/// scanning, loading, and interacting with models.
pub struct InferenceEngine {
    active_models: RwLock<HashMap<String, ModelInstanceState>>,
    /// Model registry for managing available models
    pub model_registry: Arc<ModelRegistry>,
    /// Application settings
    pub settings: Settings,
}

impl InferenceEngine {
    /// Creates a new inference engine with the specified registry.
    ///
    /// # Arguments
    ///
    /// * `registry` - Model registry to use
    /// * `settings` - Application settings
    pub fn new(registry: Arc<ModelRegistry>, settings: Settings) -> Self {
        Self {
            active_models: RwLock::new(HashMap::new()), // Initialize with an empty hashmap
            model_registry: registry,
            settings,
        }
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
    pub fn attach_model(&self, model_number: usize, user_label: Option<String>) -> Result<ModelDetails, Box<dyn Error + Send + Sync>> {
        // Generate a UUID for the model session
        let model_session_uuid = Uuid::new_v4().to_string();

        // Get the model entry from the registry.
        let model_entry = self.model_registry.get_model_by_number(model_number)?;

        // Store the original registry number. Use 0 if somehow None (shouldn't happen).
        let original_model_number = model_entry.number.unwrap_or(0);

        // Determine the effective label: Use user input or generate a random name if None.
        let effective_label = user_label.clone().unwrap_or_else(|| 
            // Generate a random name if no label is provided
            names::generate_random_name()
        ); 

        // --- Check for label conflict using the helper --- 
        {
            let models_guard = self.active_models.read()
                .map_err(|e| format!("Failed to get read lock on active_models: {}", e))?;
            // Check availability, ignoring no UUID since it's a new attachment
            self.check_label_availability(&effective_label, &models_guard, None)?;
            // Read lock is released here
        }
        // --- End of label conflict check ---

        // Create the full path to the model file
        let model_path = self.model_registry.get_model_path(&model_entry.filename);

        // Load the GGUF metadata
        let gguf_model = ModelMetadata::load(
            model_entry.label.clone(), // Registry label (can differ from user label)
            model_entry.name.clone(),
            model_entry.size.clone(),
            model_entry.architecture.clone(),
            model_entry.quantization.clone(),
            model_path.clone(),
        )?;
        let gguf_model_arc = Arc::new(gguf_model);

        // Extract details needed later
        let model_size = gguf_model_arc.size.clone();
        let model_architecture = gguf_model_arc.architecture.clone();
        let model_tensor_count = gguf_model_arc.tensors.len() as u64;
        let metadata_map: Vec<(String, String, String)> = gguf_model_arc.metadata
            .iter()
            .map(|(key, (type_str, value))| (key.clone(), type_str.clone(), value.to_string()))
            .collect();

        // Now load the model using llama_cpp
        tracing::info!("Loading model via llama_cpp: {}", model_path.display());
        let n_gpu_layers = self.settings.inference.n_gpu_layers;
        let use_mmap = self.settings.inference.use_mmap;
        let use_mlock = self.settings.inference.use_mlock;
        let llama_params = LlamaParams { n_gpu_layers, use_mmap, use_mlock, ..Default::default() };
        tracing::info!(n_gpu_layers, use_mmap, use_mlock, "Attempting to load model with LlamaParams");
        let active_model = LlamaModel::load_from_file(&model_path, llama_params)
            .map_err(|e| format!("Failed to load model with llama_cpp: {}", e))?;
        println!("---> [DEBUG] Model loaded via llama_cpp using LlamaParams: n_gpu_layers={}, use_mmap={}, use_mlock={}", n_gpu_layers, use_mmap, use_mlock);
        let active_model_arc = Arc::new(active_model);
        tracing::info!("Successfully loaded model via llama_cpp.");

        // Create the inference context
        let inference_context = InferenceContext::new(Arc::clone(&active_model_arc), &self.settings)?;

        // Create the state bundle using the effective label
        let instance_state = ModelInstanceState {
            uuid: model_session_uuid.clone(),
            metadata: Arc::clone(&gguf_model_arc),
            model: active_model_arc,
            context: inference_context,
            // Store the effective label as the user_label for this instance
            user_label: Some(effective_label.clone()), // Use the checked effective_label
            model_number: original_model_number, // Set the stored number
        };

        // Add the new model instance state to the central map.
        {
            let mut models_guard = self.active_models.write().map_err(|e| format!("Failed to get write lock on active_models: {}", e))?;
            models_guard.insert(model_session_uuid.clone(), instance_state);
            tracing::info!("Inserted new model instance with UUID: {} and effective label: {}", model_session_uuid, effective_label);
        }

        // Return the model details, ensuring user_label reflects the effective label used
        Ok(ModelDetails {
            uuid: model_session_uuid.clone(),
            number: model_entry.number,
            // Return the effective_label in the response
            user_label: Some(effective_label), // Return the checked effective_label
            name: model_entry.name, // Base model name remains the same
            size: model_size,
            architecture: model_architecture,
            quantization: model_entry.quantization, // Use quantization from registry entry
            added_date: model_entry.added_date,
            tensor_count: model_tensor_count,
            filename: model_entry.filename,
            directory: self.model_registry.models_dir.to_string_lossy().to_string(),
            metadata: metadata_map,
        })
    }

    /// Checks if *any* model is currently attached.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether a model at least one model instance is attached.
    #[allow(dead_code)]
    pub fn is_model_attached(&self) -> bool {
        self.active_models.read() // Acquire a read lock on the active_models map.
            .map(|guard| !guard.is_empty()) // Check if the map is not empty
            .unwrap_or(false) // Return false if we fail to acquire the lock.
    }

    /// Generates text using a specific attached model instance.
    ///
    /// If a `model_session_uuid` is provided, uses that specific instance for generation.
    /// If `None` is provided, uses the sole model instance *only if* exactly one is attached.
    ///
    /// # Arguments
    ///
    /// * `model_session_uuid` - The optional unique UUID of the model instance to use.
    /// * `prompt` - The input prompt for text generation.
    ///
    /// # Returns
    ///
    /// A Result containing a tuple:
    ///   - `String`: The actual UUID of the model instance used for generation.
    ///   - `Pin<Box<dyn Stream...>>`: The stream of generated text chunks.
    /// Or an error if:
    /// - The specified UUID was not found.
    /// - `None` was provided, but zero or multiple models were attached.
    /// - An inference error occurs.
    pub async fn generate(&self, model_session_uuid: Option<&str>, prompt: &str)
        -> Result<(String, Pin<Box<dyn Stream<Item = Result<String, Box<dyn Error + Send + Sync>>> + Send>>), Box<dyn Error + Send + Sync>>
    {
        // Acquire a read lock on the active_models map.
        let models_guard = self.active_models.read()
             .map_err(|e| format!("Failed to get read lock on active_models: {}", e))?;

        // Use the helper function to find the UUID and the corresponding instance.
        let uuid_str = self.find_uuid_for_identifier(model_session_uuid, &models_guard)?;

        // Get the context using the found UUID. This should always succeed if find_uuid was Ok.
        let context = models_guard.get(uuid_str)
            .map(|instance| &instance.context)
            .ok_or("Internal error: Failed to retrieve context after finding UUID.")?;

        // Clone the UUID to return it, and start processing.
        Ok((uuid_str.to_string(), context.process_input(prompt.to_string())))
        // Read lock (models_guard) is released here.
    }

    /// Detaches a model instance.
    ///
    /// If a `model_session_uuid` is provided, detaches the specific instance.
    /// If `None` is provided, detaches the sole model instance *only if* exactly one is attached.
    ///
    /// # Arguments
    ///
    /// * `model_session_uuid` - The optional unique UUID of the model instance to detach.
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error if:
    /// - The specified UUID was not found.
    /// - `None` was provided, but zero or multiple models were attached.
    pub fn drop_model(&self, model_session_uuid: Option<&str>) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Acquire a write lock on the active_models map.
        let mut models_guard = self.active_models.write()
            .map_err(|e| format!("Failed to get write lock on active_models: {}", e))?;

        // Use the helper function (with the write lock downgraded for reading) to find the UUID.
        // Note: We need the UUID as a String here because the borrow checker won't let us
        // borrow from models_guard while also holding a mutable borrow later for removal.
        let uuid_to_remove = {
            // Create a temporary immutable borrow for the helper function
            let immutable_guard = &*models_guard;
            self.find_uuid_for_identifier(model_session_uuid, immutable_guard)?.to_string()
        };

        // Attempt to remove the model instance by its UUID
        if models_guard.remove(&uuid_to_remove).is_some() {
            tracing::info!("Successfully dropped model instance with UUID: {}", uuid_to_remove);
            // The ModelInstanceState goes out of scope after remove,
            // decrementing reference counts and potentially freeing resources.
            Ok(())
        } else {
            // This case should ideally not be reached if find_uuid_for_identifier succeeded,
            // but handle defensively.
            Err(format!("Internal error: Failed to remove model instance with UUID '{}' after finding it.", uuid_to_remove).into())
        }
        // Write lock is released when models_guard goes out of scope at the end of the function.
    }

    /// Gets metadata for a specific attached model instance.
    ///
    /// If a `model_session_uuid` is provided, gets details for that specific instance.
    /// If `None` is provided, gets details for the sole model instance *only if* exactly one is attached.
    ///
    /// # Arguments
    ///
    /// * `model_session_uuid` - The optional unique UUID of the model instance.
    ///
    /// # Returns
    ///
    /// A Result containing the model details or an error if:
    /// - The specified UUID was not found.
    /// - `None` was provided, but zero or multiple models were attached.
    pub fn get_metadata(&self, model_session_uuid: Option<&str>) -> Result<ModelDetails, Box<dyn Error + Send + Sync>> {
        // Acquire a read lock on the active_models map.
        let models_guard = self.active_models.read()
             .map_err(|e| format!("Failed to get read lock on active_models: {}", e))?;

        // Use the helper function to find the UUID.
        let uuid_str = self.find_uuid_for_identifier(model_session_uuid, &models_guard)?;

        // Get the instance using the found UUID. This should succeed.
        let instance = models_guard.get(uuid_str)
            .ok_or("Internal error: Failed to retrieve instance after finding UUID.")?;

        // Helper closure to construct ModelDetails (moved inside to access instance)
        let create_model_details = |uuid: &str, instance: &ModelInstanceState| -> ModelDetails {
            let model_metadata = &instance.metadata; // Borrow instance's metadata Arc
            let metadata_vec: Vec<(String, String, String)> = model_metadata.metadata
                .iter()
                .map(|(key, (type_str, value))| (key.clone(), type_str.clone(), value.to_string()))
                .collect();

            ModelDetails {
                uuid: uuid.to_string(), // Use the actual instance UUID
                number: None, // Registry number isn't stored in instance state, set to None
                user_label: instance.user_label.clone(), // Use the instance's user label
                name: model_metadata.name.clone(),
                size: model_metadata.size.clone(),
                architecture: model_metadata.architecture.clone(),
                quantization: model_metadata.quantization.clone(),
                added_date: model_metadata.loaded_at, // Use the loaded_at time from metadata
                tensor_count: model_metadata.tensors.len() as u64,
                filename: model_metadata.path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string(),
                directory: model_metadata.path.parent()
                    .and_then(|p| p.to_str())
                    .unwrap_or("")
                    .to_string(),
                metadata: metadata_vec,
            }
        };

        // Create the details using the found uuid and instance
        Ok(create_model_details(uuid_str, instance))
        // Read lock is released when models_guard goes out of scope.
    }

    /// Gets tensor information for a specific attached model instance.
    ///
    /// If a `model_session_uuid` is provided, gets tensor info for that specific instance.
    /// If `None` is provided, gets tensor info for the sole model instance *only if* exactly one is attached.
    ///
    /// # Arguments
    ///
    /// * `model_session_uuid` - The optional unique UUID of the model instance.
    ///
    /// # Returns
    ///
    /// A Result containing a vector of tensor information or an error if:
    /// - The specified UUID was not found.
    /// - `None` was provided, but zero or multiple models were attached.
    pub fn get_tensors(&self, model_session_uuid: Option<&str>) -> Result<Vec<TensorInfo>, Box<dyn Error + Send + Sync>> {
        // Acquire a read lock on the active_models map.
        let models_guard = self.active_models.read()
             .map_err(|e| format!("Failed to get read lock on active_models: {}", e))?;

        // Use the helper function to find the UUID.
        let uuid_str = self.find_uuid_for_identifier(model_session_uuid, &models_guard)?;

        // Get the instance using the found UUID. This should succeed.
        let instance = models_guard.get(uuid_str)
            .ok_or("Internal error: Failed to retrieve instance after finding UUID.")?;

        // Clone the tensor info from the instance's metadata
        Ok(instance.metadata.tensors.clone())
        // Read lock is released when models_guard goes out of scope.
    }

    /// Renames a specific attached model instance, or the sole instance if only one exists.
    ///
    /// # Arguments
    ///
    /// * `identifier` - The optional UUID or current user_label of the model instance to rename.
    ///                  If `None`, targets the sole attached model if exactly one exists.
    /// * `new_label` - The new user_label to assign to the instance.
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error if:
    /// - No model is attached.
    /// - An identifier was provided but not found.
    /// - An identifier matched multiple models by label (ambiguous).
    /// - `None` was provided, but zero or multiple models were attached.
    pub fn rename_model(&self, identifier: Option<&str>, new_label: String) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Acquire a write lock on the active_models map.
        let mut models_guard = self.active_models.write()
            .map_err(|e| format!("Failed to get write lock on active_models: {}", e))?;

        // 1. Find the UUID of the model to rename.
        let target_uuid = {
            let immutable_guard = &*models_guard;
            self.find_uuid_for_identifier(identifier, immutable_guard)?.to_string()
        };

        // 2. Check if the new label is available (ignoring the target model itself).
        {
            let immutable_guard = &*models_guard;
            self.check_label_availability(&new_label, immutable_guard, Some(&target_uuid))?;
        }

        // 3. Get mutable access to the target instance and perform the rename.
        if let Some(instance) = models_guard.get_mut(&target_uuid) {
            let old_label = instance.user_label.clone();
            instance.user_label = Some(new_label.clone());
            tracing::info!(
                "Relabeled model instance {} (Old Label: {:?}) to New Label: {}",
                target_uuid, // Use the cloned UUID for logging
                old_label,
                new_label
            );
            Ok(())
        } else {
            // This should be unreachable if find_uuid_for_identifier succeeded.
            Err("Internal error: Failed to get mutable reference to found model for renaming.".into())
        }
        // Write lock is released when models_guard goes out of scope.
    }

    /// Checks if a given label is already in use by another attached model instance.
    /// Optionally ignores a specific UUID during the check (useful for renaming).
    /// Requires a read lock on active_models to be held by the caller.
    fn check_label_availability(
        &self,
        label_to_check: &str,
        models_map: &HashMap<String, ModelInstanceState>,
        uuid_to_ignore: Option<&str> // UUID of the model being renamed/checked
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        for (uuid, instance) in models_map.iter() {
            // Skip the check if this instance is the one we're specifically ignoring
            if uuid_to_ignore.is_some() && uuid_to_ignore.unwrap() == uuid {
                continue;
            }

            if instance.user_label.as_deref() == Some(label_to_check) {
                return Err(format!(
                    "Label '{}' is already in use by model instance {}",
                    label_to_check,
                    uuid
                ).into());
            }
        }
        Ok(())
    }

    /// Helper to find a single, unambiguous UUID based on an optional identifier (UUID or user_label).
    /// This requires a read lock on active_models to be held by the caller.
    fn find_uuid_for_identifier<'a>(
        &self,
        identifier: Option<&'a str>,
        models_map: &'a HashMap<String, ModelInstanceState> // Pass the locked map
    ) -> Result<&'a str, Box<dyn Error + Send + Sync>> {

        if models_map.is_empty() {
            return Err("No models attached.".into());
        }

        match identifier {
            // Case 1: No identifier provided - find the sole model if applicable
            None => {
                match models_map.len() {
                    1 => {
                        // Exactly one model, return its UUID (key). Safe to unwrap.
                        // Need to return a reference with the lifetime of the map
                        Ok(models_map.keys().next().unwrap().as_str())
                    }
                    0 => Err("No models attached.".into()), // Should be caught earlier, but defensive
                    _ => Err("Multiple models attached. Please specify an identifier (UUID or label).".into()),
                }
            }
            // Case 2: Specific identifier provided
            Some(id) => {
                let mut matching_uuids = Vec::new();
                for (uuid, instance) in models_map.iter() {
                    // Check UUID first (should be unique)
                    if uuid == id {
                        return Ok(uuid.as_str()); // Found exact UUID match, return immediately
                    }
                    // Check label if UUID didn't match
                    if instance.user_label.as_deref() == Some(id) {
                        // Store a reference to the UUID string owned by the map
                        matching_uuids.push(uuid.as_str());
                    }
                }

                // Process label matches
                match matching_uuids.len() {
                    0 => Err(format!("No model instance found with identifier '{}'.", id).into()),
                    1 => Ok(matching_uuids[0]), // Exactly one label match
                    _ => Err(format!("Identifier '{}' is ambiguous and matches multiple attached models by label.", id).into()),
                }
            }
        }
    }

    /// Returns basic information about all currently attached model instances.
    pub fn get_attached_models(&self) -> Result<Vec<AttachedModelInfo>, Box<dyn Error + Send + Sync>> {
        let models_guard = self.active_models.read()
            .map_err(|e| format!("Failed to get read lock on active_models: {}", e))?;

        let attached_info: Vec<AttachedModelInfo> = models_guard
            .iter()
            .map(|(uuid, instance)| AttachedModelInfo {
                uuid: uuid.clone(),
                user_label: instance.user_label.clone(),
                name: instance.metadata.name.clone(), // Get name from metadata
                number: instance.model_number,      // Get stored registry number
            })
            .collect();

        Ok(attached_info)
    }
}