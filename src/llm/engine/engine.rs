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

/// The core inference engine that manages model state and operations.
///
/// The InferenceEngine maintains thread-safe access to model state using RwLocks,
/// allowing multiple readers but exclusive writers. It provides methods for
/// scanning, loading, and interacting with models.
pub struct InferenceEngine {
    /// Currently selected model label (if any)
    pub current_model: RwLock<Option<String>>,
    /// Static GGUF metadata/tensor info for the active model
    pub active_metadata: RwLock<Option<Arc<ModelMetadata>>>,
    /// Active llama_cpp model instance used for inference
    pub active_model: RwLock<Option<Arc<LlamaModel>>>,
    /// Inference context for the current model
    pub inference_context: RwLock<Option<InferenceContext>>,
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
            current_model: RwLock::new(None),
            active_metadata: RwLock::new(None),
            active_model: RwLock::new(None),
            inference_context: RwLock::new(None),
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

        // Get the model entry from the registry. That is the model number in the registry.
        let model_entry = self.model_registry.get_model_by_number(model_number)?;
        
        // Create the full path to the model file where the model will be loaded from
        let model_path = self.model_registry.get_model_path(&model_entry.filename);

        // Load the model (using original GGUFReader based Model::load)
        let gguf_model = ModelMetadata::load(
            model_entry.label.clone(),
            model_entry.name.clone(),
            model_entry.size.clone(),
            model_entry.architecture.clone(),
            model_entry.quantization.clone(),
            model_path.clone(),
        )?;
        
        // Get metadata from the GGUF model for ModelDetails
        let metadata: Vec<(String, String, String)> = gguf_model.metadata
            .iter()
            .map(|(key, (type_str, value))| (key.clone(), type_str.clone(), value.to_string()))
            .collect();

        // Set the current model which is the label of the model
        {
            let mut current_model = self.current_model.write().map_err(|e| e.to_string())?;
            *current_model = Some(model_session_uuid.clone());
        }
        
        // Set the active metadata (GGUF info)
        let gguf_model_arc = Arc::new(gguf_model);
        {
            let mut active_metadata = self.active_metadata.write().map_err(|e| e.to_string())?;
            *active_metadata = Some(Arc::clone(&gguf_model_arc));
        }
        
        // Clone details needed for ModelDetails *before* gguf_model_arc might be moved/used elsewhere
        let model_size = gguf_model_arc.size.clone();
        let model_architecture = gguf_model_arc.architecture.clone();
        let model_tensor_count = gguf_model_arc.tensors.len() as u64;
        
        // Now, load the model again using llama_cpp for actual inference
        tracing::info!("Loading model via llama_cpp: {}", model_path.display());
        
        // Store parameters locally before creating LlamaParams
        let n_gpu_layers = self.settings.inference.n_gpu_layers;
        let use_mmap = self.settings.inference.use_mmap;
        let use_mlock = self.settings.inference.use_mlock;

        // Configure LlamaParams from local variables
        let llama_params = LlamaParams {
            n_gpu_layers, // Use local variable
            use_mmap,     // Use local variable
            use_mlock,    // Use local variable
            ..Default::default()
        };

        // Log the specific parameters being used for model loading (using tracing)
        tracing::info!(
            n_gpu_layers,
            use_mmap,
            use_mlock,
            "Attempting to load model with LlamaParams (tracing)"
        );
            
        let active_model = LlamaModel::load_from_file(&model_path, llama_params) // llama_params is moved here
            .map_err(|e| format!("Failed to load model with llama_cpp: {}", e))?;
        
        // Print confirmation AFTER successful load, using local variables
        println!(
            "---> [DEBUG] Model loaded via llama_cpp using LlamaParams: n_gpu_layers={}, use_mmap={}, use_mlock={}",
            n_gpu_layers, // Print local variable
            use_mmap,     // Print local variable
            use_mlock     // Print local variable
        );

        let active_model_arc = Arc::new(active_model);
        {
            let mut active_model = self.active_model.write().map_err(|e| e.to_string())?;
            *active_model = Some(Arc::clone(&active_model_arc));
            tracing::info!("Successfully loaded model via llama_cpp.");
        }
        
        // Create inference context with settings, passing the active llama_cpp model
        {
            let mut inference_context = self.inference_context.write().map_err(|e| e.to_string())?;
            *inference_context = Some(InferenceContext::new(active_model_arc, &self.settings)?);
        }
        
        // Return the model details (populated from gguf_model)
        Ok(ModelDetails {
            uuid: model_session_uuid.clone(),
            number: model_entry.number,
            user_label: user_label,
            name: model_entry.name,
            size: model_size,
            architecture: model_architecture,
            quantization: model_entry.quantization,
            added_date: model_entry.added_date,
            tensor_count: model_tensor_count,
            filename: model_entry.filename,
            directory: self.model_registry.models_dir.to_string_lossy().to_string(),
            metadata,
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
    /// Returns a stream of generated text chunks.
    pub async fn generate(&self, prompt: &str) 
        -> Result<Pin<Box<dyn Stream<Item = Result<String, Box<dyn Error + Send + Sync>>> + Send>>, Box<dyn Error + Send + Sync>>
    {
        if !self.is_model_attached() {
            return Err("No model attached".into());
        }
        
        // Get a read lock to access the Option<InferenceContext>
        let inference_context_guard = self.inference_context.read()
            .map_err(|e| format!("Failed to read lock inference_context: {}", e))?;
        
        // Get a reference to the context within the lock guard.
        if let Some(context) = inference_context_guard.as_ref() { 
            // Call process_input on the reference.
            // The returned stream is now 'static and doesn't borrow the guard.
            Ok(context.process_input(prompt.to_string())) 
        } else {
            Err("Inference context unexpectedly missing despite model being attachedso".into())
        }
        // Guard is dropped here, releasing the lock. Stream is independent.
    }

    /// Detaches the current model and clears all related state.
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure of the operation
    pub fn drop_model(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Check if a model is attached
        if !self.is_model_attached() {
            return Err("No model attached".into());
        }

        // Clear the current model and inference context
        {
            let mut current_model = self.current_model.write().map_err(|e| e.to_string())?;
            *current_model = None;
        }
        
        {
            let mut active_metadata = self.active_metadata.write().map_err(|e| e.to_string())?;
            *active_metadata = None;
        }
        
        {
            let mut active_model = self.active_model.write().map_err(|e| e.to_string())?;
            *active_model = None;
        }
        
        {
            let mut inference_context = self.inference_context.write().map_err(|e| e.to_string())?;
            *inference_context = None;
        }

        Ok(())
    }

    /// Gets metadata for the currently attached model.
    ///
    /// # Returns
    ///
    /// A Result containing the model details or an error if no model is attached
    pub fn get_metadata(&self) -> Result<ModelDetails, Box<dyn Error + Send + Sync>> {
        // Check if a model is attached
        if !self.is_model_attached() {
            return Err("No model attached".into());
        }

        // Get the active metadata
        let active_metadata_guard = self.active_metadata.read().map_err(|e| e.to_string())?;
        let model = active_metadata_guard.as_ref().ok_or("No model active metadata")?;

        // Get metadata from the GGUF model
        let metadata: Vec<(String, String, String)> = model.metadata
            .iter()
            .map(|(key, (type_str, value))| (key.clone(), type_str.clone(), value.to_string()))
            .collect();

        // Return the model details
        Ok(ModelDetails {
            uuid: "".to_string(), // Not relevant for current model, but the response expects it
            number: None, // Not relevant for current model
            user_label: Some(model.label.clone()),
            name: model.name.clone(),
            size: model.size.clone(),
            architecture: model.architecture.clone(),
            quantization: model.quantization.clone(),
            added_date: model.loaded_at,
            tensor_count: model.tensors.len() as u64,
            filename: model.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            directory: model.path.parent()
                .and_then(|p| p.to_str())
                .unwrap_or("")
                .to_string(),
            metadata,
        })
    }

    /// Gets tensor information for the currently attached model.
    ///
    /// # Returns
    ///
    /// A Result containing a vector of tensor information or an error if no model is attached
    pub fn get_tensors(&self) -> Result<Vec<TensorInfo>, Box<dyn Error + Send + Sync>> {
        // Check if a model is attached
        if !self.is_model_attached() {
            return Err("No model attached".into());
        }

        // Get the active metadata
        let active_metadata_guard = self.active_metadata.read().map_err(|e| e.to_string())?;
        let model = active_metadata_guard.as_ref().ok_or("No model active metadata")?;

        // Get tensor information
        Ok(model.tensors.clone())
    }
}