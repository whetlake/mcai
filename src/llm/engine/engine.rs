use std::error::Error;
use std::sync::Arc;
use crate::llm::session::InferenceContext;
use crate::llm::model::{Model, ModelDetails};
use crate::llm::registry::ModelRegistry;
use crate::gguf::TensorInfo;
use std::sync::RwLock;
use crate::config::Settings;

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
            loaded_model: RwLock::new(None),
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
    pub fn attach_model(&self, model_number: usize) -> Result<ModelDetails, Box<dyn Error + Send + Sync>> {
        // Get the model entry from the registry
        let model_entry = self.model_registry.get_model_by_number(model_number)?;
        
        // Create the full path to the model file
        let model_path = self.model_registry.get_model_path(&model_entry.filename);

        // Load the model
        let model = Model::load(
            model_entry.label.clone(),
            model_entry.name.clone(),
            model_entry.size.clone(),
            model_entry.architecture.clone(),
            model_entry.quantization.clone(),
            model_path.clone(),
        )?;
        
        // Get metadata from the model
        let metadata: Vec<(String, String, String)> = model.gguf_reader()
            .metadata
            .iter()
            .map(|(key, (type_str, value))| (key.clone(), type_str.clone(), value.to_string()))
            .collect();

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
        
        // Create inference context with settings
        {
            let mut inference_context = self.inference_context.write().map_err(|e| e.to_string())?;
            *inference_context = Some(InferenceContext::new(model_clone, &self.settings)?);
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

    /// Generates text from the current model using the provided prompt. This is the function that gets the user
    /// input and generates the response via the inference context.
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
            let mut loaded_model = self.loaded_model.write().map_err(|e| e.to_string())?;
            *loaded_model = None;
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

        // Get the loaded model
        let loaded_model = self.loaded_model.read().map_err(|e| e.to_string())?;
        let model = loaded_model.as_ref().ok_or("No model loaded")?;

        // Get metadata from the model
        let metadata: Vec<(String, String, String)> = model.gguf_reader()
            .metadata
            .iter()
            .map(|(key, (type_str, value))| (key.clone(), type_str.clone(), value.to_string()))
            .collect();

        // Return the model details
        Ok(ModelDetails {
            number: None, // Not relevant for current model
            label: model.label.clone(),
            name: model.name.clone(),
            size: model.size.clone(),
            architecture: model.architecture.clone(),
            quantization: model.quantization.clone(),
            added_date: model.loaded_at,
            tensor_count: model.gguf_reader().tensor_count,
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

        // Get the loaded model
        let loaded_model = self.loaded_model.read().map_err(|e| e.to_string())?;
        let model = loaded_model.as_ref().ok_or("No model loaded")?;

        // Get tensor information
        Ok(model.gguf_reader().tensors.clone())
    }
}