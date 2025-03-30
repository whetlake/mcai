use std::error::Error;
use std::sync::Arc;
use std::collections::HashMap;
use crate::llm::model::Model;
use crate::llm::tensor::Tensor;
use crate::llm::backend::Backend;

/// A cache for model tensors with utilities for loading and retrieving
pub struct TensorCache {
    /// Reference to the model
    model: Arc<Model>,
    /// Backend for tensor operations
    backend: Arc<Box<dyn Backend>>,
    /// Cache storage for loaded tensors
    cache: HashMap<String, Tensor>,
}

impl TensorCache {
    /// Create a new tensor cache
    pub fn new(model: Arc<Model>, backend: Arc<Box<dyn Backend>>) -> Self {
        Self {
            model,
            backend,
            cache: HashMap::new(),
        }
    }
    
    /// Get a tensor by name, loading it if not in cache
    /// 
    /// # Arguments
    /// * `tensor_name` - Name of the tensor to get
    /// 
    /// # Returns
    /// * `Result<&Tensor, Box<dyn Error + Send + Sync>>` - Reference to the tensor
    pub fn get(&mut self, tensor_name: &str) -> Result<&Tensor, Box<dyn Error + Send + Sync>> {
        // If not in cache, load it
        if !self.cache.contains_key(tensor_name) {
            self.load(tensor_name)?;
        }
        
        // Now it should be in the cache
        Ok(&self.cache[tensor_name])
    }
    
    /// Get tensor data by name, loading the tensor if not in cache
    /// 
    /// # Arguments
    /// * `tensor_name` - Name of the tensor to get data for
    /// 
    /// # Returns
    /// * `Result<Vec<f32>, Box<dyn Error + Send + Sync>>` - Tensor data as a vector
    pub fn get_data(&mut self, tensor_name: &str) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        // Get the tensor
        let tensor = self.get(tensor_name)?;
        
        // Return a copy of its data
        Ok(tensor.data().to_vec())
    }
    
    /// Load a tensor into the cache
    /// 
    /// # Arguments
    /// * `tensor_name` - Name of the tensor to load
    /// 
    /// # Returns
    /// * `Result<(), Box<dyn Error + Send + Sync>>` - Success or error
    pub fn load(&mut self, tensor_name: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Find the tensor info in the model
        let tensor_info = self.model.tensors.iter()
            .find(|t| t.name == tensor_name)
            .ok_or_else(|| format!("Tensor '{}' not found in model", tensor_name))?;
            
        // Print tensor information
        println!("Loading tensor: {}", tensor_name);
        println!("  - Dimensions: {:?}", tensor_info.dims);
        println!("  - Type: {:?}", tensor_info.data_type);
                
        // Load the actual tensor data from the model's memory map
        let start_time = std::time::Instant::now();
        
        // Create the tensor directly from the tensor info and raw model data
        let tensor = Tensor::new(
            self.model.raw_data(),
            tensor_info,
            Arc::clone(&self.backend)
        )?;
        
        let duration = start_time.elapsed();
        println!("  - Loaded tensor in {:.2?}", duration);
        
        // Store in cache
        self.cache.insert(tensor_name.to_string(), tensor);
        
        Ok(())
    }
    
    /// Preload a list of tensors
    /// 
    /// # Arguments
    /// * `tensor_names` - List of tensor names to preload
    /// 
    /// # Returns
    /// * `(usize, usize)` - (success_count, total_count)
    pub fn preload(&mut self, tensor_names: &[&str]) -> (usize, usize) {
        let mut loaded_count = 0;
        
        for &tensor_name in tensor_names {
            match self.load(tensor_name) {
                Ok(_) => {
                    loaded_count += 1;
                },
                Err(e) => {
                    println!("Warning: Failed to preload tensor '{}': {}", tensor_name, e);
                }
            }
        }
        
        println!("Preloaded {}/{} tensors", loaded_count, tensor_names.len());
        (loaded_count, tensor_names.len())
    }
}
