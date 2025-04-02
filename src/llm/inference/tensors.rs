use std::error::Error;
use std::sync::Arc;
use std::collections::HashMap;
use crate::llm::model::Model;
use crate::llm::tensor::Tensor;
use crate::llm::backend::Backend;
use rayon::prelude::*; // Import Rayon traits

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
                            
        // Load the actual tensor data from the model's memory map
        let start_time = std::time::Instant::now();
        
        // Create the tensor directly from the tensor info and raw model data
        let tensor = Tensor::new(
            self.model.raw_data(),
            tensor_info,
            Arc::clone(&self.backend)
        )?;
        
        let _duration = start_time.elapsed();
        
        // Store in cache
        self.cache.insert(tensor_name.to_string(), tensor);
        
        Ok(())
    }
    
    /// Preload a list of tensors in parallel
    /// 
    /// # Arguments
    /// * `tensor_names` - List of tensor names to preload
    /// 
    /// # Returns
    /// * `(usize, usize)` - (success_count, total_count)
    pub fn preload(&mut self, tensor_names: &[&str]) -> (usize, usize) {
        let total_count = tensor_names.len();
        println!("Starting parallel preloading of {} tensors...", total_count);

        let results: Vec<_> = tensor_names
            .par_iter() // Use parallel iterator
            .map(|&name| {
                // --- Start: Parallelizable part (similar to load) ---
                let tensor_info_result = self.model.tensors.iter()
                    .find(|t| t.name == name)
                    .ok_or_else(|| format!("Tensor '{}' not found in model", name).into());

                let tensor_result: Result<Tensor, Box<dyn Error + Send + Sync>> = match tensor_info_result {
                    Ok(tensor_info) => {
                        Tensor::new(
                            self.model.raw_data(),
                            tensor_info,
                            Arc::clone(&self.backend)
                        )
                    }
                    Err(e) => Err(e),
                };
                // --- End: Parallelizable part ---
                
                (name.to_string(), tensor_result) // Return name and result
            })
            .collect(); // Collect results from parallel tasks

        // --- Start: Sequential insertion into cache ---
        let mut loaded_count = 0;
        let mut errors = Vec::new();
        for (name, result) in results {
            match result {
                Ok(tensor) => {
                    self.cache.insert(name, tensor);
                    loaded_count += 1;
                }
                Err(e) => {
                    errors.push((name, e));
                }
            }
        }
        // --- End: Sequential insertion ---

        if !errors.is_empty() {
            println!("Warnings during preloading:");
            for (name, e) in errors {
                println!("  - Failed to preload tensor '{}': {}", name, e);
            }
        }

        println!("Finished preloading. Success: {}/{}", loaded_count, total_count);
        (loaded_count, total_count)
    }
}
