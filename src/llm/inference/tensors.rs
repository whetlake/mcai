use std::error::Error;
use std::sync::{Arc, Mutex};
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
    /// Cache storage for loaded tensors, protected by Mutex and using Arc for shared ownership
    cache: Mutex<HashMap<String, Arc<Tensor>>>,
}

impl TensorCache {
    /// Create a new tensor cache
    pub fn new(model: Arc<Model>, backend: Arc<Box<dyn Backend>>) -> Self {
        Self {
            model,
            backend,
            cache: Mutex::new(HashMap::new()), // Initialize Mutex
        }
    }
    
    /// Get a tensor by name, loading it if not in cache.
    /// Returns an Arc<Tensor> for shared ownership.
    /// Takes &self due to interior mutability.
    /// 
    /// # Arguments
    /// * `tensor_name` - Name of the tensor to get
    /// 
    /// # Returns
    /// * `Result<Arc<Tensor>, Box<dyn Error + Send + Sync>>` - Arc reference to the tensor
    pub fn get(&self, tensor_name: &str) -> Result<Arc<Tensor>, Box<dyn Error + Send + Sync>> {
        // First, try reading the cache without locking write access for long
        {
            let cache_guard = self.cache.lock()
                .map_err(|e| format!("Failed to acquire read lock on tensor cache: {}", e))?;
            if let Some(tensor_arc) = cache_guard.get(tensor_name) {
                return Ok(Arc::clone(tensor_arc)); // Return cloned Arc if found
            }
        } // Read lock released here

        // If not found, load it (potentially slow)
        // This load function will handle acquiring the write lock internally
        let tensor_arc = self.load(tensor_name)?; 
        Ok(tensor_arc)
    }
    
    /// Get tensor data by name, loading the tensor if not in cache
    /// 
    /// # Arguments
    /// * `tensor_name` - Name of the tensor to get data for
    /// 
    /// # Returns
    /// * `Result<Vec<f32>, Box<dyn Error + Send + Sync>>` - Tensor data as a vector
    pub fn get_data(&self, tensor_name: &str) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        // Get the tensor Arc
        let tensor_arc = self.get(tensor_name)?;
        
        // Return a copy of its data from the underlying tensor
        Ok(tensor_arc.data().to_vec())
    }
    
    /// Load a tensor into the cache (helper function, acquires write lock)
    /// Takes &self due to interior mutability.
    /// 
    /// # Arguments
    /// * `tensor_name` - Name of the tensor to load
    /// 
    /// # Returns
    /// * `Result<Arc<Tensor>, Box<dyn Error + Send + Sync>>` - Arc reference to the newly loaded tensor
    fn load(&self, tensor_name: &str) -> Result<Arc<Tensor>, Box<dyn Error + Send + Sync>> {
        // Acquire write lock once before checking and potentially loading
        let mut cache_guard = self.cache.lock()
            .map_err(|e| format!("Failed to acquire write lock on tensor cache for loading: {}", e))?;

        // Double-check if another thread loaded it while waiting for the lock
        if let Some(tensor_arc) = cache_guard.get(tensor_name) {
            return Ok(Arc::clone(tensor_arc));
        }

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
        
        let _duration = start_time.elapsed(); // Can keep for debugging if needed

        // Wrap in Arc
        let tensor_arc = Arc::new(tensor);
        
        // Store Arc in cache (still holding the write lock)
        cache_guard.insert(tensor_name.to_string(), Arc::clone(&tensor_arc));
        
        Ok(tensor_arc) // Return the newly created Arc
    }
    
    /// Preload a list of tensors in parallel
    /// Takes &self due to interior mutability.
    /// 
    /// # Arguments
    /// * `tensor_names` - List of tensor names to preload
    /// 
    /// # Returns
    /// * `(usize, usize)` - (success_count, total_count)
    pub fn preload(&self, tensor_names: &[&str]) -> (usize, usize) {
        let total_count = tensor_names.len();
        println!("Starting parallel preloading of {} tensors...", total_count);

        let results: Vec<_> = tensor_names
            .par_iter() // Use parallel iterator
            .map(|&name| {
                // --- Start: Parallelizable part (finding info + creating Tensor) ---
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
                
                // Return name and Result<Tensor> (not Arc yet)
                (name.to_string(), tensor_result) 
            })
            .collect(); // Collect results from parallel tasks

        // --- Start: Sequential insertion into cache (requires write lock) ---
        let mut loaded_count = 0;
        let mut errors = Vec::new();
        { // Scope for the lock guard
            let mut cache_guard = self.cache.lock()
                .expect("Failed to acquire write lock for preload insertion");

            for (name, result) in results {
                match result {
                    Ok(tensor) => {
                        // Insert Arc<Tensor> into cache
                        cache_guard.entry(name.clone()).or_insert_with(|| Arc::new(tensor));
                        loaded_count += 1;
                    }
                    Err(e) => {
                        errors.push((name, e));
                    }
                }
            }
        } // Lock released here
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
