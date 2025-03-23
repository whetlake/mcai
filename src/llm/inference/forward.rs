use std::error::Error;
use std::sync::Arc;
use std::collections::HashMap;
use crate::llm::model::Model;
use crate::llm::tensor::Tensor;
use crate::llm::backend::Backend;

/// Handles the forward pass of the neural network for token prediction
pub struct ForwardPass {
    /// Reference to the model
    model: Arc<Model>,
    /// Maximum context length (adjusted from settings)
    max_context_length: usize,
    /// Cache for frequently accessed tensors
    tensor_cache: HashMap<String, Tensor>,
    /// Backend for tensor operations
    backend: Arc<Box<dyn Backend>>,
}

impl ForwardPass {
    /// Creates a new ForwardPass instance
    pub fn new(model: Arc<Model>, backend: Arc<Box<dyn Backend>>, max_context_length: Option<usize>) -> Self {
        // Use provided max_context_length or fall back to model's training context length
        let max_context_length = max_context_length.unwrap_or(model.params.model_context_length);
        
        println!("Model parameters:");
        println!("  vocab_size: {}", model.params.vocab_size);
        println!("  hidden_dim: {}", model.params.hidden_dim);
        println!("  block_count: {}", model.params.block_count);
        println!("  head_count: {}", model.params.head_count);
        println!("  head_count_kv: {}", model.params.head_count_kv);
        println!("  model_context_length: {}", model.params.model_context_length);
        println!("  feed_forward_length: {}", model.params.feed_forward_length);
        println!("  layer_norm_rms_epsilon: {}", model.params.layer_norm_rms_epsilon);
        
        // Print detected block indexing information if available
        if let Some(index_start) = model.block_index_start {
            println!("  block_index_start: {}", index_start);
        }
        
        println!("Using max_context_length: {}", max_context_length);
        
        // Initialize empty tensor cache
        let tensor_cache = HashMap::new();
        
        let mut forward_pass = Self {
            model,
            max_context_length,
            tensor_cache,
            backend,
        };
        
        // Preload global tensors used in every forward pass
        forward_pass.preload_global_tensors();
        
        forward_pass
    }
    
    /// Preload global tensors that are used in every forward pass
    /// 
    /// While all model tensors are permanent (weights don't change during inference),
    /// some tensors are used in every forward pass (global), while others are only used
    /// when processing specific layers (layer-specific).
    /// 
    /// This method preloads the global tensors to improve performance.
    fn preload_global_tensors(&mut self) {
        println!("Preloading global tensors (used in every forward pass)...");
        
        // List of global tensors to preload
        let global_tensors = [
            // Token embedding table - needed for token embedding lookup (first step)
            "token_embd.weight",
            // Output normalization - used in the final layer (before logits)
            "output_norm.weight", 
            // Output projection - used for final logits (last step)
            "output.weight",  
        ];
        
        // Count of successfully loaded tensors
        let mut loaded_count = 0;
        
        for &tensor_name in &global_tensors {
            match self.load_tensor(tensor_name) {
                Ok(_) => {
                    loaded_count += 1;
                },
                Err(e) => {
                    println!("Warning: Failed to preload global tensor '{}': {}", tensor_name, e);
                }
            }
        }
        
        println!("Preloaded {}/{} global tensors", loaded_count, global_tensors.len());
    }
    
    /// Load a tensor from cache or from memory map
    fn load_tensor(&mut self, tensor_name: &str) -> Result<&Tensor, Box<dyn Error + Send + Sync>> {
        // Return from cache if already loaded
        if self.tensor_cache.contains_key(tensor_name) {
            return Ok(&self.tensor_cache[tensor_name]);
        }
        
        // Find the tensor info in the model
        let tensor_info = self.model.tensors.iter()
            .find(|t| t.name == tensor_name)
            .ok_or_else(|| format!("Tensor '{}' not found in model", tensor_name))?;
            
        // Print tensor information directly from TensorInfo
        println!("Loading tensor: {}", tensor_name);
        println!("  - Dimensions: {:?}", tensor_info.dims);
        println!("  - Type: {:?}", tensor_info.data_type);
        println!("  - Offset: {}", tensor_info.offset);
        
        // Calculate total elements from dimensions
        let total_elements: usize = tensor_info.dims.iter().map(|&d| d as usize).product();
        println!("  - Total elements: {}", total_elements);
        
        // Load the actual tensor data from the model's memory map
        println!("  - Reading tensor data from memory map...");
        let start_time = std::time::Instant::now();
        
        // Create the tensor directly from the tensor info and raw model data
        let tensor = Tensor::from_tensor_info(
            self.model.raw_data(),
            tensor_info,
            Arc::clone(&self.backend)
        )?;
        
        let duration = start_time.elapsed();
        println!("  - Loaded tensor in {:.2?}", duration);
        
        // Store in cache
        self.tensor_cache.insert(tensor_name.to_string(), tensor);
        
        Ok(&self.tensor_cache[tensor_name])
    }
    
    /// Convert token IDs to embeddings by looking up in the embedding table
    /// 
    /// This is the first step in the forward pass.
    /// 
    /// # Arguments
    /// * `tokens` - The input token IDs
    /// 
    /// # Returns
    /// * `Ok(Vec<Vec<f32>>)` - A sequence of embeddings, one for each token
    pub fn tokens_to_embeddings(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
        // First, limit sequence length to maximum context length
        let max_context_length = self.max_context_length;
        let hidden_dim = self.model.params.hidden_dim;
        let vocab_size = self.model.params.vocab_size;
        
        let tokens = if tokens.len() > max_context_length {
            &tokens[tokens.len() - max_context_length..]
        } else {
            tokens
        };
        
        println!("Converting {} tokens to embeddings", tokens.len());
        
        // Now load the embedding table
        let embedding_table = self.load_tensor("token_embd.weight")?;
        let embedding_data = embedding_table.data();
        
        // Lookup embeddings for each token
        let mut embeddings = Vec::with_capacity(tokens.len());
        
        for &token_id in tokens {
            let token_id = token_id as usize;
            
            // Check token ID is within range
            if token_id >= vocab_size {
                return Err(format!("Token ID {} is out of range (max: {})", 
                          token_id, vocab_size - 1).into());
            }
            
            // Extract the embedding for this token
            let start_idx = token_id * hidden_dim;
            let end_idx = start_idx + hidden_dim;
            
            // Copy the embedding from the embedding table
            let token_embedding = embedding_data[start_idx..end_idx].to_vec();
            embeddings.push(token_embedding);
        }
        
        Ok(embeddings)
    }
    
    /// Predicts the next token given the current context
    pub fn predict_next_token(&mut self, tokens: &[u32]) -> Result<u32, Box<dyn Error + Send + Sync>> {
        // Placeholder implementation
        // Will integrate tokens_to_embeddings in next step
        let next_token = 111;
        
        Ok(next_token)
    }
}
