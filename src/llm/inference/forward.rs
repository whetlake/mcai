use std::error::Error;
use std::sync::Arc;
use std::collections::HashMap;
use crate::llm::model::Model;
use crate::gguf::{TensorInfo, GGUFValueType};

/// Handles the forward pass of the neural network for token prediction
pub struct ForwardPass {
    /// Reference to the model
    model: Arc<Model>,
    /// Maximum context length (adjusted from settings)
    max_context_length: usize,
    /// Cache for frequently accessed tensors
    tensor_cache: HashMap<String, Vec<f32>>,
}

impl ForwardPass {
    /// Creates a new ForwardPass instance
    pub fn new(model: Arc<Model>, max_context_length: Option<usize>) -> Self {
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
        println!("Using max_context_length: {}", max_context_length);
        
        // Initialize empty tensor cache
        let tensor_cache = HashMap::new();
        
        let forward_pass = Self {
            model,
            max_context_length,
            tensor_cache,
        };
        
        // Print tensor summary
        
        forward_pass

    }


    
    
    /// Predicts the next token given the current context
    pub fn predict_next_token(&self, tokens: &[u32]) -> Result<u32, Box<dyn Error + Send + Sync>> {
        let next_token = 111;
        
        Ok(next_token)
    }
}
