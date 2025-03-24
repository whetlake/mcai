use std::error::Error;
use std::sync::{Arc, Mutex};
use crate::llm::model::Model;
use crate::llm::backend::Backend;
use super::TensorCache;

/// Represents a transformer block in the model.
///
/// Each transformer block consists of:
/// 1. Attention layer with self-attention mechanism
/// 2. Feed-forward network
pub struct Transformer {
    /// Reference to the model
    model: Arc<Model>,
    /// Backend for tensor operations
    backend: Arc<Box<dyn Backend>>,
    /// Number of transformer blocks
    block_count: usize,
    /// Dimension of the hidden state
    hidden_dim: usize,
    /// Number of attention heads
    head_count: usize,
    /// Number of key-value attention heads (may differ from head_count in some architectures)
    head_count_kv: usize,
    /// Size of feed-forward layer
    feed_forward_dim: usize,
    /// Epsilon value for layer normalization
    layer_norm_epsilon: f32,
    /// Reference to the shared tensor cache
    tensor_cache: Arc<Mutex<TensorCache>>,
}

impl Transformer {
    /// Creates a new Transformer instance with model parameters
    pub fn new(
        model: Arc<Model>, 
        backend: Arc<Box<dyn Backend>>,
        block_count: usize,
        hidden_dim: usize,
        head_count: usize,
        head_count_kv: usize,
        feed_forward_dim: usize,
        layer_norm_epsilon: f32,
        tensor_cache: Arc<Mutex<TensorCache>>,
    ) -> Self {
        Self {
            model,
            backend,
            block_count,
            hidden_dim,
            head_count,
            head_count_kv,
            feed_forward_dim,
            layer_norm_epsilon,
            tensor_cache,
        }
    }
    
    /// Apply RMS normalization to a vector
    fn rms_norm(&self, input: &[f32], norm_weights: &[f32]) -> Vec<f32> {
        let hidden_dim = input.len();
        
        // Calculate sum of squares
        let mut sum_squares = 0.0;
        for &val in input {
            sum_squares += val * val;
        }
        
        // Calculate RMS
        let rms = (sum_squares / hidden_dim as f32 + self.layer_norm_epsilon).sqrt();
        
        // Apply normalization with weights
        let mut output = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            output[i] = (input[i] / rms) * norm_weights[i];
        }
        
        output
    }
    
    /// Process a sequence through all transformer blocks
    /// 
    /// # Arguments
    /// * `embeddings` - The input embeddings for the sequence of tokens
    /// 
    /// # Returns
    /// * `Vec<Vec<f32>>` - The processed hidden states
    pub fn forward(&self, embeddings: &[Vec<f32>]) 
        -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>>
    {
        println!("Transformer forward pass with {} embeddings", embeddings.len());
        
        // Get the tensor cache safely using the mutex
        let mut tensor_cache = self.tensor_cache.lock()
            .map_err(|e| format!("Failed to lock tensor cache: {}", e))?;
        
        // Start with the original embeddings
        let mut hidden_states = embeddings.to_vec();
        
        // Iterate through all transformer blocks
        for block_idx in 0..self.block_count {
            println!("Processing transformer block {}/{}", block_idx + 1, self.block_count);
            
            // Load the normalization weights for this block
            let attn_norm_tensor_name = format!("blk.{}.attn_norm.weight", block_idx);
            let attn_norm_data = tensor_cache.get_data(&attn_norm_tensor_name)?;
            
            // Apply normalization to each hidden state
            let mut normalized_states = Vec::with_capacity(hidden_states.len());
            for state in &hidden_states {
                let normalized = self.rms_norm(state, &attn_norm_data);
                normalized_states.push(normalized);
            }
            
            // For now, just use the normalized states as the output
            // In future steps, we'll add the attention and feed-forward layers
            hidden_states = normalized_states;
        }
        
        println!("Completed processing through {} transformer blocks", self.block_count);
        Ok(hidden_states)
    }
    
    /// Process a sequence through all transformer blocks using an external tensor loader
    /// 
    /// # Arguments
    /// * `embeddings` - The input embeddings for the sequence of tokens
    /// * `load_tensor` - A function to load tensors from cache or model
    /// 
    /// # Returns
    /// * `Vec<Vec<f32>>` - The processed hidden states
    pub fn forward_with_loader<F>(&self, embeddings: &[Vec<f32>], mut load_tensor: F) 
        -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>>
    where
        F: FnMut(&str) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>>,
    {
        println!("Transformer forward pass with {} embeddings (using external loader)", embeddings.len());
        
        // Start with the original embeddings
        let mut hidden_states = embeddings.to_vec();
        
        // Iterate through all transformer blocks
        for block_idx in 0..self.block_count {
            println!("Processing transformer block {}/{}", block_idx + 1, self.block_count);
            
            // Load the normalization weights for this block
            let attn_norm_tensor_name = format!("blk.{}.attn_norm.weight", block_idx);
            let attn_norm_data = load_tensor(&attn_norm_tensor_name)?;
            
            // Apply normalization to each hidden state
            let mut normalized_states = Vec::with_capacity(hidden_states.len());
            for state in &hidden_states {
                let normalized = self.rms_norm(state, &attn_norm_data);
                normalized_states.push(normalized);
            }
            
            // For now, just use the normalized states as the output
            // In future steps, we'll add the attention and feed-forward layers
            hidden_states = normalized_states;
        }
        
        println!("Completed processing through {} transformer blocks", self.block_count);
        Ok(hidden_states)
    }
}
