use std::error::Error;
use std::sync::{Arc, Mutex};
use crate::llm::model::Model;
use crate::llm::backend::Backend;
use crate::llm::tensor::{Tensor, matmul, add};
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
    
    /// Process a sequence through all transformer blocks
    /// 
    /// # Arguments
    /// * `embeddings_tensor` - The input embeddings tensor with shape [seq_len, hidden_dim]
    /// 
    /// # Returns
    /// * `Result<Tensor, Box<dyn Error + Send + Sync>>` - The processed hidden states tensor
    pub fn forward(&self, embeddings_tensor: &Tensor) 
        -> Result<Tensor, Box<dyn Error + Send + Sync>>
    {
        // Check that the tensor has the expected shape [seq_len, hidden_dim]
        let shape = embeddings_tensor.shape();
        if shape.len() != 2 {
            return Err(format!("Expected embeddings tensor with 2 dimensions, got shape: {:?}", shape).into());
        }
        
        // Check that the hidden dimensions match the model's hidden dimension
        // in the model parameters
        let hidden_dim = shape[1];
        if hidden_dim != self.hidden_dim {
            return Err(format!("Expected hidden dimension {}, got {}", self.hidden_dim, hidden_dim).into());
        }
        println!("Transformer forward pass with embeddings tensor of shape {:?}", shape);
        
        // Get the tensor cache safely using the mutex to be able to
        // retrieve the relevant tensors from the cache
        let mut tensor_cache = self.tensor_cache.lock()
            .map_err(|e| format!("Failed to lock tensor cache: {}", e))?;
        
        // Handle the case where there are no blocks
        // if self.block_count == 0 {
        //     println!("No transformer blocks to process");
        //     return Ok(embeddings_tensor.clone());
        // }
        
        // For the first block, process the input tensor directly
        // for memory efficiency and so that we do not need to clone
        // the input tensor and can use embeddings_tensor directly
        let mut current_states = self.process_block(embeddings_tensor, 0, &mut tensor_cache)?;
        println!("Processed block 1/{}", self.block_count);
        
        // Process remaining blocks, using the output of each block as input to the next
        for block_idx in 1..self.block_count {
            let next_states = self.process_block(&current_states, block_idx, &mut tensor_cache)?;
            current_states = next_states;
            println!("Processed block {}/{}", block_idx + 1, self.block_count);
        }
        
        println!("Completed processing through {} transformer blocks", self.block_count);
        Ok(current_states)
    }
    
    /// Process a single transformer block
    fn process_block(&self, 
                     hidden_state: &Tensor, 
                     block_idx: usize,
                     tensor_cache: &mut TensorCache) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        println!("Processing transformer block {}/{}", block_idx + 1, self.block_count);
        
        // 1. Load the normalization weights for this block
        let attn_norm_tensor_name = format!("blk.{}.attn_norm.weight", block_idx);
        println!("  Loading tensor: {}", attn_norm_tensor_name);
        let attn_norm_tensor: &Tensor = tensor_cache.get(&attn_norm_tensor_name)
            .map_err(|e| format!("Failed to load attention norm weights for block {}: {}", block_idx, e))?;
        
        // 2. Apply RMS normalization
        println!("  Applying RMS normalization");
        let normalized_state = self.rms_norm(hidden_state, attn_norm_tensor)?;

        // 3. Apply the attention layer
        println!("  Applying attention layer");
        let query_projection = self.query_projection(&normalized_state, block_idx, tensor_cache)?;
        let key_projection = self.key_projection(&normalized_state, block_idx, tensor_cache)?;
        
        // Print a sample of the query projection tensor for debugging
        let shape = query_projection.shape();
        println!("  Query projection shape: {:?}", shape);
        
        // Get a small sample of the query projection tensor (first few values)
        let sample_size = 5.min(query_projection.data().len());
        let sample = &query_projection.data()[..sample_size];
        println!("  Query projection sample: {:?}", sample);
        
        // Calculate and print some statistics about the query projection
        if !sample.is_empty() {
            let min = sample.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = sample.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = sample.iter().sum();
            let avg = sum / sample.len() as f32;
            println!("  Query projection stats - min: {:.4}, max: {:.4}, avg: {:.4}", min, max, avg);
        }
        
        // For now, just return the normalized states
        println!("  Block {} processing complete", block_idx + 1);
        Ok(normalized_state)
    }

    /// Apply RMS normalization to a tensor
    ///
    /// RMS (Root Mean Square) normalization is a technique used in transformers
    /// to normalize the scale of embeddings while preserving their direction.
    /// 
    /// This function performs the following steps:
    /// 1. Calculate root mean square (RMS) for each row (token vector)
    /// 2. Normalize each value by dividing by (RMS + epsilon)
    /// 3. Scale by element-wise multiplication with norm_weights
    ///
    /// # Arguments
    /// * `input` - Embeddings tensor with shape [seq_len, hidden_dim]
    /// * `norm_weights` - Normalization weights tensor with shape [hidden_dim]
    ///
    /// # Returns
    /// * Result<Tensor, Box<dyn Error + Send + Sync>> - Normalized tensor
    fn rms_norm(&self, input_tensor: &Tensor, norm_weights: &Tensor) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        // Get shape information
        let shape = input_tensor.shape();
        let seq_len = shape[0];
        let hidden_dim = shape[1];
                
        // Create output tensor with same shape as the embeddings tensor
        let mut result = Tensor::zeros(shape.to_vec(), Arc::clone(&self.backend))?;
        
        // Use the backend's rms_norm implementation because it is optimized
        self.backend.rms_norm(
            input_tensor.data(),
            norm_weights.data(),
            result.data_mut(),
            seq_len,
            hidden_dim,
            self.layer_norm_epsilon,
        )?;
        
        Ok(result)
    }

    /// Project the normalized input tensor using query weights and bias
    ///
    /// This function performs the query projection step in the attention mechanism:
    /// Q = (X * W_q) + b_q
    /// where:
    /// - X is the normalized input tensor [seq_len, hidden_dim]
    /// - W_q is the query weight matrix [hidden_dim, hidden_dim]
    /// - b_q is the query bias vector [hidden_dim]
    ///
    /// # Arguments
    /// * `normalized_input` - The normalized input tensor with shape [seq_len, hidden_dim]
    /// * `block_idx` - The index of the current transformer block
    /// * `tensor_cache` - Reference to the tensor cache for loading weights
    ///
    /// # Returns
    /// * Result<Tensor, Box<dyn Error + Send + Sync>> - The projected query tensor
    fn query_projection(&self, 
                       normalized_input: &Tensor,
                       block_idx: usize,
                       tensor_cache: &mut TensorCache) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        // Load query weights and bias for this block
        let q_weight_name = format!("blk.{}.attn_q.weight", block_idx);
        let q_bias_name = format!("blk.{}.attn_q.bias", block_idx);
        
        println!("Query projection shapes:");
        println!("  normalized_input shape: {:?}", normalized_input.shape());
        
        // Load weights first
        let q_weight: &Tensor = tensor_cache.get(&q_weight_name)
            .map_err(|e: Box<dyn Error + Send + Sync>| format!("Failed to load query weights for block {}: {}", block_idx, e))?;
        println!("  q_weight shape: {:?}", q_weight.shape());
        
        // Perform matrix multiplication: X * W_q^T
        let mut query = matmul(normalized_input, q_weight, false, false)?;
        println!("  query after matmul shape: {:?}", query.shape());
        
        // Load bias and add it: (X * W_q^T) + b_q
        let q_bias: &Tensor = tensor_cache.get(&q_bias_name)
            .map_err(|e| format!("Failed to load query bias for block {}: {}", block_idx, e))?;
        println!("  q_bias shape: {:?}", q_bias.shape());
        
        query = add(&query, q_bias)?;
        println!("  final query shape: {:?}", query.shape());
        
        Ok(query)
    }

    /// Project the normalized input tensor using key weights and bias
    /// K = (X * W_k) + b_k
    fn key_projection(&self, 
                      normalized_input: &Tensor,
                      block_idx: usize,
                      tensor_cache: &mut TensorCache) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        // Load key weights and bias for this block
        let k_weight_name = format!("blk.{}.attn_k.weight", block_idx);
        let k_bias_name = format!("blk.{}.attn_k.bias", block_idx);
        
        println!("Key projection shapes:");
        println!("  normalized_input shape: {:?}", normalized_input.shape());
        
        // Load weights first
        let k_weight: &Tensor = tensor_cache.get(&k_weight_name)
            .map_err(|e: Box<dyn Error + Send + Sync>| format!("Failed to load key weights for block {}: {}", block_idx, e))?;
        println!("  k_weight shape: {:?}", k_weight.shape());
        
        // Perform matrix multiplication: X * W_k^T
        let mut key = matmul(normalized_input, k_weight, false, false)?;
        println!("  key after matmul shape: {:?}", key.shape());
        
        // Load bias and add it: (X * W_k^T) + b_k
        let k_bias: &Tensor = tensor_cache.get(&k_bias_name)
            .map_err(|e| format!("Failed to load key bias for block {}: {}", block_idx, e))?;
        println!("  k_bias shape: {:?}", k_bias.shape());
        
        key = add(&key, k_bias)?;
        println!("  final key shape: {:?}", key.shape());
        
        Ok(key)
    }
}
