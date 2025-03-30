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

        // 3. Calculate Q, K, V projections using the unified function
        println!("  Applying attention layer projections");
        let query_projection = self.projection(&normalized_state, block_idx, "q", tensor_cache)?;
        let key_projection = self.projection(&normalized_state, block_idx, "k", tensor_cache)?;
        let value_projection = self.projection(&normalized_state, block_idx, "v", tensor_cache)?;
        
        
        
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

    /// Generic projection function for Q, K, V
    fn projection(&self,
        normalized_input: &Tensor,
        block_idx: usize,
        projection_type: &str,
        tensor_cache: &mut TensorCache)
        -> Result<Tensor, Box<dyn Error + Send + Sync>>
    {
        // Load weights and bias for this block based on projection_type ("q", "k", or "v")
        let weight_name = format!("blk.{}.attn_{}.weight", block_idx, projection_type);
        let bias_name = format!("blk.{}.attn_{}.bias", block_idx, projection_type);
        
        println!("{} projection shapes:", projection_type.to_uppercase());
        println!("  normalized_input shape: {:?}", normalized_input.shape());
        
        // Load weights first
        let weight: &Tensor = tensor_cache.get(&weight_name)
            .map_err(|e: Box<dyn Error + Send + Sync>| format!("Failed to load {} weights for block {}: {}", projection_type, block_idx, e))?;
        println!("  {} weight shape: {:?}", projection_type, weight.shape());
        
        // Perform matrix multiplication: X * W
        let mut proj_result = matmul(normalized_input, weight, false, false)?;
        println!("  {} after matmul shape: {:?}", projection_type, proj_result.shape());
        
        // Load bias and add it: (X * W) + b
        let bias: &Tensor = tensor_cache.get(&bias_name)
            .map_err(|e| format!("Failed to load {} bias for block {}: {}", projection_type, block_idx, e))?;
        println!("  {} bias shape: {:?}", projection_type, bias.shape());
        
        proj_result = add(&proj_result, bias)?;
        println!("  final {} shape: {:?}", projection_type, proj_result.shape());
        
        Ok(proj_result)
    }
}
