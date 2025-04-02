use std::error::Error;
use std::sync::{Arc, Mutex};
use crate::llm::model::Model;
use crate::llm::backend::Backend;
use crate::llm::tensor::Tensor;
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
        let normalized_state = match self.rms_norm(hidden_state, attn_norm_tensor) {
            Ok(tensor) => tensor,
            Err(e) => {
                eprintln!("!!! Error returned directly from rms_norm call (block {}): {:?}", block_idx, e);
                return Err(e);
            }
        };

        // 3. Calculate Q, K, V projections using the unified function
        println!("  Applying attention layer projections");
        let mut query_projection = match self.projection(&normalized_state, block_idx, "q", tensor_cache) {
            Ok(tensor) => tensor,
            Err(e) => {
                eprintln!("!!! Error returned directly from projection('q') call (block {}): {:?}", block_idx, e);
                return Err(e);
            }
        };
        let mut key_projection = self.projection(&normalized_state, block_idx, "k", tensor_cache)?;
        let mut value_projection = self.projection(&normalized_state, block_idx, "v", tensor_cache)?;

        // Print head counts
        println!("  Number of Query Heads (head_count): {}", self.head_count);
        println!("  Number of KV Heads (head_count_kv): {}", self.head_count_kv);
        
        // --- Multi-Head Attention Steps ---
        println!("  Reshaping projections for multi-head attention");
        
        // Get sequence length
        let seq_len = normalized_state.shape()[0];

        // 1. Reshape Q in place
        let head_dim = self.hidden_dim / self.head_count;
        query_projection.reshape_in_place(vec![seq_len, self.head_count, head_dim])?;
        println!("    Query reshaped: {:?}", query_projection.shape());

        // 2. Reshape K and V in place - Note: they use head_count_kv
        let kv_dim = key_projection.shape()[1]; // Dimension of the K projection output
        if kv_dim % self.head_count_kv != 0 {
            return Err(format!("KV dimension ({}) is not divisible by KV head count ({})", kv_dim, self.head_count_kv).into());
        }
        let kv_head_dim = kv_dim / self.head_count_kv;

        key_projection.reshape_in_place(vec![seq_len, self.head_count_kv, kv_head_dim])?;
        value_projection.reshape_in_place(vec![seq_len, self.head_count_kv, kv_head_dim])?;
        println!("    Key reshaped:   {:?}", key_projection.shape());
        println!("    Value reshaped: {:?}", value_projection.shape());

        // 3. Apply RoPE to Q and K
        println!("  Applying RoPE to query and key projections");
        self.apply_rope_in_place(&mut query_projection)?;
        self.apply_rope_in_place(&mut key_projection)?;

        // 5. Permute Q, K, V for batched matrix multiplication
        // Reshape from [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        println!("  Permuting Q, K, V for attention calculation");
        let query_permuted = query_projection.permute(&[1, 0, 2])?;
        let key_permuted = key_projection.permute(&[1, 0, 2])?;
        let value_permuted = value_projection.permute(&[1, 0, 2])?; // V also needs permutation
        println!("    Query permuted: {:?}", query_permuted.shape());
        println!("    Key permuted:   {:?}", key_permuted.shape());
        println!("    Value permuted: {:?}", value_permuted.shape());

        // --- Attention Calculation ---

        // 6. Handle Grouped-Query Attention (Repeat K/V heads if necessary)
        println!("  Handling GQA (Repeating KV heads if needed)");
        let num_groups = self.head_count / self.head_count_kv;
        let key_repeated = self.backend.repeat_kv_heads(&key_permuted, num_groups)?;
        let value_repeated = self.backend.repeat_kv_heads(&value_permuted, num_groups)?;
        println!("    Key repeated:   {:?}", key_repeated.shape()); // Should be [head_count, seq_len, kv_head_dim]
        println!("    Value repeated: {:?}", value_repeated.shape()); // Should be [head_count, seq_len, kv_head_dim]

        // Ensure head dimensions match after potential repetition
        let head_dim = query_permuted.shape()[2];
        if key_repeated.shape()[2] != head_dim || value_repeated.shape()[2] != head_dim {
            return Err(format!("Head dimensions mismatch after KV repetition: Q[{}], K[{}], V[{}]",
                                head_dim, key_repeated.shape()[2], value_repeated.shape()[2]).into());
        }

        // 7. Calculate Attention Scores: Q @ Kᵀ (batch-wise)
        println!("  Calculating attention scores (Q @ K^T)");
        // Input Q: [head_count, seq_len, head_dim]
        // Input K: [head_count, seq_len, head_dim] -> Transposed K: [head_count, head_dim, seq_len]
        // Output Scores: [head_count, seq_len, seq_len]
        // Explicitly match the result instead of using '?'
        let mut attention_scores = match self.backend.bmm(
            &query_permuted,
            &key_repeated, // Use the (potentially) repeated K
            false,         // transpose_a = false
            true           // transpose_b = true (K -> Kᵀ)
        ) {
            Ok(tensor) => tensor, // If Ok, assign the tensor
            Err(e) => {
                // If Err, print the specific error and then return it
                eprintln!("!!! Error returned directly from bmm call: {:?}", e);
                return Err(e); 
            }
        };
        println!("    Attention scores shape: {:?}", attention_scores.shape());

        // 8. Scale Scores
        println!("  Scaling attention scores by 1/sqrt(head_dim)");
        let scale = (head_dim as f32).sqrt().recip(); // Calculate 1 / sqrt(head_dim)
        self.backend.scale(&mut attention_scores, scale)?;

        // 9. Apply Causal Attention Mask
        println!("  Applying causal attention mask");
        self.backend.apply_causal_mask(&mut attention_scores)?;

        // 10. Apply Softmax
        println!("  Applying softmax to scores");
        self.backend.softmax(&mut attention_scores)?;
        // attention_scores now holds probabilities
        let attention_probs = attention_scores; // Rename for clarity

        // 11. Multiply by Value: Output = Attention_Probs @ V
        println!("  Multiplying by Value");
        let weighted_values = self.backend.bmm(&attention_probs, &value_repeated, false, false)?;
        println!("    Weighted values shape: {:?}", weighted_values.shape());

        // 12. Combine Heads: Transpose and reshape back to [seq_len, hidden_dim]
        println!("  Combining heads");
        // Current shape: [head_count, seq_len, head_dim]
        // Final shape:   [seq_len, hidden_dim]
        let combined_heads = weighted_values.permute_and_reshape(
            &[1, 0, 2], // Permute axes: [H, S, D] -> [S, H, D]
            vec![seq_len, self.hidden_dim] // Target shape
        )?;
        println!("    Combined heads final shape: {:?}", combined_heads.shape());

        // (13. Final Projection - Placeholder)
        println!("  (Skipping final output projection for now)");

        // (14. Add Residual Connection - Placeholder)
        println!("  (Skipping residual connection for now)");

        // Return the combined heads tensor
        println!("  Block {} attention calculation steps 11 & 12 complete", block_idx + 1);
        // This tensor now has the shape [seq_len, hidden_dim], suitable for the next block
        Ok(combined_heads)
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
    pub(super) fn rms_norm(&self, input_tensor: &Tensor, norm_weights: &Tensor) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
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

        // Perform matrix multiplication: X * W using the backend's tensor method
        let proj_matmul_result = self.backend.matmul_tensors(normalized_input, weight, false, false)?;
        println!("  {} after matmul shape: {:?}", projection_type, proj_matmul_result.shape());

        // Load bias
        let bias: &Tensor = tensor_cache.get(&bias_name)
            .map_err(|e| format!("Failed to load {} bias for block {}: {}", projection_type, block_idx, e))?;
        println!("  {} bias shape: {:?}", projection_type, bias.shape());

        // Perform addition: (X * W) + b using the backend's tensor method
        let final_result = self.backend.add_tensors(&proj_matmul_result, bias)?;
        println!("  final {} shape: {:?}", projection_type, final_result.shape());

        Ok(final_result)
    }

    /// Applies Rotary Positional Embeddings (RoPE) to a tensor in place.
    /// Input tensor shape: [seq_len, num_heads, head_dim]
    fn apply_rope_in_place(&self, input: &mut Tensor) -> Result<(), Box<dyn Error + Send + Sync>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err("Input tensor for RoPE must have 3 dimensions [seq_len, num_heads, head_dim]".into());
        }
        let seq_len = shape[0];
        let num_heads = shape[1];
        let head_dim = shape[2];

        // Call the backend implementation
        self.backend.apply_rope(
            input.data_mut(),
            seq_len,
            num_heads,
            head_dim,
        )
    }
}
