use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use crate::llm::model::Model;
use crate::llm::backend::Backend;
use crate::llm::tensor::Tensor;
use super::TensorCache;
use rayon::prelude::*; // Import Rayon traits


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
    tensor_cache: Arc<TensorCache>,
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
        tensor_cache: Arc<TensorCache>,
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
        
        // Handle the case where there are no blocks
        // if self.block_count == 0 {
        //     println!("No transformer blocks to process");
        //     return Ok(embeddings_tensor.clone());
        // }
        
        // For the first block, process the input tensor directly
        // for memory efficiency and so that we do not need to clone
        // the input tensor and can use embeddings_tensor directly
        let mut current_states = self.process_block(embeddings_tensor, 0, &*self.tensor_cache)?;
        println!("Processed block 1/{}", self.block_count);
        
        // Process remaining blocks, using the output of each block as input to the next
        for block_idx in 1..self.block_count {
            let next_states = self.process_block(&current_states, block_idx, &*self.tensor_cache)?;
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
                     tensor_cache: &TensorCache) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        
        // 1. Load the normalization weights for this block
        let start_load_attn_norm = Instant::now();
        let attn_norm_tensor_name = format!("blk.{}.attn_norm.weight", block_idx);
        let attn_norm_tensor_arc: Arc<Tensor> = tensor_cache.get(&attn_norm_tensor_name)
            .map_err(|e| format!("Failed to load attention norm weights for block {}: {}", block_idx, e))?;
        println!("  [Block {}] Load attn_norm: {:?}", block_idx + 1, start_load_attn_norm.elapsed());
        
        // 2. Apply RMS normalization
        let start_rms_norm_1 = Instant::now();
        let normalized_state = match self.rms_norm(hidden_state, &*attn_norm_tensor_arc) {
            Ok(tensor) => tensor,
            Err(e) => {
                eprintln!("!!! Error returned directly from rms_norm call (block {}): {:?}", block_idx, e);
                return Err(e);
            }
        };
        println!("  [Block {}] RMS Norm 1 (Attn): {:?}", block_idx + 1, start_rms_norm_1.elapsed());

        // --- QKV Projections (Sequential) ---
        // Reverted to sequential as parallel execution caused contention
        let start_qkv_proj = Instant::now();

        // Call projection sequentially
        let mut query_projection = self.projection(&normalized_state, block_idx, "q", tensor_cache)?;
        let mut key_projection = self.projection(&normalized_state, block_idx, "k", tensor_cache)?;
        let mut value_projection = self.projection(&normalized_state, block_idx, "v", tensor_cache)?;

        /* // Old parallel code (removed due to contention)
        let (q_result, (k_result, v_result)) = rayon::join(
            || self.projection(&normalized_state, block_idx, "q", tensor_cache),
            || rayon::join(
                 || self.projection(&normalized_state, block_idx, "k", tensor_cache),
                 || self.projection(&normalized_state, block_idx, "v", tensor_cache)
            )
        );
        let mut query_projection = q_result?;
        let mut key_projection = k_result?;
        let mut value_projection = v_result?;
        */

        println!("  [Block {}] QKV Projections (Sequential): {:?}", block_idx + 1, start_qkv_proj.elapsed());
        // --- End QKV Projections ---

        // Get sequence length
        let seq_len = normalized_state.shape()[0];

        // --- Start QKV Reshape ---
        let start_qkv_reshape = Instant::now();
        // 1. Reshape Q in place
        let head_dim = self.hidden_dim / self.head_count;
        query_projection.reshape_in_place(vec![seq_len, self.head_count, head_dim])?;

        // 2. Reshape K and V in place - Note: they use head_count_kv
        let kv_dim = key_projection.shape()[1]; // Dimension of the K projection output
        if kv_dim % self.head_count_kv != 0 {
            return Err(format!("KV dimension ({}) is not divisible by KV head count ({})", kv_dim, self.head_count_kv).into());
        }
        let kv_head_dim = kv_dim / self.head_count_kv;

        key_projection.reshape_in_place(vec![seq_len, self.head_count_kv, kv_head_dim])?;
        value_projection.reshape_in_place(vec![seq_len, self.head_count_kv, kv_head_dim])?;
        println!("  [Block {}] QKV Reshape: {:?}", block_idx + 1, start_qkv_reshape.elapsed());
        // --- End QKV Reshape ---

        // 3. Apply RoPE to Q and K
        let start_rope = Instant::now();
        self.apply_rope_in_place(&mut query_projection)?;
        self.apply_rope_in_place(&mut key_projection)?;
        println!("  [Block {}] RoPE Apply: {:?}", block_idx + 1, start_rope.elapsed());

        // 5. Permute Q, K, V for batched matrix multiplication
        let start_qkv_permute = Instant::now();
        // Reshape from [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        let query_permuted = query_projection.permute(&[1, 0, 2])?;
        let key_permuted = key_projection.permute(&[1, 0, 2])?;
        let value_permuted = value_projection.permute(&[1, 0, 2])?; // V also needs permutation
        println!("  [Block {}] QKV Permute: {:?}", block_idx + 1, start_qkv_permute.elapsed());

        // --- Attention Calculation ---

        // 6. Handle Grouped-Query Attention (Repeat K/V heads if necessary)
        let start_kv_repeat = Instant::now();
        let num_groups = self.head_count / self.head_count_kv;
        let key_repeated = self.backend.repeat_kv_heads(&key_permuted, num_groups)?;
        let value_repeated = self.backend.repeat_kv_heads(&value_permuted, num_groups)?;
        println!("  [Block {}] KV Repeat Heads: {:?}", block_idx + 1, start_kv_repeat.elapsed());

        // Ensure head dimensions match after potential repetition
        // (Shape checks are cheap, not timing)
        let head_dim = query_permuted.shape()[2];
        if key_repeated.shape()[2] != head_dim || value_repeated.shape()[2] != head_dim {
            return Err(format!("Head dimensions mismatch after KV repetition: Q[{}], K[{}], V[{}]",
                                head_dim, key_repeated.shape()[2], value_repeated.shape()[2]).into());
        }

        // 7. Calculate Attention Scores: Q @ Kᵀ (batch-wise)
        let start_attn_scores = Instant::now();
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
        println!("  [Block {}] Attention Scores (BMM1): {:?}", block_idx + 1, start_attn_scores.elapsed());

        // 8. Scale Scores
        let start_scale = Instant::now();
        let scale = (head_dim as f32).sqrt().recip(); // Calculate 1 / sqrt(head_dim)
        self.backend.scale(&mut attention_scores, scale)?;
        println!("  [Block {}] Scale Scores: {:?}", block_idx + 1, start_scale.elapsed());

        // 9. Apply Causal Attention Mask
        let start_mask = Instant::now();
        self.backend.apply_causal_mask(&mut attention_scores)?;
        println!("  [Block {}] Apply Mask: {:?}", block_idx + 1, start_mask.elapsed());

        // 10. Apply Softmax
        let start_softmax = Instant::now();
        self.backend.softmax(&mut attention_scores)?;
        // attention_scores now holds probabilities
        let attention_probs = attention_scores; // Rename for clarity
        println!("  [Block {}] Softmax: {:?}", block_idx + 1, start_softmax.elapsed());

        // 11. Multiply by Value: Output = Attention_Probs @ V
        let start_weighted_values = Instant::now();
        let weighted_values = self.backend.bmm(&attention_probs, &value_repeated, false, false)?;
        println!("  [Block {}] Weighted Values (BMM2): {:?}", block_idx + 1, start_weighted_values.elapsed());

        // 12. Combine Heads: Transpose and reshape back to [seq_len, hidden_dim]
        let start_combine_heads = Instant::now();
        let combined_heads = weighted_values.permute_and_reshape(
            &[1, 0, 2], // Permute axes: [H, S, D] -> [S, H, D]
            vec![seq_len, self.hidden_dim] // Target shape
        )?;
        println!("  [Block {}] Combine Heads (Permute/Reshape): {:?}", block_idx + 1, start_combine_heads.elapsed());

        // 13. Attention Output Projection
        let start_attn_output_proj_load = Instant::now();
        println!("  Applying attention output projection");
        let attn_output_weight_name = format!("blk.{}.attn_output.weight", block_idx);
        let attn_output_weight_arc = tensor_cache.get(&attn_output_weight_name)
            .map_err(|e| format!("Failed to load attention output weights for block {}: {}", block_idx, e))?;
        println!("  [Block {}] Load attn_output_weight: {:?}", block_idx + 1, start_attn_output_proj_load.elapsed());

        let start_attn_output_proj_matmul = Instant::now();
        // Perform the projection: attention_output = combined_heads @ attn_output_weight
        let attention_output = self.backend.matmul_tensors(
            &combined_heads,
            &*attn_output_weight_arc, // Dereference Arc
            false, // Do not transpose combined_heads (A)
            false  // Do not transpose attn_output_weight (B)
        )?;
        println!("  [Block {}] Attention Output Projection (MatMul): {:?}", block_idx + 1, start_attn_output_proj_matmul.elapsed());

        // 14. Add First Residual Connection
        let start_residual_1 = Instant::now();
        // Add the original input of the block (before normalization) to the attention output
        let residual_after_attn: Tensor = self.backend.add_tensors(hidden_state, &attention_output)?;
        println!("  [Block {}] Add Residual 1: {:?}", block_idx + 1, start_residual_1.elapsed());

        // --- Feed-Forward Network (FFN) ---

        // 15. Normalize before FFN
        let start_ffn_norm_load = Instant::now();
        let ffn_norm_weight_name = format!("blk.{}.ffn_norm.weight", block_idx);
        let ffn_norm_weight_arc = tensor_cache.get(&ffn_norm_weight_name)
            .map_err(|e| format!("Failed to load FFN norm weights for block {}: {}", block_idx, e))?;
        println!("  [Block {}] Load ffn_norm_weight: {:?}", block_idx + 1, start_ffn_norm_load.elapsed());

        let start_rms_norm_2 = Instant::now();
        // Apply RMSNorm using the weights specific to the FFN input
        let normalized_for_ffn = self.rms_norm(&residual_after_attn, &*ffn_norm_weight_arc)?;
        println!("  [Block {}] RMS Norm 2 (FFN): {:?}", block_idx + 1, start_rms_norm_2.elapsed());

        // --- FFN Calculations Step 16 ---

        // 16a. Gate Projection (W_gate * x)
        let start_ffn_gate_proj_load = Instant::now();
        println!("  Calculating FFN gate projection");
        let ffn_gate_weight_name = format!("blk.{}.ffn_gate.weight", block_idx);
        let ffn_gate_weight_arc = tensor_cache.get(&ffn_gate_weight_name)
            .map_err(|e| format!("Failed to load FFN gate weights for block {}: {}", block_idx, e))?;
        println!("  [Block {}] Load ffn_gate_weight: {:?}", block_idx + 1, start_ffn_gate_proj_load.elapsed());

        let start_ffn_gate_proj_matmul = Instant::now();
        let gate_proj = self.backend.matmul_tensors(&normalized_for_ffn, &*ffn_gate_weight_arc, false, false)?;
        println!("    FFN Gate projection shape: {:?}", gate_proj.shape()); // Optional debug print
        println!("  [Block {}] FFN Gate Projection (MatMul): {:?}", block_idx + 1, start_ffn_gate_proj_matmul.elapsed());

        // 16b. Up Projection (W_up * x)
        let start_ffn_up_proj_load = Instant::now();
        println!("  Calculating FFN up projection");
        let ffn_up_weight_name = format!("blk.{}.ffn_up.weight", block_idx);
        let ffn_up_weight_arc = tensor_cache.get(&ffn_up_weight_name)
            .map_err(|e| format!("Failed to load FFN up weights for block {}: {}", block_idx, e))?;
        println!("  [Block {}] Load ffn_up_weight: {:?}", block_idx + 1, start_ffn_up_proj_load.elapsed());

        let start_ffn_up_proj_matmul = Instant::now();
        let up_proj = self.backend.matmul_tensors(&normalized_for_ffn, &*ffn_up_weight_arc, false, false)?;
        println!("    FFN Up projection shape: {:?}", up_proj.shape()); // Optional debug print
        println!("  [Block {}] FFN Up Projection (MatMul): {:?}", block_idx + 1, start_ffn_up_proj_matmul.elapsed());

        // 16c. Activation and Gating (SwiGLU): gate_proj * SiLU(up_proj)
        println!("  Applying SwiGLU activation: gate_proj * SiLU(up_proj)");

        // Apply SiLU activation directly to up_proj in place using the backend.
        let start_silu = Instant::now();
        let mut activated_up_proj = up_proj; // Rename for clarity, make mutable
        self.backend.silu(&mut activated_up_proj)?;
        println!("  [Block {}] SiLU Activation (In-Place): {:?}", block_idx + 1, start_silu.elapsed());

        // Perform element-wise multiplication: ffn_intermediate = gate_proj * activated_up_proj
        let start_gating_mul = Instant::now();
        let ffn_intermediate = self.backend.mul_tensors(&gate_proj, &activated_up_proj)?;
        println!("  [Block {}] FFN Gating (Mul): {:?}", block_idx + 1, start_gating_mul.elapsed());

        println!("    FFN intermediate result shape after SwiGLU: {:?}", ffn_intermediate.shape());

        // 16d. Down Projection: result * W_down
        let start_ffn_down_proj_load = Instant::now();
        println!("  Applying FFN down projection");
        let ffn_down_weight_name = format!("blk.{}.ffn_down.weight", block_idx);
        let ffn_down_weight_arc = tensor_cache.get(&ffn_down_weight_name)
            .map_err(|e| format!("Failed to load FFN down weights for block {}: {}", block_idx, e))?;
        println!("  [Block {}] Load ffn_down_weight: {:?}", block_idx + 1, start_ffn_down_proj_load.elapsed());
            
        // Perform the down projection: ffn_output = ffn_intermediate @ ffn_down_weight
        let start_ffn_down_proj_matmul = Instant::now(); // Start timer *before* matmul
        let ffn_output = self.backend.matmul_tensors(
            &ffn_intermediate, 
            &*ffn_down_weight_arc, // Dereference Arc
            false, // Do not transpose ffn_intermediate (A)
            false  // Do not transpose ffn_down_weight (B)
        )?;
        println!("  [Block {}] FFN Down Projection (MatMul): {:?}", block_idx + 1, start_ffn_down_proj_matmul.elapsed()); // End timer *after* matmul
        println!("    FFN output shape after down projection: {:?}", ffn_output.shape());

        // 17. Add Second Residual Connection
        let start_residual_2 = Instant::now();
        // Add the input that went into the FFN (residual_after_attn) 
        // to the output of the FFN (ffn_output)
        println!("  Adding second residual connection");
        let final_block_output = self.backend.add_tensors(&residual_after_attn, &ffn_output)?;
        println!("  [Block {}] Add Residual 2: {:?}", block_idx + 1, start_residual_2.elapsed());

        // Return the final output of this transformer block
        println!("  Block {} processing complete.", block_idx + 1);
        Ok(final_block_output)
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
        tensor_cache: &TensorCache)
        -> Result<Tensor, Box<dyn Error + Send + Sync>>
    {
        // Load weights and bias for this block based on projection_type ("q", "k", or "v")
        let weight_name = format!("blk.{}.attn_{}.weight", block_idx, projection_type);
        let bias_name = format!("blk.{}.attn_{}.bias", block_idx, projection_type);

        // Load weights first using the updated cache.get() which returns Arc<Tensor>
        let weight_arc: Arc<Tensor> = tensor_cache.get(&weight_name)
            .map_err(|e: Box<dyn Error + Send + Sync>| format!("Failed to load {} weights for block {}: {}", projection_type, block_idx, e))?;

        // Perform matrix multiplication: X * W using the backend's tensor method
        // Dereference the Arc to get the underlying &Tensor
        let proj_matmul_result = self.backend.matmul_tensors(normalized_input, &*weight_arc, false, false)?;

        // Load bias using the updated cache.get()
        let bias_arc: Arc<Tensor> = tensor_cache.get(&bias_name)
            .map_err(|e| format!("Failed to load {} bias for block {}: {}", projection_type, block_idx, e))?;

        // Perform addition: (X * W) + b using the backend's tensor method
        // Dereference the Arc to get the underlying &Tensor
        let final_result = self.backend.add_tensors(&proj_matmul_result, &*bias_arc)?;

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
