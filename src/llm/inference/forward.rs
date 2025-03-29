use std::error::Error;
use std::sync::{Arc, Mutex};
use crate::llm::model::Model;
use crate::llm::backend::Backend;
use super::Transformer;
use super::TensorCache;
use crate::llm::tensor::Tensor;

/// Handles the forward pass of the neural network for token prediction
pub struct ForwardPass {
    /// Reference to the model
    model: Arc<Model>,
    /// Maximum context length (adjusted from settings)
    max_context_length: usize,
    /// Tensor cache for loading and storing tensors
    tensor_cache: Arc<Mutex<TensorCache>>,
    /// Backend for tensor operations
    backend: Arc<Box<dyn Backend>>,
    /// Transformer for processing through layers
    transformer: Transformer,
}

// Constants for tensor names
const TOKEN_EMBEDDING_TENSOR: &str = "token_embd.weight";
const OUTPUT_NORM_TENSOR: &str = "output_norm.weight";
const OUTPUT_TENSOR: &str = "output.weight";

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
        
        // Initialize tensor cache and wrap it in Arc<Mutex<_>>
        let tensor_cache = TensorCache::new(Arc::clone(&model), Arc::clone(&backend));
        let shared_cache = Arc::new(Mutex::new(tensor_cache));
        
        // Create transformer with shared tensor cache
        let transformer = Transformer::new(
            Arc::clone(&model),
            Arc::clone(&backend),
            model.params.block_count,
            model.params.hidden_dim,
            model.params.head_count,
            model.params.head_count_kv,
            model.params.feed_forward_length,
            model.params.layer_norm_rms_epsilon,
            Arc::clone(&shared_cache),
        );
        
        let mut forward_pass = Self {
            model,
            max_context_length,
            tensor_cache: shared_cache,
            backend,
            transformer,
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
        // List of global tensors to preload
        let global_tensors = [
            // Token embedding table - needed for token embedding lookup (first step)
            TOKEN_EMBEDDING_TENSOR,
            // Output normalization - used in the final layer (before logits)
            OUTPUT_NORM_TENSOR, 
            // Output projection - used for final logits (last step)
            OUTPUT_TENSOR,  
        ];
        
        // Get mutable access to tensor cache
        let mut tensor_cache = self.tensor_cache.lock()
            .expect("Failed to lock tensor cache during preloading");
        
        // Preload tensors using the tensor cache
        let (loaded_count, total_count) = tensor_cache.preload(&global_tensors);
        
        println!("Preloaded {}/{} global tensors", loaded_count, total_count);
    }
    
    /// Convert token IDs to embeddings by looking up in the embedding table
    /// 
    /// This is the first step in the forward pass.
    /// 
    /// # Arguments
    /// * `tokens` - The input token IDs (already limited to context window)
    /// 
    /// # Returns
    /// * `Result<Tensor, Box<dyn Error + Send + Sync>>` - A tensor of shape [seq_len, hidden_dim] containing all token embeddings
    pub fn tokens_to_embeddings(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        let hidden_dim = self.model.params.hidden_dim;
        let vocab_size = self.model.params.vocab_size;
        let seq_len = tokens.len();
        
        println!("Converting {} tokens to embeddings", seq_len);
        
        // Get access to tensor cache
        let mut tensor_cache = self.tensor_cache.lock()
            .map_err(|e| format!("Failed to lock tensor cache: {}", e))?;
        
        // Load the embedding table tensor
        let embedding_table = tensor_cache.get( TOKEN_EMBEDDING_TENSOR)?;
        let embedding_data = embedding_table.data();
        
        // Create a tensor to hold all embeddings with shape [seq_len, hidden_dim]
        let mut embeddings_tensor = Tensor::zeros(vec![seq_len, hidden_dim], Arc::clone(&self.backend));
        
        // Lookup embeddings for each token and copy to the embeddings tensor
        for (i, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            
            // Check token ID is within range
            if token_id >= vocab_size {
                return Err(format!("Token ID {} is out of range (max: {})", 
                          token_id, vocab_size - 1).into());
            }
            
            // Extract the embedding for this token
            let start_idx = token_id * hidden_dim;
            let end_idx = start_idx + hidden_dim;
            
            // Copy the embedding from the embedding table to the embeddings tensor using slices
            let embed_start = i * hidden_dim;
            let embed_end = embed_start + hidden_dim;
            embeddings_tensor.data_mut()[embed_start..embed_end].copy_from_slice(&embedding_data[start_idx..end_idx]);
        }
        
        Ok(embeddings_tensor)
    }

    
    /// Predicts the next token given the current context
    pub fn predict_next_token(&mut self, tokens: &[u32]) -> Result<u32, Box<dyn Error + Send + Sync>> {  
        // Check if we have tokens to process
        if tokens.is_empty() {
            return Err("No tokens provided for prediction".into());
        }

        eprintln!("\n=== Starting Token Prediction ===");
        eprintln!("Input tokens: {:?}", tokens);

        // Step 1: Convert tokens to embeddings tensor
        eprintln!("Converting tokens to embeddings...");
        let embeddings = self.tokens_to_embeddings(tokens)?;
        eprintln!("Embeddings tensor shape: {:?}", embeddings.shape());
        
        // Step 2: Process embeddings through transformer blocks
        eprintln!("Processing embeddings through transformer blocks...");
        let transformer_output = self.transformer.forward(&embeddings)?;
        eprintln!("Transformer output shape: {:?}", transformer_output.shape());
        
        // For now, just return a placeholder token
        // In a real implementation, we would:
        // 1. Apply output normalization
        // 2. Project to vocabulary size
        // 3. Apply softmax
        // 4. Sample from the distribution
        eprintln!("Using placeholder token 111 for now");
        
        eprintln!("=== Token Prediction Complete ===\n");

        // Return the last token from input as a placeholder
        Ok(tokens[tokens.len() - 1])
    }
}
