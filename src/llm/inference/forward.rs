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
    tensor_cache: Arc<TensorCache>,
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
        
        // Initialize tensor cache (Mutex is now internal to TensorCache)
        let tensor_cache = TensorCache::new(Arc::clone(&model), Arc::clone(&backend));
        let shared_cache = Arc::new(tensor_cache);
        
        // --- Preload ALL tensors ---
        println!("Preloading all model tensors...");
        let all_tensor_names: Vec<&str> = model.tensors.iter().map(|ti| ti.name.as_str()).collect();
        { // Scope for using the cache directly, no lock needed here
            let (loaded_count, total_count) = shared_cache.preload(&all_tensor_names);
            println!("Preloaded {}/{} tensors", loaded_count, total_count);
        }
        // --- End Preload ---

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
        
        let forward_pass = Self {
            model,
            max_context_length,
            tensor_cache: shared_cache,
            backend,
            transformer,
        };
        
        forward_pass
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
        
        // Get access to tensor cache (no longer needs to be mutable or locked here)
        // self.tensor_cache is Arc<TensorCache>, get() takes &TensorCache
        let tensor_cache = &*self.tensor_cache; // Dereference Arc to get &TensorCache
        
        // Load the embedding table tensor using the &TensorCache reference
        let embedding_table = tensor_cache.get( TOKEN_EMBEDDING_TENSOR)?;
        let embedding_data = embedding_table.data();
        
        // Create a tensor to hold all embeddings with shape [seq_len, hidden_dim]
        let mut embeddings_tensor = Tensor::zeros(vec![seq_len, hidden_dim], Arc::clone(&self.backend))?;
        
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
        let transformer_output = self.transformer.forward(&embeddings)?; // Corrected: Pass only embeddings
        eprintln!("Transformer output shape: {:?}", transformer_output.shape());
        
        // Step 3: Apply output normalization (RMSNorm)
        eprintln!("Applying output normalization...");
        // Get access to tensor cache again (no longer needs mut or lock)
        let tensor_cache = &*self.tensor_cache; // Dereference Arc to get &TensorCache
        let norm_weights_arc = tensor_cache.get(OUTPUT_NORM_TENSOR)?; // Get Arc<Tensor>
        
        // We need the rms_norm function from Transformer because it directly uses the output of the transformer.
        // Dereference the Arc to pass &Tensor to rms_norm
        let normalized_output = self.transformer.rms_norm(&transformer_output, &*norm_weights_arc)?; 
        eprintln!("Normalized output shape: {:?}", normalized_output.shape());

        // Step 4: Project to vocabulary size

        eprintln!("=== Token Prediction Complete ===\n");

        // Return the last token from input as a placeholder
        Ok(tokens[tokens.len() - 1])
    }
}
