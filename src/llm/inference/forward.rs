use std::error::Error;
use std::sync::Arc;
use crate::llm::model::Model;

/// Handles the forward pass of the neural network for token prediction
pub struct ForwardPass {
    /// Reference to the model
    model: Arc<Model>,
    /// Cache for model parameters
    params: ModelParameters,
}

/// Cached model parameters extracted from the memory map
struct ModelParameters {
    /// Vocabulary size
    vocab_size: usize,
    /// Hidden dimension size
    hidden_dim: usize,
    /// Number of layers
    block_count: usize,
    /// Number of attention heads
    head_count: usize,
    /// Number of KV attention heads (for grouped-query attention)
    head_count_kv: usize,
    /// Context length
    context_length: usize,
    /// Feed forward length
    feed_forward_length: usize,
    /// Layer norm epsilon
    layer_norm_rms_epsilon: f32,
}

impl ForwardPass {
    /// Creates a new ForwardPass instance
    pub fn new(model: Arc<Model>) -> Self {
        // Extract model parameters from metadata
        let params = Self::get_metadata_values(&model);
        
        println!("params vocab_size: {:?}", params.vocab_size);
        println!("params hidden_dim: {:?}", params.hidden_dim);
        println!("params block_count: {:?}", params.block_count);
        println!("params head_count: {:?}", params.head_count);
        println!("params head_count_kv: {:?}", params.head_count_kv);
        println!("params context_length: {:?}", params.context_length);
        println!("params feed_forward_length: {:?}", params.feed_forward_length);
        println!("params layer_norm_rms_epsilon: {:?}", params.layer_norm_rms_epsilon);
        
        Self { 
            model,
            params,
        }
    }
    
    /// Extract model parameters from metadata
    fn get_metadata_values(model: &Arc<Model>) -> ModelParameters {
        // Get vocabulary size from tokenizer.ggml.tokens array length
        let vocab_size = match model.get_metadata_value("tokenizer.ggml.tokens") {
            Ok(value) => {
                // The value should be an array of tokens
                match &value {
                    crate::gguf::GGUFValue::Array(arr) => arr.len(),
                    _ => panic!("tokenizer.ggml.tokens is not an array"),
                }
            },
            Err(_) => panic!("Could not find tokenizer.ggml.tokens in model metadata"),
        };
        
        let hidden_dim = match model.get_metadata_value(&format!("{}.embedding_length", model.architecture.to_lowercase()))
            .or_else(|_| model.get_metadata_value("embedding_length")) {
            Ok(value) => value.to_string().parse::<usize>()
                .expect("Failed to parse embedding_length"),
            Err(_) => panic!("Could not find embedding_length in model metadata"),
        };
        
        let block_count = match model.get_metadata_value(&format!("{}.block_count", model.architecture.to_lowercase()))
            .or_else(|_| model.get_metadata_value("block_count")) {
            Ok(value) => value.to_string().parse::<usize>()
                .expect("Failed to parse block_count"),
            Err(_) => panic!("Could not find block_count in model metadata"),
        };
        
        let head_count = match model.get_metadata_value(&format!("{}.attention.head_count", model.architecture.to_lowercase()))
            .or_else(|_| model.get_metadata_value("attention.head_count")) {
            Ok(value) => value.to_string().parse::<usize>()
                .expect("Failed to parse attention.head_count"),
            Err(_) => panic!("Could not find attention.head_count in model metadata"),
        };
        
        let head_count_kv = match model.get_metadata_value(&format!("{}.attention.head_count_kv", model.architecture.to_lowercase()))
            .or_else(|_| model.get_metadata_value("attention.head_count_kv")) {
            Ok(value) => value.to_string().parse::<usize>()
                .expect("Failed to parse attention.head_count_kv"),
            Err(e) => {
                // Some models don't have KV heads (use same as regular heads)
                println!("Warning: Could not find attention.head_count_kv, using head_count instead: {}", e);
                head_count
            },
        };
        
        let context_length = match model.get_metadata_value(&format!("{}.context_length", model.architecture.to_lowercase()))
            .or_else(|_| model.get_metadata_value("context_length")) {
            Ok(value) => value.to_string().parse::<usize>()
                .expect("Failed to parse context_length"),
            Err(_) => panic!("Could not find context_length in model metadata"),
        };
        
        let feed_forward_length = match model.get_metadata_value(&format!("{}.feed_forward_length", model.architecture.to_lowercase()))
            .or_else(|_| model.get_metadata_value("feed_forward_length")) {
            Ok(value) => value.to_string().parse::<usize>()
                .expect("Failed to parse feed_forward_length"),
            Err(_) => panic!("Could not find feed_forward_length in model metadata"),
        };
        
        let layer_norm_rms_epsilon = match model.get_metadata_value(&format!("{}.attention.layer_norm_rms_epsilon", model.architecture.to_lowercase()))
            .or_else(|_| model.get_metadata_value("attention.layer_norm_rms_epsilon")) {
            Ok(value) => value.to_string().parse::<f32>()
                .expect("Failed to parse attention.layer_norm_rms_epsilon"),
            Err(_) => panic!("Could not find attention.layer_norm_rms_epsilon in model metadata"),
        };
                
        // Return the parameters
        ModelParameters {
            vocab_size,
            hidden_dim,
            block_count,
            head_count,
            head_count_kv,
            context_length,
            feed_forward_length,
            layer_norm_rms_epsilon,
        }
    }
    
    /// Predicts the next token given the current context
    pub fn predict_next_token(&self, tokens: &[u32]) -> Result<u32, Box<dyn Error + Send + Sync>> {
        let next_token = 111;
        
        Ok(next_token)
    }
} 