use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, serde::ts_seconds};

/// Cached model parameters extracted from the memory map for easy access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Number of layers
    pub block_count: usize,
    /// Number of attention heads
    pub head_count: usize,
    /// Number of KV attention heads (for grouped-query attention)
    pub head_count_kv: usize,
    /// Model's training context length (from metadata)
    pub model_context_length: usize,
    /// Feed forward length
    pub feed_forward_length: usize,
    /// Layer norm epsilon
    pub layer_norm_rms_epsilon: f32,
}

/// Detailed model information for display and API responses.
///
/// This struct contains enriched information about a model,
/// including runtime-calculated fields like tensor count and
/// absolute directory paths.
#[derive(Serialize, Deserialize)]
pub struct ModelDetails {
    /// Position in the model list (optional)
    pub number: Option<usize>,
    /// Short identifier for the model
    pub label: String,
    /// Human-readable name
    pub name: String,
    /// Size category (e.g., "7B")
    pub size: String,
    /// Model architecture
    pub architecture: String,
    /// Quantization format
    pub quantization: String,
    /// When the model was added
    #[serde(with = "ts_seconds")]
    pub added_date: DateTime<Utc>,
    /// Number of tensors in the model
    pub tensor_count: u64,
    /// Name of the model file
    pub filename: String,
    /// Absolute path to the models directory
    pub directory: String,
    /// Complete metadata from the model
    pub metadata: Vec<(String, String, String)>,
} 