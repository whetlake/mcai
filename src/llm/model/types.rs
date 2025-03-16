use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, serde::ts_seconds};

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