use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, serde::ts_seconds};

/// Represents a model entry in the registry file.
///
/// This struct contains persistent metadata about available models
/// and is serialized to/from the model_registry.json file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelEntry {
    /// Position in the model list (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number: Option<usize>,
    /// Filename of the model file (relative to models directory)
    pub filename: String,
    /// Short identifier/label for the model
    pub label: String,
    /// Human-readable name of the model
    pub name: String,
    /// Size category of the model (e.g., "7B", "13B")
    pub size: String,
    /// Architecture of the model (e.g., "LLaMA", "Mistral")
    pub architecture: String,
    /// Quantization format (e.g., "Q4_K_M", "Q5_K_M")
    pub quantization: String,
    /// File type from GGUF metadata (e.g., 15 for Q8_K)
    pub file_type: i64,
    /// Quantization version from GGUF metadata
    pub quantization_version: i64,
    /// Number of tensors in the model
    pub tensor_count: u64,
    /// When the model was added to the registry
    #[serde(with = "ts_seconds")]
    pub added_date: DateTime<Utc>,
}