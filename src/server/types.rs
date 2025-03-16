use serde::{Deserialize, Serialize};
use crate::llm::model::ModelDetails;

/// Request for text generation
#[derive(Deserialize, Serialize, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
}

/// Response for text generation
#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    pub response: String,
}

/// Generic API response wrapper
#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub status: String,
    pub data: Option<T>,
    pub message: Option<String>,
}

/// Response for current model information
#[derive(Serialize, Deserialize)]
pub struct CurrentModelResponse {
    pub model: Option<ModelDetails>,
}

/// Request to attach a model by number
#[derive(Deserialize)]
pub struct AttachModelRequest {
    pub model_number: usize,
}

/// Response after attaching a model
#[derive(Serialize)]
pub struct AttachModelResponse {
    pub name: String,
    pub label: String,
    pub greeting: String,
} 