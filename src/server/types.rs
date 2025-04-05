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
    pub user_label: Option<String>,
}

/// Response after attaching a model
#[derive(Serialize, Deserialize)]
pub struct AttachModelResponse {
    pub name: String, // Actual model name
    pub user_label: Option<String>, // Label given by the user (optional)
    pub greeting: String, // The first greeting message from the model
    pub uuid: String, // The UUID assigned to this model session
} 