use serde::{Deserialize, Serialize};
use crate::llm::model::ModelDetails;
use crate::llm::registry::ModelEntry;
use crate::gguf::TensorInfo;

/// Request for text generation
#[derive(Deserialize, Serialize, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub model_session_uuid: Option<String>,
}

/// Response for text generation
#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    pub response: String,
}

/// Generic API response wrapper
#[derive(Serialize, Deserialize, Debug)]
pub struct ApiResponse<T> {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Response for current model information
#[derive(Serialize, Deserialize)]
pub struct CurrentModelResponse {
    pub model: Option<ModelDetails>,
}

/// Request to attach a model by number
#[derive(Deserialize, Debug)]
pub struct AttachModelRequest {
    pub model_number: usize,
    pub user_label: Option<String>,
}

/// Response after attaching a model
#[derive(Serialize, Deserialize, Debug)]
pub struct AttachModelResponse {
    pub name: String, // Actual model name
    pub user_label: Option<String>, // Label given by the user (optional)
    pub greeting: String, // The first greeting message from the model
    pub uuid: String, // The UUID assigned to this model session
}

#[derive(Serialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelEntry>,
}

#[derive(Serialize)]
pub struct MetadataResponse {
    pub metadata: ModelDetails,
}

#[derive(Serialize)]
pub struct TensorsResponse {
    pub tensors: Vec<TensorInfo>,
}

// Struct for the rename request body
#[derive(Deserialize, Debug)]
pub struct RenameModelRequest {
    #[allow(dead_code)] // Silence warning as it's used by serde/axum
    pub identifier: Option<String>, // The UUID or current label to rename
    pub new_label: String,
} 