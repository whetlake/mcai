use axum::{
    Json, 
    extract::State, 
    response::IntoResponse,
    http::StatusCode,
    response::sse::{Sse, Event},
};
use std::sync::Arc;
use tracing::{info, error};
use futures::stream::StreamExt;
use std::convert::Infallible;

use crate::llm::engine::InferenceEngine;
use crate::llm::registry::{ModelEntry, ModelRegistry};
use super::types::{
    ApiResponse, 
    GenerateRequest, 
    AttachModelRequest, 
    AttachModelResponse
};

/// Returns a health check response
#[allow(dead_code)]
pub async fn health_check() -> &'static str {
    info!("Health check endpoint called");
    "MCAI is running!"
}

/// Returns a list of all available models in JSON format.
/// The models are sorted by label for consistent ordering.
pub async fn list_models(State(registry): State<Arc<ModelRegistry>>) -> impl IntoResponse {
    
    // Get the model registry
    match registry.registry.read() {
        Ok(registry) => {
            info!("Successfully retrieved model registry");
            // Convert registry values to a vector and sort by label
            let mut models: Vec<ModelEntry> = registry.values().map(|v| v.clone()).collect();
            models.sort_by(|a, b| a.label.cmp(&b.label));
            
            // Return the models as JSON
            let response = ApiResponse {
                status: "success".to_string(),
                data: Some(models),
                message: None,
            };
            
            (StatusCode::OK, Json(response))
        },
        Err(e) => {
            error!("Failed to read model registry: {}", e);
            let response = ApiResponse::<Vec<ModelEntry>> {
                status: "error".to_string(),
                data: None,
                message: Some(format!("Failed to read model registry: {}", e)),
            };
            
            (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
        }
    }
}

/// Attaches a model by its number
pub async fn attach_model(
    State(engine): State<Arc<InferenceEngine>>,
    Json(request): Json<AttachModelRequest>
) -> impl IntoResponse {
    info!("Attach model endpoint called with model number: {}", request.model_number);
   
    if request.model_number == 0 {
        error!("Invalid model number: Model numbering starts at 1");
        return Json(ApiResponse {
            status: "error".to_string(),
            data: None,
            message: Some("Invalid model number: Model numbering starts at 1".to_string()),
        });
    }

    match engine.attach_model(request.model_number, request.user_label) {
        Ok(model) => {
            info!("Successfully attached model: {}", model.name);
            let response = AttachModelResponse {
                name: model.name.clone(),
                user_label: model.user_label.clone(),
                greeting: format!("Hello, thank you for choosing {}. Type to start interaction or type 'mcai help' to see more commands.", model.name),
                uuid: model.uuid.clone()
            };
            Json(ApiResponse {
                status: "success".to_string(),
                data: Some(response),
                message: None,
            })
        },
        Err(e) => {
            error!("Failed to attach model: {}", e);
            Json(ApiResponse {
                status: "error".to_string(),
                data: None,
                message: Some(format!("Failed to attach model: {}", e)),
            })
        }
    }
}

/// Handles the generate endpoint for text generation using Server-Sent Events (SSE)
pub async fn generate(
    State(engine): State<Arc<InferenceEngine>>,
    Json(request): Json<GenerateRequest>
) -> impl IntoResponse {
    info!("SSE Generate endpoint called with prompt: {}", &request.prompt);

    match engine.generate(&request.prompt).await {
        Ok(llm_stream) => {
            let sse_stream = llm_stream.filter_map(|result| async move {
                match result {
                    Ok(piece) => Some(Ok::<_, Infallible>(Event::default().data(piece))),
                    Err(e) => {
                        error!("Error during LLM generation stream: {}", e);
                        None
                    }
                }
            });

            Sse::new(sse_stream).into_response()
        },
        Err(e) => {
            error!("Failed to initiate generation stream: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, 
             Json(ApiResponse::<()> {
                 status: "error".to_string(),
                 data: None,
                 message: Some(format!("Failed to start generation: {}", e)),
             })
            ).into_response()
        }
    }
}

/// Handles the drop endpoint for detaching the current model
pub async fn drop_model(
    State(engine): State<Arc<InferenceEngine>>
) -> impl IntoResponse {
    match engine.drop_model() {
        Ok(_) => {
            Json(ApiResponse::<()> {
                status: "success".to_string(),
                data: None,
                message: None,
            })
        },
        Err(e) => {
            Json(ApiResponse::<()> {
                status: "error".to_string(),
                data: None,
                message: Some(e.to_string()),
            })
        }
    }
}

/// Gets metadata for the currently attached model
pub async fn get_metadata(State(engine): State<Arc<InferenceEngine>>) -> impl IntoResponse {
    info!("Metadata endpoint called");

    match engine.get_metadata() {
        Ok(metadata) => {
            Json(ApiResponse {
                status: "success".to_string(),
                data: Some(metadata),
                message: None,
            })
        },
        Err(e) => {
            Json(ApiResponse {
                status: "error".to_string(),
                data: None,
                message: Some(e.to_string()),
            })
        }
    }
}

/// Gets tensor information for the currently attached model
pub async fn get_tensors(State(engine): State<Arc<InferenceEngine>>) -> impl IntoResponse {
    info!("Tensors endpoint called");

    match engine.get_tensors() {
        Ok(tensors) => {
            Json(ApiResponse {
                status: "success".to_string(),
                data: Some(tensors),
                message: None,
            })
        },
        Err(e) => {
            Json(ApiResponse {
                status: "error".to_string(),
                data: None,
                message: Some(e.to_string()),
            })
        }
    }
} 