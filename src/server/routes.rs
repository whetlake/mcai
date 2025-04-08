use axum::{
    Json, 
    extract::{State, Query},
    response::IntoResponse,
    http::StatusCode,
    response::sse::{Sse, Event}
};
use std::sync::Arc;
use tracing::{info, error};
use futures::stream::StreamExt;
use std::convert::Infallible;
use serde_json::json;
use serde::Deserialize;

use crate::llm::engine::InferenceEngine;
use crate::llm::registry::{ModelEntry, ModelRegistry};
use super::types::{
    ApiResponse, 
    GenerateRequest, 
    AttachModelRequest, 
    AttachModelResponse,
    RenameModelRequest
};
use crate::llm::model::ModelDetails;
use crate::gguf::TensorInfo;
use crate::llm::engine::AttachedModelInfo;

// Struct to capture optional identifier from query params
#[derive(Deserialize, Debug)]
pub(crate) struct OptionalIdentifierParams {
    identifier: Option<String>,
}

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
    info!(
        "SSE Generate endpoint called. Prompt: '{}', UUID: {:?}",
        &request.prompt, &request.model_session_uuid
    );

    // Pass the specific optional UUID from the body and prompt to the engine
    match engine.generate(request.model_session_uuid.as_deref(), &request.prompt).await {
        Ok((actual_uuid, llm_stream)) => {
            let sse_stream = llm_stream.filter_map(move |result| {
                let uuid_clone = actual_uuid.clone();
                async move {
                    match result {
                        Ok(piece) => {
                            let json_payload = json!({
                                "uuid": uuid_clone,
                                "text": piece
                            });
                            match serde_json::to_string(&json_payload) {
                                Ok(data_string) => Some(Ok::<_, Infallible>(Event::default().data(data_string))),
                                Err(e) => {
                                    error!("Failed to serialize SSE data to JSON: {}", e);
                                    None
                                }
                            }
                        },
                        Err(e) => {
                            error!("Error during LLM generation stream: {}", e);
                            None
                        }
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

/// Handles the drop endpoint for detaching a model instance.
/// Accepts an optional 'identifier' (UUID or label) query parameter.
pub async fn drop_model(
    State(engine): State<Arc<InferenceEngine>>,
    Query(params): Query<OptionalIdentifierParams>,
) -> impl IntoResponse {
    info!("Drop model endpoint called. Identifier: {:?}", params.identifier);

    // Call the engine's drop_model, passing the optional identifier
    match engine.drop_model(params.identifier.as_deref()) {
        Ok(_) => {
            info!("Successfully dropped model (Identifier: {:?})", params.identifier);
            Json(ApiResponse::<()> {
                status: "success".to_string(),
                data: None,
                message: None,
            })
        },
        Err(e) => {
            error!("Failed to drop model (Identifier: {:?}): {}", params.identifier, e);
            Json(ApiResponse::<()> {
                status: "error".to_string(),
                data: None,
                message: Some(e.to_string()),
            })
        }
    }
}

/// Gets metadata for a specific attached model instance.
/// Accepts an optional 'identifier' (UUID or label) query parameter.
pub async fn get_metadata(
    State(engine): State<Arc<InferenceEngine>>,
    Query(params): Query<OptionalIdentifierParams>,
) -> impl IntoResponse {
    info!("Metadata endpoint called. Identifier: {:?}", params.identifier);

    // Call the engine's get_metadata, passing the optional identifier
    match engine.get_metadata(params.identifier.as_deref()) {
        Ok(metadata) => {
            info!("Successfully retrieved metadata (Identifier: {:?})", params.identifier);
            Json(ApiResponse {
                status: "success".to_string(),
                data: Some(metadata),
                message: None,
            })
        },
        Err(e) => {
            error!("Failed to retrieve metadata (Identifier: {:?}): {}", params.identifier, e);
            Json(ApiResponse {
                status: "error".to_string(),
                data: None::<ModelDetails>,
                message: Some(e.to_string()),
            })
        }
    }
}

/// Gets tensor information for a specific attached model instance.
/// Accepts an optional 'identifier' (UUID or label) query parameter.
pub async fn get_tensors(
    State(engine): State<Arc<InferenceEngine>>,
    Query(params): Query<OptionalIdentifierParams>,
) -> impl IntoResponse {
    info!("Tensors endpoint called. Identifier: {:?}", params.identifier);

    // Call the engine's get_tensors, passing the optional identifier
    match engine.get_tensors(params.identifier.as_deref()) {
        Ok(tensors) => {
            info!("Successfully retrieved tensors (Identifier: {:?})", params.identifier);
            Json(ApiResponse {
                status: "success".to_string(),
                data: Some(tensors),
                message: None,
            })
        },
        Err(e) => {
            error!("Failed to retrieve tensors (Identifier: {:?}): {}", params.identifier, e);
            Json(ApiResponse {
                status: "error".to_string(),
                data: None::<Vec<TensorInfo>>,
                message: Some(e.to_string()),
            })
        }
    }
}

/// Renames an attached model instance.
/// Takes optional `identifier` (UUID or label) in query params.
/// Takes `new_label` in JSON body.
pub async fn rename_model_route(
    State(engine): State<Arc<InferenceEngine>>,
    Query(params): Query<OptionalIdentifierParams>,
    Json(request): Json<RenameModelRequest>,
) -> impl IntoResponse {
    info!("Rename model endpoint called. Identifier: {:?}, New Label: {}", params.identifier, request.new_label);

    // Call the engine's rename_model, passing optional identifier and new label
    match engine.rename_model(params.identifier.as_deref(), request.new_label) {
        Ok(_) => {
            info!("Successfully renamed model (Identifier: {:?})", params.identifier);
            Json(ApiResponse::<()> {
                status: "success".to_string(),
                data: None,
                message: Some("Model instance renamed successfully.".to_string()),
            })
        }
        Err(e) => {
            error!("Failed to rename model (Identifier: {:?}): {}", params.identifier, e);
            Json(ApiResponse::<()> {
                status: "error".to_string(),
                data: None,
                message: Some(e.to_string()),
            })
        }
    }
}

/// Returns information about currently attached model instances.
pub async fn list_attached_models(
    State(engine): State<Arc<InferenceEngine>>,
) -> impl IntoResponse {
    info!("List attached models endpoint called.");

    match engine.get_attached_models() {
        Ok(attached_models) => {
            info!("Successfully retrieved list of attached models.");
            let response = ApiResponse {
                status: "success".to_string(),
                data: Some(attached_models),
                message: None,
            };
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            error!("Failed to retrieve list of attached models: {}", e);
            let response = ApiResponse::<Vec<AttachedModelInfo>> {
                status: "error".to_string(),
                data: None,
                message: Some(format!("Failed to get attached models: {}", e)),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
        }
    }
} 