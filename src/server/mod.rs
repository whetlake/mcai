use axum::{routing::{get, post}, Router, Json, extract::State, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use crate::inference::InferenceEngine;
use crate::inference::{ModelDetails, ModelEntry};
use crate::gguf::TensorInfo;
use tracing::{info, warn, error};
use std::error::Error;

#[derive(Deserialize, Serialize, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
}

#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    pub response: String,
}

#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub status: String,
    pub data: Option<T>,
    pub message: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct CurrentModelResponse {
    pub model: Option<ModelDetails>,
}

/// Request to attach a model by number
#[derive(Deserialize)]
pub struct AttachModelRequest {
    pub model_number: usize,
}

#[derive(Serialize)]
struct AttachModelResponse {
    name: String,
    label: String,
    greeting: String,
}

pub struct ApiServer {
    engine: Arc<InferenceEngine>,
    host: String,
    port: u16,
}

impl ApiServer {
    pub fn new(engine: InferenceEngine, host: String, port: u16) -> Self {
        // Initialize model index at startup
        info!("Initializing model index...");
        // Scan for models at startup to see if there are new gguf files
        // in the models directory. If there are add them to the registry.
        if let Err(e) = engine.scan_models() {
            warn!("Failed to initialize model index: {}", e);
        }
        info!("Model index initialized");
        
        info!("Creating new API server on {}:{}", host, port);
        Self {
            engine: Arc::new(engine),
            host,
            port,
        }
    }

    pub async fn start(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let app_state = Arc::clone(&self.engine);
        
        let app = Router::new()
            .route("/api/v1/models", get(list_models))
            .route("/api/v1/attach", post(attach_model))
            .route("/api/v1/generate", post(generate))
            .route("/api/v1/drop", post(drop_model))
            .route("/api/v1/metadata", get(get_metadata))
            .route("/api/v1/tensors", get(get_tensors))
            .with_state(app_state);

        info!("Starting server on {}:{}", self.host, self.port);
        let listener = TcpListener::bind((self.host.as_str(), self.port)).await?;
        
        info!("Server started successfully\n");
        axum::serve(listener, app).await?;
        Ok(())
    }
}

#[allow(dead_code)]
async fn health_check() -> &'static str {
    info!("Health check endpoint called");
    "MCAI is running!"
}

/// Returns a list of all available models in JSON format.
/// The models are sorted by label for consistent ordering.
async fn list_models(State(engine): State<Arc<InferenceEngine>>) -> impl IntoResponse {
    info!("Models endpoint called");
    
    // Get the model registry
    match engine.registry.read() {
        Ok(registry) => {
            info!("Successfully retrieved model registry");
            // Convert registry values to a vector and sort by label
            let mut models: Vec<ModelEntry> = registry.values().map(|v| v.clone()).collect();
            models.sort_by(|a, b| a.label.cmp(&b.label));
            
            Json(ApiResponse {
                status: "success".to_string(),
                data: Some(models),
                message: None,
            })
        },
        Err(e) => {
            error!("Failed to read model registry: {}", e);
            Json(ApiResponse {
                status: "error".to_string(),
                data: None,
                message: Some(format!("Failed to read model registry: {}", e)),
            })
        }
    }
}

/// Attaches a model by its number
async fn attach_model(
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

    match engine.attach_model(request.model_number) {
        Ok(model) => {
            info!("Successfully attached model: {}", model.name);
            let response = AttachModelResponse {
                name: model.name.clone(),
                label: model.label.clone(),
                greeting: format!("Hello, thank you for choosing {}. Type to start interaction or type 'mcai help' to see more commands.", model.name),
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

/// Handles the generate endpoint for text generation
async fn generate(
    State(engine): State<Arc<InferenceEngine>>,
    Json(request): Json<GenerateRequest>
) -> impl IntoResponse {
    match engine.generate(&request.prompt) {
        Ok(response) => {
            Json(ApiResponse {
                status: "success".to_string(),
                data: Some(GenerateResponse { response }),
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

/// Handles the drop endpoint for detaching the current model
async fn drop_model(
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
async fn get_metadata(State(engine): State<Arc<InferenceEngine>>) -> impl IntoResponse {
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
async fn get_tensors(State(engine): State<Arc<InferenceEngine>>) -> impl IntoResponse {
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