use std::sync::Arc;
use std::error::Error;
use tokio::net::TcpListener;
use axum::{Router, routing::{get, post}};
use tracing::{info, warn};

use crate::inference::InferenceEngine;
use super::routes;

/// API Server for handling model inference requests
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
            .route("/api/v1/models", get(routes::list_models))
            .route("/api/v1/attach", post(routes::attach_model))
            .route("/api/v1/generate", post(routes::generate))
            .route("/api/v1/drop", post(routes::drop_model))
            .route("/api/v1/metadata", get(routes::get_metadata))
            .route("/api/v1/tensors", get(routes::get_tensors))
            .with_state(app_state);

        info!("Starting server on {}:{}", self.host, self.port);
        let listener = TcpListener::bind((self.host.as_str(), self.port)).await?;
        
        info!("Server started successfully\n");
        axum::serve(listener, app).await?;
        Ok(())
    }
} 