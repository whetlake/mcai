use axum::{routing::{get, post}, Router, Json, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use crate::inference::InferenceEngine;
use crate::inference::ModelDetails;
use tracing::{info, warn, error};

#[derive(Deserialize, Serialize)]
pub struct GenerateRequest {
    pub prompt: String,
}

#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    pub response: String,
}

#[derive(Serialize)]
pub struct ModelsResponse(Vec<String>);

#[derive(Serialize, Deserialize)]
pub struct CurrentModelResponse {
    pub model: Option<ModelDetails>,
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
        // Scan for models at startup
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

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let app_state = Arc::clone(&self.engine);
        
        let app = Router::new()
            .route("/", get(health_check))
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