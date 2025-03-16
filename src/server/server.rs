use std::sync::Arc;
use std::error::Error;
use tokio::net::TcpListener;
use axum::{Router, routing::{get, post}};
use tracing::info;

use crate::llm::engine::InferenceEngine;
use crate::llm::registry::ModelRegistry;
use super::routes;

/// API Server for handling model inference requests
pub struct ApiServer {
    registry: Arc<ModelRegistry>,
    engine: Arc<InferenceEngine>,
    host: String,
    port: u16,
}

impl ApiServer {
    pub fn new(registry: Arc<ModelRegistry>, engine: InferenceEngine, host: String, port: u16) -> Self {
        info!("Creating new API server on {}:{}", host, port);
        Self {
            registry: registry,
            engine: Arc::new(engine),
            host,
            port,
        }
    }

    pub async fn start(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let engine_state = Arc::clone(&self.engine);
        let registry_state = Arc::clone(&self.registry);
        
        // Create a router for engine-based routes
        let engine_routes = Router::new()
            .route("/api/v1/attach", post(routes::attach_model))
            .route("/api/v1/generate", post(routes::generate))
            .route("/api/v1/drop", post(routes::drop_model))
            .route("/api/v1/metadata", get(routes::get_metadata))
            .route("/api/v1/tensors", get(routes::get_tensors))
            .with_state(engine_state);
            
        // Create a router for registry-based routes
        let registry_routes = Router::new()
            .route("/api/v1/models", get(routes::list_models))
            .with_state(registry_state);
            
        // Merge the routers
        let app = registry_routes.merge(engine_routes);

        info!("Starting server on {}:{}", self.host, self.port);
        let listener = TcpListener::bind((self.host.as_str(), self.port)).await?;
        
        info!("Server started successfully\n");
        axum::serve(listener, app).await?;
        Ok(())
    }
} 