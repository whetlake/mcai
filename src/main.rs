use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use tracing_subscriber;
use tracing_appender;
use tracing::info;
use llm::registry::ModelRegistry;
use llm::engine::InferenceEngine;

// Declare modules
mod llm;
mod gguf;
mod server;
mod config;
mod chat;

use config::Settings;

/// Main entry point for the MCAI application
///
/// Parses command line arguments and handles two main modes of operation:
/// - Run: Starts both the inference server and an interactive chat session
/// - Serve: Starts only the inference server
///
/// # Arguments
/// The program accepts commands and arguments defined in the `cli` module
///
/// # Errors
/// Returns an error if server initialization fails or if there are IO errors
/// during the chat session
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Load settings first
    let settings = Settings::new()?;
    
    // Initialize the subscriber first, before any file operations.
    // This ensures that the log file is created before any logging is done.
    let file_appender = tracing_appender::rolling::RollingFileAppender::new(
        tracing_appender::rolling::Rotation::DAILY,
        // Use log file path from settings, or default to "logs"
        settings.logging.file.as_deref().unwrap_or_else(|| Path::new("logs")),
        "mcai",
    );
    
    // Create the log directory if it doesn't exist
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    
    // Initialize the subscriber
    tracing_subscriber::fmt()
        // Write to both console and file
        .with_writer(non_blocking)
        // Disable ANSI colors for cleaner log files
        .with_ansi(false)
        .with_line_number(true)
        .with_file(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(false)
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("MCAI Starting up...");
    info!("Initializing logging system...");
    
    let log_path = settings.logging.file.as_deref().unwrap_or_else(|| Path::new("logs"));
    std::fs::create_dir_all(log_path)?;
    let full_log_path = std::fs::canonicalize(log_path)?;
    info!("Log directory: {}", full_log_path.display());
    info!("Logging initialized");

    // Models directory location
    let models_path = std::fs::canonicalize(&settings.models.directory)?;
    info!("Models directory: {}", models_path.display());

    info!("Settings loaded");

    // Create and initialize registry
    let registry = Arc::new(ModelRegistry::new(models_path));
    
    // Scan for models
    info!("Initializing model index...");
    if let Err(e) = registry.scan_models() {
        eprintln!("Failed to scan models: {}", e);
    }
    info!("Model index initialized");

    // Create inference engine with the registry
    let engine = InferenceEngine::new(Arc::clone(&registry), settings.clone());

    // Create and start server
    let server = server::ApiServer::new(Arc::clone(&registry), engine, settings.server.host.clone(), settings.server.port);
    
    // Start server in a separate task
    tokio::spawn(async move {
        if let Err(e) = server.start().await {
            eprintln!("Server error: {}", e);
        }
    });

    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Start chat loop
    chat::chat_loop(&settings).await?;

    Ok(())
}