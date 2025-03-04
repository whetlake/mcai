use std::error::Error;
use std::path::Path;
use tracing_subscriber;
use tracing_appender;
use tracing::info;

mod gguf;
mod model;
mod inference;
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
    
    // Initialize the subscriber first, before any file operations
    let file_appender = tracing_appender::rolling::RollingFileAppender::new(
        tracing_appender::rolling::Rotation::DAILY,
        // Use log file path from settings, or default to "logs"
        settings.logging.file.as_deref().unwrap_or_else(|| Path::new("logs")),
        "mcai",
    );
    
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    
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

    // Create inference engine with models directory from settings
    let engine: inference::InferenceEngine = inference::InferenceEngine::new(models_path);

    // Create and start server
    let server = server::ApiServer::new(engine, settings.server.host.clone(), settings.server.port);
    
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