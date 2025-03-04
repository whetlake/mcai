// Required external crates for configuration management and serialization
use serde::Deserialize;
use std::path::PathBuf;
use config::{Config, ConfigError, Environment, File};

/// Configuration for model loading and management
#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    /// Directory where model files are stored
    pub directory: PathBuf,
}

/// Configuration for model inference parameters
#[derive(Debug, Deserialize, Clone)]
pub struct InferenceConfig {
    /// Controls randomness in generation (0.0-1.0)
    pub temperature: f32,
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Size of the context window for inference
    pub context_size: usize,
}

/// Configuration for the HTTP server
#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    /// Host address to bind to
    pub host: String,
    /// Port number to listen on
    pub port: u16,
    /// Whether to enable rate limiting
    pub rate_limit: bool,
    /// Requests per minute when rate limiting is enabled
    pub rate_limit_rpm: u32,
}

/// Configuration for application logging
#[derive(Debug, Deserialize, Clone)]
pub struct LoggingConfig {
    /// Log level (debug, info, warn, error)
    pub level: String,
    /// Optional log file path
    pub file: Option<PathBuf>,
}

/// Main settings struct that contains all configuration
#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    /// Model-related settings
    pub models: ModelConfig,
    /// Inference-related settings
    pub inference: InferenceConfig,
    /// Server-related settings
    pub server: ServerConfig,
    /// Logging-related settings
    pub logging: LoggingConfig,
}

/// Implementation for loading and parsing configuration
impl Settings {
    /// Creates a new Settings instance by loading config from multiple sources
    /// in the following order of precedence (highest to lowest):
    /// 1. Environment variables prefixed with MCAI_
    /// 2. Local config file (local.toml) if present
    /// 3. Default config file (default.toml)
    pub fn new() -> Result<Self, ConfigError> {
        // Check if current directory exists
        let config_dir = std::env::current_dir()
            .map_err(|e| ConfigError::Message(
                format!("Failed to get current directory: {}", e)
            ))?
            .join("config");

        // Check if config directory exists
        if !config_dir.exists() {
            return Err(ConfigError::Message(
                format!("Config directory not found at: {}", config_dir.display())
            ));
        }

        // Check if default.toml exists
        let default_config = config_dir.join("default.toml");
        if !default_config.exists() {
            return Err(ConfigError::Message(
                format!("Default configuration file not found at: {}", default_config.display())
            ));
        }

        // Create the local config path
        let local_config = config_dir.join("local.toml");

        // Convert paths to strings and keep them alive
        let default_config_path = default_config.to_string_lossy();
        let local_config_path = local_config.to_string_lossy();

        // Load and validate configuration
        let settings = Config::builder()
            .add_source(File::with_name(&default_config_path))
            .add_source(File::with_name(&local_config_path).required(false))
            .add_source(Environment::with_prefix("MCAI").separator("_"))
            .build()?
            .try_deserialize::<Settings>()?;

        // Validate settings after loading
        settings.validate()?;

        Ok(settings)
    }

    /// Validate configuration values
    fn validate(&self) -> Result<(), ConfigError> {
        // Create models directory if it doesn't exist
        if !self.models.directory.exists() {
            std::fs::create_dir_all(&self.models.directory).map_err(|e| {
                ConfigError::Message(format!(
                    "Failed to create models directory at {}: {}", 
                    self.models.directory.display(), e
                ))
            })?;
        }

        // Validate temperature range
        if !(0.0..=1.0).contains(&self.inference.temperature) {
            return Err(ConfigError::Message(
                format!("Temperature must be between 0.0 and 1.0, got: {}", self.inference.temperature)
            ));
        }

        // Validate max_tokens
        if self.inference.max_tokens == 0 {
            return Err(ConfigError::Message(
                "max_tokens must be greater than 0".to_string()
            ));
        }

        // Validate context_size
        if self.inference.context_size == 0 {
            return Err(ConfigError::Message(
                "context_size must be greater than 0".to_string()
            ));
        }

        // Validate server port range
        if !(1..=65535).contains(&self.server.port) {
            return Err(ConfigError::Message(
                format!("Port must be between 1 and 65535, got: {}", self.server.port)
            ));
        }

        // Validate rate limit
        if self.server.rate_limit && self.server.rate_limit_rpm == 0 {
            return Err(ConfigError::Message(
                "rate_limit_rpm must be greater than 0 when rate limiting is enabled".to_string()
            ));
        }

        // Validate logging level
        match self.logging.level.to_lowercase().as_str() {
            "error" | "warn" | "info" | "debug" | "trace" => Ok(()),
            _ => Err(ConfigError::Message(
                format!("Invalid logging level: {}. Must be one of: error, warn, info, debug, trace", 
                    self.logging.level)
            )),
        }?;

        // Create log file directory if configured and doesn't exist
        if let Some(log_file) = &self.logging.file {
            if let Some(parent) = log_file.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        ConfigError::Message(format!(
                            "Failed to create log directory at {}: {}", 
                            parent.display(), e
                        ))
                    })?;
                }
            }
        }

        Ok(())
    }
} 