//! # Inference Engine Module
//! 
//! The inference module provides core functionality for managing and interacting with 
//! large language models (LLMs) in GGUF format. It handles model discovery, loading,
//! tracking, and text generation operations.
//!
//! ## Key Components
//!
//! - `InferenceEngine`: The central component that manages model state and operations
//! - `ModelEntry`: Represents a model in the registry (persistent storage format)
//! - `ModelDetails`: Enriched model information for API responses and display
//!
//! ## Architecture
//!
//! The inference engine maintains thread-safe access to shared state using `RwLock`
//! to allow concurrent reads while ensuring exclusive writes. It interacts with the
//! file system to scan for models and provides an API for the server to expose
//! model operations.

use std::error::Error;
use std::sync::RwLock;  // Add this for thread-safe state
use std::path::PathBuf;
use std::fs;
use tracing::{info, error, debug};
use crate::gguf::{GGUFReader, GGUFError, is_gguf_file};  // Add GGUF parser and standalone function
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, serde::ts_seconds};
use std::thread;
use std::time::Duration;
use indicatif::{ProgressBar, ProgressStyle};
use std::fmt;

// Declare submodules
pub mod model;
pub mod inference;
pub mod tokenizer;
mod engine;  // Add this line

// Re-export types for external use
pub use model::Model;
pub use inference::InferenceContext;
pub use tokenizer::Tokenizer;
pub use engine::InferenceEngine;  // Add this line
pub use engine::{ModelEntry, ModelDetails};  // Add this line if needed

