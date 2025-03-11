//! The inference module provides functionality for loading and running LLM models.
//!
//! This module contains the core components needed for model inference:
//! 
//! - Model loading and management via [`Model`] and [`InferenceEngine`]
//! - Tokenization and text processing via the [`tokenizer`] module
//! - Inference execution via the [`inference`] module
//! - Model registry and metadata tracking via [`ModelEntry`] and [`ModelDetails`]
//!
//! The main entry point is the [`InferenceEngine`] which handles model loading,
//! registry management, and inference coordination.

// Declare submodules
pub mod model;
pub mod inference;
pub mod tokenizer;
mod engine;

// Re-export types for external use
pub use model::Model;
pub use engine::InferenceEngine;
pub use engine::{ModelEntry, ModelDetails};