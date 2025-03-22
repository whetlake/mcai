pub mod backends;
pub mod ops;
mod tensor;

// Re-export the Tensor struct and related items for easy access
pub use tensor::Tensor;