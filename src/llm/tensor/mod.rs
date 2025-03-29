pub mod ops;
pub mod tensor;
pub mod utils;

// Re-export the Tensor struct and related items for easy access
pub use tensor::Tensor;

// Re-export commonly used tensor operations
pub use ops::{mul, add, matmul, dot};
