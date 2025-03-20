pub mod backends;
pub mod ops;
mod tensor;

// Re-export the Tensor struct and related items for easy access
pub use tensor::Tensor;

use std::error::Error;
use std::sync::Arc;
use std::fmt::{self, Debug};

use self::backends::Backend;