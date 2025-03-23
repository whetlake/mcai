use std::error::Error;
use std::fmt::Debug;

pub mod cpu;
pub mod backend;

// Re-export the Backend trait and factory function
pub use backend::{Backend, create_backend};
// Re-export the CpuBackend for testing/debugging
pub use cpu::CpuBackend;
