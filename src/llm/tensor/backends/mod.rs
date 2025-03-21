use std::error::Error;
use std::fmt::Debug;

pub mod cpu;
// Re-export the CpuBackend
pub use cpu::CpuBackend;

/// A trait for tensor operation backends
pub trait Backend: Send + Sync + Debug {
    /// Perform matrix multiplication C = A * B
    /// 
    /// # Parameters
    /// * `a` - Input matrix A with shape (m, k)
    /// * `b` - Input matrix B with shape (k, n)
    /// * `c` - Output matrix C with shape (m, n)
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A and rows in B
    /// * `transpose_a` - Whether to transpose matrix A
    /// * `transpose_b` - Whether to transpose matrix B
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Perform element-wise addition C = A + B
    fn add(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
    
    /// Perform element-wise multiplication C = A * B
    fn mul(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
    
    /// Apply RMS normalization to a tensor
    fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        output: &mut [f32],
        size: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
    
    /// Apply softmax function
    fn softmax(
        &self,
        x: &[f32],
        output: &mut [f32],
        size: usize,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
    
    /// Calculate dot product between two vectors
    fn dot(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> Result<f32, Box<dyn Error + Send + Sync>>;
}

// Factory function to create a backend based on available hardware
pub fn create_backend() -> Box<dyn Backend> {
    // For now, only CPU backend is available
    Box::new(CpuBackend::new())
}
