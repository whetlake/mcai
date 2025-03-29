use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

use super::cpu::CpuBackend;
use crate::gguf::GGUFValueType;

/// Trait for backend-specific memory management
pub trait BackendMemory: Send + Sync {
    fn as_slice(&self) -> &[f32];
    fn as_mut_slice(&mut self) -> &mut [f32];
    fn to_cpu(&self) -> Vec<f32>;  // For when we need CPU data
}

/// A trait for tensor operation backends
pub trait Backend: Send + Sync + Debug {
    /// Allocate memory for a tensor in the backend's memory space
    fn allocate_memory(&self, size: usize) -> Result<Box<dyn BackendMemory>, Box<dyn Error + Send + Sync>>;
    
    /// Dequantize data directly into backend memory
    fn dequantize_to_memory(
        &self,
        data: &[u8],
        offset: usize,
        total_elements: usize,
        data_type: GGUFValueType,
        memory: &mut Box<dyn BackendMemory>
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

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
    
    /// Dequantizes tensor data from its compressed format to f32 values
    ///
    /// # Parameters
    /// * `data` - The raw tensor data
    /// * `offset` - The offset in bytes where the tensor data starts
    /// * `total_elements` - The number of elements in the tensor
    /// * `data_type` - The data type of the tensor
    ///
    /// # Returns
    /// * A vector of f32 values representing the dequantized tensor
    fn dequantize(
        &self,
        data: &[u8],
        offset: usize,
        total_elements: usize,
        data_type: GGUFValueType,
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>>;
}

// Factory function to create a backend based on available hardware
pub fn create_backend() -> Arc<Box<dyn Backend>> {
    // In the future, this can check for available hardware and select the best backend
    // For now, only CPU backend is available
    Arc::new(Box::new(CpuBackend::new()))
}
