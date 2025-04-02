use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

use super::cpu::CpuBackend;
use crate::gguf::GGUFValueType;
use crate::llm::tensor::Tensor;

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
    
    /// Apply softmax function in-place along the last axis of the tensor.
    fn softmax(
        &self,
        tensor: &mut Tensor,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
    
    /// Calculate dot product between two vectors
    fn dot(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> Result<f32, Box<dyn Error + Send + Sync>>;
    
    /// Applies the Sigmoid Linear Unit (SiLU) activation function in-place.
    /// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    fn silu(&self, tensor: &mut Tensor) -> Result<(), Box<dyn Error + Send + Sync>>;
    
    /// Applies Rotary Positional Embeddings (RoPE) in place.
    /// Operates on data assumed to be shaped [seq_len, num_heads, head_dim].
    fn apply_rope(
        &self,
        data: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Scales all elements of a tensor by a given factor, in place.
    fn scale(&self, tensor: &mut Tensor, scale_factor: f32) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Applies a causal mask to attention scores in place.
    /// Assumes scores shape: [head_count, seq_len, seq_len]
    fn apply_causal_mask(&self, scores: &mut Tensor) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Repeats Key/Value heads for Grouped-Query Attention.
    /// Input shape: [head_count_kv, seq_len, head_dim]
    /// Output shape: [head_count_kv * num_groups, seq_len, head_dim]
    fn repeat_kv_heads(
        &self,
        tensor: &Tensor,
        num_groups: usize, // head_count / head_count_kv
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>>;

    /// Performs batched matrix multiplication: C = A @ B
    /// Expects A shape [batch, m, k], B shape [batch, k, n] (or [batch, n, k] if transpose_b)
    /// Returns C shape [batch, m, n]
    fn bmm(
        &self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool, // Typically false for Q
        transpose_b: bool, // Typically true for K
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>>;

    /// Perform matrix multiplication C = A * B using Tensors, returning a new Tensor.
    fn matmul_tensors(
        &self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>>;

    /// Perform element-wise addition C = A + B using Tensors (with broadcasting for B), returning a new Tensor.
    fn add_tensors(
        &self,
        a: &Tensor, // Typically the larger tensor (e.g., matrix)
        b: &Tensor, // Typically the smaller tensor (e.g., bias vector)
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>>;

    /// Perform element-wise multiplication C = A * B using Tensors, returning a new Tensor.
    fn mul_tensors(
        &self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>>;

    /// Permutes the axes of a tensor's data.
    /// Returns a new BackendMemory buffer with the permuted data.
    fn permute(
        &self,
        data: &[f32],
        current_shape: &[usize],
        new_axes: &[usize],
    ) -> Result<Box<dyn BackendMemory>, Box<dyn Error + Send + Sync>>;

    /// Permutes axes and reshapes the tensor's data in one operation.
    /// Returns a new BackendMemory buffer with the data in the final layout.
    fn permute_and_reshape(
        &self,
        data: &[f32],
        current_shape: &[usize],
        new_axes: &[usize],
        target_shape: &[usize],
    ) -> Result<Box<dyn BackendMemory>, Box<dyn Error + Send + Sync>>;

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
