use std::error::Error;
use std::sync::Arc;

use crate::llm::backend::backend::Backend;
use crate::llm::tensor::Tensor;

/// Performs matrix multiplication C = A * B
pub fn matmul(
    a: &Tensor,
    b: &Tensor,
    c: &mut Tensor,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Get shapes
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Calculate output shape
    let m = if transpose_a { a_shape[1] } else { a_shape[0] };
    let k = if transpose_a { a_shape[0] } else { a_shape[1] };
    let n = if transpose_b { b_shape[0] } else { b_shape[1] };
    
    // Verify shapes are compatible
    if k != if transpose_b { b_shape[1] } else { b_shape[0] } {
        return Err("Matrix dimensions do not match for multiplication".into());
    }
    
    // Verify output shape
    if c.shape() != &[m, n] {
        return Err("Output tensor has incorrect shape".into());
    }
    
    // Perform matrix multiplication using backend
    a.backend().matmul(
        a.as_slice(),
        b.as_slice(),
        c.as_mut_slice(),
        m,
        n,
        k,
        transpose_a,
        transpose_b,
    )
}

/// Performs element-wise addition C = A + B
pub fn add(
    a: &Tensor,
    b: &Tensor,
    c: &mut Tensor,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Verify shapes are compatible (broadcasting will be handled by backend)
    if a.shape() != b.shape() && a.shape() != c.shape() {
        return Err("Tensor shapes are incompatible for addition".into());
    }
    
    // Perform addition using backend
    a.backend().add(a.as_slice(), b.as_slice(), c.as_mut_slice())
}

/// Performs element-wise multiplication C = A * B
pub fn mul(
    a: &Tensor,
    b: &Tensor,
    c: &mut Tensor,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Verify shapes are compatible (broadcasting will be handled by backend)
    if a.shape() != b.shape() && a.shape() != c.shape() {
        return Err("Tensor shapes are incompatible for multiplication".into());
    }
    
    // Perform multiplication using backend
    a.backend().mul(a.as_slice(), b.as_slice(), c.as_mut_slice())
}

/// Applies RMS normalization to a tensor
pub fn rms_norm(
    x: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Verify shapes
    let x_shape = x.shape();
    let hidden_size = x_shape.last().ok_or("Input tensor must have at least one dimension")?;
    
    if weight.shape() != &[*hidden_size] {
        return Err("Weight tensor has incorrect shape".into());
    }
    
    if output.shape() != x_shape {
        return Err("Output tensor has incorrect shape".into());
    }
    
    // Calculate size (number of tokens) and hidden size
    let size = x_shape.iter().take(x_shape.len() - 1).product();
    
    // Apply RMS normalization using backend
    x.backend().rms_norm(
        x.as_slice(),
        weight.as_slice(),
        output.as_mut_slice(),
        size,
        *hidden_size,
        eps,
    )
}

/// Applies softmax to a tensor
pub fn softmax(
    x: &Tensor,
    output: &mut Tensor,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Verify shapes
    if x.shape() != output.shape() {
        return Err("Output tensor has incorrect shape".into());
    }
    
    // Calculate size (total number of elements)
    let size = x.shape().iter().product();
    
    // Apply softmax using backend
    x.backend().softmax(x.as_slice(), output.as_mut_slice(), size)
}

/// Calculates the dot product of two vectors
pub fn dot(
    a: &Tensor,
    b: &Tensor,
) -> Result<f32, Box<dyn Error + Send + Sync>> {
    // Verify shapes
    if a.shape() != b.shape() {
        return Err("Tensors must have the same shape for dot product".into());
    }
    
    // Calculate dot product using backend
    a.backend().dot(a.as_slice(), b.as_slice())
} 