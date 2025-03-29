use std::error::Error;
use std::sync::Arc;

use super::Tensor;

/// Element-wise multiplication of two tensors
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// * `Result<Tensor, Box<dyn Error + Send + Sync>>` - Result of element-wise multiplication
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    // Check that shapes are compatible for multiplication
    if a.shape() != b.shape() {
        return Err(format!("Incompatible shapes for multiplication: {:?} and {:?}",
                          a.shape(), b.shape()).into());
    }
    
    // Create output tensor
    let mut result = Tensor::zeros(a.shape().to_vec(), Arc::clone(a.backend()));
    
    // Call backend mul implementation
    a.backend().mul(
        a.data(),
        b.data(),
        &mut result.data_mut(),
    )?;
    
    Ok(result)
}

/// Matrix multiplication of two tensors
///
/// # Arguments
/// * `a` - First tensor with shape [..., m, k]
/// * `b` - Second tensor with shape [..., k, n]
/// * `transpose_a` - Whether to transpose the first tensor
/// * `transpose_b` - Whether to transpose the second tensor
///
/// # Returns
/// * `Result<Tensor, Box<dyn Error + Send + Sync>>` - Result tensor with shape [..., m, n]
pub fn matmul(a: &Tensor, b: &Tensor, transpose_a: bool, transpose_b: bool) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    // Check that tensors have at least 2 dimensions
    if a.shape().len() < 2 || b.shape().len() < 2 {
        return Err("Both tensors must have at least 2 dimensions for matmul".into());
    }
      
    let m = a.shape()[a.shape().len() - 2];  // rows of A
    let k = a.shape()[a.shape().len() - 1];  // cols of A = rows of B
    let n = b.shape()[b.shape().len() - 1];  // cols of B
    
    if k != b.shape()[b.shape().len() - 2] {
        return Err(format!("Incompatible dimensions for matmul: {:?} and {:?}", 
                          a.shape(), b.shape()).into());
    }
    
    // Create output tensor
    let mut result_shape = a.shape().to_vec();
    let last_idx = result_shape.len() - 1;
    result_shape[last_idx] = n;
    let mut result = Tensor::zeros(result_shape, Arc::clone(a.backend()));
    
    // Call backend matmul implementation
    a.backend().matmul(
        a.data(),
        b.data(),
        &mut result.data_mut(),
        m,
        n,
        k,
        transpose_a,
        transpose_b,
    )?;
    
    Ok(result)
}

/// Add two tensors element-wise
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// * `Result<Tensor, Box<dyn Error + Send + Sync>>` - Result of element-wise addition
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    // Check that shapes are compatible for addition
    if a.shape() != b.shape() {
        return Err(format!("Incompatible shapes for addition: {:?} and {:?}",
                          a.shape(), b.shape()).into());
    }
    
    // Create output tensor
    let mut result = Tensor::zeros(a.shape().to_vec(), Arc::clone(a.backend()));
    
    // Call backend add implementation
    a.backend().add(
        a.data(),
        b.data(),
        &mut result.data_mut(),
    )?;
    
    Ok(result)
}

/// Calculate dot product of two 1D tensors
///
/// # Arguments
/// * `a` - First 1D tensor
/// * `b` - Second 1D tensor
///
/// # Returns
/// * `Result<f32, Box<dyn Error + Send + Sync>>` - Dot product result
pub fn dot(a: &Tensor, b: &Tensor) -> Result<f32, Box<dyn Error + Send + Sync>> {
    // Check that both tensors are 1D with the same size
    if a.shape().len() != 1 || b.shape().len() != 1 {
        return Err("Both tensors must be 1D for dot product".into());
    }
    
    if a.shape()[0] != b.shape()[0] {
        return Err(format!("Incompatible shapes for dot product: {:?} and {:?}",
                          a.shape(), b.shape()).into());
    }
    
    // Call backend dot implementation
    a.backend().dot(a.data(), b.data())
}

