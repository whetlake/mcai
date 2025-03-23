use std::error::Error;
use std::fmt;
use ndarray::{Array, Array1, Array2, ArrayView1};

use super::super::Backend;
use super::quants::dequantize::Dequantizer;
use crate::gguf::GGUFValueType;

/// CPU backend implementation using ndarray
#[derive(Clone)]
pub struct CpuBackend {
    // Configuration options could go here
}

impl CpuBackend {
    /// Create a new CPU backend instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend").finish()
    }
}

impl Backend for CpuBackend {
    /// Performs matrix multiplication C = A * B using ndarray.
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
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Create ndarray views of the input data
        let a_array = Array2::from_shape_vec((m, k), a.to_vec())?;
        let b_array = Array2::from_shape_vec((k, n), b.to_vec())?;
        
        // Handle transpose if needed
        let a_view = if transpose_a { a_array.t() } else { a_array.view() };
        let b_view = if transpose_b { b_array.t() } else { b_array.view() };
        
        // Perform matrix multiplication (this uses BLAS internally if available)
        let result = a_view.dot(&b_view);
        
        // Copy result to output buffer
        c.copy_from_slice(result.as_slice().unwrap());
        
        Ok(())
    }

    /// Performs element-wise addition C = A + B.
    ///
    /// # Parameters
    /// * `a` - First input tensor
    /// * `b` - Second input tensor
    /// * `c` - Output tensor
    /// * `len` - Number of elements in each tensor
    fn add(
        &self,
        a: &[f32],
        b: &[f32], 
        c: &mut [f32],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Create ndarray views
        let a_array = ArrayView1::from(a);
        let b_array = ArrayView1::from(b);
        
        // Perform addition
        let result = &a_array + &b_array;
        
        // Copy result to output buffer
        c.copy_from_slice(result.as_slice().unwrap());
        
        Ok(())
    }
    
    /// Performs element-wise multiplication C = A * B.
    ///
    /// # Parameters
    /// * `a` - First input tensor
    /// * `b` - Second input tensor
    /// * `c` - Output tensor
    /// * `len` - Number of elements in each tensor
    fn mul(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Create ndarray views
        let a_array = ArrayView1::from(a);
        let b_array = ArrayView1::from(b);
        
        // Perform element-wise multiplication
        let result = &a_array * &b_array;
        
        // Copy result to output buffer
        c.copy_from_slice(result.as_slice().unwrap());
        
        Ok(())
    }
    
    /// Applies RMS normalization to a tensor.
    ///
    /// # Parameters
    /// * `x` - Input tensor
    /// * `weight` - Scale factors for each hidden dimension
    /// * `output` - Output tensor
    /// * `size` - Number of sequences/rows
    /// * `hidden_size` - Size of hidden dimension
    /// * `eps` - Small constant for numerical stability
    fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        output: &mut [f32],
        size: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Reshape input into a 2D array with shape [size, hidden_size]
        let x_array = Array::from_shape_vec((size, hidden_size), x.to_vec())?;
        let weight_array = ArrayView1::from(weight);
        
        // Create output array
        let mut result = Array::zeros((size, hidden_size));
        
        // Apply RMS normalization row by row
        for (i, row) in x_array.outer_iter().enumerate() {
            // Calculate sum of squares
            let ss: f32 = row.iter().map(|&v| v * v).sum();
            
            // Calculate normalization factor
            let norm_factor = 1.0 / (ss / hidden_size as f32 + eps).sqrt();
            
            // Normalize and scale with weights
            for j in 0..hidden_size {
                result[[i, j]] = norm_factor * row[j] * weight_array[j];
            }
        }
        
        // Copy result to output buffer
        output.copy_from_slice(result.as_slice().unwrap());
        
        Ok(())
    }
    
    /// Applies softmax function to input tensor.
    ///
    /// # Parameters
    /// * `x` - Input tensor
    /// * `output` - Output tensor
    /// * `size` - Number of elements
    fn softmax(
        &self,
        x: &[f32],
        output: &mut [f32],
        size: usize,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Create ndarray view
        let x_array = ArrayView1::from(x);
        
        // Find max value for numerical stability
        let max_val = x_array.fold(f32::MIN, |max, &val| max.max(val));
        
        // Calculate exp(x - max) and sum
        let mut exp_vals = Array1::zeros(size);
        let mut sum = 0.0;
        
        for i in 0..size {
            let val = (x_array[i] - max_val).exp();
            exp_vals[i] = val;
            sum += val;
        }
        
        // Normalize by sum
        let result = exp_vals / sum;
        
        // Copy result to output buffer
        output.copy_from_slice(result.as_slice().unwrap());
        
        Ok(())
    }
    
    /// Calculates dot product between two vectors.
    ///
    /// # Parameters
    /// * `a` - First vector
    /// * `b` - Second vector
    /// * `len` - Vector length
    ///
    /// # Returns
    /// * The dot product as a f32 value
    fn dot(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> Result<f32, Box<dyn Error + Send + Sync>> {
        // Create ndarray views
        let a_array = ArrayView1::from(a);
        let b_array = ArrayView1::from(b);
        
        // Calculate dot product
        let result = a_array.dot(&b_array);
        
        Ok(result)
    }

    /// Dequantizes tensor data from its compressed format to f32 values
    fn dequantize(
        &self,
        data: &[u8],
        offset: usize,
        total_elements: usize,
        data_type: GGUFValueType,
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        // Use the CPU-specific Dequantizer to handle all tensor formats
        Dequantizer::dequantize(data, offset, total_elements, data_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matmul() {
        let backend = CpuBackend::new();
        
        // 2x3 matrix A (row-major)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 3x2 matrix B (row-major)
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        // 2x2 result matrix C (row-major)
        let mut c = vec![0.0; 4];
        
        // Perform C = A * B
        backend.matmul(&a, &b, &mut c, 2, 2, 3, false, false).unwrap();
        
        // Log the result
        println!("CPU Backend matmul result: {:?}", c);
        
        // Expected result:
        // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
        // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
        // = [58, 64]
        //   [139, 154]
        
        // Check if the result matches expectations
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }
    
    #[test]
    fn test_add() {
        let backend = CpuBackend::new();
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        backend.add(&a, &b, &mut c).unwrap();
        
        assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0]);
    }
    
    #[test]
    fn test_mul() {
        let backend = CpuBackend::new();
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        backend.mul(&a, &b, &mut c).unwrap();
        
        assert_eq!(c, vec![5.0, 12.0, 21.0, 32.0]);
    }
}
