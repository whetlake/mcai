use std::error::Error;
use std::fmt;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis, IxDyn, s};
use std::sync::Arc;

use super::super::backend::{Backend, BackendMemory};
use super::quants::dequantize::Dequantizer;
use crate::gguf::GGUFValueType;
use crate::llm::tensor::Tensor;
use ndarray::{Array3, ArrayD, ArrayView3, stack};

/// CPU-specific memory implementation
pub struct CpuMemory {
    data: Vec<f32>,
}

impl BackendMemory for CpuMemory {
    fn as_slice(&self) -> &[f32] {
        &self.data
    }
    
    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
    
    fn to_cpu(&self) -> Vec<f32> {
        self.data.clone()
    }
}

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
    fn allocate_memory(&self, size: usize) -> Result<Box<dyn BackendMemory>, Box<dyn Error + Send + Sync>> {
        Ok(Box::new(CpuMemory {
            data: vec![0.0; size]
        }))
    }

    fn dequantize_to_memory(
        &self,
        data: &[u8],
        offset: usize,
        total_elements: usize,
        data_type: GGUFValueType,
        memory: &mut Box<dyn BackendMemory>
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Dequantize into a temporary vector first
        let dequantized_data = self.dequantize(data, offset, total_elements, data_type)?;
        
        // Copy the dequantized data into the backend memory
        memory.as_mut_slice().copy_from_slice(&dequantized_data);
        
        Ok(())
    }

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
        // println!("Matrix multiplication shapes:");
        // println!("  Matrix A: {}x{}", m, k);
        // println!("  Matrix B: {}x{}", k, n);
        // println!("  Output C: {}x{}", m, n);
        // println!("  Transpose A: {}", transpose_a);
        // println!("  Transpose B: {}", transpose_b);
        // println!("  Input A length: {}", a.len());
        // println!("  Input B length: {}", b.len());
        // println!("  Output C length: {}", c.len());

        // Create ndarray views of the input data
        let a_array = Array2::from_shape_vec((m, k), a.to_vec())?;
        let b_array = Array2::from_shape_vec((k, n), b.to_vec())?;
        
        // Handle transpose if needed
        let a_view = if transpose_a { a_array.t() } else { a_array.view() };
        let b_view = if transpose_b { b_array.t() } else { b_array.view() };
        
        // Perform matrix multiplication (this uses BLAS internally if available)
        let result = a_view.dot(&b_view);
        
        // Copy result to output buffer using assign to handle potential non-contiguity
        let mut c_array = ArrayViewMut2::from_shape((m, n), c)?;
        c_array.assign(&result);
        
        Ok(())
    }

    /// Performs element-wise addition C = A + B with broadcasting support
    ///
    /// # Parameters
    /// * `a` - First input tensor (can be matrix)
    /// * `b` - Second input tensor (can be vector)
    /// * `c` - Output tensor
    fn add(
        &self,
        a: &[f32],
        b: &[f32], 
        c: &mut [f32],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Determine shapes based on slice lengths
        let b_len = b.len();
        if a.len() % b_len != 0 {
            return Err("Shape mismatch for broadcasting add".into());
        }
        let a_rows = a.len() / b_len;
        let output_shape = (a_rows, b_len);

        // Create ndarray views with correct shapes
        let a_array = ArrayView2::from_shape(output_shape, a)?;
        let b_array = ArrayView1::from(b); // Bias is 1D
        
        // Perform addition with broadcasting
        let result = &a_array + &b_array;
        
        // Copy result to output buffer using assign
        let mut c_array = ArrayViewMut2::from_shape(output_shape, c)?;
        c_array.assign(&result);
        
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
    /// * `x` - Input tensor of shape [size, hidden_size]
    /// * `weight` - Scale factors of shape [hidden_size]
    /// * `output` - Output tensor of shape [size, hidden_size]
    /// * `size` - Number of sequences/rows
    /// * `hidden_size` - Size of hidden dimension
    /// * `eps` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Process
    /// For each row (token embedding):
    /// 1. Calculate RMS: sqrt(mean(x²))
    /// 2. Normalize: x / (RMS + eps)
    /// 3. Scale: normalized * weight
    fn rms_norm(
        &self,
        x: &[f32], // Input tensor of shape [size, hidden_size]
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
        
        // Process each row seperately or each token separately
        for (i, row) in x_array.outer_iter().enumerate() {
            // Step 1: Calculate RMS
            // Square each element and take mean
            let mean_square = row.iter().map(|&v| v * v).sum::<f32>() / hidden_size as f32;
            let rms = mean_square.sqrt() + eps;  // Add epsilon after square root
            
            // Step 2 & 3: Normalize and scale using ndarray operations
            let mut normalized = row.to_owned();
            normalized /= rms;
            result.slice_mut(s![i, ..]).assign(&(normalized * &weight_array));
        }
        
        // Copy result to output buffer
        output.copy_from_slice(result.as_slice().unwrap());
        
        Ok(())
    }
    
    /// Applies softmax function in-place along the last axis.
    fn softmax(
        &self,
        tensor: &mut Tensor,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let shape = tensor.shape();
        if shape.len() < 1 {
            return Ok(()); // No-op for empty or 0D tensor
        }
        let last_axis = Axis(shape.len() - 1);
        let mut array_view = ndarray::ArrayViewMutD::from_shape(IxDyn(shape), tensor.data_mut())?;

        // Iterate over all possible slices along the last dimension
        for mut row in array_view.axis_iter_mut(last_axis) {
            // Apply 1D softmax to this row (slice)
            let max_val = row.fold(f32::NEG_INFINITY, |max, &val| max.max(val));
            let mut sum = 0.0f32;
            
            // Calculate exp(x - max) and sum
            row.mapv_inplace(|x| {
                let val = (x - max_val).exp(); // First subtrack the maximum for numerical stability
                sum += val; // add up the sum
                val // update the value with the exponential
            });
            
            // Check for sum == 0 to avoid division by zero (e.g., if all inputs were -inf)
            // There is one consideration here: be aware that this might impact the output for that specific
            // token position, as it will receive no contextual information from attention. But in practice
            // this is unlikely to happen, because this sum 0.0 should never occur.
            if sum == 0.0 {
                // If sum is zero (e.g., all masked inputs were -inf),
                // assign zero probability everywhere.
                row.fill(0.0);
            } else {
                // Normalize by sum
                 row.mapv_inplace(|x| x / sum);
            }
        }

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
        Ok(a_array.dot(&b_array))
    }

    /// Applies Rotary Positional Embeddings (RoPE) using ndarray.
    fn apply_rope(
        &self,
        data: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Create a mutable 3D view of the data
        let mut view = ArrayViewMut3::from_shape((seq_len, num_heads, head_dim), data)?;

        let rope_dim = head_dim; // Apply RoPE to the full head dimension
        let theta_base = 10000.0f32;

        for pos in 0..seq_len {
            for h in 0..num_heads {
                // Get mutable access to the head slice for this position and head
                let mut head_slice = view.slice_mut(s![pos, h, ..]); 
                for i in (0..rope_dim).step_by(2) {
                    // Calculate theta for this dimension pair
                    let theta = theta_base.powf(-((i as f32) / (rope_dim as f32)));
                    let m_theta = pos as f32 * theta;
                    let cos_m_theta = m_theta.cos();
                    let sin_m_theta = m_theta.sin();

                    // Get the pair of values
                    let v0 = head_slice[i];
                    let v1 = head_slice[i + 1];

                    // Apply rotation
                    *head_slice.get_mut(i).unwrap()     = v0 * cos_m_theta - v1 * sin_m_theta;
                    *head_slice.get_mut(i + 1).unwrap() = v0 * sin_m_theta + v1 * cos_m_theta;
                }
            }
        }

        Ok(())
    }

    /// Permutes the axes of a tensor's data using ndarray.
    fn permute(
        &self,
        data: &[f32],
        current_shape: &[usize],
        new_axes: &[usize],
    ) -> Result<Box<dyn BackendMemory>, Box<dyn Error + Send + Sync>> {
        // Create an ndarray view with the original shape
        let view = ndarray::ArrayViewD::from_shape(IxDyn(current_shape), data)?;

        // Permute the axes
        let permuted_view = view.permuted_axes(IxDyn(new_axes));

        // Create a new owned array with the data in the new permuted order (makes it contiguous)
        let permuted_array: ndarray::ArrayD<f32> = permuted_view.to_owned();

        // Convert the owned array's data into a Vec<f32>
        let permuted_data = permuted_array.into_raw_vec_and_offset().0;

        // Allocate new backend memory and copy the contiguous permuted data into it
        let mut new_memory = self.allocate_memory(permuted_data.len())?;
        new_memory.as_mut_slice().copy_from_slice(&permuted_data);

        Ok(new_memory)
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

    /// Scales all elements of a tensor by a given factor, in place.
    fn scale(&self, tensor: &mut Tensor, scale_factor: f32) -> Result<(), Box<dyn Error + Send + Sync>> {
        let data = tensor.data_mut();
        for val in data {
            *val *= scale_factor;
        }
        Ok(())
    }

    /// Applies a causal mask to attention scores in place.
    fn apply_causal_mask(&self, scores: &mut Tensor) -> Result<(), Box<dyn Error + Send + Sync>> {
        let shape = scores.shape();
        if shape.len() != 3 {
            return Err("Scores tensor must be 3D for causal masking".into());
        }
        let head_count = shape[0];
        let seq_len = shape[1];
        if shape[2] != seq_len {
            return Err(format!("Scores tensor must be square in the last two dimensions, got {:?}", shape).into());
        }

        let scores_data = scores.data_mut();

        for h in 0..head_count {
            for i in 0..seq_len { // Query sequence position
                for j in 0..seq_len { // Key sequence position
                    if j > i {
                        let index = h * seq_len * seq_len + i * seq_len + j;
                        if index < scores_data.len() {
                           scores_data[index] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Repeats Key/Value heads for Grouped-Query Attention.
    fn repeat_kv_heads(
        &self,
        tensor: &Tensor,
        num_groups: usize,
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        if num_groups == 1 {
            // No repetition needed, return a clone of the original tensor
            return Ok(tensor.clone());
        }

        let shape = tensor.shape();
        if shape.len() != 3 {
            return Err("Input tensor must be 3D [head_count_kv, seq_len, head_dim]".into());
        }
        let head_count_kv = shape[0];
        let seq_len = shape[1];
        let head_dim = shape[2];

        let new_head_count = head_count_kv * num_groups;
        let new_shape = vec![new_head_count, seq_len, head_dim];
        let total_elements = new_shape.iter().product();

        // Create input view
        let input_view = ArrayView3::from_shape((head_count_kv, seq_len, head_dim), tensor.data())?;
        
        // Create output array
        let mut output_array = Array3::<f32>::zeros((new_head_count, seq_len, head_dim));

        // Copy heads
        for kv_h in 0..head_count_kv {
            let head_slice = input_view.slice(s![kv_h, .., ..]);
            for g in 0..num_groups {
                output_array.slice_mut(s![kv_h * num_groups + g, .., ..]).assign(&head_slice);
            }
        }

        // Convert to Vec<f32>
        let output_data = output_array.into_raw_vec_and_offset().0;

        // Allocate memory and copy
        let mut new_memory = self.allocate_memory(total_elements)?;
        new_memory.as_mut_slice().copy_from_slice(&output_data);

        // Create new tensor
        Tensor::from_backend_memory(new_memory, new_shape, Arc::clone(tensor.backend()))
    }

    /// Performs batched matrix multiplication: C = A @ B
    fn bmm(
        &self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 3 || b_shape.len() != 3 {
            return Err("Inputs for bmm must be 3D tensors".into());
        }
        if a_shape[0] != b_shape[0] { // Batch sizes must match
            return Err(format!("Batch dimensions must match for bmm: {:?} vs {:?}", a_shape, b_shape).into());
        }

        let batch_size = a_shape[0]; // Size of the batch dimension (number of heads)
        let m = a_shape[1]; // Number of rows in each matrix of A (seq_len for Q)
        let k_a = a_shape[2]; // Number of columns in each matrix of A (head_dim for Q)
        let k_b = b_shape[1]; // Number of rows in each matrix of B (seq_len for K)
        let n = b_shape[2]; // Number of columns in each matrix of B (head_dim for K)

        // Validate dimensions for multiplication
        if transpose_b { // A[batch, m, k_a] @ B[batch, k_b, n].T (shape [batch, n, k_b])
            // Inner dimension k_a of A must match dimension n of B (which becomes the inner dim of B.T)
            if k_a != n { // Corrected check
                 // Corrected error message format
                 return Err(format!("Inner dimensions must match for A @ B.T (A:k={}, B:n={}): {:?} vs {:?}", k_a, n, a_shape, b_shape).into());
            }
        } else { // A[batch, m, k_a] @ B[batch, k_b, n]
            // Inner dimension k_a of A must match dimension k_b of B
            if k_a != k_b { // Correct check for this case
                 // Corrected error message format
                return Err(format!("Inner dimensions must match for A @ B (A:k={}, B:k={}): {:?} vs {:?}", k_a, k_b, a_shape, b_shape).into());
            }
        }
        
        // Determine output shape's last dimension 'n_out'
        let output_n = if transpose_b { k_b } else { n };

        let output_shape = vec![batch_size, m, output_n];
        let total_elements = output_shape.iter().product();

        let a_array = ArrayView3::from_shape((batch_size, m, k_a), a.data())
            .map_err(|e| format!("Error creating view for A in bmm: {}", e))?;
        let b_array = ArrayView3::from_shape((batch_size, k_b, n), b.data())
            .map_err(|e| format!("Error creating view for B in bmm: {}", e))?;

        let mut result_arrays = Vec::with_capacity(batch_size); // Store owned arrays here

        // Perform matmul for each item in the batch
        for i in 0..batch_size {
            //println!("      [bmm] Processing batch item {}", i);
            let a_slice = a_array.slice(s![i, .., ..]);
            let b_slice = b_array.slice(s![i, .., ..]);
            
            // Handle transposition
            let a_view = if transpose_a { unimplemented!("Transpose A not yet handled in bmm") } else { a_slice };
            let b_view = if transpose_b { b_slice.t() } else { b_slice.view() };
            
            // Check final shapes before dot product
            if a_view.shape()[1] != b_view.shape()[0] {
                 return Err(format!("Shape mismatch before dot product in batch {}: A_slice {:?}, B_view {:?}", i, a_view.shape(), b_view.shape()).into());
            }
            
            // Calculate result for this batch item
            //println!("      [bmm] Performing dot product for batch item {}", i);
            let c_slice: Array2<f32> = a_view.dot(&b_view); //.map_err(|e| format!("Error during dot product in bmm batch {}: {}", i, e))?;
            // Note: .dot() on views doesn't typically return a Result unless maybe BLAS fails?
            // Let's assume it panics or returns the array directly for now. If BLAS errors are possible, proper handling would be needed.

            result_arrays.push(c_slice); // Push owned array
        }
        // Create views from the owned arrays for stacking
        let result_views: Vec<_> = result_arrays.iter().map(|a| a.view()).collect();

        // Stack results into a 3D array
        let output_array = stack(Axis(0), &result_views)
            .map_err(|e| format!("Error during stack operation in bmm: {}", e))?; // Add error context here

        let output_data = output_array.into_raw_vec_and_offset().0; // Use updated method

        // Allocate memory and copy
        let mut new_memory = self.allocate_memory(total_elements)?;
        new_memory.as_mut_slice().copy_from_slice(&output_data);

        // Create new tensor
        Tensor::from_backend_memory(new_memory, output_shape, Arc::clone(a.backend()))
    }

    /// Permutes axes and reshapes the tensor's data in one operation.
    fn permute_and_reshape(
        &self,
        data: &[f32],
        current_shape: &[usize],
        new_axes: &[usize],
        target_shape: &[usize],
    ) -> Result<Box<dyn BackendMemory>, Box<dyn Error + Send + Sync>> {
        // Create an ndarray view with the original shape
        let view = ndarray::ArrayViewD::from_shape(IxDyn(current_shape), data)?;

        // Permute the axes
        let permuted_view = view.permuted_axes(IxDyn(new_axes));

        // Create a new owned array with the data in the permuted order (makes it contiguous)
        let permuted_array: ndarray::ArrayD<f32> = permuted_view.to_owned();

        // Validate the total number of elements matches the target shape
        let expected_elements: usize = target_shape.iter().product();
        if permuted_array.len() != expected_elements {
            return Err(format!(
                "Permute/Reshape error: Element count mismatch. Permuted has {} elements, target shape {:?} requires {}",
                permuted_array.len(), target_shape, expected_elements
            ).into());
        }

        // Convert the owned array's data into a Vec<f32>
        let permuted_data = permuted_array.into_raw_vec_and_offset().0;

        // Allocate new backend memory (size based on target_shape) and copy the contiguous permuted data
        let mut new_memory = self.allocate_memory(expected_elements)?;
        new_memory.as_mut_slice().copy_from_slice(&permuted_data);

        Ok(new_memory)
    }

    /// Perform matrix multiplication C = A * B using Tensors, returning a new Tensor.
    fn matmul_tensors(
        &self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err("matmul_tensors currently only supports 2D tensors".into());
        }

        // Extract dimensions based on potential transposition
        let m = if transpose_a { a_shape[1] } else { a_shape[0] };
        let k_a = if transpose_a { a_shape[0] } else { a_shape[1] };
        let k_b = if transpose_b { b_shape[1] } else { b_shape[0] };
        let n = if transpose_b { b_shape[0] } else { b_shape[1] };

        // Verify shapes are compatible for multiplication
        if k_a != k_b {
            return Err(format!(
                "Matrix multiplication dimension mismatch: A({}) vs B({}) (A shape: {:?}, B shape: {:?}, transpose_a: {}, transpose_b: {})",
                k_a, k_b, a_shape, b_shape, transpose_a, transpose_b
            ).into());
        }

        // Determine output shape
        let output_shape = vec![m, n];

        // Create output tensor (backend is cloned from tensor 'a')
        let mut result_tensor = Tensor::zeros(output_shape, Arc::clone(a.backend()))?;

        // Perform matrix multiplication using the backend's slice-based method
        self.matmul(
            a.data(),
            b.data(),
            result_tensor.data_mut(),
            m, n, k_a, // Use k_a as the inner dimension 'k'
            transpose_a,
            transpose_b,
        )?;

        Ok(result_tensor)
    }

    /// Perform element-wise addition C = A + B using Tensors (with broadcasting for B), returning a new Tensor.
    fn add_tensors(
        &self,
        a: &Tensor, // Typically the larger tensor (e.g., matrix)
        b: &Tensor, // Typically the smaller tensor (e.g., bias vector)
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        // Check for valid shapes: either identical shapes or B can be broadcast onto A
        let is_identical_shape = a_shape == b_shape;
        let is_broadcastable = b_shape.len() == 1 && a_shape.last() == b_shape.first() && a_shape.len() > 0;

        if !is_identical_shape && !is_broadcastable {
            return Err(format!(
                "Shape mismatch for add_tensors: A {:?} and B {:?} are not identical or broadcastable",
                a_shape, b_shape
            ).into());
        }

        // Output tensor has the same shape as A
        let mut result_tensor = Tensor::zeros(a_shape.to_vec(), Arc::clone(a.backend()))?;

        // Perform addition using the backend's slice-based method
        self.add(
            a.data(),
            b.data(), // Bias vector data
            result_tensor.data_mut(),
        )?;

        Ok(result_tensor)
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
