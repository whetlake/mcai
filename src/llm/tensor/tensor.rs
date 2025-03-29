use std::error::Error;
use std::sync::Arc;
use std::fmt::{self, Debug};

use crate::llm::backend::Backend;
use crate::gguf::TensorInfo;

/// A tensor representing a multi-dimensional array
#[derive(Clone)]
pub struct Tensor {
    /// The raw data as a contiguous array of f32 values
    data: Vec<f32>,
    /// The shape of the tensor (e.g., [batch_size, seq_len, hidden_size])
    shape: Vec<usize>,
    /// The backend used for operations on this tensor
    backend: Arc<Box<dyn Backend>>,
}

impl Tensor {
    /// Create a new tensor with the given data and shape
    pub fn new(data: &[u8], tensor_info: &TensorInfo, backend: Arc<Box<dyn Backend>>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let shape: Vec<usize> = tensor_info.dims.iter().map(|&d| d as usize).collect();
        
        // Calculate total elements from dimensions
        let total_elements: usize = shape.iter().product();

        // Use the backend's dequantize method to convert raw data to f32
        let offset = tensor_info.offset as usize;
        let dequantized_data = backend.dequantize(
            data, 
            offset, 
            total_elements, 
            tensor_info.data_type
        )?;
        
        // Check that data size matches shape
        if dequantized_data.len() != total_elements {
            return Err(format!("Dequantized data length ({}) does not match shape {:?} (expected {})", 
                               dequantized_data.len(), shape, total_elements).into());
        }
        
        Ok(Self { data: dequantized_data, shape, backend })
    }
    
    
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: Vec<usize>, backend: Arc<Box<dyn Backend>>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
            backend,
        }
    }
    
    /// Create a new tensor filled with ones
    pub fn ones(shape: Vec<usize>, backend: Arc<Box<dyn Backend>>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape,
            backend,
        }
    }
    
    /// Get a reference to the tensor's raw data
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    
    /// Get a mutable reference to the tensor's raw data
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the size (total number of elements) in the tensor
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Reshape the tensor to a new shape with the same total number of elements
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(format!("Cannot reshape tensor of size {} to shape {:?} (size {})",
                              self.size(), new_shape, new_size).into());
        }
        
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            backend: Arc::clone(&self.backend),
        })
    }
    
    /// Get a reference to the tensor's backend
    pub fn backend(&self) -> &Arc<Box<dyn Backend>> {
        &self.backend
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {{ shape: {:?}, data: truncated }}", self.shape)
    }
} 