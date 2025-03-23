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
    pub fn new(data: Vec<f32>, shape: Vec<usize>, backend: Arc<Box<dyn Backend>>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Check that data size matches shape
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(format!("Data length ({}) does not match shape {:?} (expected {})", 
                               data.len(), shape, expected_size).into());
        }
        
        Ok(Self { data, shape, backend })
    }
    
    /// Create a tensor by loading and dequantizing from raw model data
    pub fn from_tensor_info(
        data: &[u8],
        tensor_info: &TensorInfo,
        backend: Arc<Box<dyn Backend>>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Create the shape from dimensions
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
        
        // Create a new tensor with the dequantized data
        Self::new(dequantized_data, shape, backend)
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
    
    /// Perform matrix multiplication with another tensor
    pub fn matmul(&self, other: &Self) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Check that shapes are compatible for matmul
        if self.shape.len() < 2 || other.shape.len() < 2 {
            return Err("Both tensors must have at least 2 dimensions for matmul".into());
        }
        
        let m = self.shape[self.shape.len() - 2];  // rows of A
        let k = self.shape[self.shape.len() - 1];  // cols of A = rows of B
        let n = other.shape[other.shape.len() - 1]; // cols of B
        
        if k != other.shape[other.shape.len() - 2] {
            return Err(format!("Incompatible dimensions for matmul: {:?} and {:?}", 
                              self.shape, other.shape).into());
        }
        
        // Create output tensor
        let mut result_shape = self.shape.clone();
        let last_idx = result_shape.len() - 1;
        result_shape[last_idx] = n;
        let mut result = Self::zeros(result_shape, Arc::clone(&self.backend));
        
        // Call backend matmul implementation
        self.backend.matmul(
            &self.data,
            &other.data,
            &mut result.data,
            m,
            n,
            k,
            false,
            false,
        )?;
        
        Ok(result)
    }
    
    /// Add another tensor to this one
    pub fn add(&self, other: &Self) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Check that shapes are compatible for addition
        if self.shape != other.shape {
            return Err(format!("Incompatible shapes for addition: {:?} and {:?}",
                              self.shape, other.shape).into());
        }
        
        // Create output tensor
        let mut result = Self::zeros(self.shape.clone(), Arc::clone(&self.backend));
        
        // Call backend add implementation
        self.backend.add(
            &self.data,
            &other.data,
            &mut result.data,
        )?;
        
        Ok(result)
    }
    
    /// Multiply element-wise with another tensor
    pub fn mul(&self, other: &Self) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Check that shapes are compatible for multiplication
        if self.shape != other.shape {
            return Err(format!("Incompatible shapes for multiplication: {:?} and {:?}",
                              self.shape, other.shape).into());
        }
        
        // Create output tensor
        let mut result = Self::zeros(self.shape.clone(), Arc::clone(&self.backend));
        
        // Call backend mul implementation
        self.backend.mul(
            &self.data,
            &other.data,
            &mut result.data,
        )?;
        
        Ok(result)
    }
    
    /// Calculate dot product with another tensor
    pub fn dot(&self, other: &Self) -> Result<f32, Box<dyn Error + Send + Sync>> {
        // Check that both tensors are 1D with the same size
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err("Both tensors must be 1D for dot product".into());
        }
        
        if self.shape[0] != other.shape[0] {
            return Err(format!("Incompatible shapes for dot product: {:?} and {:?}",
                             self.shape, other.shape).into());
        }
        
        // Call backend dot implementation
        self.backend.dot(&self.data, &other.data)
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