use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::llm::backend::backend::{Backend, BackendMemory};
use crate::gguf::TensorInfo;

/// A tensor that can be operated on by a backend
pub struct Tensor {
    /// The backend this tensor belongs to
    backend: Arc<Box<dyn Backend>>,
    /// The tensor's data in backend-specific memory
    memory: Box<dyn BackendMemory>,
    /// The tensor's shape
    shape: Vec<usize>,
    /// The tensor's name
    name: String,
}

impl Tensor {
    /// Create a new tensor from raw data and tensor info
    pub fn new(
        data: &[u8],
        info: &TensorInfo,
        backend: Arc<Box<dyn Backend>>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Calculate total number of elements
        let total_elements = info.dims.iter().map(|&d| d as usize).product();
        
        // Allocate memory in the backend
        let mut memory = backend.allocate_memory(total_elements)?;
        
        // Dequantize data directly into backend memory
        backend.dequantize_to_memory(
            data,
            info.offset as usize,
            total_elements,
            info.data_type,
            &mut memory
        )?;
        
        Ok(Self {
            backend,
            memory,
            shape: info.dims.iter().map(|&d| d as usize).collect(),
            name: info.name.clone(),
        })
    }

    /// Create a new tensor filled with zeros
    pub fn zeros(shape: Vec<usize>, backend: Arc<Box<dyn Backend>>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let total_elements = shape.iter().product();
        let memory = backend.allocate_memory(total_elements)?;
        
        Ok(Self {
            backend,
            memory,
            shape,
            name: String::new(),
        })
    }

    /// Get the tensor's shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the tensor's name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the tensor's data as a slice
    pub fn data(&self) -> &[f32] {
        self.memory.as_slice()
    }

    /// Get the tensor's data as a mutable slice
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.memory.as_mut_slice()
    }

    /// Convert the tensor's data to CPU format
    pub fn to_cpu(&self) -> Vec<f32> {
        self.memory.to_cpu()
    }

    /// Get the backend this tensor belongs to
    pub fn backend(&self) -> &Arc<Box<dyn Backend>> {
        &self.backend
    }

    /// Reshapes the tensor's shape metadata in place.
    /// Requires the total number of elements to remain the same.
    /// This operation is very cheap as it does not copy data.
    pub fn reshape_in_place(&mut self, new_shape: Vec<usize>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let current_elements: usize = self.shape.iter().product();
        let new_elements: usize = new_shape.iter().product();

        if current_elements != new_elements {
            return Err(format!(
                "Cannot reshape tensor in place with {} elements (shape: {:?}) to {} elements (shape: {:?})",
                current_elements, self.shape, new_elements, new_shape
            )
            .into());
        }

        // Just update the shape metadata
        self.shape = new_shape;
        Ok(())
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("name", &self.name)
            .finish()
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let mut memory = self.backend.allocate_memory(self.shape.iter().product())
            .expect("Failed to allocate memory for tensor clone");
            
        // Copy data from original tensor to new memory
        // Ensure we are using the correct method to access data on BackendMemory
        memory.as_mut_slice().copy_from_slice(self.memory.as_slice());
            
        Self {
            backend: Arc::clone(&self.backend),
            memory,
            shape: self.shape.clone(),
            name: self.name.clone(),
        }
    }
}