use std::error::Error;
use crate::gguf::{GGUFValueType, TensorInfo};
use crate::llm::quants::dequantize::Dequantizer;

/// Provides utilities for working with tensor data
pub struct TensorUtils;

impl TensorUtils {
    /// Converts tensor data from raw bytes to f32 values
    ///
    /// # Arguments
    /// * `data` - Raw bytes from memory map
    /// * `offset` - Offset in the data where the tensor begins
    /// * `tensor_info` - Information about the tensor dimensions and type
    ///
    /// # Returns
    /// * `Vec<f32>` - Tensor data converted to f32 values
    pub fn convert_tensor_data(
        data: &[u8], 
        offset: usize, 
        tensor_info: &TensorInfo
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        let total_elements: usize = tensor_info.dims.iter().map(|&d| d as usize).product();
        
        // Use the specialized Dequantizer to handle all tensor formats
        Dequantizer::dequantize(data, offset, total_elements, tensor_info.data_type)
    }
    

    /// Calculates the total bytes needed for a tensor, including handling quantized formats
    ///
    /// # Arguments
    /// * `tensor_info` - Information about the tensor
    ///
    /// # Returns
    /// * `usize` - Total bytes needed for this tensor in memory
    pub fn calculate_tensor_size(tensor_info: &TensorInfo) -> Result<usize, Box<dyn Error + Send + Sync>> {
        let total_elements: i64 = tensor_info.dims.iter().map(|&d| d as i64).product();
        
        match tensor_info.data_type {
            // K-quant formats (newer formats from llama.cpp)
            GGUFValueType::Q2_K => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 2 / 8 + 2)) as usize)
            },
            GGUFValueType::Q3_K_M | GGUFValueType::Q3_K_L | GGUFValueType::Q3_K_S => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 3 / 8 + 2)) as usize)
            },
            GGUFValueType::Q4_K_M | GGUFValueType::Q4_K_S => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 4 / 8 + 2)) as usize)
            },
            GGUFValueType::Q5_K_M | GGUFValueType::Q5_K_S => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 5 / 8 + 2)) as usize)
            },
            GGUFValueType::Q6_K => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 6 / 8 + 2)) as usize)
            },
            
            // Original quant formats
            GGUFValueType::Q4_0 => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 4 / 8 + 1)) as usize)
            },
            GGUFValueType::Q4_1 => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 4 / 8 + 2)) as usize)
            },
            GGUFValueType::Q5_0 => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 5 / 8 + 1)) as usize)
            },
            GGUFValueType::Q5_1 => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size * 5 / 8 + 2)) as usize)
            },
            GGUFValueType::Q8_0 => {
                let block_size = 32;
                let blocks = (total_elements + block_size - 1) / block_size;
                Ok((blocks * (block_size + 1)) as usize)
            },
            
            // Standard non-quantized formats - simple calculation
            GGUFValueType::FLOAT32 => Ok((total_elements * 4) as usize),
            GGUFValueType::UINT8 | GGUFValueType::INT8 => Ok(total_elements as usize),
            GGUFValueType::UINT16 | GGUFValueType::INT16 => Ok((total_elements * 2) as usize),
            GGUFValueType::UINT32 | GGUFValueType::INT32 => Ok((total_elements * 4) as usize),
            GGUFValueType::UINT64 | GGUFValueType::INT64 => Ok((total_elements * 8) as usize),
            
            // Unsupported types
            _ => Err(format!("Unsupported data type for size calculation: {:?}", tensor_info.data_type).into()),
        }
    }
} 