use std::error::Error;
use crate::gguf::GGUFValueType;
use super::formats::get_format_by_gguf_type;

/// Provides utilities for dequantizing model weights from their compressed formats
pub struct Dequantizer;
impl Dequantizer {
    /// Dequantizes a tensor from its compressed format to f32 values
    ///
    /// # Arguments
    /// * `data` - The raw tensor data
    /// * `offset` - The offset in bytes where the tensor data starts
    /// * `total_elements` - The number of elements in the tensor
    /// * `data_type` - The data type of the tensor
    ///
    /// # Returns
    /// * A vector of f32 values representing the dequantized tensor
    pub fn dequantize(
        data: &[u8],
        offset: usize,
        total_elements: usize,
        data_type: GGUFValueType,
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
                
        // Try to get a format implementation for this data type
        if let Some(format) = get_format_by_gguf_type(data_type) {
            // Use the format implementation to dequantize the data
            let mut result = Vec::with_capacity(total_elements);
            let mut current_offset = offset;
            
            // Start timer for performance measurement
            let start_time = std::time::Instant::now();
            
            // Dequantize the data
            format.dequantize(data, &mut current_offset, total_elements, &mut result)?;
            
            // Calculate elapsed time
            let elapsed = start_time.elapsed();

            println!("Dequantization completed in {:.2?} of type {:?}", elapsed, data_type);
            
            Ok(result)
        } else {
            // No format implementation found for this data type
            Err(format!("Unsupported data type for dequantization: {:?}. No format implementation available.", data_type).into())
        }
    }
}
