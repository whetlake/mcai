use std::error::Error;
use crate::gguf::GGUFValueType;
use super::FormatImpl;
use super::super::utils::{parallel_process, check_data_availability};

/// UINT8 format - 8-bit unsigned integers
#[derive(Clone)]
pub struct Uint8Format {}

impl Uint8Format {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Uint8Format {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::UINT8
    }
    
    fn name(&self) -> &'static str {
        "UINT8"
    }
    
    fn clone_box(&self) -> Box<dyn FormatImpl> {
        Box::new(self.clone())
    }

    fn dequantize(
        &self,
        data: &[u8],
        offset: &mut usize,
        num_elements: usize,
        result: &mut Vec<f32>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // For UINT8, we need 1 byte per element
        let bytes_needed = num_elements;
        
        // Calculate sizes for reporting
        let actual_size_mb = bytes_needed as f32 / (1024.0 * 1024.0);
        let f32_size_mb = (num_elements * 4) as f32 / (1024.0 * 1024.0);
        let compression_ratio = f32_size_mb / actual_size_mb;
        
        println!("UINT8 Format Size Details:");
        println!("  Total bytes needed: {} ({:.4} MB)", bytes_needed, actual_size_mb);
        println!("  Equivalent F32 size: {:.4} MB", f32_size_mb);
        println!("  Compression ratio: {:.2}x", compression_ratio);
        
        // Ensure we have enough data using the utility function
        check_data_availability(data, *offset, bytes_needed, "UINT8")?;
        
        // Resize result vector to hold all elements
        result.resize(num_elements, 0.0);
        
        // Get the data slice we need
        let source_data = &data[*offset..*offset + bytes_needed];
        
        // Adaptive block size based on tensor size
        // Larger blocks reduce threading overhead for smaller tensors
        // Smaller blocks provide better parallelism for larger tensors
        let block_size = if bytes_needed > 32 * 1024 * 1024 {
            // For tensors > 32MB, use smaller blocks (8K elements)
            8 * 1024
        } else if bytes_needed > 4 * 1024 * 1024 {
            // For tensors 4-32MB, use medium blocks (16K elements)
            16 * 1024
        } else {
            // For smaller tensors, use larger blocks (32K elements)
            32 * 1024
        };
        
        // Calculate number of blocks
        let num_blocks = (num_elements + block_size - 1) / block_size;
        
        // More aggressive threshold for parallelization
        // Use multiple threads only when there's enough data to justify the overhead
        // 2MB is a reasonable threshold for UINT8 data
        let parallel_threshold = (2 * 1024 * 1024) / block_size; // ~2MB in blocks
        
        // Use parallel_process to handle parallelization with auto thread detection
        parallel_process(
            source_data,
            num_blocks,
            block_size,
            block_size, // For UINT8, bytes_per_block equals elements_per_block
            parallel_threshold,
            0,   // Auto-detect thread count
            result.as_mut_slice(),
            dequantize_uint8_blocks
        );
        
        // Update offset
        *offset += bytes_needed;
        
        Ok(())
    }
}

/// Function to dequantize a range of UINT8 values with optimized memory access
#[inline(always)]
fn dequantize_uint8_blocks(
    data: &[u8],
    base_offset: usize,
    start_block: usize,
    end_block: usize,
    bytes_per_block: usize,
    result: &mut [f32]
) {
    // For UINT8, element count equals byte count
    let elements_per_block = bytes_per_block;
    
    // Calculate start and end positions in elements
    let start_element = start_block * elements_per_block;
    let end_element = if end_block * elements_per_block > result.len() {
        result.len()  // Handle the last chunk which might be partial
    } else {
        end_block * elements_per_block
    };
    
    // Calculate byte offset and element count
    let element_count = end_element - start_element;
    let byte_offset = base_offset + start_block * bytes_per_block;
    
    // Process the entire range at once - simpler approach
    // For UINT8 format, we just need to convert each byte to f32
    for i in 0..element_count {
        result[start_element + i] = data[byte_offset + i] as f32;
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Uint8Format::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_uint8_format() {
        let format = Uint8Format::new();
        assert_eq!(format.name(), "UINT8");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::UINT8 as u32);
        
        // Test dequantization with some sample data
        let data = vec![0, 1, 2, 3, 4, 255];
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 6, &mut result).unwrap();
        
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 255.0]);
        assert_eq!(offset, 6);
    }
    
    #[test]
    fn test_uint8_not_enough_data() {
        let format = Uint8Format::new();
        let data = vec![0, 1, 2]; // Only 3 bytes
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Try to dequantize 5 elements when only 3 are available
        let err = format.dequantize(&data, &mut offset, 5, &mut result).unwrap_err();
        assert!(err.to_string().contains("Not enough data"));
    }
    
    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_deliberately_failing() {
        let format = Uint8Format::new();
        let data = vec![1, 2, 3];
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 3, &mut result).unwrap();
        
        // This assertion is deliberately wrong - the result should be [1.0, 2.0, 3.0]
        // but we're asserting it's [10.0, 20.0, 30.0]
        assert_eq!(result, vec![10.0, 20.0, 30.0], "This test is designed to fail to verify that tests are actually validating functionality");
    }
} 