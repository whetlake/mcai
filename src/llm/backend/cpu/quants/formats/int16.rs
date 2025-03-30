use std::error::Error;
use crate::gguf::GGUFValueType;
use super::FormatImpl;
use super::super::utils::{parallel_process, check_data_availability};

/// INT16 format - 16-bit signed integers
#[derive(Clone)]
pub struct INT16Format {}

impl INT16Format {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for INT16Format {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::INT16
    }
    
    fn name(&self) -> &'static str {
        "INT16"
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
        // For INT16, we need 2 bytes per element
        let bytes_needed = num_elements * 2;
        
        // Calculate sizes for reporting
        let actual_size_mb = bytes_needed as f32 / (1024.0 * 1024.0);
        let f32_size_mb = (num_elements * 4) as f32 / (1024.0 * 1024.0);
        let compression_ratio = f32_size_mb / actual_size_mb;
        
        // println!("INT16 Format Size Details:");
        // println!("  Total bytes needed: {} ({:.4} MB)", bytes_needed, actual_size_mb);
        // println!("  Equivalent F32 size: {:.4} MB", f32_size_mb);
        // println!("  Compression ratio: {:.2}x", compression_ratio);
        
        // Ensure we have enough data using the utility function
        check_data_availability(data, *offset, bytes_needed, "INT16")?;
        
        // Check alignment - offset should be aligned to 2 bytes for i16 access
        if *offset % 2 != 0 {
            return Err("Misaligned offset for INT16 data. Offset must be a multiple of 2 bytes.".into());
        }
        
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
        // 2MB is a reasonable threshold for INT16 data
        let parallel_threshold = (2 * 1024 * 1024) / (2 * block_size); // ~2MB in blocks
        
        // Use parallel_process to handle parallelization with auto thread detection
        parallel_process(
            source_data,
            num_blocks,
            block_size,
            block_size * 2, // For INT16, bytes_per_block = 2 * elements_per_block
            parallel_threshold,
            0,   // Auto-detect thread count
            result.as_mut_slice(),
            dequantize_int16_blocks
        );
        
        // Update offset
        *offset += bytes_needed;
        
        Ok(())
    }
}

/// Function to dequantize a range of INT16 values with optimized memory access
#[inline(always)]
fn dequantize_int16_blocks(
    data: &[u8],
    base_offset: usize,
    start_block: usize,
    end_block: usize,
    bytes_per_block: usize,
    result: &mut [f32]
) {
    // For INT16, element count is half the byte count
    let elements_per_block = bytes_per_block / 2;
    
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
    // For INT16 format, we need to convert each pair of bytes to i16 and then to f32
    for i in 0..element_count {
        let idx = byte_offset + i * 2;
        
        // Read 2 bytes and convert to i16
        let bytes = [data[idx], data[idx+1]];
        let i16_val = i16::from_le_bytes(bytes);
        
        result[start_element + i] = i16_val as f32;
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(INT16Format::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_int16_format() {
        let format = INT16Format::new();
        assert_eq!(format.name(), "INT16");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::INT16 as u32);
        
        // Test dequantization with some sample data
        let mut data = Vec::new();
        
        // Positive values: 0, 1, 32767
        data.extend_from_slice(&0i16.to_le_bytes());
        data.extend_from_slice(&1i16.to_le_bytes());
        data.extend_from_slice(&32767i16.to_le_bytes());
        
        // Negative values: -1, -2, -32768
        data.extend_from_slice(&(-1i16).to_le_bytes());
        data.extend_from_slice(&(-2i16).to_le_bytes());
        data.extend_from_slice(&(-32768i16).to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 6, &mut result).unwrap();
        
        assert_eq!(result, vec![0.0, 1.0, 32767.0, -1.0, -2.0, -32768.0]);
        assert_eq!(offset, 12); // 6 * 2 bytes
    }
    
    #[test]
    fn test_int16_not_enough_data() {
        let format = INT16Format::new();
        let mut data = Vec::new();
        data.extend_from_slice(&1i16.to_le_bytes());
        data.extend_from_slice(&2i16.to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Try to dequantize 3 elements when only 2 are available
        let err = format.dequantize(&data, &mut offset, 3, &mut result).unwrap_err();
        assert!(err.to_string().contains("Not enough data"));
    }
    
    #[test]
    fn test_int16_misaligned_offset() {
        let format = INT16Format::new();
        let data = vec![0, 1, 2, 3, 4, 5];
        let mut result = Vec::new();
        let mut offset = 1; // Misaligned offset
        
        let err = format.dequantize(&data, &mut offset, 2, &mut result).unwrap_err();
        assert!(err.to_string().contains("Misaligned offset"));
    }
    
    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_deliberately_failing() {
        let format = INT16Format::new();
        let mut data = Vec::new();
        data.extend_from_slice(&1i16.to_le_bytes());
        data.extend_from_slice(&2i16.to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 2, &mut result).unwrap();
        
        // This assertion is deliberately wrong
        assert_eq!(result, vec![10.0, 20.0], "This test is designed to fail to verify that tests are actually validating functionality");
    }
} 