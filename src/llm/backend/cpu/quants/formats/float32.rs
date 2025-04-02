use std::error::Error;
use std::ptr;
use crate::gguf::GGUFValueType;
use super::FormatImpl;
use super::super::utils::{parallel_process, check_data_availability};

/// Float32 format - direct storage of 32-bit floating point values
#[derive(Clone)]
pub struct Float32Format;

impl Float32Format {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Float32Format {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::FLOAT32
    }
    
    fn name(&self) -> &'static str {
        "FLOAT32"
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
        // For FLOAT32, we need 4 bytes per element
        let bytes_needed = num_elements * 4;
        
        // Calculate sizes for reporting
        let _actual_size_mb = bytes_needed as f32 / (1024.0 * 1024.0);
        
        // println!("FLOAT32 Format Size Details:");
        // println!("  Total bytes needed: {} ({:.4} MB)", bytes_needed, actual_size_mb);
        // println!("  Compression ratio: 1.00x (uncompressed)");
        
        // Ensure we have enough data using the utility function
        check_data_availability(data, *offset, bytes_needed, "FLOAT32")?;
        
        // Check alignment - offset should be aligned to 4 bytes for f32 access
        if *offset % 4 != 0 {
            return Err("Misaligned offset for FLOAT32 data. Offset must be a multiple of 4 bytes.".into());
        }
        
        // Resize result vector to hold all elements
        result.resize(num_elements, 0.0);
        
        // Get the data slice we need
        let source_data = &data[*offset..*offset + bytes_needed];
        
        // Check if we can do a direct aligned memory copy (optimal case)
        // Both source and destination need to be 4-byte aligned
        let can_direct_copy = (source_data.as_ptr() as usize) % 4 == 0 && 
                            (result.as_ptr() as usize) % 4 == 0;
        
        if can_direct_copy {
            // SAFETY: We've verified alignment, size, and non-overlap
            // This is the fastest approach - direct memory copy
            unsafe {
                ptr::copy_nonoverlapping(
                    source_data.as_ptr() as *const f32,
                    result.as_mut_ptr(),
                    num_elements
                );
            }
        } else {
            // For larger tensors, parallel processing is beneficial
            // For smaller ones, single-threaded is more efficient
            
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
            
            // More aggressive threshold for parallelization - only use multiple threads
            // for tensors that are large enough to benefit
            // 12MB is a reasonable threshold where threading overhead is justified
            let parallel_threshold = (3 * 1024 * 1024) / (4 * block_size); // ~3MB in blocks
            
            // Use parallel_process to handle parallelization with auto thread detection
            parallel_process(
                source_data,
                num_blocks,
                block_size,
                block_size * 4, // bytes_per_block = 4 * elements_per_block
                parallel_threshold,
                0,   // Auto-detect thread count
                result.as_mut_slice(),
                dequantize_float32_blocks
            );
        }
        
        // Update offset
        *offset += bytes_needed;
        
        Ok(())
    }
}

/// Function to dequantize a range of FLOAT32 values with optimized memory access
#[inline(always)]
fn dequantize_float32_blocks(
    data: &[u8],
    base_offset: usize,
    start_block: usize,
    end_block: usize,
    bytes_per_block: usize,
    result: &mut [f32]
) {
    // Get element count per block
    let elements_per_block = bytes_per_block / 4;
    
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
    
    // Copy the entire assigned range at once - this is more efficient
    // than processing block by block for float32 data
    unsafe {
        // This is safe because:
        // 1. We've verified we have enough data in the main dequantize function
        // 2. We're handling the last partial block correctly
        // 3. Float32 data is just a reinterpretation of the bytes
        let src_ptr = data.as_ptr().add(byte_offset) as *const f32;
        let dst_ptr = result.as_mut_ptr().add(start_element);
        
        // Direct memory copy
        std::ptr::copy_nonoverlapping(
            src_ptr,
            dst_ptr,
            element_count
        );
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Float32Format::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_float32_format() {
        let format = Float32Format::new();
        assert_eq!(format.name(), "FLOAT32");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::FLOAT32 as u32);
        
        // Test dequantization with some sample data
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 3, &mut result).unwrap();
        
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        assert_eq!(offset, 12); // 3 * 4 bytes
    }
    
    #[test]
    fn test_float32_format_misaligned() {
        let format = Float32Format::new();
        
        // Create test data with one byte padding to cause misalignment
        let mut data = vec![0u8]; // Padding byte
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 1; // Start at the misaligned position
        
        // This should return an error due to misalignment
        let err = format.dequantize(&data, &mut offset, 2, &mut result).unwrap_err();
        assert!(err.to_string().contains("Misaligned offset"));
    }
    
    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_deliberately_failing() {
        let format = Float32Format::new();
        
        // Create valid test data
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        // The function should succeed
        format.dequantize(&data, &mut offset, 2, &mut result).unwrap();
        
        // This assertion is deliberately wrong - the result should be [1.0, 2.0]
        // but we're asserting it's [3.0, 4.0]
        assert_eq!(result, vec![3.0, 4.0], "This test is designed to fail to verify that tests are actually validating functionality");
    }
} 