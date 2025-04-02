use std::error::Error;
use crate::gguf::GGUFValueType;
use super::{FormatImpl, QK_K};
use super::super::utils::{f16_to_f32, parallel_process, check_data_availability};

/// Q4_K_S format - 4-bit quantization with K=256 (small variant)
/// 
/// This format stores 4-bit quantized values with a simpler bit-packing scheme
/// than the medium variant, trading efficiency for simplicity.
/// 
/// Format structure per block (256 elements):
/// - 128 bytes for quantized values (4 bits per value, 2 values per byte)
/// - 2 bytes for min value (f16)
/// - 2 bytes for d value (max-min scale, f16)
/// Total: 132 bytes per 256 elements (compared to 1024 bytes for F32)
#[derive(Clone)]
pub struct Q4KSFormat {}

impl Q4KSFormat {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Q4KSFormat {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::Q4_K_S
    }
    
    fn name(&self) -> &'static str {
        "Q4_K_S"
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
        // Q4_K_S works on blocks of QK_K (256) elements
        if num_elements % QK_K != 0 {
            return Err(format!("For Q4_K_S format, number of elements must be a multiple of {}, got {}", QK_K, num_elements).into());
        }
        
        let num_blocks = num_elements / QK_K;
        
        // Calculate size needed for each block:
        // - QK_K/2 bytes for quantized values (4 bits per value)
        // - 2 bytes (1 f16) for min value
        // - 2 bytes (1 f16) for d value (max - min)
        let bytes_per_block = QK_K/2 + 2 + 2;
        let bytes_needed = num_blocks * bytes_per_block;
        
        // Calculate sizes for reporting
        let actual_size_mb = bytes_needed as f32 / (1024.0 * 1024.0);
        let f32_size_mb = (num_elements * 4) as f32 / (1024.0 * 1024.0);
        let _compression_ratio = f32_size_mb / actual_size_mb;
        
        // println!("Q4_K_S Format Size Details:");
        // println!("  Number of blocks: {}", num_blocks);
        // println!("  Bytes per block: {} bytes", bytes_per_block);
        // println!("  Total bytes needed: {} ({:.4} MB)", bytes_needed, actual_size_mb);
        // println!("  Equivalent F32 size: {:.4} MB", f32_size_mb);
        // println!("  Compression ratio: {:.2}x", compression_ratio);
        
        // Ensure we have enough data using utility function
        check_data_availability(data, *offset, bytes_needed, "Q4_K_S")?;
        
        // Check alignment - offset should be aligned to at least 2 bytes for f16 access
        if *offset % 2 != 0 {
            return Err("Misaligned offset for Q4_K_S data. Offset must be a multiple of 2 bytes.".into());
        }
        
        // Preallocate the result vector with zeros
        result.resize(num_elements, 0.0);
        
        // Extract the data we need to process
        // For memory mapped files, we're only reading the portion we need
        let base_offset = *offset;
        let data_slice = &data[base_offset..base_offset + bytes_needed];
        
        // For very large tensors, use smaller blocks for better thread distribution
        // This helps avoid out of bounds errors when dividing work among threads
        let optimal_block_count = if num_blocks > 1_000_000 {
            // For extremely large tensors, use smaller blocks per thread to avoid memory issues
            64
        } else if num_blocks > 100_000 {
            // For large tensors
            128
        } else if num_blocks > 10_000 {
            // For medium tensors
            192
        } else {
            // For smaller tensors
            256
        };
        
        // Use the parallel_process utility to handle parallelization
        // Pass 0 for num_threads to auto-detect the optimal thread count
        parallel_process(
            data_slice,
            num_blocks,
            QK_K,
            bytes_per_block,
            optimal_block_count, // Adaptive threshold for parallel processing
            0,   // Auto-detect thread count
            result.as_mut_slice(),
            dequantize_blocks
        );
        
        // Update offset for next read
        *offset += bytes_needed;
        
        Ok(())
    }
}

/// Function to dequantize a range of blocks - extracted to allow parallel processing
#[inline(always)]
fn dequantize_blocks(
    data: &[u8],
    base_offset: usize, 
    start_block: usize, 
    end_block: usize, 
    bytes_per_block: usize,
    result: &mut [f32]
) {
    // Constants for better readability
    const QS_SIZE: usize = QK_K/2; // 4 bits per value (2 values per byte)
    
    // Process each block in range
    for i in 0..(end_block - start_block) {
        let block_idx = start_block + i;
        let block_offset = base_offset + block_idx * bytes_per_block;
        
        // Calculate offsets for different parts of the block
        let qs_offset = block_offset;
        let min_offset = qs_offset + QS_SIZE;
        let d_offset = min_offset + 2;
        
        // Get min value (f16 stored as 2 bytes)
        let min_val = f16_to_f32(&[data[min_offset], data[min_offset + 1]]);
        
        // Get d value (max - min, f16 stored as 2 bytes)
        let d = f16_to_f32(&[data[d_offset], data[d_offset + 1]]);
        
        // Calculate output position in the result slice
        let output_offset = i * QK_K;
        
        // Process the entire block with efficient method
        process_block(
            &data[qs_offset..qs_offset + QS_SIZE],
            min_val,
            d,
            &mut result[output_offset..output_offset + QK_K]
        );
    }
}

/// Process a block of 256 quantized values
/// 
/// Each 4-bit value q is dequantized as: min_val + q * d / 15
/// where 15 is the maximum value representable in 4 bits (0xF)
#[inline(always)]
fn process_block(
    qs_data: &[u8],
    min_val: f32,
    d: f32,
    output: &mut [f32]
) {
    // Process all values in the block
    // Each byte contains two 4-bit values
    for i in 0..qs_data.len() {
        let byte = qs_data[i];
        
        // Extract the two 4-bit values
        let qs1 = byte & 0x0F;
        let qs2 = (byte >> 4) & 0x0F;
        
        // Calculate the final values
        // Formula: min_val + q * d / 15 (normalized by max possible 4-bit value)
        let val1 = min_val + (qs1 as f32) * d / 15.0;
        let val2 = min_val + (qs2 as f32) * d / 15.0;
        
        // Store in output buffer
        let out_idx = i * 2;
        output[out_idx] = val1;
        output[out_idx + 1] = val2;
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Q4KSFormat::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_q4_k_s_format_properties() {
        let format = Q4KSFormat::new();
        assert_eq!(format.name(), "Q4_K_S");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::Q4_K_S as u32);
    }

    #[test]
    fn test_q4_k_s_requires_block_multiple() {
        let format = Q4KSFormat::new();
        let data = vec![0u8; 1000]; // Some arbitrary data
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Try with non-multiple of QK_K
        let err = format.dequantize(&data, &mut offset, 100, &mut result).unwrap_err();
        assert!(err.to_string().contains("must be a multiple of"));
    }
    
    #[test]
    fn test_q4_k_s_not_enough_data() {
        let format = Q4KSFormat::new();
        let data = vec![0u8; 50]; // Not enough for even one block
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Try with one block (256 elements)
        let err = format.dequantize(&data, &mut offset, QK_K, &mut result).unwrap_err();
        assert!(err.to_string().contains("Not enough data"));
    }
    
    #[test]
    fn test_q4_k_s_misaligned_offset() {
        let format = Q4KSFormat::new();
        let data = vec![0u8; 1000]; // Plenty of data
        let mut result = Vec::new();
        let mut offset = 1; // Misaligned offset
        
        // Try with one block
        let err = format.dequantize(&data, &mut offset, QK_K, &mut result).unwrap_err();
        assert!(err.to_string().contains("Misaligned offset"));
    }
    
    #[test]
    fn test_q4_k_s_dequantization() {
        // Create a simple test block with known values
        let mut data = Vec::new();
        
        // Create 128 bytes for quantized values (256 4-bit values)
        // Pattern: alternating 0x12 and 0x34
        for _ in 0..64 {
            data.push(0x12); // 0001 0010 in binary
            data.push(0x34); // 0011 0100 in binary
        }
        
        // Add min_val as f16 (0.0)
        data.extend_from_slice(&[0, 0]);
        
        // Add d as f16 (1.0)
        data.extend_from_slice(&[0, 60]); // 1.0 in f16 format
        
        let format = Q4KSFormat::new();
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, QK_K, &mut result).unwrap();
        
        // Check a few values from the result
        // For 0x12: 
        // - First 4 bits (2) = min_val + 2 * d / 15 = 0.0 + 2 * 1.0 / 15 = 0.133...
        // - Second 4 bits (1) = min_val + 1 * d / 15 = 0.0 + 1 * 1.0 / 15 = 0.066...
        assert!((result[0] - 0.133).abs() < 0.01, "Expected ~0.133, got {}", result[0]);
        assert!((result[1] - 0.067).abs() < 0.01, "Expected ~0.067, got {}", result[1]);
        
        // For 0x34:
        // - First 4 bits (4) = min_val + 4 * d / 15 = 0.0 + 4 * 1.0 / 15 = 0.266...
        // - Second 4 bits (3) = min_val + 3 * d / 15 = 0.0 + 3 * 1.0 / 15 = 0.2
        assert!((result[2] - 0.267).abs() < 0.01, "Expected ~0.267, got {}", result[2]);
        assert!((result[3] - 0.2).abs() < 0.01, "Expected ~0.2, got {}", result[3]);
    }
} 