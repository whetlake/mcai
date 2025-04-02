use std::error::Error;
use crate::gguf::GGUFValueType;
use super::{FormatImpl, QK_K};
use super::super::utils::{f16_to_f32, parallel_process, check_data_availability};

/// Q3_K_M format - 3-bit quantization with K=256 (medium variant)
/// 
/// This is a specialized format that stores 3-bit quantized values with 
/// a complex bit-packing scheme to save memory.
#[derive(Clone)]
pub struct Q3KMFormat {}

impl Q3KMFormat {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Q3KMFormat {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::Q3_K_M
    }
    
    fn name(&self) -> &'static str {
        "Q3_K_M"
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
        // Q3_K_M works on blocks of QK_K (256) elements
        if num_elements % QK_K != 0 {
            return Err(format!("For Q3_K_M format, number of elements must be a multiple of {}, got {}", QK_K, num_elements).into());
        }
        
        let num_blocks = num_elements / QK_K;
        
        // Calculate size needed for each block:
        // - QK_K/8 bytes for high mask (1 bit per value)
        // - QK_K/4 bytes for quantized values (2 bits per value) 
        // - 12 bytes for scales (16 scales at 6 bits each, packed)
        // - 2 bytes (1 f16) for d value
        let bytes_per_block = QK_K/8 + QK_K/4 + 12 + 2;
        let bytes_needed = num_blocks * bytes_per_block;
        
        // Calculate sizes for reporting
        let actual_size_mb = bytes_needed as f32 / (1024.0 * 1024.0);
        let f32_size_mb = (num_elements * 4) as f32 / (1024.0 * 1024.0);
        let _compression_ratio = f32_size_mb / actual_size_mb;
        
        // println!("Q3_K_M Format Size Details:");
        // println!("  Number of blocks: {}", num_blocks);
        // println!("  Bytes per block: {} bytes", bytes_per_block);
        // println!("  Total bytes needed: {} ({:.4} MB)", bytes_needed, actual_size_mb);
        // println!("  Equivalent F32 size: {:.4} MB", f32_size_mb);
        // println!("  Compression ratio: {:.2}x", compression_ratio);
        
        // Ensure we have enough data using utility function
        check_data_availability(data, *offset, bytes_needed, "Q3_K_M")?;
        
        // Check alignment - offset should be aligned to at least 2 bytes for f16 access
        if *offset % 2 != 0 {
            return Err("Misaligned offset for Q3_K_M data. Offset must be a multiple of 2 bytes.".into());
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
/// Optimized version with better memory access patterns and reduced computation
#[inline(always)]
fn dequantize_blocks(
    data: &[u8],
    base_offset: usize, 
    start_block: usize, 
    end_block: usize, 
    bytes_per_block: usize,
    result: &mut [f32]
) {
    // Pre-allocate scales array
    let mut scales = [0.0f32; 16];
    
    // Constants for better readability
    const HMASK_SIZE: usize = QK_K/8;
    const QS_SIZE: usize = QK_K/4;
    const SCALES_SIZE: usize = 12;
    
    // Process each block in range
    for i in 0..(end_block - start_block) {
        let block_idx = start_block + i;
        let block_offset = base_offset + block_idx * bytes_per_block;
        
        // Calculate offsets for different parts of the block
        let hmask_offset = block_offset;
        let qs_offset = hmask_offset + HMASK_SIZE;
        let scales_offset = qs_offset + QS_SIZE;
        let d_offset = scales_offset + SCALES_SIZE;
        
        // Get d value (f16 stored as 2 bytes)
        let d = f16_to_f32(&[data[d_offset], data[d_offset + 1]]);
        
        // Decode 16 scales
        decode_scales_fast(&data, scales_offset, &mut scales);
        
        // Calculate output position in the result slice
        let output_offset = i * QK_K;
        
        // Process the entire block at once with faster method
        process_block_fast(
            data,
            hmask_offset,
            qs_offset,
            &scales,
            d,
            &mut result[output_offset..output_offset + QK_K]
        );
    }
}

/// Faster method to process an entire block of 256 values
#[inline(always)]
fn process_block_fast(
    data: &[u8],
    hmask_offset: usize,
    qs_offset: usize,
    scales: &[f32; 16],
    d: f32,
    output: &mut [f32]
) {
    // Read all 32 bytes of high mask bits at once - QK_K/8 = 32 bytes
    let hmask_data = &data[hmask_offset..hmask_offset + QK_K/8];
    
    // Process all 64 quant bytes (containing 256 2-bit values)
    for group_idx in 0..16 {
        // Get scale for this group
        let dscale = d * scales[group_idx];
        
        // Calculate offsets
        let group_start = group_idx * 16;
        
        // Process in batches of 4 (one byte of qs)
        for j in 0..4 {
            let base_idx = group_start + j * 4;
            
            // Get hmask byte containing 8 bits for our values
            let hmask_byte_idx = base_idx / 8;
            
            // Safety check to avoid index out of bounds
            if hmask_byte_idx >= hmask_data.len() - 1 {
                // We're at the boundary, handle with care
                let hmask_byte = hmask_data[hmask_byte_idx];
                let hmask_next_byte = if hmask_byte_idx + 1 < hmask_data.len() {
                    hmask_data[hmask_byte_idx + 1]
                } else {
                    0 // Use 0 for bits beyond available data
                };
                
                // Process as before...
                let hmask_word = ((hmask_next_byte as u16) << 8) | (hmask_byte as u16);
                process_quad(base_idx, hmask_word, data[qs_offset + base_idx/4], dscale, output);
            } else {
                // Normal case, data safely available
                let hmask_byte = hmask_data[hmask_byte_idx];
                let hmask_next_byte = hmask_data[hmask_byte_idx + 1];
                
                // Construct a 16-bit word to simplify bit extraction
                let hmask_word = ((hmask_next_byte as u16) << 8) | (hmask_byte as u16);
                
                // Get qs byte containing 4 2-bit values
                let qs_byte = data[qs_offset + base_idx/4];
                
                // Process 4 elements at once
                process_quad(base_idx, hmask_word, qs_byte, dscale, output);
            }
        }
    }
}

/// Process 4 elements at once (extracted to avoid code duplication)
#[inline(always)]
fn process_quad(
    base_idx: usize,
    hmask_word: u16,
    qs_byte: u8,
    dscale: f32,
    output: &mut [f32]
) {
    // Process 4 elements at once (unrolled for speed)
    let bit_shift = (base_idx % 8);
    let hmask_bits = (hmask_word >> bit_shift) & 0xF; // Get 4 bits at once
    
    // Element 0
    let qh_bit = (hmask_bits) & 1;
    let qh = 1 - qh_bit;
    let qs_bits = qs_byte & 0x3;
    let q = (qs_bits as i8) - ((qh as i8) << 2);
    output[base_idx] = dscale * (q as f32);
    
    // Element 1
    let qh_bit = (hmask_bits >> 1) & 1;
    let qh = 1 - qh_bit;
    let qs_bits = (qs_byte >> 2) & 0x3;
    let q = (qs_bits as i8) - ((qh as i8) << 2);
    output[base_idx + 1] = dscale * (q as f32);
    
    // Element 2
    let qh_bit = (hmask_bits >> 2) & 1;
    let qh = 1 - qh_bit;
    let qs_bits = (qs_byte >> 4) & 0x3;
    let q = (qs_bits as i8) - ((qh as i8) << 2);
    output[base_idx + 2] = dscale * (q as f32);
    
    // Element 3
    let qh_bit = (hmask_bits >> 3) & 1;
    let qh = 1 - qh_bit;
    let qs_bits = (qs_byte >> 6) & 0x3;
    let q = (qs_bits as i8) - ((qh as i8) << 2);
    output[base_idx + 3] = dscale * (q as f32);
}

/// Ultra-fast scale decoding
#[inline(always)]
fn decode_scales_fast(data: &[u8], scales_offset: usize, scales: &mut [f32; 16]) {
    // First pass: get low 4 bits of each scale (from first 8 bytes)
    for i in 0..8 {
        let byte = data[scales_offset + i];
        
        // Extract and adjust the lower and upper 4 bits (-8 bias for each)
        scales[i*2] = ((byte & 0x0F) as i8 - 8) as f32;
        scales[i*2+1] = (((byte >> 4) & 0x0F) as i8 - 8) as f32;
    }
    
    // Second pass: add high 2 bits to each scale (from last 4 bytes)
    for i in 0..4 {
        let byte = data[scales_offset + 8 + i];
        let idx = i * 4;
        
        // Apply high bits - bits 0-1 go to scales[idx]
        scales[idx] += ((byte & 0x03) << 4) as f32;
        
        // Apply high bits - bits 2-3 go to scales[idx+1]
        scales[idx+1] += (((byte >> 2) & 0x03) << 4) as f32;
        
        // Apply high bits - bits 4-5 go to scales[idx+2]
        scales[idx+2] += (((byte >> 4) & 0x03) << 4) as f32;
        
        // Apply high bits - bits 6-7 go to scales[idx+3]
        scales[idx+3] += (((byte >> 6) & 0x03) << 4) as f32;
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Q3KMFormat::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_q3_k_m_format_properties() {
        let format = Q3KMFormat::new();
        assert_eq!(format.name(), "Q3_K_M");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::Q3_K_M as u32);
    }

    #[test]
    fn test_q3_k_m_requires_block_multiple() {
        let format = Q3KMFormat::new();
        let data = vec![0u8; 1000]; // Some arbitrary data
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Try with non-multiple of QK_K
        let err = format.dequantize(&data, &mut offset, 100, &mut result).unwrap_err();
        assert!(err.to_string().contains("must be a multiple of"));
    }
    
    #[test]
    fn test_q3_k_m_not_enough_data() {
        let format = Q3KMFormat::new();
        let data = vec![0u8; 50]; // Not enough for even one block
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Try with one block (256 elements)
        let err = format.dequantize(&data, &mut offset, QK_K, &mut result).unwrap_err();
        assert!(err.to_string().contains("Not enough data"));
    }
    
    #[test]
    fn test_q3_k_m_misaligned_offset() {
        let format = Q3KMFormat::new();
        let data = vec![0u8; 1000]; // Plenty of data
        let mut result = Vec::new();
        let mut offset = 1; // Misaligned offset
        
        // Try with one block
        let err = format.dequantize(&data, &mut offset, QK_K, &mut result).unwrap_err();
        assert!(err.to_string().contains("Misaligned offset"));
    }
} 