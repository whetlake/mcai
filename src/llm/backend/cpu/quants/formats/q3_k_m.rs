use std::error::Error;
use crate::gguf::GGUFValueType;
use super::{FormatImpl, QK_K};
use super::super::utils::f16_to_f32;

/// Q3_K_M format - 3-bit quantization with K=256 (medium variant)
/// 
/// This is a specialized format that stores 3-bit quantized values with 
/// a complex bit-packing scheme to save memory.
#[derive(Clone)]
pub struct Q3KMFormat;

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
        let theoretical_size_mb = (num_elements * 3) as f64 / 8.0 / (1024.0 * 1024.0);
        let actual_size_mb = bytes_needed as f64 / (1024.0 * 1024.0);
        let f32_size_mb = (num_elements * 4) as f64 / (1024.0 * 1024.0);
        let compression_ratio = f32_size_mb / actual_size_mb;
        
        println!("Q3_K_M Format Size Details:");
        println!("  Number of blocks: {}", num_blocks);
        println!("  Bytes per block: {} bytes", bytes_per_block);
        println!("  Total bytes needed: {} ({:.4} MB)", bytes_needed, actual_size_mb);
        println!("  Theoretical 3-bit size: {:.4} MB", theoretical_size_mb);
        println!("  Equivalent F32 size: {:.4} MB", f32_size_mb);
        println!("  Compression ratio: {:.2}x", compression_ratio);
        
        // Ensure we have enough data
        if *offset + bytes_needed > data.len() {
            let available = if data.len() > *offset {
                data.len() - *offset
            } else {
                0
            };
            return Err(format!("Not enough data to read Q3_K_M values. Need {} bytes, but only have {}", 
                              bytes_needed, available).into());
        }
        
        // Check alignment - offset should be aligned to at least 2 bytes for f16 access
        if *offset % 2 != 0 {
            return Err("Misaligned offset for Q3_K_M data. Offset must be a multiple of 2 bytes.".into());
        }
        
        // Preallocate result vector
        result.reserve(num_elements);
        
        // Process each block
        for _ in 0..num_blocks {
            // Extract components from the block
            let hmask_offset = *offset;
            let qs_offset = hmask_offset + QK_K/8;
            let scales_offset = qs_offset + QK_K/4;
            let d_offset = scales_offset + 12;
            
            // Get d value (f16 stored as 2 bytes)
            let d_bytes = &data[d_offset..d_offset + 2];
            // Use the utility function instead of our custom implementation
            let d = f16_to_f32(d_bytes);
            
            // Extract 12 bytes of scale data
            let scale_bytes = &data[scales_offset..scales_offset + 12];
            
            // Decode 16 scales (each 6-bit, ranging from -32 to 31 after adjustment)
            let mut scales = [0.0f32; 16];
            decode_scales(scale_bytes, &mut scales);
            
            // Process each group of 16 values
            for group in 0..16 {
                let scale = scales[group];
                
                // Process 16 values in this group
                for i in 0..16 {
                    let idx = group * 16 + i;
                    
                    // Get 2-bit value from qs
                    let byte_idx = idx / 4;
                    let bit_shift = (idx % 4) * 2;
                    let qs_bits = (data[qs_offset + byte_idx] >> bit_shift) & 0x3;
                    
                    // Get high bit from hmask
                    let mask_byte_idx = idx / 8;
                    let mask_bit_shift = idx % 8;
                    let qh_bit = (data[hmask_offset + mask_byte_idx] >> mask_bit_shift) & 0x1;
                    
                    // Combine bits to get 3-bit signed value (-4 to 3)
                    // Strange behavior: qh bit is actually inverted from what you'd expect
                    let qh = 1 - qh_bit; // XOR with 1 to flip 0->1, 1->0
                    let q = (qs_bits as i8) - ((qh as i8) << 2);
                    
                    // Final value is d * scale * q
                    let value = d * scale * (q as f32);
                    result.push(value);
                }
            }
            
            // Update offset for next block
            *offset += bytes_per_block;
        }
        
        Ok(())
    }
}

/// Decode the 16 scale values from 12 bytes
/// Scales are packed at 6-bit each in this pattern:
///  0: IIIIAAAA
///  1: JJJJBBBB
///  2: KKKKCCCC
///  3: LLLLDDDD
///  4: MMMMEEEE
///  5: NNNNFFFF
///  6: OOOOGGGG
///  7: PPPPHHHH
///  8: MMIIEEAA
///  9: NNJJFFBB
/// 10: OOKKGGCC
/// 11: PPLLHHDD
fn decode_scales(scale_bytes: &[u8], scales: &mut [f32; 16]) {
    // Extract lower 4 bits and higher 2 bits for each scale
    let mut lower_bits = [0u8; 16];
    let mut higher_bits = [0u8; 16];
    
    // Extract lower 4 bits from first 8 bytes
    for i in 0..8 {
        lower_bits[i*2] = scale_bytes[i] & 0x0F;
        lower_bits[i*2+1] = (scale_bytes[i] >> 4) & 0x0F;
    }
    
    // Extract higher 2 bits from last 4 bytes
    for i in 0..4 {
        higher_bits[i*4] = scale_bytes[8+i] & 0x03;
        higher_bits[i*4+1] = (scale_bytes[8+i] >> 2) & 0x03;
        higher_bits[i*4+2] = (scale_bytes[8+i] >> 4) & 0x03;
        higher_bits[i*4+3] = (scale_bytes[8+i] >> 6) & 0x03;
    }
    
    // Combine bits and adjust range (-32 to 31)
    for i in 0..16 {
        let combined = lower_bits[i] | (higher_bits[i] << 4);
        let adjusted = (combined as i8) - 32;
        scales[i] = adjusted as f32;
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