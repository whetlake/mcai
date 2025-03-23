use crate::gguf::GGUFValueType;
use crate::llm::quants::dequantize::Dequantizer;
use crate::llm::quants::formats::QK_K;
use crate::llm::quants::formats::q3_k_m::Q3KMFormat;
use crate::llm::quants::formats::FormatImpl;

#[cfg(test)]
mod q3_k_m_tests {
    use super::*;
    
    // Helper function to create a sample Q3_K_M block
    fn create_sample_block() -> Vec<u8> {
        // A block consists of:
        // - QK_K/8 (32) bytes for hmask
        // - QK_K/4 (64) bytes for qs
        // - 12 bytes for scales
        // - 2 bytes for d (f16)
        
        let mut block = Vec::with_capacity(32 + 64 + 12 + 2);
        
        // hmask - all zeros (0x00) means qh bits are 0
        block.extend(vec![0x00; 32]);
        
        // qs - alternating patterns of 01 01 01 01
        block.extend(vec![0x55; 64]);
        
        // scales - 12 bytes to encode 16 scales
        // Setting all scales to 0 for simplicity (adjust by -32 in the algorithm)
        block.extend(vec![0x20; 8]); // First 8 bytes (low 4 bits)
        block.extend(vec![0x00; 4]);  // Last 4 bytes (high 2 bits)
        
        // d value - f16 representation of 1.0 (0x3C00)
        block.extend_from_slice(&[0x00, 0x3C]);
        
        block
    }
    
    #[test]
    fn test_q3_k_m_format() {
        let format = Q3KMFormat::new();
        assert_eq!(format.name(), "Q3_K_M");
        assert_eq!(format.gguf_type(), GGUFValueType::Q3_K_M);
        
        // Create a sample block
        let mut data = create_sample_block();
        
        // Test dequantization
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Dequantize one block (QK_K=256 elements)
        format.dequantize(&data, &mut offset, QK_K, &mut result).unwrap();
        
        // Check result size
        assert_eq!(result.len(), QK_K);
        
        // Check offset advanced correctly
        assert_eq!(offset, 32 + 64 + 12 + 2);
        
        // With our specific block setup:
        // - hmask is all 0s, which means qh=1 (since we flip it)
        // - qs is 01 pattern (value 1)
        // - scales are 0
        // - d is 1.0
        // So each value should be 1.0 * 0 * (1 - 4) = -3.0
        
        // Check a sampling of values
        for i in 0..10 {
            assert_eq!(result[i], -3.0);
        }
    }
    
    #[test]
    fn test_q3_k_m_with_varied_block() {
        let format = Q3KMFormat::new();
        
        // Create a more varied block
        let mut block = Vec::with_capacity(32 + 64 + 12 + 2);
        
        // hmask - alternating 0s and 1s
        block.extend(vec![0xAA; 32]);
        
        // qs - alternating 00, 01, 10, 11 patterns
        for i in 0..64 {
            block.push((i % 4) as u8);
        }
        
        // scales - varying values
        for i in 0..8 {
            // Lower 4 bits of scale i*2, upper 4 bits of scale i*2+1
            block.push((i & 0x0F) | ((i+1) << 4));
        }
        
        // Higher 2 bits
        for i in 0..4 {
            block.push(0x55); // 01 01 01 01 bit pattern
        }
        
        // d value - f16 representation of 2.0 (0x4000)
        block.extend_from_slice(&[0x00, 0x40]);
        
        // Test dequantization
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Dequantize one block
        format.dequantize(&block, &mut offset, QK_K, &mut result).unwrap();
        
        // Check result size
        assert_eq!(result.len(), QK_K);
        
        // The values should vary based on our input pattern
        // We're not checking exact values since they're complex to calculate by hand,
        // but we can check that the pattern varies.
        let mut has_different_values = false;
        for i in 1..result.len() {
            if (result[i] - result[0]).abs() > 0.001 {
                has_different_values = true;
                break;
            }
        }
        assert!(has_different_values);
    }
    
    #[test]
    fn test_q3_k_m_through_dequantizer() {
        // Test the format through the main Dequantizer
        
        // Create a sample block
        let mut data = create_sample_block();
        
        // Dequantize through the Dequantizer interface
        let result = Dequantizer::dequantize(
            &data, 
            0, 
            QK_K, 
            GGUFValueType::Q3_K_M
        ).unwrap();
        
        // Check result size
        assert_eq!(result.len(), QK_K);
        
        // Check a sampling of values
        for i in 0..10 {
            assert_eq!(result[i], -3.0);   
        }
    }
    
    #[test]
    fn test_q3_k_m_multiple_blocks() {
        let format = Q3KMFormat::new();
        
        // Create two identical blocks
        let mut block = create_sample_block();
        let mut another_block = create_sample_block();
        block.append(&mut another_block);
        
        // Test dequantization
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Dequantize two blocks
        format.dequantize(&block, &mut offset, QK_K * 2, &mut result).unwrap();
        
        // Check result size
        assert_eq!(result.len(), QK_K * 2);
        
        // Check offset advanced correctly
        assert_eq!(offset, (32 + 64 + 12 + 2) * 2);
        
        // All values should be the same since we used the same block twice
        for i in 0..10 {
            assert_eq!(result[i], result[i + QK_K]);
        }
    }
    
    // Also test error conditions
    
    #[test]
    fn test_q3_k_m_not_enough_data() {
        let format = Q3KMFormat::new();
        
        // Create a partial block
        let block = vec![0u8; 50]; // Not enough for a full block
        
        // Test dequantization
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Attempt to dequantize
        let err = format.dequantize(&block, &mut offset, QK_K, &mut result).unwrap_err();
        assert!(err.to_string().contains("Not enough data"));
    }
    
    #[test]
    fn test_q3_k_m_wrong_element_count() {
        let format = Q3KMFormat::new();
        
        // Create a complete block
        let block = create_sample_block();
        
        // Test dequantization with invalid element count
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Attempt to dequantize non-multiple of QK_K
        let err = format.dequantize(&block, &mut offset, 100, &mut result).unwrap_err();
        assert!(err.to_string().contains("must be a multiple of"));
    }
} 