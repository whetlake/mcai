use crate::gguf::GGUFValueType;
use crate::llm::quants::dequantize::Dequantizer;
use crate::llm::quants::formats::QK_K;
use crate::llm::quants::formats::q4_k_s::Q4KSFormat;
use crate::llm::quants::formats::FormatImpl;

#[cfg(test)]
mod q4_k_s_tests {
    use super::*;
    
    // Helper function to create a sample Q4_K_S block
    fn create_sample_block() -> Vec<u8> {
        // A block consists of:
        // - QK_K/2 (128) bytes for quantized values (4 bits per value)
        // - 2 bytes for min value (f16)
        // - 2 bytes for d value (f16)
        
        let mut block = Vec::with_capacity(128 + 2 + 2);
        
        // Quantized values - all 0x55 (0101 0101) pattern
        // This gives alternating 5 and 5 values
        block.extend(vec![0x55; 128]);
        
        // min value - f16 representation of 0.0 (0x0000)
        block.extend_from_slice(&[0x00, 0x00]);
        
        // d value - f16 representation of 1.0 (0x3C00)
        block.extend_from_slice(&[0x00, 0x3C]);
        
        block
    }
    
    #[test]
    fn test_q4_k_s_format() {
        let format = Q4KSFormat::new();
        assert_eq!(format.name(), "Q4_K_S");
        assert_eq!(format.gguf_type(), GGUFValueType::Q4_K_S);
        
        // Create a sample block
        let data = create_sample_block();
        
        // Test dequantization
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Dequantize one block (QK_K=256 elements)
        format.dequantize(&data, &mut offset, QK_K, &mut result).unwrap();
        
        // Check result size
        assert_eq!(result.len(), QK_K);
        
        // Check offset advanced correctly
        assert_eq!(offset, 128 + 2 + 2);
        
        // With our specific block setup:
        // - Quantized values are 0x55 = 0101 0101 in binary
        // - min is 0.0
        // - d is 1.0
        // So each 4-bit value 5 (0101) should dequantize to: 0.0 + 5 * 1.0 / 15 = 0.333...
        
        // Check a sampling of values
        for i in 0..10 {
            assert!((result[i] - 0.333).abs() < 0.01, "Expected ~0.333, got {}", result[i]);
        }
    }
    
    #[test]
    fn test_q4_k_s_with_varied_block() {
        let format = Q4KSFormat::new();
        
        // Create a more varied block
        let mut block = Vec::with_capacity(128 + 2 + 2);
        
        // Quantized values - varying patterns
        for i in 0..128 {
            block.push((i % 16) as u8); // 0x00, 0x01, ..., 0x0F pattern
        }
        
        // min value - f16 representation of -1.0 (0xBC00)
        block.extend_from_slice(&[0x00, 0xBC]);
        
        // d value - f16 representation of 2.0 (0x4000)
        block.extend_from_slice(&[0x00, 0x40]);
        
        // Test dequantization
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Dequantize one block
        format.dequantize(&block, &mut offset, QK_K, &mut result).unwrap();
        
        // Check result size
        assert_eq!(result.len(), QK_K);
        
        // For this block, values should range from:
        // -1.0 + 0 * 2.0 / 15 = -1.0 to -1.0 + 15 * 2.0 / 15 = 1.0
        
        // Check a few specific values based on our pattern
        // Byte 0 has values 0 and 0
        assert!((result[0] - (-1.0)).abs() < 0.01, "Expected -1.0, got {}", result[0]);
        assert!((result[1] - (-1.0)).abs() < 0.01, "Expected -1.0, got {}", result[1]);
        
        // Byte 7 has values 7 and 0
        assert!((result[14] - (-0.067)).abs() < 0.01, "Expected -0.067, got {}", result[14]);
        assert!((result[15] - (-1.0)).abs() < 0.01, "Expected -1.0, got {}", result[15]);
        
        // Byte 15 has values 15 and 0
        assert!((result[30] - 1.0).abs() < 0.01, "Expected 1.0, got {}", result[30]);
        assert!((result[31] - (-1.0)).abs() < 0.01, "Expected -1.0, got {}", result[31]);
        
        // Verify we have varying values
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
    fn test_q4_k_s_through_dequantizer() {
        // Test the format through the main Dequantizer
        
        // Create a sample block
        let data = create_sample_block();
        
        // Dequantize through the Dequantizer interface
        let result = Dequantizer::dequantize(
            &data, 
            0, 
            QK_K, 
            GGUFValueType::Q4_K_S
        ).unwrap();
        
        // Check result size
        assert_eq!(result.len(), QK_K);
        
        // Check a sampling of values
        for i in 0..10 {
            assert!((result[i] - 0.333).abs() < 0.01, "Expected ~0.333, got {}", result[i]);
        }
    }
    
    #[test]
    fn test_q4_k_s_multiple_blocks() {
        let format = Q4KSFormat::new();
        
        // Create two different blocks
        let mut block1 = create_sample_block(); // Has 0x55 pattern
        
        // Second block with different values
        let mut block2 = Vec::with_capacity(128 + 2 + 2);
        block2.extend(vec![0xAA; 128]); // 0xAA = 1010 1010 (values 10 and 10)
        block2.extend_from_slice(&[0x00, 0x00]); // min 0.0
        block2.extend_from_slice(&[0x00, 0x3C]); // d 1.0
        
        // Combine blocks
        let mut combined = block1;
        combined.append(&mut block2);
        
        // Test dequantization
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Dequantize two blocks
        format.dequantize(&combined, &mut offset, QK_K * 2, &mut result).unwrap();
        
        // Check result size
        assert_eq!(result.len(), QK_K * 2);
        
        // Check offset advanced correctly
        assert_eq!(offset, (128 + 2 + 2) * 2);
        
        // First block should have values around 0.333 (from 5/15)
        assert!((result[0] - 0.333).abs() < 0.01, "Expected ~0.333, got {}", result[0]);
        
        // Second block should have values around 0.667 (from 10/15)
        assert!((result[QK_K] - 0.667).abs() < 0.01, "Expected ~0.667, got {}", result[QK_K]);
    }
    
    // Also test error conditions
    
    #[test]
    fn test_q4_k_s_not_enough_data() {
        let format = Q4KSFormat::new();
        
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
    fn test_q4_k_s_wrong_element_count() {
        let format = Q4KSFormat::new();
        
        // Create a complete block
        let block = create_sample_block();
        
        // Test dequantization with invalid element count
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Attempt to dequantize non-multiple of QK_K
        let err = format.dequantize(&block, &mut offset, 100, &mut result).unwrap_err();
        assert!(err.to_string().contains("must be a multiple of"));
    }
    
    #[test]
    fn test_q4_k_s_misaligned_offset() {
        let format = Q4KSFormat::new();
        
        // Create a block
        let block = create_sample_block();
        
        // Test with misaligned offset
        let mut result = Vec::new();
        let mut offset = 1; // Not aligned to 2 bytes
        
        // Attempt to dequantize
        let err = format.dequantize(&block, &mut offset, QK_K, &mut result).unwrap_err();
        assert!(err.to_string().contains("Misaligned offset"));
    }
    
    #[test]
    fn test_q4_k_s_edge_values() {
        let format = Q4KSFormat::new();
        
        // Create a block with edge case values
        let mut block = Vec::with_capacity(128 + 2 + 2);
        
        // First half with value 0, second half with value 15 (max)
        block.extend(vec![0x00; 64]); // 0000 0000
        block.extend(vec![0xFF; 64]); // 1111 1111
        
        // min value -10.0 as f16
        block.extend_from_slice(&[0x00, 0xD2]); 
        
        // d value 20.0 as f16
        block.extend_from_slice(&[0x00, 0x48]);
        
        // Test dequantization
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Dequantize one block
        format.dequantize(&block, &mut offset, QK_K, &mut result).unwrap();
        
        // Check min value (first elements should be -10 + 0*20/15 = -10)
        assert!((result[0] - (-10.0)).abs() < 0.01, "Expected -10.0, got {}", result[0]);
        
        // Check max value (later elements should be -10 + 15*20/15 = 10)
        assert!((result[128] - 10.0).abs() < 0.01, "Expected 10.0, got {}", result[128]);
    }
} 