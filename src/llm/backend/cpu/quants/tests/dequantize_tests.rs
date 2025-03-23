use super::super::dequantize::Dequantizer;
use crate::gguf::GGUFValueType;

#[cfg(test)]
mod dequantize_tests {
    use super::*;

    #[test]
    fn test_dequantize_with_unsupported_type() {
        // Try to dequantize with a data type that doesn't have a format implementation
        // We know from the format registry that Q4_0 isn't implemented
        let result = Dequantizer::dequantize(
            &[0u8, 1, 2, 3], // Some sample data
            0,               // Offset
            4,               // Number of elements
            GGUFValueType::Q4_0, // Not implemented in the FORMAT_REGISTRY
        );
        
        // Should return an error
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unsupported data type"));
        }
    }
    
    #[test]
    fn test_dequantize_not_enough_data() {
        // Test with float32 format which requires 4 bytes per element
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7]; // Only 8 bytes (2 float32s)
        
        // Try to dequantize more elements than we have data for
        let result = Dequantizer::dequantize(
            &data,
            0,
            3, // Asking for 3 elements (12 bytes) but only have 8
            GGUFValueType::FLOAT32,
        );
        
        // Should return an error
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough data"));
        }
    }
    
    #[test]
    fn test_dequantize_misaligned_data() {
        // Test formats that require alignment
        
        // Test float32 with misaligned offset
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        
        // Try to dequantize from offset 1 (misaligned for float32)
        let result = Dequantizer::dequantize(
            &data,
            1, // Misaligned offset (not multiple of 4)
            2,
            GGUFValueType::FLOAT32,
        );
        
        // Should return an error
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Misaligned offset"));
        }
        
        // Test int16 with misaligned offset
        let result = Dequantizer::dequantize(
            &data,
            1, // Misaligned offset (not multiple of 2)
            4,
            GGUFValueType::INT16,
        );
        
        // Should return an error
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Misaligned offset"));
        }
        
        // Test uint16 with misaligned offset
        let result = Dequantizer::dequantize(
            &data,
            1, // Misaligned offset (not multiple of 2)
            4,
            GGUFValueType::UINT16,
        );
        
        // Should return an error
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Misaligned offset"));
        }
    }
    
    #[test]
    fn test_dequantize_zero_elements() {
        // Test dequantizing zero elements (should work but return empty vector)
        let data = vec![0u8, 1, 2, 3];
        
        let result = Dequantizer::dequantize(
            &data,
            0,
            0, // Zero elements
            GGUFValueType::INT8,
        );
        
        assert!(result.is_ok());
        if let Ok(values) = result {
            assert!(values.is_empty());
        }
    }
    
    #[test]
    fn test_int8_not_enough_data() {
        // Test with INT8 format which requires 1 byte per element
        let data = vec![0u8, 1, 2, 3]; // Only 4 bytes
        
        // Try to dequantize more elements than we have data for
        let result = Dequantizer::dequantize(
            &data,
            0,
            10, // Asking for 10 elements but only have 4
            GGUFValueType::INT8,
        );
        
        // Should return an error
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough data"));
        }
    }
    
    #[test]
    fn test_uint8_not_enough_data() {
        // Test with UINT8 format which requires 1 byte per element
        let data = vec![0u8, 1, 2, 3]; // Only 4 bytes
        
        // Try to dequantize more elements than we have data for
        let result = Dequantizer::dequantize(
            &data,
            0,
            10, // Asking for 10 elements but only have 4
            GGUFValueType::UINT8,
        );
        
        // Should return an error
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough data"));
        }
    }
    
    #[test]
    fn test_dequantize_with_large_offset() {
        // Test with an offset that is beyond the data bounds
        let data = vec![0u8, 1, 2, 3];
        
        let result = Dequantizer::dequantize(
            &data,
            10, // Offset beyond data size
            1,
            GGUFValueType::UINT8,
        );
        
        // Should return an error
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough data"));
        }
    }
}
