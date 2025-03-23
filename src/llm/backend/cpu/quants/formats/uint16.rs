use std::error::Error;
use crate::gguf::GGUFValueType;
use super::FormatImpl;

/// UINT16 format - direct storage of 16-bit unsigned integer values
#[derive(Clone)]
pub struct Uint16Format;

impl Uint16Format {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Uint16Format {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::UINT16
    }
    
    fn name(&self) -> &'static str {
        "UINT16"
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
        // Calculate size for reporting
        let bytes_needed = num_elements * 2; // 2 bytes per uint16
        let actual_size_mb = bytes_needed as f64 / (1024.0 * 1024.0);
        let f32_size_mb = (num_elements * 4) as f64 / (1024.0 * 1024.0);
        let compression_ratio = f32_size_mb / actual_size_mb;
        
        println!("UINT16 Format Size Details:");
        println!("  Total bytes needed: {} ({:.4} MB)", bytes_needed, actual_size_mb);
        println!("  Equivalent F32 size: {:.4} MB", f32_size_mb);
        println!("  Compression ratio: {:.2}x", compression_ratio);
        
        // Ensure we have enough data
        if *offset + bytes_needed > data.len() {
            let available = if data.len() > *offset {
                data.len() - *offset
            } else {
                0
            };
            return Err(format!("Not enough data to read UINT16 values. Need {} bytes, but only have {}", 
                              bytes_needed, available).into());
        }
        
        // Check alignment - offset should be aligned to at least 2 bytes
        if *offset % 2 != 0 {
            return Err("Misaligned offset for UINT16 data. Offset must be a multiple of 2 bytes.".into());
        }
        
        // Preallocate result vector
        result.reserve(num_elements);
        
        // Read each uint16 value
        for i in 0..num_elements {
            let base_idx = *offset + i * 2;
            // Create array of bytes in little-endian order
            let bytes = [data[base_idx], data[base_idx + 1]];
            // Convert bytes to u16, then to f32
            let value = u16::from_le_bytes(bytes) as f32;
            result.push(value);
        }
        
        // Update offset
        *offset += bytes_needed;
        
        Ok(())
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Uint16Format::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_uint16_format() {
        let format = Uint16Format::new();
        assert_eq!(format.name(), "UINT16");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::UINT16 as u32);
        
        // Test dequantization with some sample data
        let mut data = Vec::new();
        
        // Values: 0, 1, 100, 32768, 65535
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&100u16.to_le_bytes());
        data.extend_from_slice(&32768u16.to_le_bytes());
        data.extend_from_slice(&65535u16.to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 5, &mut result).unwrap();
        
        assert_eq!(result, vec![0.0, 1.0, 100.0, 32768.0, 65535.0]);
        assert_eq!(offset, 10); // 5 * 2 bytes
    }
} 