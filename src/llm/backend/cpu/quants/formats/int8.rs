use std::error::Error;
use crate::gguf::GGUFValueType;
use super::FormatImpl;

/// INT8 format - 8-bit signed integers
#[derive(Clone)]
pub struct Int8Format;

impl Int8Format {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Int8Format {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::INT8
    }
    
    fn name(&self) -> &'static str {
        "INT8"
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
        let bytes_needed = num_elements;
        let actual_size_mb = bytes_needed as f64 / (1024.0 * 1024.0);
        let f32_size_mb = (num_elements * 4) as f64 / (1024.0 * 1024.0);
        let compression_ratio = f32_size_mb / actual_size_mb;
        
        println!("INT8 Format Size Details:");
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
            return Err(format!("Not enough data to read INT8 values. Need {} bytes, but only have {}", 
                              bytes_needed, available).into());
        }
        
        // Preallocate result vector
        result.reserve(num_elements);
        
        // Convert each byte to a signed int8, then cast to f32
        for i in 0..num_elements {
            let val = data[*offset + i] as i8 as f32;
            result.push(val);
        }
        
        // Update offset
        *offset += num_elements;
        
        Ok(())
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Int8Format::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_int8_format() {
        let format = Int8Format::new();
        assert_eq!(format.name(), "INT8");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::INT8 as u32);
        
        // Test dequantization with some sample data
        // Positive values
        let mut data = vec![0, 1, 2, 3, 127];
        // Negative values (-1, -2, -128)
        data.extend_from_slice(&[255, 254, 128]);
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 8, &mut result).unwrap();
        
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 127.0, -1.0, -2.0, -128.0]);
        assert_eq!(offset, 8);
    }
} 