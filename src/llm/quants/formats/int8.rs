use std::error::Error;
use crate::gguf::GGUFValueType;
use crate::llm::quants::formats::FormatImpl;

/// INT8 format - direct storage of 8-bit signed integer values
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
        // Ensure we have enough data
        if *offset + num_elements > data.len() {
            return Err("Not enough data to read INT8 values".into());
        }
        
        // Get a slice of all the data at once
        let values = &data[*offset..*offset + num_elements];
        
        // Pre-allocate space in the result vector
        result.reserve(num_elements);
        
        // Convert all bytes to signed i8 first, then to f32
        result.extend(values.iter().map(|&value| (value as i8) as f32));
        
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