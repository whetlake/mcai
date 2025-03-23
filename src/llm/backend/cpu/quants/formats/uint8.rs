use std::error::Error;
use crate::gguf::GGUFValueType;
use super::FormatImpl;

/// UINT8 format - direct storage of 8-bit unsigned integer values
#[derive(Clone)]
pub struct Uint8Format;

impl Uint8Format {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Uint8Format {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::UINT8
    }
    
    fn name(&self) -> &'static str {
        "UINT8"
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
            return Err("Not enough data to read UINT8 values".into());
        }
        
        // Get a slice of all the data at once
        let values = &data[*offset..*offset + num_elements];
        
        // Pre-allocate space in the result vector
        result.reserve(num_elements);
        
        // Convert all bytes to f32 at once
        result.extend(values.iter().map(|&value| value as f32));
        
        // Update offset
        *offset += num_elements;
        
        Ok(())
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Uint8Format::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_uint8_format() {
        let format = Uint8Format::new();
        assert_eq!(format.name(), "UINT8");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::UINT8 as u32);
        
        // Test dequantization with some sample data
        let data = vec![0, 1, 2, 3, 4, 255];
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 6, &mut result).unwrap();
        
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 255.0]);
        assert_eq!(offset, 6);
    }
} 