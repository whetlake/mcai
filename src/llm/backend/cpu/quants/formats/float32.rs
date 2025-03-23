use std::error::Error;
use crate::gguf::GGUFValueType;
use super::FormatImpl;

/// Float32 format - direct storage of 32-bit floating point values
#[derive(Clone)]
pub struct Float32Format;

impl Float32Format {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Float32Format {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::FLOAT32
    }
    
    fn name(&self) -> &'static str {
        "FLOAT32"
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
        let bytes_needed = num_elements * 4; // 4 bytes per f32
        let actual_size_mb = bytes_needed as f64 / (1024.0 * 1024.0);
        let compression_ratio = 1.0; // No compression for f32
        
        println!("FLOAT32 Format Size Details:");
        println!("  Total bytes needed: {} ({:.4} MB)", bytes_needed, actual_size_mb);
        println!("  Compression ratio: {:.2}x", compression_ratio);
        
        // Ensure we have enough data
        if *offset + bytes_needed > data.len() {
            let available = if data.len() > *offset {
                data.len() - *offset
            } else {
                0
            };
            return Err(format!("Not enough data to read FLOAT32 values. Need {} bytes, but only have {}", 
                              bytes_needed, available).into());
        }
        
        // Check alignment - for float32, offset must be a multiple of 4 bytes
        if *offset % 4 != 0 {
            return Err("Misaligned offset for FLOAT32 data. Offset must be a multiple of 4 bytes.".into());
        }
        
        // Get a slice of all the data at once
        let data_slice = &data[*offset..*offset + bytes_needed];
        
        // Pre-allocate space in the result vector
        result.reserve(num_elements);
        
        // SAFETY: This is safe because:
        // 1. We've verified we have enough data
        // 2. We've verified the offset is properly aligned
        // 3. We're interpreting raw bytes as f32, which is a transparent operation
        // 4. We're working with a direct copy, not modifying the original data
        unsafe {
            // View the raw bytes as a slice of f32 values
            let float_slice = std::slice::from_raw_parts(
                data_slice.as_ptr() as *const f32,
                num_elements
            );
            
            // Copy all values into our result vector
            result.extend_from_slice(float_slice);
        }
        
        // Update offset
        *offset += bytes_needed;
        
        Ok(())
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Float32Format::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_float32_format() {
        let format = Float32Format::new();
        assert_eq!(format.name(), "FLOAT32");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::FLOAT32 as u32);
        
        // Test dequantization with some sample data
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 3, &mut result).unwrap();
        
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        assert_eq!(offset, 12); // 3 * 4 bytes
    }
} 