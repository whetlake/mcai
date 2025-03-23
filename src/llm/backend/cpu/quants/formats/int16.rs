use std::error::Error;
use crate::gguf::GGUFValueType;
use super::FormatImpl;

/// INT16 format - direct storage of 16-bit signed integer values
#[derive(Clone)]
pub struct Int16Format;

impl Int16Format {
    pub fn new() -> Self {
        Self {}
    }
}

impl FormatImpl for Int16Format {
    fn gguf_type(&self) -> GGUFValueType {
        GGUFValueType::INT16
    }
    
    fn name(&self) -> &'static str {
        "INT16"
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
        // Calculate bytes needed
        let bytes_needed = num_elements * 2; // 2 bytes per i16
        
        // Ensure we have enough data
        if *offset + bytes_needed > data.len() {
            return Err("Not enough data to read INT16 values".into());
        }
        
        // Check alignment - for int16, offset must be a multiple of 2 bytes
        if *offset % 2 != 0 {
            return Err("Misaligned offset for INT16 data. Offset must be a multiple of 2 bytes.".into());
        }
        
        // Get a slice of all the data at once
        let data_slice = &data[*offset..*offset + bytes_needed];
        
        // Pre-allocate space in the result vector
        result.reserve(num_elements);
        
        // Process all data based on endianness
        if cfg!(target_endian = "little") {
            // On little-endian systems, we can use unsafe for better performance
            // SAFETY: This is safe because:
            // 1. We've verified we have enough data
            // 2. We've verified the offset is properly aligned
            // 3. We're working with a direct copy, not modifying the original data
            unsafe {
                // View the raw bytes as a slice of i16 values
                let i16_slice = std::slice::from_raw_parts(
                    data_slice.as_ptr() as *const i16,
                    num_elements
                );
                
                // Convert all i16 values to f32
                result.extend(i16_slice.iter().map(|&value| value as f32));
            }
        } else {
            // On big-endian systems, we need to manually handle the byte order
            for chunk in data_slice.chunks_exact(2) {
                let value = i16::from_le_bytes([chunk[0], chunk[1]]);
                result.push(value as f32);
            }
        }
        
        // Update offset
        *offset += bytes_needed;
        
        Ok(())
    }
}

/// Create a new boxed instance of this format
pub fn create_format() -> Box<dyn FormatImpl> {
    Box::new(Int16Format::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_int16_format() {
        let format = Int16Format::new();
        assert_eq!(format.name(), "INT16");
        assert_eq!(format.gguf_type() as u32, GGUFValueType::INT16 as u32);
        
        // Test dequantization with some sample data
        let mut data = Vec::new();
        
        // Positive values: 0, 1, 32767
        data.extend_from_slice(&0i16.to_le_bytes());
        data.extend_from_slice(&1i16.to_le_bytes());
        data.extend_from_slice(&32767i16.to_le_bytes());
        
        // Negative values: -1, -2, -32768
        data.extend_from_slice(&(-1i16).to_le_bytes());
        data.extend_from_slice(&(-2i16).to_le_bytes());
        data.extend_from_slice(&(-32768i16).to_le_bytes());
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        format.dequantize(&data, &mut offset, 6, &mut result).unwrap();
        
        assert_eq!(result, vec![0.0, 1.0, 32767.0, -1.0, -2.0, -32768.0]);
        assert_eq!(offset, 12); // 6 * 2 bytes
    }
} 