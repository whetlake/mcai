// Explicit module imports for better IDE support
pub mod float32;
pub mod int8;
pub mod uint8;
pub mod int16;
pub mod uint16;
pub mod q3_k_m;
pub mod q4_k_s;

// Common constants for all formats
pub const QK_K: usize = 256;

use std::error::Error;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use crate::gguf::GGUFValueType;


/// Trait that all format implementations must implement
pub trait FormatImpl: Send + Sync + 'static {
    /// Returns the GGUF value type this implementation handles
    fn gguf_type(&self) -> GGUFValueType;
    
    /// Name of the format
    #[allow(unused)]
    fn name(&self) -> &'static str;
    
    /// Dequantize the data
    fn dequantize(&self, 
                data: &[u8],    
                offset: &mut usize, 
                num_elements: usize, 
                result: &mut Vec<f32>
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
    
    /// Clone this format implementation
    fn clone_box(&self) -> Box<dyn FormatImpl>;
}

// When adding a new format:
// 1. Create your format file (e.g., q4_0.rs) 
// 2. Add it as a pub mod above
// 3. Add it to the formats list below

// Registry of format implementations
static FORMAT_REGISTRY: Lazy<Mutex<Vec<Box<dyn FormatImpl>>>> = Lazy::new(|| {
    let mut registry = Vec::new();
    
    // Register all implemented formats
    registry.push(float32::create_format());
    registry.push(int8::create_format());
    registry.push(uint8::create_format());
    registry.push(int16::create_format());
    registry.push(uint16::create_format());
    registry.push(q3_k_m::create_format());
    registry.push(q4_k_s::create_format());
    // Add new formats here when implemented
    
    Mutex::new(registry)
});

/// Get a format implementation by GGUF value type
pub fn get_format_by_gguf_type(value_type: GGUFValueType) -> Option<Box<dyn FormatImpl>> {
    let registry = FORMAT_REGISTRY.lock().unwrap();
    
    for format in registry.iter() {
        // Compare value types
        if format.gguf_type() as u32 == value_type as u32 {
            // Create a new instance by cloning the implementation
            return Some(format.clone_box());
        }
    }
    
    None
}