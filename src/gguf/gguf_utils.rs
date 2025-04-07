use std::fs::File;
use std::path::Path;
use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};
use super::types::{GGUFValue, GGUFError};
use std::error::Error;
// use tracing::{info, error, debug};

/// The magic number that identifies GGUF files
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in ASCII

/// Read a string value from the file
/// If skip_content is true, just read the length and skip the content
pub fn read_string(file: &mut File, version: u32, skip_content: bool) -> Result<String, Box<dyn Error + Send + Sync>> {
    // Read string length
    let str_len = if version >= 3 {
        file.read_u64::<LittleEndian>()?
    } else {
        file.read_u32::<LittleEndian>()? as u64
    };
    
    if skip_content {
        // Skip string content without reading it
        file.seek(SeekFrom::Current(str_len as i64))?;
        Ok(String::new()) // Return empty string
    } else {
        // Read string content
        let mut buffer = vec![0u8; str_len as usize];
        file.read_exact(&mut buffer)?;
        
        // Convert to String
        let string = String::from_utf8(buffer)
            .map_err(|e| Box::new(GGUFError::InvalidFormat(format!("Invalid UTF-8 in string: {}", e))))?;
        
        Ok(string)
    }
}

/// Read a GGUF value of the specified type from the file
pub fn read_value_by_type(file: &mut File, value_type: u32, version: u32) -> Result<GGUFValue, Box<dyn Error + Send + Sync>> {
    match value_type {
        0 => { // UINT8
            let val = file.read_u8()?;
            Ok(GGUFValue::Int(val as i64))
        },
        1 => { // INT8
            let val = file.read_i8()?;
            Ok(GGUFValue::Int(val as i64))
        },
        2 => { // UINT16
            let val = file.read_u16::<LittleEndian>()?;
            Ok(GGUFValue::Int(val as i64))
        },
        3 => { // INT16
            let val = file.read_i16::<LittleEndian>()?;
            Ok(GGUFValue::Int(val as i64))
        },
        4 => { // UINT32
            let val = file.read_u32::<LittleEndian>()?;
            Ok(GGUFValue::Int(val as i64))
        },
        5 => { // INT32
            let val = file.read_i32::<LittleEndian>()?;
            Ok(GGUFValue::Int(val as i64))
        },
        6 => { // FLOAT32
            let val = file.read_f32::<LittleEndian>()?;
            Ok(GGUFValue::Float(val))
        },
        7 => { // BOOL
            let val = file.read_u8()? != 0;
            Ok(GGUFValue::Bool(val))
        },
        8 => { // STRING
            let s: String = read_string(file, version, false)?;
            Ok(GGUFValue::String(s))
        },
        10 => { // UINT64
            let val = file.read_u64::<LittleEndian>()?;
            Ok(GGUFValue::Int(val as i64))
        },
        11 => { // INT64
            let val = file.read_i64::<LittleEndian>()?;
            Ok(GGUFValue::Int(val))
        },
        12 => { // FLOAT64
            let val = file.read_f64::<LittleEndian>()?;
            // Store as f32 since we don't have a dedicated f64 type
            Ok(GGUFValue::Float(val as f32))
        },
        _ => {
            Err(Box::new(GGUFError::InvalidFormat(
                format!("Unknown value type: {}", value_type)
            )))
        }
    }
}

/// Checks if a file at the given path is a GGUF format file by verifying its magic number.
///
/// # Arguments
///
/// * `path` - Path to the file to check
///
/// # Returns
///
/// `true` if the file exists and has a valid GGUF magic number, `false` otherwise
pub fn is_gguf_file<P: AsRef<Path>>(path: P) -> bool {
    if let Ok(mut file) = File::open(path) {
        if let Ok(magic) = file.read_u32::<LittleEndian>() {
            return magic == GGUF_MAGIC;
        }
    }
    false
}

/// Get the size in bytes of a specific GGUF value type
#[allow(dead_code)]
pub fn get_type_size(value_type: u32, _version: u32) -> Result<u64, Box<dyn Error + Send + Sync>> {
    match value_type {
        0 => Ok(1), // UINT8
        1 => Ok(1), // INT8
        2 => Ok(2), // UINT16
        3 => Ok(2), // INT16
        4 => Ok(4), // UINT32
        5 => Ok(4), // INT32
        6 => Ok(4), // FLOAT32
        7 => Ok(1), // BOOL
        8 => Err(Box::new(GGUFError::InvalidFormat(
            format!("Cannot determine fixed size for variable-length string type")))), // STRING
        10 => Ok(8), // UINT64
        11 => Ok(8), // INT64
        12 => Ok(8), // FLOAT64
        _ => Err(Box::new(GGUFError::InvalidFormat(
            format!("Unknown value type: {}", value_type)
        )))
    }
} 