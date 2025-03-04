use std::fmt::{self};
use std::error::Error;

/// GGUF metadata value types that can be stored in a GGUF file
#[derive(Clone)]
pub enum GGUFValue {
    /// String value type for text data
    String(String),
    /// Integer value type for whole numbers, stored as i64
    Int(i64), 
    /// Float value type for decimal numbers, stored as f32
    Float(f32),
    /// Boolean value type for true/false values
    Bool(bool),
    /// Array value type for sequences of other GGUF values
    Array(Vec<GGUFValue>),
    /// Truncated array value type for sequences of other GGUF values
    TruncatedArray(Vec<GGUFValue>, u64),
}

impl GGUFValue {
    /// Converts any GGUF value to its string representation
    ///
    /// # Returns
    /// A String containing the value formatted as text
    pub fn to_string(&self) -> String {
        match self {
            GGUFValue::String(s) => s.clone(),
            GGUFValue::Int(i) => i.to_string(),
            GGUFValue::Float(f) => f.to_string(),
            GGUFValue::Bool(b) => b.to_string(),
            GGUFValue::Array(arr) => {
                format!("[{}]", arr.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))
            }
            GGUFValue::TruncatedArray(arr, total) => {
                format!("[{} ... out of {}]", arr.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "), total)
            }
        }
    }
}

// Add this impl to support better debug formatting for the GGUFValue enum
impl fmt::Debug for GGUFValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GGUFValue::String(s) => write!(f, "String({:?})", s),
            GGUFValue::Int(i) => write!(f, "Int({})", i),
            GGUFValue::Float(fl) => write!(f, "Float({})", fl),
            GGUFValue::Bool(b) => write!(f, "Bool({})", b),
            GGUFValue::Array(arr) => {
                if arr.is_empty() {
                    write!(f, "Array([])")
                } else if arr.len() <= 3 {
                    write!(f, "Array({:?})", arr)
                } else {
                    write!(f, "Array([{:?}, {:?}, {:?}, ...and {} more])", 
                           &arr[0], &arr[1], &arr[2], arr.len() - 3)
                }
            },
            GGUFValue::TruncatedArray(arr, total) => {
                if arr.is_empty() {
                    write!(f, "Array([] ...out of {})", total)
                } else if arr.len() <= 3 {
                    write!(f, "Array({:?} ...out of {})", arr, total)
                } else {
                    write!(f, "Array([{:?}, {:?}, {:?}, ...] out of {} total)", 
                           &arr[0], &arr[1], &arr[2], total)
                }
            }
        }
    }
} 

/// Custom error types for GGUF operations
#[derive(Debug)]
pub enum GGUFError {
    /// Wraps std::io::Error for file operations
    IoError(std::io::Error),
    /// Invalid format errors with a message
    InvalidFormat(String),
    /// Missing metadata key errors
    MetadataNotFound(String),
}

/// Implements Display trait for GGUFError for error reporting
impl fmt::Display for GGUFError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GGUFError::IoError(e) => write!(f, "I/O error: {}", e),
            GGUFError::InvalidFormat(msg) => write!(f, "Invalid GGUF format: {}", msg),
            GGUFError::MetadataNotFound(key) => write!(f, "Metadata key not found: {}", key),
        }
    }
}

/// Implements Error trait to allow GGUFError to be used as a standard error type
impl Error for GGUFError {}

/// Allows automatic conversion from std::io::Error to GGUFError
impl From<std::io::Error> for GGUFError {
    fn from(err: std::io::Error) -> Self {
        GGUFError::IoError(err)
    }
}

// Add back the Display implementation alongside Debug
impl fmt::Display for GGUFValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GGUFValue::String(s) => write!(f, "{}", s),
            GGUFValue::Int(i) => write!(f, "{}", i),
            GGUFValue::Float(fl) => write!(f, "{}", fl),
            GGUFValue::Bool(b) => write!(f, "{}", b),
            GGUFValue::Array(arr) => {
                write!(f, "[")?;
                for (i, value) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")? }
                    write!(f, "{}", value)?;
                }
                write!(f, "]")
            },
            GGUFValue::TruncatedArray(arr, total) => {
                write!(f, "[")?;
                for (i, value) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")? }
                    write!(f, "{}", value)?;
                }
                write!(f, " ... out of {}]", total)
            }
        }
    }
}

/// Value type identifiers from the GGUF format specification
#[derive(Debug, Clone, Copy)]
pub enum GGUFValueType {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
}

impl GGUFValueType {
    /// Convert the value type to a string representation
    pub fn type_string(&self) -> String {
        match self {
            GGUFValueType::STRING => "String",
            GGUFValueType::ARRAY => "Array",
            GGUFValueType::BOOL => "Bool",
            GGUFValueType::FLOAT32 | GGUFValueType::FLOAT64 => "Float",
            _ => "Int", // All other numeric types
        }.to_string()
    }
}

// Add From<u32> implementation for GGUFValueType
impl From<u32> for GGUFValueType {
    fn from(value: u32) -> Self {
        match value {
            0 => GGUFValueType::UINT8,
            1 => GGUFValueType::INT8,
            2 => GGUFValueType::UINT16,
            3 => GGUFValueType::INT16,
            4 => GGUFValueType::UINT32,
            5 => GGUFValueType::INT32,
            6 => GGUFValueType::FLOAT32,
            7 => GGUFValueType::BOOL,
            8 => GGUFValueType::STRING,
            9 => GGUFValueType::ARRAY,
            10 => GGUFValueType::UINT64,
            11 => GGUFValueType::INT64,
            12 => GGUFValueType::FLOAT64,
            _ => panic!("Invalid GGUF value type: {}", value),
        }
    }
}