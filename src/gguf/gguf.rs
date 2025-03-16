use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::collections::BTreeMap;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Seek, SeekFrom, Read};
use crate::gguf::is_gguf_file;
use super::types::{GGUFValue, GGUFError, GGUFValueType, TensorInfo};
use crate::gguf::gguf_utils;
use tracing::{info, error, debug};

pub struct GGUFReader {
    /// Path to the GGUF file
    pub path: PathBuf,
    /// Whether the file is a valid GGUF file
    pub is_valid_gguf: bool,
    /// Number of tensors in the file
    pub tensor_count: u64,
    /// Metadata key-value pairs
    pub metadata: BTreeMap<String, (String, GGUFValue)>,
    /// Information about each tensor
    pub tensors: Vec<TensorInfo>,
    /// File type from metadata (e.g., 15 for Q8_K)
    pub file_type: i64,
    /// Quantization version from metadata
    pub quantization_version: i64,
}

/// The magic number that identifies GGUF files
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in ASCII

impl GGUFReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let path_box = Box::from(path.as_ref());
        
        if !is_gguf_file(&path_box) {
            return Err(Box::new(GGUFError::InvalidFormat("Invalid magic number".into())));
        }
        
        // Open file and parse the GGUF format
        let mut file = File::open(&path_box)?;
        
        // Skip past the magic number
        file.seek(SeekFrom::Start(4))?;
        
        // Read version
        let version = file.read_u32::<LittleEndian>()?;
        
        // Read tensor count
        let tensor_count = if version >= 3 {
            file.read_u64::<LittleEndian>()?
        } else {
            file.read_u32::<LittleEndian>()? as u64
        };

        // Read metadata count
        let metadata_count = if version >= 3 {
            file.read_u64::<LittleEndian>()?
        } else {
            file.read_u32::<LittleEndian>()? as u64
        };

        debug!("Reading GGUF file: {} tensors, {} metadata entries", tensor_count, metadata_count);
        
        // Parse all metadata. This is necessary to update the model registry.
        let mut metadata: BTreeMap<String, (String, GGUFValue)> = BTreeMap::new();
        let mut actual_count = 0;
        let mut file_type = 0i64;
        let mut quantization_version = 0i64;

        // Read metadata items, but stop early if issues occur
        for i in 0..metadata_count {
            match read_metadata_kv(&mut file, version) {
                Ok((key, type_str, value)) => {
                    metadata.insert(key.clone(), (type_str, value.clone()));
                    actual_count += 1;
                    if key == "general.file_type" {
                        if let Some(v) = value.as_int() {
                            file_type = v;
                        }
                    } else if key == "general.quantization_version" {
                        if let Some(v) = value.as_int() {
                            quantization_version = v;
                        }
                    }
                },
                Err(e) => {
                    error!("Reached end of metadata at index {} of reported {}: {}", 
                            i, metadata_count, e);
                    break;
                }
            }
        }

        info!("Successfully read {}/{} metadata entries from GGUF file", actual_count, metadata_count);
        
        // Read tensor information
        let tensors = read_tensor_info(&mut file, tensor_count, version)?;
        
        Ok(Self {
            path: path_box.to_path_buf(),
            is_valid_gguf: true,
            tensor_count,
            metadata,
            tensors,
            file_type,
            quantization_version,
        })
    }

    pub fn get_metadata_value(&self, key: &str) -> Result<GGUFValue, Box<dyn Error + Send + Sync>> {
        match self.metadata.get(key) {
            Some((_, value)) => Ok(value.clone()),
            None => Err(Box::new(GGUFError::MetadataNotFound(key.to_string())))
        }
    }

}

fn read_metadata_kv(file: &mut File, version: u32) -> Result<(String, String, GGUFValue), Box<dyn Error + Send + Sync>> {
    // Read key length
    let key_length = if version >= 3 {
        file.read_u64::<LittleEndian>()?
    } else {
        file.read_u32::<LittleEndian>()? as u64
    };
    
    let mut key_bytes = vec![0u8; key_length as usize];
    file.read_exact(&mut key_bytes)?;
    let key = String::from_utf8(key_bytes)
        .map_err(|e| Box::new(GGUFError::InvalidFormat(format!("Invalid UTF-8 in key: {}", e))))?;
    // So now we have the value name, ie the metadata key name or metadata label
    
    // Read value type
    let value_type = file.read_u32::<LittleEndian>()?;
    let value_type_enum: GGUFValueType = value_type.into();

    // Get type string
    let type_str = value_type_enum.type_string();

    // Handle based on value type
    match value_type_enum {
        GGUFValueType::ARRAY => {
            // First read the element type and array length
            let element_type = file.read_u32::<LittleEndian>()?;
            let arr_len = if version >= 3 {
                file.read_u64::<LittleEndian>()?
            } else {
                file.read_u32::<LittleEndian>()? as u64
            };
            
            // Read full array
            let mut array = Vec::with_capacity(arr_len as usize);
            
            // Single loop to handle all elements
            for i in 0..arr_len {
                if element_type == 8 { // STRING
                    let str_val = gguf_utils::read_string(file, version, false)?;
                    array.push(GGUFValue::String(str_val));
                } else {
                    let value = gguf_utils::read_value_by_type(file, element_type, version)?;
                    array.push(value);
                }
            }
            
            // Return the full array
            Ok((key, type_str, GGUFValue::Array(array)))
        },
        _ => {
            let value = gguf_utils::read_value_by_type(file, value_type, version)?;
            Ok((key, type_str, value))
        }
    }
}


fn read_tensor_info(file: &mut File, tensor_count: u64, version: u32) -> Result<Vec<TensorInfo>, Box<dyn Error + Send + Sync>> {
    let mut tensors = Vec::with_capacity(tensor_count as usize);

    for _ in 0..tensor_count {
        // Read tensor name
        let name = gguf_utils::read_string(file, version, false)?;
        
        // Read number of dimensions
        let n_dims = if version >= 3 {
            file.read_u32::<LittleEndian>()?
        } else {
            file.read_u32::<LittleEndian>()?
        };

        // Read dimensions
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = if version >= 3 {
                file.read_u64::<LittleEndian>()?
            } else {
                file.read_u32::<LittleEndian>()? as u64
            };
            dims.push(dim);
        }

        // Read data type
        let data_type = file.read_u32::<LittleEndian>()?;

        // Read offset
        let offset = if version >= 3 {
            file.read_u64::<LittleEndian>()?
        } else {
            file.read_u32::<LittleEndian>()? as u64
        };

        tensors.push(TensorInfo {
            name,
            n_dims,
            dims,
            data_type: data_type.into(),
            offset,
        });
    }

    Ok(tensors)
}