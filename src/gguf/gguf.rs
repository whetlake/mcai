use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::collections::BTreeMap;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Seek, SeekFrom, Read};
use crate::gguf::is_gguf_file;
use super::types::{GGUFValue, GGUFError, GGUFValueType};
use crate::gguf::gguf_utils;
use comfy_table::*;
use tracing::{info, error, debug};

/// A reader for GGUF (GPT-Generated Unified Format) model files.
///
/// The GGUFReader provides functionality to read and parse GGUF format files,
/// which are used to store large language models. It handles reading the file
/// header, metadata, and tensor information.
///
/// # Fields
///
/// * `path` - The path to the GGUF file
/// * `is_valid_gguf` - Whether the file is a valid GGUF file
/// * `tensor_count` - Number of tensors in the model
/// * `metadata` - Key-value pairs of metadata stored in the file
///
/// # Examples
///
/// ```rust
/// use mcai::gguf::GGUFReader;
///
/// let reader = GGUFReader::new("path/to/model.gguf")?;
/// let model_name = reader.get_metadata_value("general.name")?;
/// ```
pub struct GGUFReader {
    pub path: Box<Path>,
    pub is_valid_gguf: bool,
    pub tensor_count: u64,
    pub metadata: BTreeMap<String, (String, GGUFValue)>,
}

/// The magic number that identifies GGUF files
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in ASCII

impl GGUFReader {
    /// Creates a new GGUF reader for the specified file path.
    ///
    /// This function attempts to open and parse a GGUF file, reading its header
    /// and metadata. It validates the file format and extracts key information
    /// about the model.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file to read
    ///
    /// # Returns
    ///
    /// * `Result<Self, Box<dyn Error + Send + Sync>>` - A new GGUFReader instance
    ///   or an error if the file cannot be read or is not a valid GGUF file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The file cannot be opened
    /// * The file is not a valid GGUF file
    /// * The file header cannot be read
    /// * The metadata cannot be parsed
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

        // Read metadata items, but stop early if issues occur
        for i in 0..metadata_count {
            match read_metadata_kv(&mut file, version) {
                Ok((key, type_str, value)) => {
                    metadata.insert(key, (type_str, value));
                    actual_count += 1;
                },
                Err(e) => {
                    error!("Reached end of metadata at index {} of reported {}: {}", 
                            i, metadata_count, e);
                    break;
                }
            }
        }

        info!("Successfully read {}/{} metadata entries from GGUF file", actual_count, metadata_count);
        
        Ok(Self {
            path: path_box,
            is_valid_gguf: true,
            tensor_count,
            metadata,
        })
    }

    /// Retrieves a metadata value by its key.
    ///
    /// This method looks up a metadata value in the GGUF file using its key.
    /// The key should be in the format "category.name" (e.g., "general.name").
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up in the metadata
    ///
    /// # Returns
    ///
    /// * `Result<GGUFValue, Box<dyn Error + Send + Sync>>` - The metadata value
    ///   or an error if the key is not found
    ///
    /// # Examples
    ///
    /// ```rust
    /// let reader = GGUFReader::new("model.gguf")?;
    /// let model_name = reader.get_metadata_value("general.name")?;
    /// ```
    pub fn get_metadata_value(&self, key: &str) -> Result<GGUFValue, Box<dyn Error + Send + Sync>> {
        match self.metadata.get(key) {
            Some((_, value)) => Ok(value.clone()),
            None => Err(Box::new(GGUFError::MetadataNotFound(key.to_string())))
        }
    }

    /// Displays the metadata in a formatted table.
    ///
    /// This method prints all metadata key-value pairs in a nicely formatted
    /// table using the comfy-table crate. It's useful for debugging and
    /// displaying model information.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let reader = GGUFReader::new("model.gguf")?;
    /// reader.print_metadata_table();
    /// ```
    pub fn print_metadata_table(&self) {
        let mut table = Table::new();
        table
            .set_header(vec!["Label", "Type", "Value"])
            .load_preset(comfy_table::presets::UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic);

        for (key, (type_str, value)) in &self.metadata {
            let value_str = match value {
                GGUFValue::TruncatedArray(arr, total) => {
                    if arr.is_empty() {
                        format!("[] ... out of {}", total)
                    } else {
                        let preview: Vec<_> = arr.iter().take(3).map(|v| v.to_string()).collect();
                        format!("[{}{}] ... out of {}", 
                            preview.join(", "),
                            if arr.len() > 3 { ", ..." } else { "" },
                            total)
                    }
                },
                _ => format!("{}", value)
            };

            table.add_row(vec![key, type_str, &value_str]);
        }

        println!("{}", table);
    }
}

/// Reads a metadata key-value pair from the file.
///
/// This function parses a single metadata entry from the GGUF file, including
/// the key, value type, and value itself. It handles different GGUF versions
/// and value types appropriately.
///
/// # Arguments
///
/// * `file` - The file to read from
/// * `version` - The GGUF file version
///
/// # Returns
///
/// * `Result<(String, String, GGUFValue), Box<dyn Error + Send + Sync>>` - A tuple
///   containing the key, type string, and value, or an error if reading fails
///
/// # Errors
///
/// Returns an error if:
/// * The key cannot be read
/// * The value type is invalid
/// * The value cannot be parsed
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
            
            // Prepare preview array
            let preview_count = std::cmp::min(5, arr_len as usize);
            let mut preview = Vec::with_capacity(preview_count);
            
            // Single loop to handle all elements
            for i in 0..arr_len {
                if element_type == 8 { // STRING
                    // Read or skip based on whether we want this element in the preview
                    let read_full = i < preview_count as u64;
                    let str_val = gguf_utils::read_string(file, version, !read_full)?;
                    
                    // Add to preview if needed
                    if read_full {
                        preview.push(GGUFValue::String(str_val));
                    }
                } else {
                    // Handle non-string types
                    if i < preview_count as u64 {
                        // Read and store full value for preview
                        let value = gguf_utils::read_value_by_type(file, element_type, version)?;
                        preview.push(value);
                    } else {
                        // Skip using type size
                        let element_size = gguf_utils::get_type_size(element_type, version)?;
                        file.seek(SeekFrom::Current(element_size as i64))?;
                    }
                }
            }
            
            // Return the truncated array with preview
            Ok((key, type_str, GGUFValue::TruncatedArray(preview, arr_len)))
        },
        _ => {
            let value = gguf_utils::read_value_by_type(file, value_type, version)?;
            Ok((key, type_str, value))
        }
    }
}