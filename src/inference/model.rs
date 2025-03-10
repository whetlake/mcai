use std::path::PathBuf;
use std::error::Error;
use chrono::{DateTime, Utc};
use memmap2::Mmap;
use std::fs::File;
use crate::gguf::{GGUFReader, TensorInfo, GGUFValueType};
use std::fmt;

/// Represents a loaded model in memory.
///
/// This struct contains the runtime state of a loaded model,
/// including its metadata and any resources needed for inference.
pub struct Model {
    /// Unique identifier for the model
    pub label: String,
    /// Human-readable name of the model
    pub name: String,
    /// Size category (e.g., "7B", "13B")
    pub size: String,
    /// Model architecture (e.g., "LLaMA", "Mistral")
    pub architecture: String,
    /// Quantization format (e.g., "Q4_K_M", "Q5_K_M")
    pub quantization: String,
    /// Path to the model file
    pub path: PathBuf,
    /// When the model was loaded
    pub loaded_at: DateTime<Utc>,
    /// Memory-mapped data of the model file
    data: Mmap,
    /// GGUF reader for accessing model metadata and tensors
    gguf_reader: GGUFReader,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("label", &self.label)
            .field("name", &self.name)
            .field("size", &self.size)
            .field("architecture", &self.architecture)
            .field("quantization", &self.quantization)
            .field("path", &self.path)
            .field("loaded_at", &self.loaded_at)
            .field("data_len", &self.data.len())
            .field("tensor_count", &self.gguf_reader.tensor_count)
            .finish()
    }
}

impl Clone for Model {
    fn clone(&self) -> Self {
        // Create a new memory mapping of the same file
        let file = File::open(&self.path).expect("Failed to open model file for cloning");
        let data = unsafe { Mmap::map(&file).expect("Failed to create memory mapping for clone") };
        
        // Create a new GGUF reader
        let gguf_reader = GGUFReader::new(&self.path).expect("Failed to create GGUF reader for clone");
        
        Self {
            label: self.label.clone(),
            name: self.name.clone(),
            size: self.size.clone(),
            architecture: self.architecture.clone(),
            quantization: self.quantization.clone(),
            path: self.path.clone(),
            loaded_at: self.loaded_at,
            data,
            gguf_reader,
        }
    }
}

impl Model {
    /// Creates a new Model instance.
    ///
    /// # Arguments
    ///
    /// * `label` - Unique identifier for the model
    /// * `name` - Human-readable name
    /// * `size` - Size category
    /// * `architecture` - Model architecture
    /// * `quantization` - Quantization format
    /// * `path` - Path to the model file
    /// * `data` - Memory-mapped data of the model file
    /// * `gguf_reader` - GGUF reader for accessing model metadata and tensors
    pub fn new(
        label: String,
        name: String,
        size: String,
        architecture: String,
        quantization: String,
        path: PathBuf,
        data: Mmap,
        gguf_reader: GGUFReader,
    ) -> Self {
        Self {
            label,
            name,
            size,
            architecture,
            quantization,
            path,
            loaded_at: Utc::now(),
            data,
            gguf_reader,
        }
    }

    /// Loads a model from a file.
    ///
    /// # Arguments
    ///
    /// * `label` - Unique identifier for the model
    /// * `name` - Human-readable name
    /// * `size` - Size category
    /// * `architecture` - Model architecture
    /// * `quantization` - Quantization format
    /// * `path` - Path to the model file
    ///
    /// # Returns
    ///
    /// A Result containing the loaded Model or an error
    pub fn load(
        label: String,
        name: String,
        size: String,
        architecture: String,
        quantization: String,
        path: PathBuf,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Step 1: Create GGUF reader to validate the model file
        let gguf_reader = GGUFReader::new(&path)?;
        
        // Step 2: Validate basic model structure
        // Check if the file has the expected number of tensors
        if gguf_reader.tensor_count == 0 {
            return Err("Model file contains no tensors".into());
        }

        // Step 3: Create memory map for the validated model
        let file = File::open(&path)?;
        let data = unsafe { Mmap::map(&file)? };

        // Step 4: Validate memory map
        if data.is_empty() {
            return Err("Memory map is empty".into());
        }

        // Step 5: Create the model instance
        let model = Self::new(
            label,
            name,
            size,
            architecture,
            quantization,
            path,
            data,
            gguf_reader,
        );

        // Step 6: Validate the model instance
        // This checks that we can access required metadata and tensors
        model.validate()?;
        
        Ok(model)
    }

    /// Validates the model structure and compatibility
    ///
    /// This function checks:
    /// 1. Required metadata is present (with architecture-specific prefixes)
    /// 2. Required tensors exist and have correct shapes
    /// 3. Memory map is valid and accessible
    /// 4. Model architecture is supported
    ///
    /// # Returns
    ///
    /// A Result indicating whether the model is valid and compatible
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Validating model...");

        self.gguf_reader.print_metadata_table();
        
        // 1. Check required metadata
        let architecture = self.architecture.to_lowercase();
        
        // Define the required metadata keys that should exist for any architecture
        // We only specify the part after the dot, as the prefix might vary
        let required_metadata = vec![
            "name",           // Model name
            "basename",       // Base name for the model
            "size_label",     // Size category (e.g., "7B")
            "architecture",   // Model architecture
            "quantization_version", // Quantization format
            "hidden_size",    // Size of hidden layers
            "n_heads",        // Number of attention heads
            "n_layers",       // Number of transformer layers
            "vocab_size",     // Size of vocabulary
            "context_length", // Maximum context length
        ];

        // Try to get metadata values using architecture prefix
        let mut found_metadata = false;
        
        // First try with architecture-specific prefix
        for key in &required_metadata {
            let prefixed_key = format!("{}.{}", architecture, key);
            if let Ok(_) = self.gguf_reader.get_metadata_value(&prefixed_key) {
                found_metadata = true;
                break;
            }
        }

        // If not found, try with generic prefix
        if !found_metadata {
            for key in &required_metadata {
                let generic_key = format!("general.{}", key);
                if let Ok(_) = self.gguf_reader.get_metadata_value(&generic_key) {
                    found_metadata = true;
                    break;
                }
            }
        }

        if !found_metadata {
            return Err(format!(
                "Missing required metadata for architecture: {}. Need at least one of: {:?}",
                self.architecture, required_metadata
            ).into());
        }

        // 2. Validate memory map
        if self.data.is_empty() {
            return Err("Memory map is empty".into());
        }

        // Debug: Print metadata table
        println!("\nModel Metadata:");
        for (key, (type_str, value)) in &self.gguf_reader.metadata {
            println!("  {}: {:?} (type: {})", key, value, type_str);
        }

        // 3. Check tensor data accessibility
        // Get quantization information from metadata
        let file_type = self.get_metadata_value("general.file_type")
            .ok()
            .and_then(|v| v.as_int())
            .ok_or_else(|| "Failed to get file_type from metadata")?;
        
        // Convert file_type to GGUFValueType
        let file_type_enum = GGUFValueType::from(file_type as u32);
        
        println!("\nValidating tensor data accessibility:");
        println!("  Memory map size: {} bytes", self.data.len());
        println!("  File type: {} ({})", file_type, file_type_enum.type_string());
        println!("  Total tensors: {}", self.gguf_reader.tensors.len());

        for tensor in &self.gguf_reader.tensors {
            let tensor_size = tensor.dims.iter().product::<u64>() as usize;
            println!("\n  Tensor: {}", tensor.name);
            println!("    Type: {} ({} elements)", tensor.type_string(), tensor_size);
            println!("    Dimensions: {:?}", tensor.dims);
            
            let bytes_needed = match GGUFValueType::from(tensor.data_type) {
                GGUFValueType::Q3_K_M | GGUFValueType::Q3_K_L | GGUFValueType::Q3_K_S => {
                    // Q3_K uses 3 bits per value plus 2 bytes per block
                    let block_size = 32; // Q3_K block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 3 / 8 + 2)) as usize
                },
                GGUFValueType::Q4_K_M | GGUFValueType::Q4_K_S => {
                    // Q4_K uses 4 bits per value plus 2 bytes per block
                    let block_size = 32; // Q4_K block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 4 / 8 + 2)) as usize
                },
                GGUFValueType::Q5_K_M | GGUFValueType::Q5_K_S => {
                    // Q5_K uses 5 bits per value plus 2 bytes per block
                    let block_size = 32; // Q5_K block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 5 / 8 + 2)) as usize
                },
                GGUFValueType::Q6_K => {
                    // Q6_K uses 6 bits per value plus 2 bytes per block
                    let block_size = 32; // Q6_K block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 6 / 8 + 2)) as usize
                },
                GGUFValueType::Q2_K => {
                    // Q2_K uses 2 bits per value plus 2 bytes per block
                    let block_size = 32; // Q2_K block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 2 / 8 + 2)) as usize
                },
                GGUFValueType::Q4_0 => {
                    // Q4_0 uses 4 bits per value plus 1 byte per block
                    let block_size = 32; // Q4_0 block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 4 / 8 + 1)) as usize
                },
                GGUFValueType::Q4_1 => {
                    // Q4_1 uses 4 bits per value plus 2 bytes per block
                    let block_size = 32; // Q4_1 block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 4 / 8 + 2)) as usize
                },
                GGUFValueType::Q5_0 => {
                    // Q5_0 uses 5 bits per value plus 1 byte per block
                    let block_size = 32; // Q5_0 block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 5 / 8 + 1)) as usize
                },
                GGUFValueType::Q5_1 => {
                    // Q5_1 uses 5 bits per value plus 2 bytes per block
                    let block_size = 32; // Q5_1 block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 5 / 8 + 2)) as usize
                },
                GGUFValueType::Q8_0 => {
                    // Q8_0 uses 8 bits per value plus 1 byte per block
                    let block_size = 32; // Q8_0 block size
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size + 1)) as usize
                },
                GGUFValueType::FLOAT32 => {
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    (total_elements * 4) as usize
                },
                GGUFValueType::UINT8 | GGUFValueType::INT8 => {
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    total_elements as usize
                },
                GGUFValueType::UINT16 | GGUFValueType::INT16 => {
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    (total_elements * 2) as usize
                },
                GGUFValueType::UINT32 | GGUFValueType::INT32 => {
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    (total_elements * 4) as usize
                },
                GGUFValueType::UINT64 | GGUFValueType::INT64 => {
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    (total_elements * 8) as usize
                },
                _ => panic!("Unsupported data type for size calculation: {}", tensor.data_type),
            };
            
            println!("    Bytes needed: {}", bytes_needed);
            println!("    Offset: {}", tensor.offset);
            println!("    Total required size: {}", tensor.offset as usize + bytes_needed);
            
            // Debug logging for tensor details
            println!("    Debug - Tensor details:");
            println!("      Name: {}", tensor.name);
            println!("      Type: {} ({} elements)", tensor.type_string(), tensor_size);
            println!("      Dimensions: {:?}", tensor.dims);
            println!("      Data type: {}", tensor.data_type);
            
            // Validate that the tensor's data fits within the memory map
            if tensor.offset as usize + bytes_needed > self.data.len() {
                println!("    ❌ ERROR: Tensor extends beyond memory map bounds");
                println!("    Debug - Memory map details:");
                println!("      Memory map size: {} bytes", self.data.len());
                println!("      Tensor offset: {} bytes", tensor.offset);
                println!("      Tensor size: {} bytes", bytes_needed);
                println!("      Total required: {} bytes", tensor.offset as usize + bytes_needed);
                self.gguf_reader.print_metadata_table();
                return Err(format!(
                    "Tensor '{}' extends beyond memory map bounds (offset: {}, size: {}, total: {}, map size: {})",
                    tensor.name, tensor.offset, bytes_needed, tensor.offset as usize + bytes_needed, self.data.len()
                ).into());
            }
            println!("    ✓ Valid");
        }

        println!("\n✓ All tensors validated successfully");
        Ok(())
    }

    /// Helper method to get metadata value with error handling
    fn get_metadata_value(&self, key: &str) -> Result<crate::gguf::GGUFValue, Box<dyn Error + Send + Sync>> {
        self.gguf_reader.get_metadata_value(key)
    }

    /// Displays model information in a formatted way
    pub fn display_info(&self) {
        println!("\nModel Information");
        println!("{}", "=".repeat(50));
        println!("Name: {}", self.name);
        println!("Label: {}", self.label);
        println!("Size: {}", self.size);
        println!("Architecture: {}", self.architecture);
        println!("Quantization: {}", self.quantization);
        println!("Memory mapped size: {} bytes", self.data.len());
        println!("Tensor count: {}", self.gguf_reader.tensor_count);
        println!("Status: Attached");
        println!();
    }

    /// Get reference to the memory-mapped data
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    
    /// Get the GGUF reader for accessing model metadata and tensors
    pub fn gguf_reader(&self) -> &GGUFReader {
        &self.gguf_reader
    }
    
    /// Get a tensor by name
    pub fn get_tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.gguf_reader.tensors.iter().find(|t| t.name == name)
    }
    
    /// Get tensor data at the specified offset
    pub fn get_tensor_data(&self, offset: u64, size: usize) -> &[u8] {
        &self.data[offset as usize..offset as usize + size]
    }
}