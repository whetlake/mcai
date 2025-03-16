use std::path::PathBuf;
use std::error::Error;
use chrono::{DateTime, Utc};
use memmap2::Mmap;
use std::fs::File;
use std::collections::BTreeMap;
use crate::gguf::{GGUFReader, GGUFValueType, TensorInfo, GGUFValue, GGUFError};
use tracing;

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
    /// metadata
    pub metadata: BTreeMap<String, (String, GGUFValue)>,
    /// tensors
    pub tensors: Vec<TensorInfo>,
}

impl Model {
    pub fn new(
        label: String,
        name: String,
        size: String,
        architecture: String,
        quantization: String,
        path: PathBuf,
        data: Mmap,
        metadata: BTreeMap<String, (String, GGUFValue)>,
        tensors: Vec<TensorInfo>,
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
            metadata,
            tensors,
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
        let model: Model = Self::new(
            label,
            name,
            size,
            architecture,
            quantization,
            path,
            data,
            gguf_reader.metadata,
            gguf_reader.tensors,
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
        tracing::info!("Validating model...");

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
            if let Ok(_) = self.get_metadata_value(&prefixed_key) {
                found_metadata = true;
                break;
            }
        }

        // If not found, try with generic prefix
        if !found_metadata {
            for key in &required_metadata {
                let generic_key = format!("general.{}", key);
                if let Ok(_) = self.get_metadata_value(&generic_key) {
                    found_metadata = true;
                    break;
                }
            }
        }

        if !found_metadata {
            tracing::error!("Missing required metadata for architecture: {}", self.architecture);
            return Err(format!(
                "Missing required metadata for architecture: {}. Need at least one of: {:?}",
                self.architecture, required_metadata
            ).into());
        }

        // 2. Validate memory map
        if self.data.is_empty() {
            tracing::error!("Memory map is empty");
            return Err("Memory map is empty".into());
        }

        // 3. Check tensor data accessibility
        tracing::info!("Validating tensor data accessibility for {} tensors...", self.tensors.len());

        for tensor in &self.tensors {
            
            let bytes_needed = match GGUFValueType::from(tensor.data_type) {
                GGUFValueType::Q3_K_M | GGUFValueType::Q3_K_L | GGUFValueType::Q3_K_S => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 3 / 8 + 2)) as usize
                },
                GGUFValueType::Q4_K_M | GGUFValueType::Q4_K_S => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 4 / 8 + 2)) as usize
                },
                GGUFValueType::Q5_K_M | GGUFValueType::Q5_K_S => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 5 / 8 + 2)) as usize
                },
                GGUFValueType::Q6_K => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 6 / 8 + 2)) as usize
                },
                GGUFValueType::Q2_K => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 2 / 8 + 2)) as usize
                },
                GGUFValueType::Q4_0 => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 4 / 8 + 1)) as usize
                },
                GGUFValueType::Q4_1 => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 4 / 8 + 2)) as usize
                },
                GGUFValueType::Q5_0 => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 5 / 8 + 1)) as usize
                },
                GGUFValueType::Q5_1 => {
                    let block_size = 32;
                    let total_elements = tensor.dims.iter().map(|&d| d as i64).product::<i64>();
                    let blocks = (total_elements + block_size - 1) / block_size;
                    (blocks * (block_size * 5 / 8 + 2)) as usize
                },
                GGUFValueType::Q8_0 => {
                    let block_size = 32;
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
                _ => {
                    tracing::error!("Unsupported data type for tensor '{}': {}", tensor.name, tensor.data_type);
                    return Err(format!("Unsupported data type for size calculation: {}", tensor.data_type).into());
                }
            };
            
            if tensor.offset as usize + bytes_needed > self.data.len() {
                tracing::error!("Tensor '{}' extends beyond memory map bounds", tensor.name);
                return Err(format!(
                    "Tensor '{}' extends beyond memory map bounds (offset: {}, size: {}, total: {}, map size: {})",
                    tensor.name, tensor.offset, bytes_needed, tensor.offset as usize + bytes_needed, self.data.len()
                ).into());
            }
        }

        tracing::info!("Model validation completed successfully");
        Ok(())
    }

    /// Gets a metadata value from the model's metadata by key
    /// 
    /// # Arguments
    /// * `key` - The metadata key to look up
    /// 
    /// # Returns
    /// * `Ok(GGUFValue)` - The metadata value if found
    /// * `Err` - If the key is not found in the metadata
    pub fn get_metadata_value(&self, key: &str) -> Result<GGUFValue, Box<dyn Error + Send + Sync>> {
        match self.metadata.get(key) {
            Some((_, value)) => Ok(value.clone()),
            None => Err(Box::new(GGUFError::MetadataNotFound(key.to_string())))
        }
    }
    
    // /// Get the GGUF reader for accessing model metadata and tensors
    // pub fn gguf_reader(&self) -> &GGUFReader {
    //     &self.gguf_reader
    // }

}