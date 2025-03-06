use std::path::PathBuf;
use std::error::Error;
use chrono::{DateTime, Utc};
use memmap2::Mmap;
use std::fs::File;
use crate::gguf::{GGUFReader, TensorInfo};
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
        // Open the file and create a memory map
        let file = File::open(&path)?;
        let data = unsafe { Mmap::map(&file)? };
        
        // Create a GGUF reader for accessing model metadata and tensors
        let gguf_reader = GGUFReader::new(&path)?;
        
        // Create and return the model instance
        Ok(Self::new(
            label,
            name,
            size,
            architecture,
            quantization,
            path,
            data,
            gguf_reader,
        ))
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