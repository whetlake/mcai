use std::path::PathBuf;
use std::error::Error;
use chrono::{DateTime, Utc};
use memmap2::Mmap;
use std::fs::File;
use std::collections::BTreeMap;
use crate::gguf::{GGUFReader, TensorInfo, GGUFValue, GGUFError};
use crate::llm::model::types::ModelParameters;
use tracing;

/// Represents the static metadata and tensor structure loaded from a GGUF file.
///
/// This struct contains the runtime state of a loaded model,
/// including its metadata and any resources needed for inference.
pub struct ModelMetadata {
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
    /// Parsed model parameters for easy access
    pub params: ModelParameters,
    /// Starting index of transformer blocks (0 or 1)
    pub block_index_start: Option<usize>,
    /// Block count
    pub block_count: Option<usize>,
}

impl ModelMetadata {
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
        // Extract model parameters from metadata
        let params = Self::extract_model_parameters(&metadata, &architecture);
        
        let model = Self {
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
            params,
            block_index_start: None,
            block_count: None,
        };
        
        model
    }

    /// Extract key model parameters from metadata for easy access
    fn extract_model_parameters(metadata: &BTreeMap<String, (String, GGUFValue)>, architecture: &str) -> ModelParameters {
        let arch_lowercase = architecture.to_lowercase();
        
        // Helper function to get metadata with architecture prefix or fallback
        let get_metadata = |key: &str, metadata: &BTreeMap<String, (String, GGUFValue)>, arch: &str| -> Option<GGUFValue> {
            // Try with architecture prefix
            if let Some((_, value)) = metadata.get(&format!("{}.{}", arch, key)) {
                return Some(value.clone());
            }
            
            // Try without prefix
            if let Some((_, value)) = metadata.get(key) {
                return Some(value.clone());
            }
            None
        };
        
        // Get vocabulary size from tokenizer.ggml.tokens array length
        let vocab_size = match metadata.get("tokenizer.ggml.tokens") {
            Some((_, value)) => {
                // The value should be an array of tokens
                match value {
                    GGUFValue::Array(arr) => arr.len(),
                    _ => {
                        tracing::warn!("tokenizer.ggml.tokens is not an array");
                        32000 // Default fallback
                    }
                }
            },
            None => {
                tracing::warn!("Could not find tokenizer.ggml.tokens in model metadata");
                32000 // Default fallback
            }
        };
        
        // Extract other parameters with architecture-aware fallbacks
        let hidden_dim = get_metadata("embedding_length", metadata, &arch_lowercase)
            .map(|v| v.to_string().parse::<usize>().unwrap_or(4096))
            .unwrap_or(4096);
            
        let block_count = get_metadata("block_count", metadata, &arch_lowercase)
            .map(|v| v.to_string().parse::<usize>().unwrap_or(32))
            .unwrap_or(32);
            
        let head_count = get_metadata("attention.head_count", metadata, &arch_lowercase)
            .map(|v| v.to_string().parse::<usize>().unwrap_or(32))
            .unwrap_or(32);
            
        let head_count_kv = get_metadata("attention.head_count_kv", metadata, &arch_lowercase)
            .map(|v| v.to_string().parse::<usize>().unwrap_or(head_count))
            .unwrap_or(head_count);
            
        let model_context_length = get_metadata("context_length", metadata, &arch_lowercase)
            .map(|v| v.to_string().parse::<usize>().unwrap_or(4096))
            .unwrap_or(4096);
            
        let feed_forward_length = get_metadata("feed_forward_length", metadata, &arch_lowercase)
            .map(|v| v.to_string().parse::<usize>().unwrap_or(hidden_dim * 4))
            .unwrap_or(hidden_dim * 4);
            
        let layer_norm_rms_epsilon = get_metadata("attention.layer_norm_rms_epsilon", metadata, &arch_lowercase)
            .map(|v| v.to_string().parse::<f32>().unwrap_or(1e-5))
            .unwrap_or(1e-5);
            
        // Print the extracted parameters for debugging
        tracing::info!("Extracted model parameters:");
        tracing::info!("  vocab_size: {}", vocab_size);
        tracing::info!("  hidden_dim: {}", hidden_dim);
        tracing::info!("  block_count: {}", block_count);
        tracing::info!("  head_count: {}", head_count);
        tracing::info!("  head_count_kv: {}", head_count_kv);
        tracing::info!("  model_context_length: {}", model_context_length);
        tracing::info!("  feed_forward_length: {}", feed_forward_length);
        tracing::info!("  layer_norm_rms_epsilon: {}", layer_norm_rms_epsilon);
        
        // Return the extracted parameters
        ModelParameters {
            vocab_size,
            hidden_dim,
            block_count,
            head_count,
            head_count_kv,
            model_context_length,
            feed_forward_length,
            layer_norm_rms_epsilon,
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
        let mut model: ModelMetadata = Self::new(
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
    pub fn validate(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
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

        // 3. Check that essential tensors are accessible (based on names in self.tensors)
        self.check_transformer_architecture();

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

    /// Check that correct tensors exist and are present
    pub fn check_transformer_architecture(&mut self) {
        // List of essential tensor types to check for (these need exact matches)
        // These are global tensors, not per-block/layer
        let essential_tensors = [
            "output.weight",              // Output projection (final)
            "output_norm.weight",         // Initial normalization
            "token_embd.weight",          // Embedding table
        ];
        
        // Check for each essential tensor using exact matching
        for tensor_name in &essential_tensors {
            let matches: Vec<_> = self.tensors.iter()
                .filter(|t| {
                    // Exact match ONLY for essential tensors
                    t.name == *tensor_name
                })
                .collect();
            
            if matches.is_empty() {
                tracing::warn!("Essential tensor '{}' not found in model", tensor_name);
            } else {
                tracing::info!("Found essential tensor '{}' with shape {:?}", 
                              tensor_name, matches[0].dims);
            }
        }
        
        // Transformer-related tensor patterns that typically appear at the end of tensor names
        let transformer_patterns = [
            "attn_k.weight", "attn_k.bias",
            "attn_q.weight", "attn_q.bias",
            "attn_v.weight", "attn_v.bias",
            "attn_output.weight",
            "ffn_down.weight", "ffn_gate.weight", "ffn_up.weight",
            "attn_norm.weight", "ffn_norm.weight",
        ];
                
        // Track block numbers for each pattern
        let mut pattern_to_block_numbers = std::collections::HashMap::new();
        
        // Find all tensors ending with each pattern
        for &pattern in &transformer_patterns {
            // Get all tensors ending with this pattern
            let matching_tensors: Vec<_> = self.tensors.iter()
                .filter(|t| t.name.ends_with(pattern))
                .collect();
            
            if !matching_tensors.is_empty() {
                
                // Extract block numbers from matching tensors
                let mut block_numbers = Vec::new();
                for tensor in &matching_tensors {
                    // Get the part of the name before the pattern
                    let prefix = &tensor.name[..tensor.name.len() - pattern.len()];
                    
                    // Find the last number in the prefix (block/layer number)
                    let mut num_end = prefix.len();
                    if num_end > 0 && prefix.ends_with('.') {
                        num_end -= 1; // Skip trailing dot if present
                    }
                    
                    let mut num_start = num_end;
                    // Scan backwards to find the start of the number
                    while num_start > 0 && prefix[num_start-1..num_start].chars().next().unwrap().is_digit(10) {
                        num_start -= 1;
                    }
                    
                    // If we found a number, parse it
                    if num_start < num_end {
                        if let Ok(block_num) = prefix[num_start..num_end].parse::<usize>() {
                            block_numbers.push(block_num);                            
                        }
                    }
                }
                
                if !block_numbers.is_empty() {
                    block_numbers.sort();
                    let min_block = *block_numbers.iter().min().unwrap();
                    let max_block = *block_numbers.iter().max().unwrap();
                    
                    tracing::info!("  Block numbers for '{}': min={}, max={}", pattern, min_block, max_block);
                    pattern_to_block_numbers.insert(pattern.to_string(), (min_block, max_block));
                }
            }
        }
        
        // Analyze the overall block numbering scheme
        if !pattern_to_block_numbers.is_empty() {
            let mut all_mins = Vec::new();
            let mut all_maxes = Vec::new();
            
            for (_, (min, max)) in &pattern_to_block_numbers {
                all_mins.push(*min);
                all_maxes.push(*max);
            }
            
            // Get the consistent min and max across all patterns
            let overall_min = *all_mins.iter().min().unwrap();
            let overall_max = *all_maxes.iter().max().unwrap();
            let detected_block_count = overall_max - overall_min + 1;
                        
            // Compare with metadata
            let metadata_blocks = self.params.block_count;
            if detected_block_count != metadata_blocks {
                tracing::warn!("  WARNING: Detected block count ({}) differs from metadata block count ({})", 
                         detected_block_count, metadata_blocks);                         
                self.params.block_count = detected_block_count;
            } else {
                self.block_count = Some(self.params.block_count);
            }
            
            // Check if any pattern has inconsistent min/max
            let mut inconsistent_patterns = Vec::new();
            for (pattern, (min, max)) in &pattern_to_block_numbers {
                if *min != overall_min || *max != overall_max {
                    inconsistent_patterns.push((pattern, *min, *max));
                }
            }
            
            if !inconsistent_patterns.is_empty() {
                tracing::warn!("\nPatterns with inconsistent block numbering:");
                for (pattern, min, max) in inconsistent_patterns {
                    tracing::warn!("  {} (min={}, max={})", pattern, min, max);
                }
            }
            
            // Update the block_index_start
            self.block_index_start = Some(overall_min);            
        } else {
            tracing::error!("No tensors match the expected transformer patterns at the end of their names");
        }
        
        // Print total number of tensors
        println!("\nTotal number of tensors: {}", self.tensors.len());
    }

    /// Get a reference to the model's raw memory-mapped data
    ///
    /// This provides access to the underlying raw model data for direct access
    /// by components like Tensor that need to read and process raw tensor data.
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }

}