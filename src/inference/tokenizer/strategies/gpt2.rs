/// Useful for understanding the tokenizer:
/// https://huggingface.co/hkeshhk/bpetokenizer/blob/main/tokenizer/bpetokenizer/tokenizer.py

use std::error::Error;
use std::collections::BTreeMap;
use crate::gguf::GGUFValue;
use crate::inference::tokenizer::TokenizerStrategy;
use crate::inference::tokenizer::utilities::{PATTERN, BYTES_TO_UNICODE};


/// Configuration specific to GPT2 tokenizer
#[derive(Debug, Clone)]
pub struct GPT2TokenizerConfig {
    /// Whether to add BOS token
    pub add_bos_token: bool,
    /// Whether to add EOS token
    pub add_eos_token: bool,
    /// BOS token ID
    pub bos_token_id: u32,
    /// EOS token ID
    pub eos_token_id: u32,
    /// Padding token ID (same as EOS in GPT2)
    pub padding_token_id: u32,
}

pub struct GPT2Tokenizer {
    vocabulary: BTreeMap<String, u32>,
    reverse_vocabulary: BTreeMap<u32, String>,
    merges: Vec<String>,
    config: GPT2TokenizerConfig,
}

impl GPT2Tokenizer {
    pub fn new(metadata: &BTreeMap<String, (String, GGUFValue)>) -> Result<Self, Box<dyn Error + Send + Sync>> {

        // Parse configuration from metadata
        let mut config = GPT2TokenizerConfig {
            add_bos_token: false,
            add_eos_token: false,
            bos_token_id: 0,
            eos_token_id: 0,
            padding_token_id: 0,
        };

        // Load configuration from metadata
        if let Some((_, value)) = metadata.get("tokenizer.ggml.add_bos_token") {
            config.add_bos_token = value.to_string().parse().unwrap_or(false);
        }
        if let Some((_, value)) = metadata.get("tokenizer.ggml.add_eos_token") {
            config.add_eos_token = value.to_string().parse().unwrap_or(false);
        }
        if let Some((_, value)) = metadata.get("tokenizer.ggml.bos_token_id") {
            config.bos_token_id = value.to_string().parse().unwrap_or(1);
        }
        if let Some((_, value)) = metadata.get("tokenizer.ggml.eos_token_id") {
            config.eos_token_id = value.to_string().parse().unwrap_or(2);
        }
        if let Some((_, value)) = metadata.get("tokenizer.ggml.padding_token_id") {
            config.padding_token_id = value.to_string().parse().unwrap_or(2);
        }

        // Load vocabulary
        let mut vocabulary = BTreeMap::new();
        let mut reverse_vocabulary = BTreeMap::new();
        if let Some((_, value)) = metadata.get("tokenizer.ggml.tokens") {            
            // Get the array from metadata
            let tokens: Vec<String> = match value {
                GGUFValue::Array(arr) => {
                    arr.iter().map(|v| match v {
                        GGUFValue::String(s) => Ok(s.clone()),
                        _ => Err(format!("Invalid token type: {:?}", v))
                    }).collect::<Result<Vec<String>, String>>()?
                },
                _ => {
                    return Err("Tokenizer tokens must be an array".into());
                }
            };
                                    
            // Create vocabulary mappings
            for (i, token) in tokens.into_iter().enumerate() {
                vocabulary.insert(token.clone(), i as u32);
                reverse_vocabulary.insert(i as u32, token);
            }
        } else {
            return Err("GPT-2 tokenizer requires vocabulary in metadata".into());
        }

        // Load BPE merges
        let merges = if let Some((_, value)) = metadata.get("tokenizer.ggml.merges") {
            // Get the array from metadata
            if let GGUFValue::Array(arr) = value {
                arr.iter().map(|v| match v {
                    GGUFValue::String(s) => s.clone(),
                    _ => v.to_string()
                }).collect()
            } else {
                return Err("Tokenizer merges must be an array".into());
            }
        } else {
            return Err("GPT-2 tokenizer requires BPE merges in metadata".into());
        };

        Ok(Self {
            vocabulary,
            reverse_vocabulary,
            merges,
            config,
        })
    }

    /// Preprocesses text according to GPT-2 rules
    fn preprocess_text(&self, text: &str) -> Vec<String> {
        println!("Preprocessing text: {:?}", text);
        // First split using the regex pattern
        let parts: Vec<String> = PATTERN.find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect();
        println!("Parts after regex split: {:?}", parts);

        // Then convert each part's bytes to unicode
        let unicode_parts: Vec<String> = parts.iter()
            .map(|part| {
                let bytes = part.as_bytes();
                bytes.iter()
                    .map(|&b| BYTES_TO_UNICODE.get(&b).unwrap_or(&char::REPLACEMENT_CHARACTER))
                    .collect()
            })
            .collect();
        println!("Parts after unicode conversion: {:?}", unicode_parts);
        
        unicode_parts
    }

    /// Applies BPE encoding to a piece of text
    fn bpe_encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        // Start with individual characters
        let mut parts: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        
        // Keep merging until no more merges can be applied
        while parts.len() > 1 {
            let mut min_idx = None;
            let mut min_rank = None;
            
            // Look at each adjacent pair from left to right
            for i in 0..parts.len() - 1 {
                let pair = format!("{} {}", parts[i], parts[i + 1]);
                
                // Check if this pair exists in our merges list
                let rank = self.merges.iter().position(|merge| merge == &pair);
                    
                // If we found this pair and its rank is lower than our current minimum
                if let Some(r) = rank {
                    if min_rank.is_none() || r < min_rank.unwrap() {
                        min_idx = Some(i);
                        min_rank = Some(r);
                    }
                }
            }
            
            // If we found no pairs to merge, we're done
            if min_idx.is_none() {
                break;
            }
            
            // Apply the merge with lowest rank
            let i = min_idx.unwrap();
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }
        
        // Convert final parts to token IDs
        let mut tokens = Vec::new();
        for part in &parts {
            if let Some(&token_id) = self.vocabulary.get(part) {
                tokens.push(token_id);
            } else {
                return Err("Token not found in vocabulary".into());
            }
        }
        
        Ok(tokens)
    }
}

impl TokenizerStrategy for GPT2Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        let mut tokens = Vec::new();
        
        // Add BOS token if configured
        if self.config.add_bos_token {
            tokens.push(self.config.bos_token_id);
        }
        
        // Split text into initial tokens using regex
        let parts = self.preprocess_text(text);
        
        // Process each part
        for part in parts {
            let part_tokens = self.bpe_encode(&part)?;
            tokens.extend(part_tokens);
        }
        
        // Add EOS token if configured
        if self.config.add_eos_token {
            tokens.push(self.config.eos_token_id);
        }

        println!("Tokens: {:?}", tokens);

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut text = String::new();
        let mut skip_next_space = false;
        
        for (i, &token_id) in tokens.iter().enumerate() {
            // Skip special tokens
            if token_id == self.config.bos_token_id || token_id == self.config.eos_token_id {
                continue;
            }
            
            if let Some(token_text) = self.reverse_vocabulary.get(&token_id) {
                // Handle spacing
                if i > 0 && !skip_next_space && !text.ends_with(' ') {
                    text.push(' ');
                }
                
                text.push_str(token_text);
                
                // Check if we should skip the next space
                skip_next_space = token_text.ends_with(|c: char| c.is_ascii_punctuation());
            }
        }
        
        Ok(text)
    }

    fn get_eos_token_id(&self) -> u32 {
        self.config.eos_token_id
    }
} 