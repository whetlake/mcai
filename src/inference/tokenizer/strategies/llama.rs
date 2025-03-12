use std::error::Error;
use std::collections::BTreeMap;
use crate::gguf::GGUFValue;
use crate::inference::tokenizer::TokenizerStrategy;

/// Configuration specific to LLaMA tokenizer
#[derive(Debug, Clone)]
pub struct LLaMATokenizerConfig {
    /// Whether to add BOS token (typically true for LLaMA)
    pub add_bos_token: bool,
    /// Whether to add EOS token (typically true for LLaMA)
    pub add_eos_token: bool,
    /// BOS token ID (typically 1)
    pub bos_token_id: u32,
    /// EOS token ID (typically 2)
    pub eos_token_id: u32,
    /// Padding token ID (typically 0)
    pub padding_token_id: u32,
}

pub struct LLaMATokenizer {
    vocabulary: BTreeMap<String, u32>,
    reverse_vocabulary: BTreeMap<u32, String>,
    config: LLaMATokenizerConfig,
}

impl LLaMATokenizer {
    pub fn new(metadata: &BTreeMap<String, (String, GGUFValue)>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Parse configuration from metadata
        let mut config = LLaMATokenizerConfig {
            add_bos_token: true, // LLaMA models typically use BOS
            add_eos_token: true, // LLaMA models typically use EOS
            bos_token_id: 1,
            eos_token_id: 2,
            padding_token_id: 0,
        };

        // Load configuration from metadata
        if let Some((_, value)) = metadata.get("tokenizer.ggml.add_bos_token") {
            config.add_bos_token = value.to_string().parse().unwrap_or(true);
        }
        if let Some((_, value)) = metadata.get("tokenizer.ggml.add_eos_token") {
            config.add_eos_token = value.to_string().parse().unwrap_or(true);
        }
        if let Some((_, value)) = metadata.get("tokenizer.ggml.bos_token_id") {
            config.bos_token_id = value.to_string().parse().unwrap_or(1);
        }
        if let Some((_, value)) = metadata.get("tokenizer.ggml.eos_token_id") {
            config.eos_token_id = value.to_string().parse().unwrap_or(2);
        }

        // Load vocabulary
        let mut vocabulary = BTreeMap::new();
        let mut reverse_vocabulary = BTreeMap::new();
        if let Some((_, value)) = metadata.get("tokenizer.ggml.tokens") {
            if let Ok(tokens) = serde_json::from_str::<Vec<String>>(&value.to_string()) {
                for (i, token) in tokens.into_iter().enumerate() {
                    vocabulary.insert(token.clone(), i as u32);
                    reverse_vocabulary.insert(i as u32, token);
                }
            }
        } else {
            return Err("LLaMA tokenizer requires vocabulary in metadata".into());
        }

        Ok(Self {
            vocabulary,
            reverse_vocabulary,
            config,
        })
    }
}

impl TokenizerStrategy for LLaMATokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        let mut tokens = Vec::new();
        
        // Add BOS token if configured
        if self.config.add_bos_token {
            tokens.push(self.config.bos_token_id);
        }
        
        // TODO: Implement proper SentencePiece tokenization for LLaMA
        // For now, just split on spaces and lookup in vocabulary
        for word in text.split_whitespace() {
            if let Some(&token_id) = self.vocabulary.get(word) {
                tokens.push(token_id);
            } else {
                // Handle unknown tokens by splitting into characters
                for c in word.chars() {
                    if let Some(&token_id) = self.vocabulary.get(&c.to_string()) {
                        tokens.push(token_id);
                    } else {
                        tokens.push(self.config.padding_token_id); // Use padding token for unknown chars
                    }
                }
            }
        }
        
        // Add EOS token if configured
        if self.config.add_eos_token {
            tokens.push(self.config.eos_token_id);
        }
        
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut text = String::new();
        
        for &token_id in tokens {
            // Skip special tokens
            if token_id == self.config.bos_token_id || token_id == self.config.eos_token_id {
                continue;
            }
            
            if let Some(token_text) = self.reverse_vocabulary.get(&token_id) {
                if !text.is_empty() && !text.ends_with(' ') {
                    text.push(' ');
                }
                text.push_str(token_text);
            }
        }
        
        Ok(text)
    }
} 