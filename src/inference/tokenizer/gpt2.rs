use std::error::Error;
use std::collections::BTreeMap;
use crate::gguf::GGUFValue;
use super::{TokenizerStrategy, TokenizerConfig};

pub struct GPT2Tokenizer {
    vocabulary: BTreeMap<String, u32>,
    reverse_vocabulary: BTreeMap<u32, String>,
    merges: Vec<(String, String)>,
    config: TokenizerConfig,
}

impl GPT2Tokenizer {
    pub fn new(metadata: &BTreeMap<String, (String, GGUFValue)>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Parse configuration from metadata
        let mut config = TokenizerConfig {
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
            let eos_id = value.to_string().parse().unwrap_or(2);
            config.eos_token_id = eos_id;
            config.padding_token_id = eos_id; // For GPT-2, padding token is same as EOS
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
            return Err("GPT-2 tokenizer requires vocabulary in metadata".into());
        }

        // Load BPE merges
        let mut merges = Vec::new();
        if let Some((_, value)) = metadata.get("tokenizer.ggml.merges") {
            if let Ok(merge_list) = serde_json::from_str::<Vec<String>>(&value.to_string()) {
                for merge in merge_list {
                    if let Some((first, second)) = merge.split_once(' ') {
                        merges.push((first.to_string(), second.to_string()));
                    }
                }
            }
        } else {
            return Err("GPT-2 tokenizer requires BPE merges in metadata".into());
        }

        Ok(Self {
            vocabulary,
            reverse_vocabulary,
            merges,
            config,
        })
    }

    /// Preprocesses text according to GPT-2 rules
    fn preprocess_text(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current_word = String::new();

        for c in text.chars() {
            match c {
                ' ' | '\t' | '\n' | '\r' => {
                    if !current_word.is_empty() {
                        result.push(current_word);
                        current_word = String::new();
                    }
                },
                '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' => {
                    if !current_word.is_empty() {
                        result.push(current_word);
                        current_word = String::new();
                    }
                    result.push(c.to_string());
                },
                _ => {
                    current_word.push(c);
                }
            }
        }

        if !current_word.is_empty() {
            result.push(current_word);
        }

        result
    }

    /// Applies BPE encoding to a single word
    fn bpe_encode(&self, word: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        let mut word = word.to_string();
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        
        // Apply merges until no more can be applied
        loop {
            let mut best_pair = None;
            let mut best_idx = None;
            
            // Find the first merge rule that can be applied
            for (i, pair) in parts.windows(2).enumerate() {
                let pair_str = format!("{}{}", pair[0], pair[1]);
                if self.vocabulary.contains_key(&pair_str) {
                    best_pair = Some(pair_str);
                    best_idx = Some(i);
                    break;
                }
            }
            
            // If no merge rule can be applied, we're done
            if best_pair.is_none() {
                break;
            }
            
            // Apply the merge
            if let (Some(pair), Some(idx)) = (best_pair, best_idx) {
                parts[idx] = pair;
                parts.remove(idx + 1);
            }
        }
        
        // Convert parts to token IDs
        let mut tokens = Vec::new();
        for part in parts {
            if let Some(&token_id) = self.vocabulary.get(&part) {
                tokens.push(token_id);
            } else {
                // Handle unknown tokens
                tokens.push(self.config.padding_token_id);
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
        
        // Tokenize the input text
        let words = self.preprocess_text(text);
        for word in words {
            let word_tokens = self.bpe_encode(&word)?;
            tokens.extend(word_tokens);
        }
        
        // Add EOS token if configured
        if self.config.add_eos_token {
            tokens.push(self.config.eos_token_id);
        }
        
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
} 