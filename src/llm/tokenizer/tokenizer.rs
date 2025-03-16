use std::error::Error;
use std::collections::BTreeMap;
use crate::gguf::GGUFValue;
use super::types::{TokenizerType, determine_tokenizer_type};
use super::strategy::TokenizerStrategy;
use super::strategies::GPT2Tokenizer;

/// A tokenizer that converts text to tokens and back
pub struct Tokenizer {
    strategy: Box<dyn TokenizerStrategy>,
}

impl Tokenizer {
    /// Creates a new tokenizer for the given model architecture and metadata
    pub fn new(architecture: String, metadata: &BTreeMap<String, (String, GGUFValue)>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let tokenizer_type = determine_tokenizer_type(&architecture, metadata);
        
        // Create the appropriate strategy based on type
        let strategy: Box<dyn TokenizerStrategy> = match tokenizer_type {
            TokenizerType::GPT2 => Box::new(GPT2Tokenizer::new(metadata)?),
            _ => return Err(format!("Unsupported tokenizer type: {:?}", tokenizer_type).into())
        };

        Ok(Self {
            strategy,
        })
    }

    // Delegate encode/decode to the strategy
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        self.strategy.encode(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>> {
        self.strategy.decode(tokens)
    }

    /// Get the EOS (End of Sequence) token ID
    pub fn get_eos_token_id(&self) -> u32 {
        self.strategy.get_eos_token_id()
    }
}