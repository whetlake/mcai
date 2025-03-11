use std::error::Error;
use std::collections::BTreeMap;
use crate::gguf::GGUFValue;
use crate::inference::tokenizer_types::{TokenizerType, determine_tokenizer_type};

/// Represents a token in the vocabulary
#[derive(Debug, Clone)]
pub struct Token {
    /// The text representation of the token
    pub text: String,
    /// The unique ID of the token in the vocabulary
    pub id: u32,
}

/// Configuration for tokenizer behavior
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Whether to add BOS token
    pub add_bos_token: bool,
    /// Whether to add EOS token
    pub add_eos_token: bool,
    /// BOS token ID
    pub bos_token_id: u32,
    /// EOS token ID
    pub eos_token_id: u32,
    /// Padding token ID
    pub padding_token_id: u32,
}

/// A tokenizer that converts text to tokens and back
pub struct Tokenizer {
    strategy: Box<dyn TokenizerStrategy>,
    tokenizer_type: TokenizerType,
    architecture: String,
}

pub trait TokenizerStrategy {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>>;
    fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>>;
}

pub struct GPT2Tokenizer {
    vocabulary: BTreeMap<String, u32>,
    reverse_vocabulary: BTreeMap<u32, String>,
    merges: Vec<(String, String)>,
    config: TokenizerConfig,
}

pub struct LLaMATokenizer {
    vocabulary: BTreeMap<String, u32>,
    reverse_vocabulary: BTreeMap<u32, String>,
    config: TokenizerConfig,
}

impl Tokenizer {
    /// Creates a new tokenizer for the given model architecture and metadata
    pub fn new(architecture: String, metadata: &BTreeMap<String, (String, GGUFValue)>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let tokenizer_type = determine_tokenizer_type(&architecture, metadata);
        
        // Create the appropriate strategy based on type
        let strategy: Box<dyn TokenizerStrategy> = match tokenizer_type {
            TokenizerType::GPT2 => Box::new(GPT2Tokenizer::new(metadata)?),
            TokenizerType::LLaMA => Box::new(LLaMATokenizer::new(metadata)?),
            TokenizerType::Mistral => Box::new(MistralTokenizer::new(metadata)?),
            TokenizerType::BERT => Box::new(BERTTokenizer::new(metadata)?),
            TokenizerType::Generic => Box::new(GenericTokenizer::new(metadata)?),
        };

        Ok(Self {
            strategy,
            tokenizer_type,
            architecture,
        })
    }

    // Delegate encode/decode to the strategy
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        self.strategy.encode(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>> {
        self.strategy.decode(tokens)
    }
}