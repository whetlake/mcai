use std::error::Error;

/// Trait defining the interface for all tokenizer implementations
pub trait TokenizerStrategy: Send + Sync {
    /// Convert text into token IDs
    fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>>;
    
    /// Convert token IDs back into text
    fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>>;

    /// Get the EOS (End of Sequence) token ID
    fn get_eos_token_id(&self) -> u32;
} 