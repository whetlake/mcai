use std::error::Error;
use std::collections::HashMap;

/// Represents a token in the vocabulary
#[derive(Debug, Clone)]
pub struct Token {
    /// The text representation of the token
    pub text: String,
    /// The unique ID of the token in the vocabulary
    pub id: u32,
}

/// A tokenizer that converts text to tokens and back
pub struct Tokenizer {
    /// Mapping from text to token ID
    vocabulary: HashMap<String, u32>,
    /// Mapping from token ID to text
    reverse_vocabulary: HashMap<u32, String>,
    /// Special tokens for the model
    special_tokens: HashMap<String, u32>,
    /// The model architecture this tokenizer is for
    architecture: String,
}

impl Tokenizer {
    /// Creates a new tokenizer for the given model architecture
    pub fn new(architecture: String) -> Self {
        let mut special_tokens = HashMap::new();
        
        // Set special tokens based on architecture
        match architecture.as_str() {
            "LLaMA" | "Mistral" => {
                special_tokens.insert("<s>".to_string(), 1);  // Start of sequence
                special_tokens.insert("</s>".to_string(), 2); // End of sequence
                special_tokens.insert("<unk>".to_string(), 0); // Unknown token
            },
            _ => {
                // Default special tokens
                special_tokens.insert("<unk>".to_string(), 0);
                special_tokens.insert("<s>".to_string(), 1);
                special_tokens.insert("</s>".to_string(), 2);
            }
        }
        
        Self {
            vocabulary: HashMap::new(),
            reverse_vocabulary: HashMap::new(),
            special_tokens,
            architecture,
        }
    }

    /// Loads the vocabulary from a file
    pub fn load_vocab(&mut self, vocab_file: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        // TODO: Implement vocabulary loading from file
        // For now, just add some dummy tokens
        let dummy_tokens = vec![
            ("the", 3),
            ("a", 4),
            ("an", 5),
            ("and", 6),
            ("or", 7),
            ("but", 8),
            ("in", 9),
            ("on", 10),
            ("at", 11),
            ("to", 12),
        ];
        
        for (text, id) in dummy_tokens {
            self.vocabulary.insert(text.to_string(), id);
            self.reverse_vocabulary.insert(id, text.to_string());
        }
        
        Ok(())
    }

    /// Converts text to tokens
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        let mut tokens = Vec::new();
        
        // Add start token
        if let Some(&start_id) = self.special_tokens.get("<s>") {
            tokens.push(start_id);
        }
        
        // Simple whitespace-based tokenization for now
        for word in text.split_whitespace() {
            if let Some(&token_id) = self.vocabulary.get(word) {
                tokens.push(token_id);
            } else if let Some(&unk_id) = self.special_tokens.get("<unk>") {
                tokens.push(unk_id);
            }
        }
        
        // Add end token
        if let Some(&end_id) = self.special_tokens.get("</s>") {
            tokens.push(end_id);
        }
        
        Ok(tokens)
    }

    /// Converts tokens back to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut words = Vec::new();
        
        for &token_id in tokens {
            if let Some(text) = self.reverse_vocabulary.get(&token_id) {
                words.push(text.clone());
            } else if let Some((_, text)) = self.special_tokens.iter()
                .find(|(_, &id)| id == token_id) {
                // Skip special tokens in output
                continue;
            } else {
                return Err("Unknown token ID".into());
            }
        }
        
        Ok(words.join(" "))
    }
} 