use std::error::Error;
use std::sync::Arc;
use crate::llm::model::Model;
use crate::llm::tokenizer::Tokenizer;
use crate::config::Settings;

/// Context for running inference with the model
pub struct InferenceContext {
    /// The loaded model
    model: Arc<Model>,
    /// The tokenizer for this model
    tokenizer: Tokenizer,
    /// Current context window
    context: Vec<u32>,
    /// Maximum context size
    max_context_size: usize,
    /// Temperature for sampling (0.0-1.0)
    temperature: f32,
    /// Maximum number of tokens to generate
    max_tokens: usize,
}

impl InferenceContext {
    pub fn new(model: Arc<Model>, settings: &Settings) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Create tokenizer using model metadata
        let tokenizer = Tokenizer::new(model.architecture.clone(), &model.metadata)?;
        
        Ok(Self {
            model,
            tokenizer,
            context: Vec::new(),
            max_context_size: settings.inference.context_size,
            temperature: settings.inference.temperature,
            max_tokens: settings.inference.max_tokens,
        })
    }

    /// Processes input text and generates a response
    pub fn process_input(&mut self, input: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        // Step 1: Tokenize input
        let input_tokens = self.tokenizer.encode(input)?;
        
        // Step 2: Update context window
        self.update_context(&input_tokens)?;
        
        // Step 3: Generate response tokens
        let response_tokens = self.generate_tokens()?;

        // Step 4: Decode response
        let response = self.tokenizer.decode(&response_tokens)?;

        
        Ok(response)
    }


    fn update_context(&mut self, new_tokens: &[u32]) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Add new tokens to context
        self.context.extend(new_tokens.iter().cloned());
        
        // Trim context if it exceeds max size
        if self.context.len() > self.max_context_size {
            let excess = self.context.len() - self.max_context_size;
            self.context.drain(..excess);
        }
        
        Ok(())
    }

    /// Generates new tokens based on the current context
    fn generate_tokens(&mut self) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        let mut generated_tokens = Vec::new();
        let mut current_context = self.context.clone();
        
        // Get EOS token ID from tokenizer metadata
        let eos_token_id = self.tokenizer.get_eos_token_id();
        
        // Generate tokens until we hit max_tokens or EOS
        while generated_tokens.len() < self.max_tokens {
            // Get next token prediction
            let next_token = self.predict_next_token(&current_context)?;
            
            // Add to generated tokens
            generated_tokens.push(next_token);
            
            // Update context for next prediction
            current_context.push(next_token);
            
            // Check for EOS token
            if next_token == eos_token_id {
                break;
            }
        }
        
        Ok(generated_tokens)
    }

    /// Predicts the next token given the current context
    fn predict_next_token(&self, context: &[u32]) -> Result<u32, Box<dyn Error + Send + Sync>> {
        let next_token = 111;
        
        Ok(next_token)
    }

} 