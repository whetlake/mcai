use std::error::Error;
use crate::inference::model::Model;
use crate::gguf::TensorInfo;
use crate::inference::tokenizer::Tokenizer;
use crate::config::Settings;

/// Context for running inference with the model
pub struct InferenceContext {
    /// The loaded model
    model: Model,
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
    pub fn new(model: Model, settings: &Settings) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Get model metadata and create tokenizer
        let metadata = &model.gguf_reader().metadata;
        let tokenizer = Tokenizer::new(model.architecture.clone(), metadata)?;
        
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
        
        // Generate tokens until we hit max_tokens or EOS
        while generated_tokens.len() < self.max_tokens {
            // Get next token prediction
            let next_token = self.predict_next_token(&current_context)?;
            
            // Add to generated tokens
            generated_tokens.push(next_token);
            
            // Update context for next prediction
            current_context.push(next_token);
            
            // Check for EOS token (assuming it's 2)
            if next_token == 2 {
                break;
            }
        }
        
        Ok(generated_tokens)
    }

    /// Predicts the next token given the current context
    fn predict_next_token(&self, context: &[u32]) -> Result<u32, Box<dyn Error + Send + Sync>> {
        // Get the model's output logits
        let logits = self.get_logits(context)?;
        
        // Apply temperature sampling
        let next_token = self.sample_from_logits(&logits)?;
        
        Ok(next_token)
    }

    /// Gets the model's output logits for the current context
    fn get_logits(&self, context: &[u32]) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        // TODO: Implement actual model inference
        // For now, return dummy logits
        Ok(vec![0.0; 32000]) // Assuming vocab size of 32000
    }

    /// Samples a token from the logits using temperature
    fn sample_from_logits(&self, logits: &[f32]) -> Result<u32, Box<dyn Error + Send + Sync>> {
        // TODO: Implement proper sampling
        // For now, just return a dummy token
        Ok(1)
    }

    /// Gets the current context size
    pub fn context_size(&self) -> usize {
        self.context.len()
    }

    /// Clears the current context
    pub fn clear_context(&mut self) {
        self.context.clear();
    }
    
    /// Gets a reference to the model
    pub fn model(&self) -> &Model {
        &self.model
    }
    
    /// Gets a tensor by name from the model
    pub fn get_tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.model.get_tensor_by_name(name)
    }

    /// Sets the temperature for sampling
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.max(0.0).min(1.0);
    }

    /// Sets the maximum number of tokens to generate
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }
} 