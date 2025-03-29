use std::sync::Arc;
use std::error::Error;
use crate::llm::model::Model;
use crate::llm::tokenizer::Tokenizer;
use crate::config::Settings;
use crate::llm::inference::ForwardPass;
use crate::llm::backend;

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
    /// Forward pass for token prediction
    forward_pass: ForwardPass,
}

impl InferenceContext {
    pub fn new(model: Arc<Model>, settings: &Settings) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Create the tokenizer
        let tokenizer = Tokenizer::new(model.architecture.clone(), &model.metadata)?;
        
        // Get requested context size from settings, or use model's default
        let requested_context_size = settings.inference.context_size;
        
        // Make sure we don't exceed the model's maximum context length
        let max_context_size = if requested_context_size > model.params.model_context_length {
            println!("WARNING: Requested context size {} exceeds model's maximum context length {}, using model's maximum",
                requested_context_size, model.params.model_context_length);
            model.params.model_context_length
        } else {
            requested_context_size
        };
        
        // Create a backend for tensor operations
        let backend = backend::create_backend();
        
        // Create forward pass with the adjusted context length
        let forward_pass: ForwardPass = ForwardPass::new(Arc::clone(&model), backend, Some(max_context_size));
        
        Ok(Self {
            model,
            tokenizer,
            context: Vec::new(),
            max_context_size,
            temperature: settings.inference.temperature,
            max_tokens: settings.inference.max_tokens,
            forward_pass,
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

        println!("Input tokens: {:?}", input_tokens);
        println!("Response output: {}", response);

        
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
        let max_predictions = 10;  // Limit to 10 predictions for testing
        eprintln!("\n=== Starting Token Generation ===");
        eprintln!("Initial context: {:?}", current_context);
        
        while generated_tokens.len() < max_predictions {
            eprintln!("\nGenerating token {}/{}", generated_tokens.len() + 1, max_predictions);
            
            // Get next token prediction
            let next_token = self.predict_next_token(&current_context)?;
            eprintln!("Predicted token: {}", next_token);
            
            // Add to generated tokens
            generated_tokens.push(next_token);
            
            // Update context for next prediction
            current_context.push(next_token);
            
            // Check for EOS token
            if next_token == eos_token_id {
                eprintln!("EOS token detected, stopping generation");
                break;
            }
        }
        
        eprintln!("\n=== Token Generation Complete ===");
        eprintln!("Generated {} tokens", generated_tokens.len());
        eprintln!("Final context: {:?}", current_context);
        
        Ok(generated_tokens)
    }

    /// Predicts the next token given the current context
    fn predict_next_token(&mut self, context: &[u32]) -> Result<u32, Box<dyn Error + Send + Sync>> {
        // Use the ForwardPass's predict_next_token method directly
        self.forward_pass.predict_next_token(context)
    }

} 