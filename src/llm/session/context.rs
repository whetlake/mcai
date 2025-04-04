use std::sync::Arc;
use std::error::Error;
use llama_cpp::{LlamaModel, LlamaSession, SessionParams, Token};
use llama_cpp::standard_sampler::StandardSampler;
use crate::config::Settings;
use std::sync::Mutex;

/// Context for running inference with the model, managing the llama_cpp session.
pub struct InferenceContext {
    llama_model: Arc<LlamaModel>,
    session: Mutex<LlamaSession>,
    /// Maximum context size
    max_context_size: usize,
    /// Temperature for sampling (0.0-1.0) - currently unused, sampler takes defaults
    temperature: f32,
    /// Maximum number of tokens to generate
    max_tokens: usize,
}

impl InferenceContext {
    pub fn new(llama_model_arc: Arc<LlamaModel>, settings: &Settings) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Get requested context size from settings, or use model's default
        // Note: llama_cpp session params might have its own way to determine max context
        let n_ctx = settings.inference.context_size as u32; // Convert to u32 for SessionParams
        let session_params = SessionParams {
            n_ctx: n_ctx, // Pass u32 directly
            n_batch: 512, // Default batch size, can be configured
            // Set other SessionParams fields as needed (e.g., seeds, threads)
            ..Default::default() // Use defaults for other params
        };

        // Create the LlamaSession
        tracing::info!("Creating LlamaSession with context size: {}", n_ctx);
        let session = llama_model_arc.create_session(session_params)
            .map_err(|e| format!("Failed to create LlamaSession: {}", e))?;
        tracing::info!("LlamaSession created successfully.");

        // Store n_ctx for reference (max_context_size might differ from session's internal limit)
        let max_context_size = n_ctx as usize;

        Ok(Self {
            llama_model: llama_model_arc,
            session: Mutex::new(session),
            max_context_size,
            temperature: settings.inference.temperature,
            max_tokens: settings.inference.max_tokens,
        })
    }

    /// Processes input text and generates a response
    /// Uses the internal LlamaSession to handle tokenization, inference, and decoding.
    pub fn process_input(&mut self, input: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        tracing::info!("Processing input: \"{}\"", input);

        // Lock the session mutex to get mutable access
        let mut session_guard = self.session.lock()
            .map_err(|e| format!("Failed to lock session mutex: {}", e))?;

        // Feed the prompt - Use the guard to call the method
        session_guard.advance_context(input)
            .map_err(|e| format!("Failed to advance context: {}", e))?;
        tracing::info!("Context advanced with input prompt.");

        let max_tokens = self.max_tokens; // Use configured max tokens
        // Start completion generation - Use the guard
        let sampler = StandardSampler::default(); // Can configure sampler here
        let completions_handle = session_guard.start_completing_with(sampler, max_tokens)?; // Handle potential error first
        tracing::info!("Started completion generation for max {} tokens.", max_tokens);

        // Step 3: Collect generated tokens (ignoring potential errors and decoding for now)
        let mut generated_tokens: Vec<u32> = Vec::new(); // Store token IDs
        let mut generated_count = 0;

        // Iterate directly on the handle, assuming it yields Token directly
        // (Error handling for the stream might be implicit or require different handling)
        for token in completions_handle {
            // TODO: Add error handling for the stream if necessary
            generated_tokens.push(token.0 as u32); // Access tuple struct field 0 for the ID, cast to u32
            generated_count += 1;
            if generated_count >= max_tokens {
                tracing::warn!("Reached max token limit ({}) during generation.", max_tokens);
                break;
            }
        }

        println!(); // Add a newline after generation finishes
        tracing::info!("Completion finished. Collected {} tokens (IDs: {:?}).", generated_count, generated_tokens);

        // Step 4: Decode the generated tokens into a response string
        let mut response_string = String::new();
        for token_id in generated_tokens {
            // Convert u32 ID back to Token tuple struct (i32,)
            let token = Token(token_id as i32);
            // Use the llama_model to decode the Token struct
            // Assuming token_to_piece takes Token and returns Vec<u8> directly (or similar)
            // Error handling needs verification based on actual API
            let bytes = self.llama_model.token_to_piece(token); // Assuming direct Vec<u8> return
            // TODO: Verify error handling for token_to_piece if it can fail

            // Assuming token_to_piece returns String directly
            response_string.push_str(&bytes); // Append the resulting string piece
        }
        Ok(response_string) // Return the fully decoded string
    }
} 