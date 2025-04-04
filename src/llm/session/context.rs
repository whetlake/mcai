use std::sync::Arc;
use std::error::Error;
use llama_cpp::{LlamaModel, LlamaSession, SessionParams, TokensToStrings};
use llama_cpp::standard_sampler::StandardSampler;
use crate::config::Settings;
use tokio::sync::Mutex;
use futures::stream::{Stream, StreamExt};
use async_stream::stream;
use std::pin::Pin;

/// Context for running inference with the model, managing the llama_cpp session.
#[derive(Clone)]
pub struct InferenceContext {
    active_model: Arc<LlamaModel>,
    session: Arc<Mutex<LlamaSession>>,
    /// Maximum context size
    max_context_size: usize,
    /// Temperature for sampling (0.0-1.0) - currently unused, sampler takes defaults
    temperature: f32,
    /// Maximum number of tokens to generate
    max_tokens: usize,
}

impl InferenceContext {
    pub fn new(active_model_arc: Arc<LlamaModel>, settings: &Settings) -> Result<Self, Box<dyn Error + Send + Sync>> {
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
        let session = active_model_arc.create_session(session_params)
            .map_err(|e| format!("Failed to create LlamaSession: {}", e))?;
        tracing::info!("LlamaSession created successfully.");

        // Store n_ctx for reference (max_context_size might differ from session's internal limit)
        let max_context_size = n_ctx as usize;

        Ok(Self {
            active_model: active_model_arc,
            session: Arc::new(Mutex::new(session)),
            max_context_size,
            temperature: settings.inference.temperature,
            max_tokens: settings.inference.max_tokens,
        })
    }

    /// Processes input text and returns a stream of generated response chunks.
    pub fn process_input(&self, input: String) 
        -> Pin<Box<dyn Stream<Item = Result<String, Box<dyn Error + Send + Sync>>> + Send>>
    {
        let active_model = Arc::clone(&self.active_model);
        let session_arc_mutex = Arc::clone(&self.session);
        let max_tokens = self.max_tokens;

        Box::pin(stream! {
            tracing::info!("Processing input for streaming: \"{}\"", input);

            let mut session_guard = session_arc_mutex.lock().await;
            
            if let Err(e) = session_guard.advance_context(&input) {
                yield Err(format!("Failed to advance context: {}", e).into());
                return;
            }
            tracing::info!("Context advanced with input prompt.");

            let sampler = StandardSampler::default();
            let completions_handle = match session_guard.start_completing_with(sampler, max_tokens) {
                Ok(handle) => handle,
                Err(e) => {
                    yield Err(format!("Failed to start completion: {}", e).into());
                    return;
                }
            };
            tracing::info!("Started completion generation for max {} tokens.", max_tokens);

            // Wrap the handle with TokensToStrings.
            let mut token_string_stream = TokensToStrings::new(completions_handle, (*active_model).clone());

            // Keep guard locked while iterating.
            let mut generated_count = 0;
            // Iterate the TokensToStrings stream using StreamExt::next explicitly
            while let Some(string_piece) = StreamExt::next(&mut token_string_stream).await {
                // TokensToStrings yields String directly.
                yield Ok(string_piece);

                generated_count += 1;
                // Approx limit check
                if generated_count >= max_tokens { 
                    tracing::warn!("Reached approx max token/chunk limit ({}) during generation.", max_tokens);
                    break;
                }
            }
            // Guard dropped here
            
            println!();
            tracing::info!("Completion finished. Generated approx {} chunks.", generated_count);
        })
    }
} 