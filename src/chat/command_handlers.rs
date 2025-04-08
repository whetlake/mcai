use crate::llm::engine::AttachedModelInfo;
use crate::server::types::{ApiResponse, AttachModelResponse};
use crate::chat::display::{
    display_attached_models, display_model_metadata, display_models_table, display_tensor_info,
};
use colored::*;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::io::{stdout, Write};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

// Define a struct to represent the SSE JSON payload
#[derive(Deserialize, Debug)]
struct GenerateResponseChunk {
    uuid: String,
    text: String,
}


// Define context struct to hold shared resources and state references
/// Represents the context for chat command operations
///s
/// This struct holds references to shared resources and state that are needed
/// across different command handlers in the chat interface.
///
/// # Fields
///
/// * `client` - HTTP client used for making API requests to the server
/// * `server_url` - Base URL of the server API
/// * `model_attached` - Mutable reference to a boolean indicating if a model is currently attached
/// * `current_model_label` - Mutable reference to the optional label of the currently active model
/// * `current_model_uuid` - Mutable reference to the optional UUID of the currently active model
/// * `interrupt` - Arc<AtomicBool> to handle interrupt signals
pub(super) struct ChatContext<'a> {
    pub client: &'a Client,
    pub server_url: &'a str,
    pub model_attached: &'a mut bool,
    pub current_model_label: &'a mut Option<String>,
    pub current_model_uuid: &'a mut Option<String>,
    pub interrupt: Arc<AtomicBool>,
}

// Add color constants needed by handlers
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

// --- Moved Command Handler Functions (Marked pub(super)) ---

pub(super) async fn handle_list_models(context: &ChatContext<'_>) {
    match context.client.get(format!("{}/api/v1/models", context.server_url)).send().await {
        Ok(response) => match response.text().await {
            Ok(text) => display_models_table(&text),
            Err(e) => println!("Error reading response: {}", e),
        },
        Err(e) => println!("Error requesting models: {}", e),
    }
}

pub(super) async fn handle_list_attached(context: &ChatContext<'_>) {
    match context.client.get(format!("{}/api/v1/attached", context.server_url)).send().await {
        Ok(response) => match response.text().await {
            Ok(text) => display_attached_models(&text),
            Err(e) => println!("Error reading response: {}", e),
        },
        Err(e) => println!("Error requesting attached models: {}", e),
    }
}

pub(super) async fn handle_drop_model(
    context: &mut ChatContext<'_>,
    // TODO: Add identifier parameter later if needed for `drop <id>`
) {
     // Need to handle optional identifier for drop
     let url = format!("{}/api/v1/drop", context.server_url); // Add identifier as query param later
     match context.client.post(url).send().await {
         Ok(response) => match response.text().await {
             Ok(text) => {
                 if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                     if json.get("status").and_then(|s| s.as_str()) == Some("success") {
                         println!("Model dropped successfully");
                         // If the dropped model was the active one, reset state
                         // TODO: Need logic to check if the *current* context's model was dropped
                         *context.model_attached = false;
                         *context.current_model_label = None;
                         *context.current_model_uuid = None; // Clear the UUID
                     } else if let Some(message) = json.get("message").and_then(|m| m.as_str()) {
                         println!("Error: {}", message);
                     } else {
                         println!("Failed to drop model");
                     }
                 } else {
                     println!("Failed to parse drop response");
                 }
             }
             Err(e) => println!("Error reading drop response: {}", e),
         },
         Err(e) => println!("Error sending drop request: {}", e),
     }
}

pub(super) fn handle_detach_model(
    context: &mut ChatContext<'_>,
) {
    println!("Detaching model context. Model remains loaded in the background.");
    println!("Use 'attach new <number> [label]' or 'attach <label|uuid>' to reactivate a context.");
    *context.model_attached = false;
    *context.current_model_label = None;
    *context.current_model_uuid = None; // Clear the UUID
}

pub(super) async fn handle_attach_new(
    context: &mut ChatContext<'_>,
    args: &[&str], // Arguments after "attach new "
) {
    if args.is_empty() {
        println!("Usage: attach new <model_number> [label]");
        return;
    }
    if let Ok(model_number) = args[0].parse::<usize>() {
        let user_provided_label: Option<String> = if args.len() >= 2 {
            Some(args[1..].join(" "))
        } else {
            None
        };

        let request_body = serde_json::json!({
            "model_number": model_number,
            "user_label": user_provided_label
        });

        match context.client.post(format!("{}/api/v1/attach", context.server_url))
            .json(&request_body)
            .send()
            .await {
            Ok(response) => match response.text().await {
                 Ok(text) => {
                     if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                         if json.get("status").and_then(|s| s.as_str()) == Some("success") {
                             if let Ok(model_data) = serde_json::from_value::<AttachModelResponse>(
                                 json.get("data").unwrap_or(&serde_json::Value::Null).clone()
                             ) {
                                 let display_label = model_data.user_label.as_ref().unwrap_or(&model_data.name);
                                 println!("{BOLD}[{}]{RESET} {}",
                                     display_label.yellow(),
                                     model_data.greeting.bright_cyan()
                                 );
                                 *context.model_attached = true;
                                 *context.current_model_label = Some(display_label.clone());
                                 *context.current_model_uuid = Some(model_data.uuid);
                             } else {
                                 println!("{}Error: Successfully attached model, but failed to parse model details from response.{}", YELLOW, RESET);
                                 *context.model_attached = true; // Still attached
                                 *context.current_model_label = user_provided_label.or_else(|| Some(format!("Model {}", model_number)));
                             }
                         } else if let Some(message) = json.get("message").and_then(|m| m.as_str()) {
                             println!("Error: {}", message);
                         } else {
                             println!("Failed to attach model (unknown reason)");
                         }
                     } else {
                         println!("Failed to parse server response as JSON");
                     }
                 },
                 Err(e) => println!("Error reading response body: {}", e),
             },
             Err(e) => println!("Error sending attach request: {}", e),
        }
    } else {
        println!("Invalid model number: {}", args[0]);
    }
}

// Handles both 'attach <id>' (when detached) and 'mcai switch <id>' (when attached)
pub(super) async fn handle_activate_context(
    context: &mut ChatContext<'_>,
    target_identifier: &str,
) {
     println!("Attempting to activate context for: {}", target_identifier);

     match context.client.get(format!("{}/api/v1/attached", context.server_url)).send().await {
         Ok(response) => match response.text().await {
             Ok(text) => {
                 match serde_json::from_str::<ApiResponse<Vec<AttachedModelInfo>>>(&text) {
                     Ok(parsed_response) => {
                         if parsed_response.status == "success" {
                             if let Some(attached_models) = parsed_response.data {
                                 let mut found_model: Option<&AttachedModelInfo> = None;
                                 let mut potential_matches = 0;
                                 // Find the target model
                                 for model in &attached_models {
                                     let matches_uuid = model.uuid == target_identifier;
                                     let matches_label = model.user_label.as_deref() == Some(target_identifier);
                                     if matches_uuid || matches_label {
                                         if found_model.is_none() { found_model = Some(model); }
                                         else { if matches_label { potential_matches = 2; break; } if matches_uuid { found_model = Some(model); } }
                                         if matches_uuid { potential_matches = 1; break; }
                                         potential_matches = 1;
                                     }
                                 }
                                 // Process result
                                 if potential_matches == 1 && found_model.is_some() {
                                     let model_info = found_model.unwrap();
                                     let display_label = model_info.user_label.as_ref().unwrap_or(&model_info.name);
                                     println!("Activated model context: [{}]", display_label.yellow());
                                     *context.model_attached = true; // Ensure state is attached
                                     *context.current_model_label = Some(display_label.clone());
                                     *context.current_model_uuid = Some(model_info.uuid.clone()); // Store the UUID
                                 } else if potential_matches > 1 {
                                     println!("{}Error: Identifier '{}' is ambiguous and matches multiple models by label.{}", YELLOW, target_identifier, RESET);
                                 } else {
                                     println!("{}Error: No attached model found with identifier '{}'.{}", YELLOW, target_identifier, RESET);
                                 }
                             } else { println!("{}Error: No attached models listed by server (data field missing or null).{}", YELLOW, RESET); }
                         } else { println!("{}Error fetching attached models: {}{}", YELLOW, parsed_response.message.unwrap_or_else(|| "Unknown server error".to_string()), RESET); }
                     },
                     Err(e) => println!("{}Error parsing attached models response: {}{}", YELLOW, e, RESET),
                 }
             },
             Err(e) => println!("{}Error reading attached models response: {}{}", YELLOW, e, RESET),
         },
         Err(e) => println!("{}Error requesting attached models: {}{}", YELLOW, e, RESET),
     }
}


pub(super) async fn handle_get_metadata(context: &ChatContext<'_> /*, identifier: Option<&str> */) {
    // TODO: Add identifier query param if needed
    let url = format!("{}/api/v1/metadata", context.server_url);
    match context.client.get(url).send().await {
        Ok(response) => match response.text().await {
            Ok(text) => display_model_metadata(&text),
            Err(e) => println!("Error reading metadata response: {}", e),
        },
        Err(e) => println!("Error requesting metadata: {}", e),
    }
}

pub(super) async fn handle_get_tensors(context: &ChatContext<'_> /*, identifier: Option<&str> */) {
    // TODO: Add identifier query param if needed
    let url = format!("{}/api/v1/tensors", context.server_url);
    match context.client.get(url).send().await {
         Ok(response) => match response.text().await {
             Ok(text) => display_tensor_info(&text),
             Err(e) => println!("Error reading tensor response: {}", e),
         },
         Err(e) => println!("Error requesting tensors: {}", e),
    }
}

pub(super) async fn handle_generate(
    context: &mut ChatContext<'_>,
    prompt: &str,
) {
    if let Some(label) = context.current_model_label.as_ref() {
        if let Some(uuid) = context.current_model_uuid.as_ref() {
            let url: String = format!("{}/api/v1/generate", context.server_url);
            // Send the active UUID
            let request_body = serde_json::json!({
                "prompt": prompt,
                "model_session_uuid": uuid // Send the current UUID
            });

            match context.client.post(&url).json(&request_body).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        let mut byte_stream = response.bytes_stream();
                        // Print a newline to separate the user input from the model output
                        print!("\n{BOLD}[{}]{RESET} ", label.yellow());
                        stdout().flush().unwrap();

                        let mut interrupted = false; // Track if stream was interrupted

                        while let Some(chunk_result) = byte_stream.next().await {
                            // Check for interrupt signal
                            if context.interrupt.load(std::sync::atomic::Ordering::SeqCst) {
                                println!("\n{}[Generation Interrupted]{}", YELLOW, RESET);
                                interrupted = true;
                                break; // Exit the streaming loop
                            }

                            match chunk_result {
                                Ok(bytes) => {
                                    let text_chunk = String::from_utf8_lossy(&bytes);
                                    for line in text_chunk.lines() {
                                        if line.starts_with("data:") {
                                            let potential_payload = &line[5..];
                                            let data_str = potential_payload.strip_prefix(' ').unwrap_or(potential_payload);
                                            match serde_json::from_str::<GenerateResponseChunk>(data_str) {
                                                Ok(chunk_data) => {
                                                    if !chunk_data.text.trim().is_empty() {
                                                        print!("{}", chunk_data.text.bright_cyan());
                                                        stdout().flush().unwrap();
                                                    }
                                                }
                                                Err(e) => eprintln!("\n{}[SSE Parse Error]: Failed to parse chunk '{}': {}{}", YELLOW, data_str, e, RESET),
                                            }
                                        }
                                    }
                                },
                                Err(e) => { println!("{}Error reading stream chunk: {}{}", YELLOW, e, RESET); break; }
                            }
                        }

                        // Only print final newline if not interrupted (already printed one in interrupt message)
                        if !interrupted {
                            println!(); // Final newline
                        }
                    } else {
                        let status = response.status();
                        match response.text().await {
                             Ok(text) => println!("{}Error: Server returned status {}. Response: {}{}", YELLOW, status, text, RESET),
                             Err(e) => println!("{}Error: Server returned status {} but failed to read response body: {}{}", YELLOW, status, e, RESET),
                        }
                    }
                },
                Err(e) => println!("{}Error sending generate request: {}{}", YELLOW, e, RESET),
            }
        } else {
            println!("{}Internal Error: Current model label is set ('{}'), but UUID is missing.{}", YELLOW, label, RESET);
        }
    } else {
         println!("Internal Error: No current model label set while model_attached is true.");
    }
}

// --- Rename Model Handler ---

pub(super) async fn handle_rename_model(
    context: &mut ChatContext<'_>,
    old_identifier: Option<&str>, // Optional: UUID or current label
    new_label: &str,
) {
    // Body contains only the new label
    let request_body = serde_json::json!({
        "new_label": new_label
    });

    let base_url = format!("{}/api/v1/rename", context.server_url);
    
    // Build the request using reqwest builder
    let mut request_builder = context.client.post(&base_url).json(&request_body);

    // Add the identifier as a query parameter *only if* it's provided
    if let Some(identifier) = old_identifier {
        request_builder = request_builder.query(&[("identifier", identifier)]);
    }

    match request_builder.send().await {
        Ok(response) => match response.text().await {
            Ok(text) => {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if json.get("status").and_then(|s| s.as_str()) == Some("success") {
                        println!("Model instance successfully renamed to '{}'", new_label);

                        // --- Update active context if it was the one renamed ---
                        let mut context_updated = false;

                        // Check if rename targeted the currently active UUID
                        if let Some(active_uuid) = context.current_model_uuid.as_deref() {
                             // Use old_identifier directly if Some, otherwise this check won't match when None was passed
                            if old_identifier.is_some() && old_identifier.unwrap() == active_uuid {
                                *context.current_model_label = Some(new_label.to_string());
                                context_updated = true;
                                println!("Active context label updated to '{}'.", new_label);
                            }
                        }

                        // Check if rename targeted the currently active label (if UUID didn't match or wasn't the identifier)
                        if !context_updated {
                            if let Some(active_label) = context.current_model_label.as_deref() {
                                 // Use old_identifier directly if Some
                                if old_identifier.is_some() && old_identifier.unwrap() == active_label {
                                     *context.current_model_label = Some(new_label.to_string());
                                     context_updated = true;
                                     println!("Active context label updated to '{}'.", new_label);
                                }
                            }
                        }

                        // Check if rename applied implicitly because no identifier was given AND a model is active
                        if !context_updated && old_identifier.is_none() {
                             if *context.model_attached { // Check if a model context is actually active
                                 *context.current_model_label = Some(new_label.to_string());
                                 println!("Active context label updated to '{}'.", new_label);
                                 // No need to update UUID here, as renaming doesn't change it.
                             }
                        }
                        // --- End context update ---

                    } else if let Some(message) = json.get("message").and_then(|m| m.as_str()) {
                        println!("Error renaming model: {}", message); // Server-side check errors appear here
                    } else {
                        println!("Failed to rename model (unknown server error).");
                    }
                } else {
                    println!("Failed to parse rename response from server.");
                }
            }
            Err(e) => println!("Error reading rename response: {}", e),
        },
        Err(e) => println!("Error sending rename request: {}", e),
    }
}
