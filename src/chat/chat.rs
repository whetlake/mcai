use std::error::Error;
use rustyline::DefaultEditor;
use crate::config::Settings;
use std::io::{Write, stdout};
use reqwest;
use colored::*;
use futures::StreamExt;
// Use the shared definition from server types
use crate::server::types::AttachModelResponse;
use super::display::{display_models_table, display_model_metadata, display_tensor_info};

// Add color constants
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const BRIGHT_CYAN: &str = "\x1b[96m";
const RESET: &str = "\x1b[0m";
const YELLOW: &str = "\x1b[33m";
const BOLD: &str = "\x1b[1m";

fn print_help(model_attached: bool) {
    println!("\n{CYAN}MCAI Chat Commands{RESET}");
    println!("{BRIGHT_CYAN}{}{RESET}", "=".repeat(50));
    if model_attached {
        println!("{GREEN}mcai exit, mcai bye, mcai quit{RESET} - Exit the chat");
        println!("{GREEN}mcai help{RESET} - Show this help message"); 
        println!("{GREEN}mcai clear{RESET} - Clear the screen");
        println!("{GREEN}mcai models{RESET} - Display available models");
        println!("{GREEN}mcai metadata{RESET} - Display current model metadata");
        println!("{GREEN}mcai tensors{RESET} - Display current model tensors");
        println!("{GREEN}mcai drop{RESET} - Detach the current model");
    } else {
        println!("{GREEN}exit, bye, quit{RESET} - Exit the chat");
        println!("{GREEN}help{RESET} - Show this help message");
        println!("{GREEN}clear{RESET} - Clear the screen");
        println!("{GREEN}models{RESET} - Display available models");
        println!("{GREEN}attach <model_number>{RESET} - Attach a model by its number");
    }
    println!();
}

pub async fn chat_loop(settings: &Settings) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Starting chat session");
    let mut model_attached = false;
    let mut current_model_label: Option<String> = None;
    print_help(model_attached);

    let mut rl = DefaultEditor::new()?;
    let client = reqwest::Client::new();
    let server_url = format!("http://{}:{}", settings.server.host, settings.server.port);

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(input) => {
                // Process exit commands
                let trimmed = input.trim().to_lowercase();
                let exit_commands = if model_attached {
                    ["mcai exit", "mcai bye", "mcai quit"]
                } else {
                    ["exit", "bye", "quit"]
                };

                if exit_commands.contains(&trimmed.as_str()) {
                    println!("Goodbye!");
                    break;
                }

                // Process other commands
                match trimmed.as_str() {
                    // Print help
                    cmd if (model_attached && cmd == "mcai help") || (!model_attached && cmd == "help") => {
                        print_help(model_attached)
                    },
                    // Clear screen
                    cmd if (model_attached && cmd == "mcai clear") || (!model_attached && cmd == "clear") => {
                        // Clear screen with ANSI escape codes
                        print!("\x1B[2J\x1B[1;1H"); 
                        // Ensure the escape code is sent immediately
                        std::io::stdout().flush().unwrap();
                    },
                    // Display models
                    cmd if (model_attached && cmd == "mcai models") || (!model_attached && cmd == "models") => {
                        match client.get(format!("{}/api/v1/models", server_url)).send().await {
                            Ok(response) => {
                                match response.text().await {
                                    Ok(text) => {
                                        // Use the display_models_table function instead of just printing the raw text
                                        display_models_table(&text);
                                    },
                                    Err(e) => {
                                        println!("Error reading response: {}", e);
                                    }
                                }
                            },
                            Err(e) => {
                                println!("Error requesting models: {}", e);
                            }
                        }
                    },
                    // Drop model (only when a model is attached)
                    cmd if model_attached && cmd == "mcai drop" => {
                        match client.post(format!("{}/api/v1/drop", server_url)).send().await {
                            Ok(response) => {
                                match response.text().await {
                                    Ok(text) => {
                                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                            if json.get("status").and_then(|s| s.as_str()) == Some("success") {
                                                println!("Model detached successfully");
                                                model_attached = false;
                                                current_model_label = None;
                                            } else if let Some(message) = json.get("message").and_then(|m| m.as_str()) {
                                                println!("Error: {}", message);
                                            } else {
                                                println!("Failed to detach model");
                                            }
                                        } else {
                                            println!("Failed to parse response");
                                        }
                                    },
                                    Err(e) => {
                                        println!("Error reading response: {}", e);
                                    }
                                }
                            },
                            Err(e) => {
                                println!("Error sending request: {}", e);
                            }
                        }
                    },
                    // Attach model (only when no model is attached)
                    cmd if !model_attached && cmd.starts_with("attach ") => {
                        let parts: Vec<&str> = cmd.split_whitespace().collect();
                        // Expecting "attach <number>" or "attach <number> <label...>"
                        if parts.len() < 2 {
                            println!("Usage: attach <model_number> [label]");
                            continue;
                        }

                        let model_number_str = parts[1];
                        let user_provided_label: Option<String> = if parts.len() >= 3 {
                            Some(parts[2..].join(" ")) // Join remaining parts for multi-word labels
                        } else {
                            None
                        };
                        
                        if let Ok(model_number) = model_number_str.parse::<usize>() {
                            
                            let request_body = serde_json::json!({
                                "model_number": model_number,
                                "user_label": user_provided_label
                            });
                            
                            match client.post(format!("{}/api/v1/attach", server_url))
                                .json(&request_body)
                                .send()
                                .await {
                                Ok(response) => {
                                    match response.text().await {
                                        Ok(text) => {
                                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                                if json.get("status").and_then(|s| s.as_str()) == Some("success") {
                                                    if let Ok(model_data) = serde_json::from_value::<AttachModelResponse>(
                                                        json.get("data").unwrap_or(&serde_json::Value::Null).clone()
                                                    ) {
                                                        
                                                        // --- Corrected Logic (Take 3) ---
                                                        // 1. Determine the label for the initial greeting AND the ongoing chat prompt:
                                                        //    Use the label from the server response (`model_data.label`) if available (`Some`), 
                                                        //    otherwise use the model name (`model_data.name`).
                                                        let display_label = model_data.user_label.as_ref().unwrap_or(&model_data.name);

                                                        // Print the model greeting using the determined display label
                                                        println!("{BOLD}[{}]{RESET} {}", 
                                                            display_label.yellow(), // Use display_label here
                                                            model_data.greeting.bright_cyan()
                                                        );
                                                        
                                                        model_attached = true;
                                                        // Store the same label for future messages
                                                        // We clone display_label because it's a reference (&String) from unwrap_or, 
                                                        // and current_model_label needs an owned String.
                                                        current_model_label = Some(display_label.clone()); // Use display_label here
                                                    } else {
                                                        // Handle case where data exists but doesn't match AttachModelResponse
                                                        println!("{}Error: Successfully attached model, but failed to parse model details from response.{}", YELLOW, RESET);
                                                        // Still mark as attached, but maybe use a placeholder label?
                                                        model_attached = true;
                                                        current_model_label = user_provided_label.or_else(|| Some(format!("Model {}", model_number))); // Fallback label
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
                                        Err(e) => {
                                            println!("Error reading response body: {}", e);
                                        }
                                    }
                                },
                                Err(e) => {
                                    println!("Error sending attach request: {}", e);
                                }
                            }
                        } else {
                            println!("Invalid model number: {}", model_number_str);
                        }
                    },
                    // Get metadata (only when a model is attached)
                    cmd if model_attached && cmd == "mcai metadata" => {
                        match client.get(format!("{}/api/v1/metadata", server_url)).send().await {
                            Ok(response) => {
                                match response.text().await {
                                    Ok(text) => {
                                        display_model_metadata(&text);
                                    },
                                    Err(e) => {
                                        println!("Error reading response: {}", e);
                                    }
                                }
                            },
                            Err(e) => {
                                println!("Error requesting metadata: {}", e);
                            }
                        }
                    },
                    // Get tensors (only when a model is attached)
                    cmd if model_attached && cmd == "mcai tensors" => {
                        match client.get(format!("{}/api/v1/tensors", server_url)).send().await {
                            Ok(response) => {
                                match response.text().await {
                                    Ok(text) => {
                                        display_tensor_info(&text);
                                    },
                                    Err(e) => {
                                        println!("Error reading response: {}", e);
                                    }
                                }
                            },
                            Err(e) => {
                                println!("Error requesting tensors: {}", e);
                            }
                        }
                    },
                    _ => {
                        // Normal chat input processing
                        if input.trim().is_empty() {
                            // Skip processing for empty lines
                            continue;
                        }
                        
                        if model_attached {
                            if let Some(label) = &current_model_label {                               
                                let url = format!("{}/api/v1/generate", server_url);
                                let request_body = serde_json::json!({ "prompt": input });

                                match client.post(&url).json(&request_body).send().await {
                                    Ok(response) => {
                                        if response.status().is_success() {
                                            // Make stream mutable to call next()
                                            let mut byte_stream = response.bytes_stream(); 

                                            // Print the model label once before the stream starts
                                            print!("\n{BOLD}[{}]{RESET} ", label.yellow()); 
                                            // Flush immediately to ensure label appears before stream data
                                            stdout().flush().unwrap(); 

                                            // Process the stream of byte chunks
                                            while let Some(chunk_result) = byte_stream.next().await {
                                                match chunk_result {
                                                    Ok(bytes) => {
                                                        let text_chunk = String::from_utf8_lossy(&bytes);
                                                        for line in text_chunk.lines() {
                                                            // DEBUG: Print the raw line if it starts with data:
                                                            if line.starts_with("data:") {
                                                                // eprintln!("[RAW SSE LINE]: {:?}", line);

                                                                // Get payload after "data:", removing AT MOST one leading space
                                                                let potential_payload = &line[5..];
                                                                let data_str = potential_payload.strip_prefix(' ').unwrap_or(potential_payload);

                                                                // Only print if the final data_str is not just whitespace
                                                                if !data_str.trim().is_empty() { 
                                                                    // Print the processed data string
                                                                    print!("{}", data_str.bright_cyan()); 
                                                                    stdout().flush().unwrap();
                                                                }
                                                            }
                                                        }
                                                    },
                                                    Err(e) => {
                                                        // Print error and stop processing stream for this request
                                                        println!("{}Error reading stream chunk: {}{}", YELLOW, e, RESET);
                                                        break; 
                                                    }
                                                }
                                            }
                                            // Print a final newline after the stream is fully consumed
                                            println!();

                                        } else {
                                            // Handle non-successful HTTP status codes
                                            let status = response.status();
                                            match response.text().await {
                                                Ok(text) => {
                                                    println!("{}Error: Server returned status {}. Response: {}{}", YELLOW, status, text, RESET);
                                                },
                                                Err(e) => {
                                                    println!("{}Error: Server returned status {} but failed to read response body: {}{}", YELLOW, status, e, RESET);
                                                }
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        println!("{}Error sending request: {}{}", YELLOW, e, RESET);
                                    }
                                }
                            }
                        } else {
                            println!("No model attached. Use 'attach <model_number>' to start a chat.");
                        }
                    }
                }
                
                // Add line to history
                let _ = rl.add_history_entry(&input);
            },
            Err(_) => {
                println!("Goodbye!");
                break;
            },
        }
    }
    Ok(())
}