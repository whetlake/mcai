use std::error::Error;
use rustyline::DefaultEditor;
use crate::config::Settings;
use std::io::Write;
use reqwest;
use serde::Deserialize;
use colored::*;
// Add color constants
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const BRIGHT_CYAN: &str = "\x1b[96m";
const RESET: &str = "\x1b[0m";
const YELLOW: &str = "\x1b[33m";
const BOLD: &str = "\x1b[1m";

// Import the display module
mod display;
use display::{display_models_table, display_model_metadata, display_tensor_info};

// Add this struct to deserialize the model response
#[derive(Deserialize)]
struct AttachModelResponse {
    name: String,
    label: String,
    greeting: String,
}

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
                        if parts.len() != 2 {
                            println!("Usage: attach <model_number>");
                            continue;
                        }
                        
                        if let Ok(model_number) = parts[1].parse::<usize>() {
                            
                            let request_body = serde_json::json!({
                                "model_number": model_number
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
                                                        
                                                        // Print the model greeting
                                                        println!("{BOLD}[{}]{RESET} {}", 
                                                            model_data.label.yellow(), 
                                                            model_data.greeting.bright_cyan()
                                                        );
                                                        
                                                        model_attached = true;
                                                        // Store the model label for future messages
                                                        current_model_label = Some(model_data.label);
                                                    }
                                                } else if let Some(message) = json.get("message").and_then(|m| m.as_str()) {
                                                    println!("Error: {}", message);
                                                } else {
                                                    println!("Failed to attach model");
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
                        } else {
                            println!("Invalid model number: {}", parts[1]);
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
                                // Send the message to the inference engine
                                let url = format!("http://{}:{}/api/v1/generate", settings.server.host, settings.server.port);
                                let client = reqwest::Client::new();
                                let request_body = serde_json::json!({
                                    "prompt": input
                                });
                                
                                match client.post(&url).json(&request_body).send().await {
                                    Ok(response) => {
                                        match response.text().await {
                                            Ok(text) => {
                                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                                    if let Some(data) = json.get("data") {
                                                        if let Some(response_text) = data.get("response").and_then(|r| r.as_str()) {
                                                            // Print the model's response
                                                            println!("{BOLD}[{}]{RESET} {}", label.yellow(), response_text.bright_cyan());
                                                        } else {
                                                            println!("Error: Invalid response format");
                                                        }
                                                    } else if let Some(message) = json.get("message").and_then(|m| m.as_str()) {
                                                        println!("Error: {}", message);
                                                    } else {
                                                        println!("Failed to parse response");
                                                    }
                                                } else {
                                                    println!("Failed to parse response as JSON");
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