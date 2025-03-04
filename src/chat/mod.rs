use std::error::Error;
use rustyline::DefaultEditor;
use crate::config::Settings;
use std::io::Write;
use reqwest;
// Add color constants
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const BRIGHT_CYAN: &str = "\x1b[96m";
const RESET: &str = "\x1b[0m";

fn print_help(model_attached: bool) {
    println!("\n{CYAN}MCAI Chat Commands{RESET}");
    println!("{BRIGHT_CYAN}{}{RESET}", "=".repeat(50));
    if model_attached {
        println!("{GREEN}mcai exit, mcai bye, mcai quit{RESET} - Exit the chat");
        println!("{GREEN}mcai help{RESET} - Show this help message"); 
        println!("{GREEN}mcai clear{RESET} - Clear the screen");
        println!("{GREEN}mcai models{RESET} - Display available models");
    } else {
        println!("{GREEN}exit, bye, quit{RESET} - Exit the chat");
        println!("{GREEN}help{RESET} - Show this help message");
        println!("{GREEN}clear{RESET} - Clear the screen");
        println!("{GREEN}models{RESET} - Display available models");
    }
    println!();
}

pub async fn chat_loop(settings: &Settings) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Starting chat session");
    let model_attached = false; // This should be set based on model attachment status
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
                        match client.get(format!("{}/models", server_url)).send().await {
                            Ok(response) => {
                                match response.text().await {
                                    Ok(text) => {
                                        println!("{}", text);
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
                    _ => {
                        // Normal chat input processing
                        if input.trim().is_empty() {
                            // Skip processing for empty lines, but don't add any newlines
                            // This ensures the prompt appears on the next line without skipping
                        } else {
                            // Process non-empty input
                            print!("{}\n", input);
                            std::io::stdout().flush().unwrap();
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