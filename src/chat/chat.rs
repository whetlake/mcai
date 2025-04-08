use std::error::Error;
use rustyline::DefaultEditor;
use crate::config::Settings;
use std::io::Write;
use reqwest;
use colored::*;
use serde::Deserialize;

// Import items from the sibling module
use super::command_handlers::{ 
    ChatContext, 
    handle_list_models, 
    handle_list_attached, 
    handle_drop_model, 
    handle_detach_model, 
    handle_attach_new, 
    handle_activate_context, 
    handle_get_metadata, 
    handle_get_tensors, 
    handle_generate,
}; 

// Remove direct use, rely on mod.rs declaration - This comment is now incorrect
// use super::command_handlers::ChatContext; 

// Add color constants
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const BRIGHT_CYAN: &str = "\x1b[96m";
const RESET: &str = "\x1b[0m";
const YELLOW: &str = "\x1b[33m";
const BOLD: &str = "\x1b[1m";

fn print_help(model_attached: bool) {

    println!("\n{CYAN}MCAI Chat Commands{RESET}");
    println!("{BRIGHT_CYAN}{}{RESET}", "=".repeat(60));

    if model_attached {
        println!("{GREEN}mcai exit, mcai bye, mcai quit{RESET} - Exit the chat");
        println!("{GREEN}mcai help{RESET}                   - Show this help message");
        println!("{GREEN}mcai clear{RESET}                  - Clear the screen");
        println!("{GREEN}mcai models{RESET}                 - List models available in the registry");
        println!("{GREEN}mcai attached{RESET}               - List currently attached model instances");
        println!("{GREEN}mcai attach new <model#> [label]{RESET} - Attach a new model instance by registry number");
        println!("{GREEN}mcai switch <label|uuid>{RESET}     - Switch context for an existing attached model");
        println!("{GREEN}mcai metadata [@label|uuid]{RESET} - Display metadata for the specified (or current) model");
        println!("{GREEN}mcai tensors [@label|uuid]{RESET}  - Display tensors for the specified (or current) model");
        println!("{GREEN}mcai drop [@label|uuid]{RESET}     - Detach the specified (or the only) model instance");
        println!("{GREEN}mcai detach{RESET}                 - Deactivate the current model context (keeps model loaded)");
        println!("{GREEN}mcai rename [@old|uuid] <new>{RESET}- Rename attached model. Uses current if no identifier given & only one model attached.");
        // println!("{GREEN}@<label> <your prompt...>{RESET}  - Send prompt directly to the model with specified label");
        // println!("(If no @label is specified, the prompt goes to the current default model context)");

    } else {
        println!("{GREEN}exit, bye, quit{RESET} - Exit the chat");
        println!("{GREEN}help{RESET}          - Show this help message");
        println!("{GREEN}clear{RESET}         - Clear the screen");
        println!("{GREEN}models{RESET}        - List models available in the registry");
        println!("{GREEN}attached{RESET}      - List currently attached model instances (if any)");
        println!("{GREEN}attach new <model#> [label]{RESET} - Attach a model by its registry number, optionally assign a label");
        println!("{GREEN}attach <label|uuid>{RESET}     - Reactivate context for an existing attached model");
    }
    println!();
}

// --- Main Chat Loop ---

pub async fn chat_loop(settings: &Settings) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Starting chat session");
    let mut model_attached = false;
    let mut current_model_label: Option<String> = None;
    let mut current_model_uuid: Option<String> = None;
    print_help(model_attached);

    let mut rl = DefaultEditor::new()?;
    let client = reqwest::Client::new();
    let server_url = format!("http://{}:{}", settings.server.host, settings.server.port);

    loop {
        let prompt_prefix = if model_attached {
            "[you] > ".to_string()
        } else {
            "> ".to_string()
        };
        let readline = rl.readline(&prompt_prefix);

        match readline {
            Ok(input) => {
                let input_trimmed = input.trim();
                if input_trimmed.is_empty() {
                    continue;
                }

                // Process the input into a lowercase string
                let command_lowercase = input_trimmed.to_lowercase();

                // Exit the chat if exit command is received
                let exit_commands = if model_attached { ["mcai exit", "mcai bye", "mcai quit"] } else { ["exit", "bye", "quit"] };
                if exit_commands.contains(&command_lowercase.as_str()) {
                    println!("Goodbye!");
                    break;
                }

                // Create context (Type path uses imported name)
                let mut context = ChatContext {
                    client: &client,
                    server_url: &server_url,
                    model_attached: &mut model_attached,
                    current_model_label: &mut current_model_label,
                    current_model_uuid: &mut current_model_uuid,
                };

                let mut command_handled = true;

                match command_lowercase.as_str() {
                    cmd if (*context.model_attached && cmd == "mcai help") || (!*context.model_attached && cmd == "help") => {
                        print_help(*context.model_attached)
                    },
                    cmd if (*context.model_attached && cmd == "mcai clear") || (!*context.model_attached && cmd == "clear") => {
                        print!("\x1B[2J\x1B[1;1H");
                        std::io::stdout().flush().unwrap();
                    },
                    cmd if (*context.model_attached && cmd == "mcai models") || (!*context.model_attached && cmd == "models") => {
                        handle_list_models(&context).await
                    },
                    cmd if (*context.model_attached && cmd == "mcai attached") || (!*context.model_attached && cmd == "attached") => {
                        handle_list_attached(&context).await
                    },
                    "mcai drop" if *context.model_attached => {
                        handle_drop_model(&mut context).await
                    },
                    "mcai detach" if *context.model_attached => {
                        handle_detach_model(&mut context)
                    },
                    "mcai metadata" if *context.model_attached => {
                        handle_get_metadata(&context).await
                    },
                    "mcai tensors" if *context.model_attached => {
                        handle_get_tensors(&context).await
                    },
                    _ => {
                        if *context.model_attached {
                            if command_lowercase.starts_with("mcai attach new ") {
                                let prefix_len = "mcai attach new ".len();
                                let args: Vec<&str> = input_trimmed[prefix_len..].split_whitespace().collect();
                                handle_attach_new(&mut context, &args).await;
                            } else if command_lowercase.starts_with("mcai switch ") {
                                let target_identifier = input_trimmed["mcai switch ".len()..].trim();
                                if !target_identifier.is_empty() {
                                    handle_activate_context(&mut context, target_identifier).await;
                                } else {
                                    println!("Usage: mcai switch <label|uuid>");
                                }
                            } else {
                                command_handled = false;
                            }
                        } else {
                            if command_lowercase.starts_with("attach new ") {
                                let prefix_len: usize = "attach new ".len();
                                let args: Vec<&str> = input_trimmed[prefix_len..].split_whitespace().collect();
                                handle_attach_new(&mut context, &args).await;
                            } else if command_lowercase.starts_with("attach ") {
                                let target_identifier = input_trimmed["attach ".len()..].trim();
                                if !target_identifier.is_empty() {
                                    handle_activate_context(&mut context, target_identifier).await;
                                } else {
                                    println!("Usage: attach <label|uuid>");
                                }
                            } else {
                                command_handled = false;
                            }
                        }
                    }
                }

                if !command_handled && !input_trimmed.is_empty() {
                    if *context.model_attached {
                        if input_trimmed.starts_with('@') {
                            println!("Targeted prompt (e.g., @label <prompt>) not yet implemented.");
                        } else {
                            handle_generate(&mut context, input_trimmed).await;
                        }
                    } else {
                        println!("No model context active. Use 'attach new <num>' or 'attach <id>' to start.");
                    }
                }

                if !input_trimmed.is_empty() {
                    let _ = rl.add_history_entry(input_trimmed);
                }
            },
            Err(_) => {
                println!("Goodbye!");
                break;
            },
        }
    }
    Ok(())
}