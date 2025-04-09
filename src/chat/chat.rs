use std::error::Error;
use rustyline::DefaultEditor;
use crate::config::Settings;
use std::io::Write;
use reqwest;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::signal;

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
    handle_rename_model,
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

    // --- Interrupt Handling Setup ---
    let interrupt_signal = Arc::new(AtomicBool::new(false));
    let interrupt_for_task = Arc::clone(&interrupt_signal); // Clone Arc for the task

    tokio::spawn(async move {
        // Loop to handle multiple Ctrl+C signals
        loop {
            match signal::ctrl_c().await {
                Ok(()) => {
                    // Signal received, set the flag
                    println!("\n{}[Ctrl+C detected. Press Enter to continue or type next command.]{}", YELLOW, RESET);
                    interrupt_for_task.store(true, Ordering::SeqCst);
                    // The flag will be checked by handle_generate or before the next readline call.
                    // No need to break the loop; await ctrl_c() again.
                }
                Err(err) => {
                    eprintln!("{}Error listening for Ctrl-C signal: {}. Exiting signal handler task.{}", YELLOW, err, RESET);
                    break; // Exit the loop if there's an error listening
                }
            }
        }
    });
    // --- End Interrupt Handling Setup ---

    loop {
        // --- Check and Reset Interrupt Flag Before Prompt ---
        if interrupt_signal.load(Ordering::SeqCst) {
            interrupt_signal.store(false, Ordering::SeqCst);
            println!(); // Print a newline for cleaner prompt display after interrupt message
        }
        // --- End Check ---

        // Restore the original prompt logic
        let prompt_prefix = if model_attached {
            "[you] > ".to_string() // Use "[you] > " when attached
        } else {
            "> ".to_string()
        };

        let readline = rl.readline(&prompt_prefix); // Use the string directly

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

                // Create context (pass the interrupt Arc)
                let mut context = ChatContext {
                    client: &client,
                    server_url: &server_url,
                    model_attached: &mut model_attached,
                    current_model_label: &mut current_model_label,
                    current_model_uuid: &mut current_model_uuid,
                    interrupt: Arc::clone(&interrupt_signal), // Clone Arc for context
                };

                let mut command_handled = true;

                // --- Command Matching ---
                match command_lowercase.as_str() {
                    // Handle exact matches that are context-dependent
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
                    "mcai detach" if *context.model_attached => {
                        handle_detach_model(&mut context)
                    },
                    // Handle commands that take optional identifiers differently
                    cmd if cmd.starts_with("mcai drop") && *context.model_attached => {
                        let identifier = input_trimmed["mcai drop".len()..].trim();
                        if identifier.is_empty() {
                            handle_drop_model(&mut context, None).await; // Drop current context model
                        } else {
                            handle_drop_model(&mut context, Some(identifier)).await; // Drop specific model
                        }
                    },
                     cmd if cmd.starts_with("drop ") && !*context.model_attached => {
                        let identifier = input_trimmed["drop ".len()..].trim();
                         if !identifier.is_empty() {
                            handle_drop_model(&mut context, Some(identifier)).await; // Drop specific model when detached
                         } else {
                             println!("Usage: drop <identifier> (when no model context is active)");
                         }
                    },
                    cmd if cmd.starts_with("mcai metadata") && *context.model_attached => {
                         let identifier = input_trimmed["mcai metadata".len()..].trim();
                         if identifier.is_empty() {
                            handle_get_metadata(&context, None).await; // Metadata for current context model
                         } else {
                             // Allow specifying model even when attached, e.g. `mcai metadata other_label`
                             handle_get_metadata(&context, Some(identifier)).await;
                         }
                    },
                     cmd if cmd.starts_with("metadata ") && !*context.model_attached => {
                         let identifier = input_trimmed["metadata ".len()..].trim();
                         if !identifier.is_empty() {
                             handle_get_metadata(&context, Some(identifier)).await; // Metadata for specific model when detached
                         } else {
                             println!("Usage: metadata <identifier> (when no model context is active)");
                         }
                     },
                    cmd if cmd.starts_with("mcai tensors") && *context.model_attached => {
                         let identifier = input_trimmed["mcai tensors".len()..].trim();
                          if identifier.is_empty() {
                            handle_get_tensors(&context, None).await; // Tensors for current context model
                         } else {
                             handle_get_tensors(&context, Some(identifier)).await;
                         }
                    },
                     cmd if cmd.starts_with("tensors ") && !*context.model_attached => {
                         let identifier = input_trimmed["tensors ".len()..].trim();
                          if !identifier.is_empty() {
                             handle_get_tensors(&context, Some(identifier)).await; // Tensors for specific model when detached
                          } else {
                              println!("Usage: tensors <identifier> (when no model context is active)");
                          }
                     },

                     // Handle other non-exact matches
                    _ => {
                        if *context.model_attached {
                            // If model is already attached, handle remaining commands
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
                            } else if command_lowercase.starts_with("mcai rename ") {
                                let args_str = input_trimmed["mcai rename ".len()..].trim();
                                let args: Vec<&str> = args_str.split_whitespace().collect();
                                match args.len() {
                                    1 => {
                                        let new_label = args[0];
                                        handle_rename_model(&mut context, None, new_label).await;
                                    }
                                    2 => {
                                        let old_identifier = args[0];
                                        let new_label = args[1];
                                        handle_rename_model(&mut context, Some(old_identifier), new_label).await;
                                    }
                                    _ => {
                                        println!("Usage: mcai rename [@old_identifier|uuid] <new_label>");
                                        println!("       (If no identifier is given, attempts to rename the currently active model)");
                                    }
                                }
                            } else {
                                command_handled = false; // Not a recognized 'mcai' command when attached
                            }
                        } else {
                            // If model is not attached
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
                            } else if command_lowercase.starts_with("rename ") {
                                let args_str = input_trimmed["rename ".len()..].trim();
                                let args: Vec<&str> = args_str.split_whitespace().collect();
                                if args.len() == 2 {
                                    let old_identifier = args[0];
                                    let new_label = args[1];
                                    handle_rename_model(&mut context, Some(old_identifier), new_label).await;
                                } else {
                                    println!("Usage: rename <old_identifier|uuid> <new_label>");
                                    println!("       (You must specify the model to rename when no context is active)");
                                }
                            } else {
                                command_handled = false; // Not a recognized command when detached
                            }
                        }
                    }
                }
                // --- End Command Matching ---

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
            Err(rustyline::error::ReadlineError::Interrupted) => {
                if model_attached {
                    println!("{}[Interrupted. Type 'mcai exit' or 'mcai quit' to exit completly. Press Enter for new prompt.]{}", YELLOW, RESET);
                } else {
                    println!("{}[Interrupted. Type 'exit' or 'quit' to exit completly. Press Enter for new prompt.]{}", YELLOW, RESET);
                }
                 // The interrupt_signal flag might also be set by the tokio task,
                 // the check at the start of the loop will handle resetting it.
                continue;
            },
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("Goodbye! (EOF)");
                break;
            }
            Err(err) => {
                println!("{}Error reading input: {}{}", YELLOW, err, RESET);
                break;
            },
        }
    }
    Ok(())
}