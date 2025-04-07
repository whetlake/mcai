// Declare the display submodule
mod display;

// Declare the chat submodule (containing the chat_loop logic)
mod chat;

// Declare the command_handlers module
mod command_handlers;

// Re-export the public function from the chat submodule
pub use chat::chat_loop;
