pub mod routes;
pub mod server;
// Make the types module public so it can be used by other parts of the crate (like chat)
pub mod types;

pub use server::ApiServer;