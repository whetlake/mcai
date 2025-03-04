use serde_json::Value;
use comfy_table::{Table, Cell, ContentArrangement};
use colored::*;

/// Displays a table of available models with colorful formatting.
/// 
/// # Arguments
/// 
/// * `json_response` - The JSON response from the server containing model data
pub fn display_models_table(json_response: &str) {
    println!("Processing model data...");
    
    if let Ok(value) = serde_json::from_str::<Value>(json_response) {
        if let Some(data) = value.get("data").and_then(|d| d.as_array()) {
            if data.is_empty() {
                println!("{}", "No models found in registry".yellow());
                return;
            }
            
            println!("Found {} models", data.len());
            let mut table = Table::new();
            table
                .set_header(vec![
                    Cell::new("Label").fg(comfy_table::Color::Cyan),
                    Cell::new("Name").fg(comfy_table::Color::Cyan),
                    Cell::new("Size").fg(comfy_table::Color::Cyan),
                    Cell::new("Architecture").fg(comfy_table::Color::Cyan),
                    Cell::new("Quantization").fg(comfy_table::Color::Cyan),
                    Cell::new("Tensors").fg(comfy_table::Color::Cyan),
                    Cell::new("Added Date").fg(comfy_table::Color::Cyan),
                ])
                .load_preset(comfy_table::presets::UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic);

            for model in data {
                if let (Some(label), Some(name), Some(size), Some(arch), Some(quant), Some(tensors), Some(added)) = (
                    model.get("label").and_then(|v| v.as_str()),
                    model.get("name").and_then(|v| v.as_str()),
                    model.get("size").and_then(|v| v.as_str()),
                    model.get("architecture").and_then(|v| v.as_str()),
                    model.get("quantization").and_then(|v| v.as_str()),
                    model.get("tensor_count").and_then(|v| v.as_u64()),
                    model.get("added_date").and_then(|v| v.as_str()),
                ) {
                    table.add_row(vec![
                        label.yellow(),
                        name.green(),
                        size.blue(),
                        arch.magenta(),
                        quant.cyan(),
                        tensors.to_string().bright_white(),
                        added.bright_black(),
                    ]);
                }
            }

            println!("\n{}", table);
            println!("{}", "=".repeat(100).bright_black());
        } else {
            println!("{}", "No models found".red());
        }
    } else {
        println!("{}", "Failed to parse model data".red());
    }
} 