use serde_json::Value;
use comfy_table::{Table, Cell, ContentArrangement, Attribute, CellAlignment};
use colored::*;
use chrono::{DateTime, Utc, TimeZone};

/// Displays a table of available models with colorful formatting.
/// 
/// # Arguments
/// 
/// * `json_response` - The JSON response from the server containing model data
pub fn display_models_table(json_response: &str) {
    // Remove debug output
    // println!("Processing model data...");
    
    if let Ok(value) = serde_json::from_str::<Value>(json_response) {
        if let Some(data) = value.get("data") {
            if let Some(data_array) = data.as_array() {
                if data_array.is_empty() {
                    println!("{}", "No models found in registry".yellow());
                    return;
                }
                
                let model_count = data_array.len();
                
                let mut table = Table::new();
                table
                    .set_header(vec![
                        Cell::new("#").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                        Cell::new("Label").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                        Cell::new("Name").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                        Cell::new("Size").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                        Cell::new("Architecture").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                        Cell::new("Quantization").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                        Cell::new("Tensors").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                        Cell::new("Added Date").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                    ])
                    .load_preset(comfy_table::presets::UTF8_FULL)
                    // Use dynamic content arrangement
                    .set_content_arrangement(ContentArrangement::Dynamic);

                let mut rows_added = 0;
                for (i, model) in data_array.iter().enumerate() {
                    // Get all the required fields
                    let label = model.get("label").and_then(|v| v.as_str());
                    let name = model.get("name").and_then(|v| v.as_str());
                    let size = model.get("size").and_then(|v| v.as_str());
                    let arch = model.get("architecture").and_then(|v| v.as_str());
                    let quant = model.get("quantization").and_then(|v| v.as_str());
                    let tensors = model.get("tensor_count").and_then(|v| v.as_u64());
                    
                    // Get the model number from the JSON if available, otherwise use the index
                    let model_number = model.get("number")
                        .and_then(|v| v.as_u64())
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| (i + 1).to_string());
                    
                    // Handle added_date as either a number (timestamp) or a string
                    let added_date = if let Some(timestamp) = model.get("added_date").and_then(|v| v.as_i64()) {
                        // Convert timestamp to formatted date string
                        match Utc.timestamp_opt(timestamp, 0) {
                            chrono::offset::LocalResult::Single(dt) => Some(dt.format("%Y-%m-%d %H:%M:%S").to_string()),
                            _ => None
                        }
                    } else {
                        // Try as string
                        model.get("added_date").and_then(|v| v.as_str()).map(|s| s.to_string())
                    };
                    
                    if let (Some(label), Some(name), Some(size), Some(arch), Some(quant), Some(tensors), Some(added)) = 
                        (label, name, size, arch, quant, tensors, added_date) {
                        table.add_row(vec![
                            Cell::new(model_number).fg(comfy_table::Color::White).set_alignment(CellAlignment::Center),
                            Cell::new(label).fg(comfy_table::Color::Yellow).set_alignment(CellAlignment::Center),
                            Cell::new(name).fg(comfy_table::Color::Green),
                            Cell::new(size).fg(comfy_table::Color::Blue).set_alignment(CellAlignment::Center),
                            Cell::new(arch).fg(comfy_table::Color::Magenta).set_alignment(CellAlignment::Center),
                            Cell::new(quant).fg(comfy_table::Color::Cyan).set_alignment(CellAlignment::Center),
                            Cell::new(tensors.to_string()).fg(comfy_table::Color::White).set_alignment(CellAlignment::Right),
                            Cell::new(added).fg(comfy_table::Color::DarkGrey),
                        ]);
                        rows_added += 1;
                    }
                }

                println!("\n{}", table);
                println!("{}", "=".repeat(100).bright_black());
                println!("{}", format!("Total models: {}", model_count).bright_green());
            } else {
                println!("{}", "No models found in registry".yellow());
            }
        } else {
            println!("{}", "No models found in registry".yellow());
        }
    } else {
        println!("{}", "Failed to parse model data".red());
    }
} 