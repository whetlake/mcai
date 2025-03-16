use serde_json::Value;
use comfy_table::{Table, Cell, ContentArrangement, Attribute, CellAlignment};
use colored::*;
use chrono::{Utc, TimeZone};

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
                    }
                }

                println!("\n{}", table);
                println!("{}", "=".repeat(100).bright_black());
                println!("{}", format!("Total models: {}\n", model_count).bright_green());
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

/// Displays model metadata in a formatted table
/// 
/// # Arguments
/// 
/// * `json_response` - The JSON response from the server containing model metadata
pub fn display_model_metadata(json_response: &str) {
    if let Ok(value) = serde_json::from_str::<Value>(json_response) {
        if let Some(data) = value.get("data") {
            // Print model information header
            println!("\n{}", "Model Information".cyan().bold());
            println!("{}", "=".repeat(50).bright_black());

            // Print basic model information
            if let Some(name) = data.get("name").and_then(|v| v.as_str()) {
                println!("{}: {}", "Name".yellow(), name);
            }
            if let Some(label) = data.get("label").and_then(|v| v.as_str()) {
                println!("{}: {}", "Label".yellow(), label);
            }
            if let Some(arch) = data.get("architecture").and_then(|v| v.as_str()) {
                println!("{}: {}", "Architecture".yellow(), arch);
            }
            if let Some(size) = data.get("size").and_then(|v| v.as_str()) {
                println!("{}: {}", "Size".yellow(), size);
            }
            if let Some(quant) = data.get("quantization").and_then(|v| v.as_str()) {
                println!("{}: {}", "Quantization".yellow(), quant);
            }
            if let Some(tensors) = data.get("tensor_count").and_then(|v| v.as_u64()) {
                println!("{}: {}", "Tensor Count".yellow(), tensors);
            }
            if let Some(filename) = data.get("filename").and_then(|v| v.as_str()) {
                println!("{}: {}", "Filename".yellow(), filename);
            }

            // Create metadata table
            if let Some(metadata) = data.get("metadata").and_then(|v| v.as_array()) {
                println!("\n{}", "Metadata Details".cyan().bold());
                println!("{}", "=".repeat(50).bright_black());

                let mut table = Table::new();
                table.set_header(vec![
                    Cell::new("Key").fg(comfy_table::Color::Yellow).add_attribute(Attribute::Bold),
                    Cell::new("Type").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                    Cell::new("Value").fg(comfy_table::Color::Green).add_attribute(Attribute::Bold),
                ])
                .load_preset(comfy_table::presets::UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic);

                for item in metadata {
                    if let (Some(key), Some(type_str), Some(value)) = (
                        item.get(0).and_then(|v| v.as_str()),
                        item.get(1).and_then(|v| v.as_str()),
                        item.get(2).and_then(|v| v.as_str())
                    ) {
                        // If the type is Array, truncate the display
                        let display_value = if type_str == "Array" {
                            // Check if it's an array by looking for [ at the start
                            if value.starts_with('[') && value.ends_with(']') {
                                // Extract the array elements
                                let elements: Vec<&str> = value[1..value.len()-1]
                                    .split(',')
                                    .map(|s| s.trim())
                                    .collect();
                                
                                if elements.len() > 5 {
                                    // Take first 5 elements and add ellipsis with total count
                                    let preview: Vec<_> = elements.iter().take(5).map(|s| *s).collect();
                                    format!("[{} ... out of {}]", 
                                        preview.join(", "), 
                                        elements.len())
                                } else {
                                    value.to_string()
                                }
                            } else {
                                value.to_string()
                            }
                        } else {
                            value.to_string()
                        };

                        table.add_row(vec![
                            Cell::new(key).fg(comfy_table::Color::Yellow),
                            Cell::new(type_str).fg(comfy_table::Color::Cyan),
                            Cell::new(display_value).fg(comfy_table::Color::Green),
                        ]);
                    }
                }

                println!("{table}\n");
            }
        } else if let Some(message) = value.get("message").and_then(|m| m.as_str()) {
            println!("{}: {}", "Error".red(), message);
        }
    } else {
        println!("{}", "Failed to parse metadata response".red());
    }
}

/// Displays tensor information in a formatted table
/// 
/// # Arguments
/// 
/// * `json_response` - The JSON response from the server containing tensor information
pub fn display_tensor_info(json_response: &str) {
    if let Ok(value) = serde_json::from_str::<Value>(json_response) {
        if let Some(data) = value.get("data") {
            // Print header
            println!("\n{}", "Tensor Information".cyan().bold());
            println!("{}", "=".repeat(50).bright_black());

            // Create tensor table
            if let Some(tensors) = data.as_array() {
                let mut table = Table::new();
                table.set_header(vec![
                    Cell::new("Name").fg(comfy_table::Color::Yellow).add_attribute(Attribute::Bold),
                    Cell::new("Dimensions").fg(comfy_table::Color::Cyan).add_attribute(Attribute::Bold),
                    Cell::new("Type").fg(comfy_table::Color::Green).add_attribute(Attribute::Bold),
                    Cell::new("Size").fg(comfy_table::Color::Magenta).add_attribute(Attribute::Bold),
                ])
                .load_preset(comfy_table::presets::UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic);

                for tensor in tensors {
                    if let (Some(name), Some(dims), Some(data_type), Some(n_dims)) = (
                        tensor.get("name").and_then(|v| v.as_str()),
                        tensor.get("dims").and_then(|v| v.as_array()),
                        tensor.get("data_type").and_then(|v| v.as_str()),
                        tensor.get("n_dims").and_then(|v| v.as_u64())
                    ) {
                        // Calculate total size
                        let total_size: u64 = dims.iter()
                            .filter_map(|d| d.as_u64())
                            .product();

                        // Format dimensions
                        let dims_str = dims.iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(" × ");

                        table.add_row(vec![
                            Cell::new(name).fg(comfy_table::Color::Yellow),
                            Cell::new(dims_str).fg(comfy_table::Color::Cyan),
                            Cell::new(data_type).fg(comfy_table::Color::Green),
                            Cell::new(total_size.to_string()).fg(comfy_table::Color::Magenta),
                        ]);
                    }
                }

                println!("{table}\n");
            } else {
                println!("{}", "No tensor information available".yellow());
            }
        } else if let Some(message) = value.get("message").and_then(|m| m.as_str()) {
            println!("{}: {}", "Error".red(), message);
        }
    } else {
        println!("{}", "Failed to parse tensor information".red());
    }
} 