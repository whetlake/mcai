/// Utilities for working with quantized formats
use std::sync::Arc;
use std::thread;
use std::error::Error;

/// Convert a half-precision float (16-bit) to a single-precision float (32-bit)
///
/// # Arguments
/// * `bytes` - A slice of 2 bytes representing the half-precision float in little-endian format
///
/// # Returns
/// * The converted 32-bit floating point value
///
/// IEEE 754 half-precision binary floating-point format:
/// Sign bit: 1 bit
/// Exponent width: 5 bits
/// Significand precision: 11 bits (10 explicitly stored)
pub fn f16_to_f32(bytes: &[u8]) -> f32 {
    // Convert bytes to a u16 in little-endian format
    let half = u16::from_le_bytes([bytes[0], bytes[1]]);

    // Extract components
    let sign = (half >> 15) & 1;
    let exponent = (half >> 10) & 0x1F;
    let mantissa = half & 0x3FF;

    // Special cases
    match exponent {
        0 => {
            // Zero or subnormal number
            if mantissa == 0 {
                return if sign == 0 { 0.0 } else { -0.0 };
            }
            // Subnormal number
            let sign_f = if sign == 0 { 1.0 } else { -1.0 };
            return sign_f * mantissa as f32 * 2.0f32.powi(-24);
        }
        0x1F => {
            // Infinity or NaN
            if mantissa == 0 {
                return if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY };
            }
            return f32::NAN;
        }
        _ => {
            // Normal number
            let sign_f = if sign == 0 { 1.0 } else { -1.0 };
            let exponent_f = 2.0f32.powi((exponent as i32) - 15);
            let mantissa_f = 1.0 + (mantissa as f32) / 1024.0;
            return sign_f * exponent_f * mantissa_f;
        }
    }
}

/// Checks if there is enough data available for a dequantization operation
///
/// # Arguments
/// * `data` - The data slice to check
/// * `offset` - Current position in the data
/// * `bytes_needed` - Number of bytes required
/// * `format_name` - Name of the format for error message
///
/// # Returns
/// * Ok(()) if enough data is available, otherwise Err with descriptive message
pub fn check_data_availability(
    data: &[u8],
    offset: usize,
    bytes_needed: usize,
    format_name: &str
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if offset + bytes_needed > data.len() {
        let available = if data.len() > offset {
            data.len() - offset
        } else {
            0
        };
        
        Err(format!("Not enough data to read {} values. Need {} bytes, but only have {}", 
                   format_name, bytes_needed, available).into())
    } else {
        Ok(())
    }
}

/// Processes data in parallel using multiple threads when beneficial
///
/// # Arguments
/// * `data` - The data to process
/// * `num_blocks` - Total number of blocks to process
/// * `elements_per_block` - Number of elements in each processing block
/// * `bytes_per_block` - Size of each block in bytes
/// * `threshold_blocks` - Minimum number of blocks to use multithreading
/// * `num_threads` - Maximum number of threads to use (0 for auto-detection)
/// * `result` - Pre-allocated buffer for results
/// * `process_fn` - Function to process a range of blocks
///
/// This handles the decision of whether to use single or multi-threaded processing,
/// the division of work, and thread synchronization.
pub fn parallel_process<F>(
    data: &[u8],
    num_blocks: usize,
    elements_per_block: usize,
    bytes_per_block: usize,
    threshold_blocks: usize,
    num_threads: usize,
    result: &mut [f32],
    process_fn: F
) where
    F: Fn(&[u8], usize, usize, usize, usize, &mut [f32]) + Send + Sync + Copy + 'static,
{
    // Determine the actual number of threads to use
    let actual_threads = if num_threads == 0 {
        // Auto-detect available threads
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        num_threads
    };
    
    // Determine whether to use single-threaded or multi-threaded approach
    if actual_threads <= 1 || num_blocks <= threshold_blocks {
        // Single-threaded approach for small workloads
        process_fn(data, 0, 0, num_blocks, bytes_per_block, result);
    } else {
        // Multi-threaded approach for larger workloads
        
        // Calculate blocks per thread
        let num_threads_to_use = std::cmp::min(actual_threads, num_blocks);
        let blocks_per_thread = (num_blocks + num_threads_to_use - 1) / num_threads_to_use;
        
        // Create vector to hold thread handles
        let mut handles = Vec::with_capacity(num_threads_to_use);
        
        // Convert to Arc for thread safety
        let data_arc = Arc::new(data.to_vec());
        
        // Spawn threads to process blocks in parallel
        for thread_idx in 0..num_threads_to_use {
            let start_block = thread_idx * blocks_per_thread;
            let end_block = std::cmp::min(start_block + blocks_per_thread, num_blocks);
            
            if start_block >= end_block {
                continue;
            }
            
            // Clone the Arc to share data reference
            let thread_data = Arc::clone(&data_arc);
            
            // Get a mutable slice of the result for this thread's portion
            let thread_result = unsafe {
                let ptr = result.as_mut_ptr().add(start_block * elements_per_block);
                std::slice::from_raw_parts_mut(ptr, (end_block - start_block) * elements_per_block)
            };
            
            // Spawn thread
            let handle = thread::spawn(move || {
                process_fn(&thread_data, 0, start_block, end_block, bytes_per_block, thread_result);
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_to_float() {
        // Test zero
        assert_eq!(f16_to_f32(&[0, 0]), 0.0);
        
        // Test one
        assert_eq!(f16_to_f32(&[0, 60]), 1.0);
        
        // Test negative number
        assert_eq!(f16_to_f32(&[0, 0xC0]), -2.0);
        
        // Test common value (5.5)
        // IEEE 754 half-precision for 5.5 = 0x4580 (sign=0, exp=1+14=15, mantissa=0b1011000000=0x180)
        // In little-endian: [0x80, 0x45]
        assert_eq!(f16_to_f32(&[0x80, 0x45]), 5.5);
    }
    
    #[test]
    fn test_check_data_availability() {
        let data = vec![0u8; 100];
        
        // Should be OK 
        assert!(check_data_availability(&data, 0, 100, "TEST").is_ok());
        assert!(check_data_availability(&data, 50, 50, "TEST").is_ok());
        
        // Should fail
        let err = check_data_availability(&data, 50, 51, "TEST").unwrap_err();
        assert!(err.to_string().contains("Need 51 bytes, but only have 50"));
        
        let err = check_data_availability(&data, 100, 1, "TEST").unwrap_err();
        assert!(err.to_string().contains("Need 1 bytes, but only have 0"));
        
        let err = check_data_availability(&data, 101, 1, "TEST").unwrap_err();
        assert!(err.to_string().contains("Need 1 bytes, but only have 0"));
    }
} 