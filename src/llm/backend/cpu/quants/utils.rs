/// Utilities for working with quantized formats

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
} 