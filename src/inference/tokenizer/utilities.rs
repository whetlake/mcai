use std::collections::BTreeMap;
use once_cell::sync::Lazy;
use regex::Regex;

/// The regex pattern used by tiktoken for initial text splitting
pub static PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)('s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+").unwrap()
});

/// Mapping from bytes to unicode strings, avoiding whitespace/control characters
pub static BYTES_TO_UNICODE: Lazy<BTreeMap<u8, char>> = Lazy::new(|| {
    let mut bs: Vec<u8> = Vec::new();
    // Range 33-126 is printable ASCII
    bs.extend(33..=126);
    // Range 161-172 + 174-255 is printable Latin-1 Supplement
    bs.extend(161..=172);
    bs.extend(174..=255);
    
    let mut cs = bs.clone();
    let mut n = 0u16;
    
    // Add remaining bytes, mapping to Unicode private use area
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push((n + 256) as u8);
            n += 1;
        }
    }
    
    // Create the mapping
    bs.into_iter()
        .zip(cs.into_iter().map(|n| char::from_u32(n as u32).unwrap()))
        .collect()
});

/// Reverse mapping from unicode strings to bytes
pub static UNICODE_TO_BYTES: Lazy<BTreeMap<char, u8>> = Lazy::new(|| {
    BYTES_TO_UNICODE.iter().map(|(&k, &v)| (v, k)).collect()
});
