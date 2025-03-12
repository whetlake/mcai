use std::collections::BTreeMap;
use once_cell::sync::Lazy;
use regex::Regex;

/// The regex pattern used by tiktoken for initial text splitting
pub static PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)('s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+").unwrap()
});

/// Mapping from bytes to unicode strings, avoiding whitespace/control characters
pub static BYTES_TO_UNICODE: Lazy<BTreeMap<u8, char>> = Lazy::new(|| {
    let mut mapping = BTreeMap::new();
    
    // First collect the list of bytes we want to map directly
    let mut bs = Vec::new();
    // ASCII printable characters (33-126)
    bs.extend(33..=126);
    // Latin-1 Supplement part 1 (161-172)
    bs.extend(161..=172);
    // Latin-1 Supplement part 2 (174-255)
    bs.extend(174..=255);
    
    // First map all the bytes in bs to themselves
    for &b in &bs {
        mapping.insert(b, char::from_u32(b as u32).unwrap());
    }
    
    // Then map all remaining bytes (0-255) to Unicode private use area (starting at 256)
    let mut n = 0;
    for b in 0..=255 {
        if !bs.contains(&b) {
            mapping.insert(b, char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }
    
    mapping
});

/// Reverse mapping from unicode strings to bytes
pub static UNICODE_TO_BYTES: Lazy<BTreeMap<char, u8>> = Lazy::new(|| {
    BYTES_TO_UNICODE.iter().map(|(&k, &v)| (v, k)).collect()
});
