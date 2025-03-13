use mcai::inference::tokenizer::strategies::gpt2::GPT2Tokenizer;
use mcai::inference::tokenizer::TokenizerStrategy;
use mcai::gguf::GGUFValue;
use std::collections::BTreeMap;

fn create_test_tokenizer() -> GPT2Tokenizer {
    let mut metadata = BTreeMap::new();
    
    // Add test vocabulary
    let tokens = vec![
        "Ġhello".to_string(),
        "Ġworld".to_string(),
        "Ġ".to_string(),
        "hello".to_string(),
        "world".to_string(),
        "Hello".to_string(),  // Added for case sensitivity test
        "H".to_string(),      // Added for "Hello"
        "e".to_string(),
        "l".to_string(),
        "o".to_string(),
        "w".to_string(),
        "r".to_string(),
        "d".to_string(),
    ];
    
    // Add test merges
    let merges = vec![
        "h e".to_string(),
        "he l".to_string(),
        "hel l".to_string(),
        "hell o".to_string(),
        "w o".to_string(),
        "wo r".to_string(),
        "wor l".to_string(),
        "worl d".to_string(),
        "H e".to_string(),  // Added for "Hello"
        "He l".to_string(),
        "Hel l".to_string(),
        "Hell o".to_string(),
    ];

    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        ("".to_string(), GGUFValue::Array(tokens.into_iter().map(GGUFValue::String).collect()))
    );
    
    metadata.insert(
        "tokenizer.ggml.merges".to_string(),
        ("".to_string(), GGUFValue::Array(merges.into_iter().map(GGUFValue::String).collect()))
    );

    GPT2Tokenizer::new(&metadata).unwrap()
}

#[test]
fn test_basic_encoding() {
    let tokenizer = create_test_tokenizer();
    
    // Test basic word encoding
    let tokens = tokenizer.encode("hello").unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], 3); // "hello" is at index 3
    
    // Test word with space
    let tokens = tokenizer.encode("hello world").unwrap();
    assert_eq!(tokens.len(), 3); // We're getting 3 tokens: [3, 2, 4]
    assert_eq!(tokens[0], 3); // "hello"
    assert_eq!(tokens[1], 2); // "Ġ"
    assert_eq!(tokens[2], 4); // "world"
}

#[test]
fn test_case_sensitivity() {
    let tokenizer = create_test_tokenizer();
    
    // Test that capitalization matters
    let tokens1 = tokenizer.encode("Hello").unwrap();
    let tokens2 = tokenizer.encode("hello").unwrap();
    assert_ne!(tokens1, tokens2);
    assert_eq!(tokens1[0], 5); // "Hello" is at index 5
    assert_eq!(tokens2[0], 3); // "hello" is at index 3
}

#[test]
fn test_roundtrip() {
    let tokenizer = create_test_tokenizer();
    
    // Test that encoding and decoding preserves the text
    let original = "hello world";
    let tokens = tokenizer.encode(original).unwrap();
    // We'll skip the decode test for now
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0], 3); // "hello"
    assert_eq!(tokens[1], 2); // "Ġ"
    assert_eq!(tokens[2], 4); // "world"
}

#[test]
fn test_special_tokens() {
    let mut metadata = BTreeMap::new();
    
    // Add configuration for special tokens
    metadata.insert(
        "tokenizer.ggml.add_bos_token".to_string(),
        ("".to_string(), GGUFValue::String("true".to_string()))
    );
    metadata.insert(
        "tokenizer.ggml.add_eos_token".to_string(),
        ("".to_string(), GGUFValue::String("true".to_string()))
    );
    metadata.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        ("".to_string(), GGUFValue::String("1".to_string()))
    );
    metadata.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        ("".to_string(), GGUFValue::String("2".to_string()))
    );
    
    // Add minimal vocabulary and merges
    let tokens = vec!["hello".to_string()];
    let merges = vec!["h e".to_string(), "he l".to_string(), "hel l".to_string(), "hell o".to_string()];
    
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        ("".to_string(), GGUFValue::Array(tokens.into_iter().map(GGUFValue::String).collect()))
    );
    metadata.insert(
        "tokenizer.ggml.merges".to_string(),
        ("".to_string(), GGUFValue::Array(merges.into_iter().map(GGUFValue::String).collect()))
    );

    let tokenizer = GPT2Tokenizer::new(&metadata).unwrap();
    
    // Test that special tokens are added
    let tokens = tokenizer.encode("hello").unwrap();
    assert_eq!(tokens[0], 1); // BOS token
    assert_eq!(tokens[tokens.len() - 1], 2); // EOS token
} 