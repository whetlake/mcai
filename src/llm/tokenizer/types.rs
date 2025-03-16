use std::fmt;
use std::collections::BTreeMap;
use crate::gguf::GGUFValue;

/// Represents different types of tokenizers supported by the system
#[derive(Debug, Clone, PartialEq)]
pub enum TokenizerType {
    /// BERT style WordPiece tokenizer
    BERT,
    /// RoBERTa style BPE tokenizer
    RoBERTa,
    /// GPT-2 style BPE tokenizer
    GPT2,
    /// LLaMA style SentencePiece tokenizer
    LLaMA,
    /// Mistral style SentencePiece tokenizer
    Mistral,
    /// Falcon style BPE tokenizer
    Falcon,
    /// MPT style BPE tokenizer
    MPT,
    /// T5 style SentencePiece tokenizer
    T5,
    /// BART style BPE tokenizer
    BART,
    /// XLM-RoBERTa style SentencePiece tokenizer
    XLMRoBERTa,
    /// CodeLlama style SentencePiece tokenizer
    CodeLlama,
    /// BLOOM style BPE tokenizer
    BLOOM,
    /// OPT style BPE tokenizer
    OPT,
    /// Generic fallback tokenizer
    Generic,
}

impl fmt::Display for TokenizerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerType::BERT => write!(f, "BERT"),
            TokenizerType::RoBERTa => write!(f, "RoBERTa"),
            TokenizerType::GPT2 => write!(f, "GPT-2"),
            TokenizerType::LLaMA => write!(f, "LLaMA"),
            TokenizerType::Mistral => write!(f, "Mistral"),
            TokenizerType::Falcon => write!(f, "Falcon"),
            TokenizerType::MPT => write!(f, "MPT"),
            TokenizerType::T5 => write!(f, "T5"),
            TokenizerType::BART => write!(f, "BART"),
            TokenizerType::XLMRoBERTa => write!(f, "XLM-RoBERTa"),
            TokenizerType::CodeLlama => write!(f, "CodeLlama"),
            TokenizerType::BLOOM => write!(f, "BLOOM"),
            TokenizerType::OPT => write!(f, "OPT"),
            TokenizerType::Generic => write!(f, "Generic"),
        }
    }
}

/// Determines the tokenizer type from architecture and metadata
pub fn determine_tokenizer_type(architecture: &str, metadata: &BTreeMap<String, (String, GGUFValue)>) -> TokenizerType {
    // First check if tokenizer model is explicitly specified in metadata
    if let Some((_, value)) = metadata.get("tokenizer.ggml.model") {
        println!("Found tokenizer model in metadata: {:?}", value.to_string());
        match value.to_string().to_lowercase().as_str() {
            "gpt2" => return TokenizerType::GPT2,
            "llama" => return TokenizerType::LLaMA,
            "mistral" => return TokenizerType::Mistral,
            "falcon" => return TokenizerType::Falcon,
            "mpt" => return TokenizerType::MPT,
            "codellama" => return TokenizerType::CodeLlama,
            "bert" => return TokenizerType::BERT,
            "roberta" => return TokenizerType::RoBERTa,
            "t5" => return TokenizerType::T5,
            "bart" => return TokenizerType::BART,
            "xlm-roberta" => return TokenizerType::XLMRoBERTa,
            "bloom" => return TokenizerType::BLOOM,
            "opt" => return TokenizerType::OPT,
            _ => {}
        }
    }

    // If not specified, infer from architecture
    match architecture.to_lowercase().as_str() {
        "gpt2" | "gpt-2" => TokenizerType::GPT2,
        "llama" | "llama2" => TokenizerType::LLaMA,
        "mistral" => TokenizerType::Mistral,
        "falcon" => TokenizerType::Falcon,
        "mpt" => TokenizerType::MPT,
        "codellama" => TokenizerType::CodeLlama,
        "bert" => TokenizerType::BERT,
        "roberta" => TokenizerType::RoBERTa,
        "t5" => TokenizerType::T5,
        "bart" => TokenizerType::BART,
        "xlm-roberta" => TokenizerType::XLMRoBERTa,
        "bloom" => TokenizerType::BLOOM,
        "opt" => TokenizerType::OPT,
        _ => TokenizerType::Generic,
    }
} 