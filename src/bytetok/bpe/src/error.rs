use std::fmt;

use crate::types::Token;

/// Controls how UTF-8 decoding errors are handled.
///
/// Mirrors Python's `bytes.decode(errors=...)` semantics.
/// Unknown token IDs always produce errors regardless of mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorMode {
    /// Raise an error on invalid UTF-8 like Python's "strict".
    Strict,
    /// Replace invalid UTF-8 sequences with U+FFFD (like Python's "replace").
    Replace,
}

impl ErrorMode {
    /// Parse from a Python-style error mode string.
    ///
    /// Returns `None` for unrecognised values.
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "strict" => Some(Self::Strict),
            "replace" => Some(Self::Replace),
            _ => None,
        }
    }
}

/// Errors that can occur during token decoding.
#[derive(Debug)]
pub(crate) enum DecodeError {
    /// Token ID not found in vocabulary.
    UnknownToken(Token),
    /// Decoded bytes are not valid UTF-8.
    InvalidUtf8(std::string::FromUtf8Error),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownToken(t) => write!(f, "unknown token id: {t}"),
            Self::InvalidUtf8(e) => write!(f, "invalid UTF-8 in decoded bytes: {e}"),
        }
    }
}

/// Errors that can occur during text encoding.
#[derive(Debug)]
pub(crate) enum EncodeError {
    /// Regex engine failed during text splitting (e.g. backtracking limit exceeded).
    RegexMatch(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegexMatch(msg) => write!(f, "regex match failed: {msg}"),
        }
    }
}
/// Errors that can occur when processing special tokens.
#[derive(Debug)]
pub(crate) enum SpecialTokenError {
    /// Token Id already exists in vocabulary.
    IllegalToken(Token)
}

impl fmt::Display for SpecialTokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IllegalToken(tok) => write!(f, "token already exists: {tok}"),
        }
    }
}


/// Errors that can occur when initializng a Tokenizer.
#[derive(Debug)]
pub(crate) enum TokenizerInitError{
    InvalidPattern(fancy_regex::Error),
    InvalidSpecialToken(SpecialTokenError)
}

impl From<fancy_regex::Error> for TokenizerInitError{
    fn from(e: fancy_regex::Error) -> Self {
        Self::InvalidPattern(e)
    }
}

impl From<SpecialTokenError> for TokenizerInitError{
    fn from(e: SpecialTokenError) -> Self {
        Self::InvalidSpecialToken(e)
    }
}



