//! Type aliases for BPE training and encoding.
//!
//! These type aliases provide semantic clarity and type safety throughout the codebase.

/// Represents a token identifier in the vocabulary.
///
/// Token IDs are assigned sequentially, starting from 0 for base tokens (e.g., bytes 0-255)
/// and incrementing for each learned merge operation.
pub type Token = usize;

/// Position of a token in a token sequence.
///
/// Used to index into the doubly-linked list structure during training.
pub type TextIdx = usize;

/// Frequency count for token pairs during training.
///
/// Tracks how many times a token pair appears in the current sequence.
pub type TokenFreq = usize;
