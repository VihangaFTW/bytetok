//! Type aliases and shared types for BPE training and encoding.
//!
//! These type aliases provide semantic clarity throughout the codebase.

/// Represents a token identifier in the vocabulary.
///
/// Token IDs are assigned sequentially, starting from 0 for base tokens (e.g., bytes 0-255)
/// and incrementing for each learned merge operation.
pub(crate) type Token = usize;

/// Position of a token in a token sequence.
///
/// Used to index into the doubly-linked list structure during training.
pub(crate) type TextIdx = usize;

/// Frequency count for token pairs during training.
///
/// Tracks how many times a token pair appears in the current sequence.
pub(crate) type TokenFreq = usize;

/// Merge order indicates when a merge rule was learned during training.
///
/// Lower values represent earlier merges (e.g., 0 = first merge, 1 = second merge).
pub(crate) type MergeOrder = usize;

/// A sequence of raw bytes.
///
/// Used for representing text as byte sequences before tokenization.
pub(crate) type ByteSeq = Vec<u8>;

/// A pair of adjacent tokens.
///
/// Used as a key for looking up merge rules during encoding and for
/// tracking pair frequencies during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TokenPair(pub(crate) Token, pub(crate) Token);

