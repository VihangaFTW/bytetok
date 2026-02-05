//! Fast BPE (Byte-Pair Encoding) trainer using Algorithm 2
//! from "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
//!
//! This is a PyO3 extension module providing Python bindings.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

use pyo3::prelude::*;
mod types;

mod encoder;

pub mod trainer;
use trainer::BPETrainer;

use crate::encoder::BPEEncoder;

/// Python wrapper for BPE trainer.
///
/// Provides an efficient Rust-based implementation of Byte Pair Encoding training
/// that can be used from Python via PyO3 bindings.
///
/// # Example (Python)
///
/// ```python
/// from bytetok._bpe_rs import RustBPETrainer
///
/// # Initialize with token sequence and next token ID
/// trainer = RustBPETrainer([0, 1, 0, 1, 2], next_token_id=3)
///
/// # Train with 10 merges
/// trainer.train(10)
///
/// # Get final tokens
/// tokens = trainer.get_tokens()
///
/// # Get merge history
/// merges = trainer.get_merge_history()
/// ```
#[pyclass]
pub struct RustBPETrainer {
    trainer: BPETrainer,
}

#[pymethods]
impl RustBPETrainer {
    /// Creates a new BPE trainer from an initial token sequence.
    ///
    /// Args:
    ///     tokens: Initial sequence of tokens (e.g., byte values 0-255).
    ///     next_token_id: The token ID to use for the first merge (e.g., 256 for bytes).
    ///
    /// Returns:
    ///     A new RustBPETrainer instance.
    #[new]
    fn new(tokens: Vec<usize>, next_token_id: usize) -> Self {
        RustBPETrainer {
            trainer: BPETrainer::new(&tokens, next_token_id),
        }
    }

    /// Trains the BPE model by performing the specified number of merges.
    ///
    /// Args:
    ///     num_merges: Number of merge operations to perform.
    ///
    /// Note:
    ///     Training will stop early if no more pairs can be merged.
    fn train(&mut self, num_merges: usize) {
        self.trainer.train(num_merges);
    }

    /// Performs a single merge operation on the most frequent token pair.
    ///
    /// Returns:
    ///     True if a merge was performed, False if no pairs remain to merge.
    fn merge_step(&mut self) -> bool {
        self.trainer.merge_step()
    }

    /// Returns the current token sequence after all merges.
    ///
    /// Returns:
    ///     A list of token IDs representing the encoded sequence.
    fn get_tokens(&self) -> Vec<usize> {
        self.trainer.get_encodings()
    }

    /// Returns the complete history of merge operations.
    ///
    /// Returns:
    ///     A list of tuples: ((left_token, right_token), merged_token).
    ///     The order represents the sequence in which merges were learned.
    fn get_merge_history(&self) -> Vec<((usize, usize), usize)> {
        self.trainer.get_merge_history()
    }

    /// Prints the current state of the trainer for debugging.
    ///
    /// Outputs the current token sequence and the top 5 most frequent pairs.
    fn print_state(&self) {
        self.trainer.print_state();
    }
}

/// Python wrapper for BPE encoder.
///
/// Applies learned BPE merge rules to encode token sequences efficiently.
/// The encoder uses merge rules learned during training to compress sequences.
///
/// # Example (Python)
///
/// ```python
/// from bytetok._bpe_rs import RustBPEEncoder
///
/// # Create encoder from merge history
/// merge_history = [((0, 1), 256), ((256, 0), 257)]
/// encoder = RustBPEEncoder(merge_history)
///
/// # Encode token sequence
/// tokens = [0, 1, 0]
/// encoded = encoder.encode(tokens)  # Returns [257]
///
/// # Check if pair can be merged
/// can_merge = encoder.can_merge(0, 1)  # Returns True
/// ```
#[pyclass]
pub struct RustBPEEncoder {
    encoder: BPEEncoder,
}

#[pymethods]
impl RustBPEEncoder {
    /// Creates a new BPE encoder from merge history.
    ///
    /// Args:
    ///     merge_history: List of merge rules as ((left_token, right_token), merged_token).
    ///                   The order determines merge priority (earlier = higher priority).
    ///
    /// Returns:
    ///     A new RustBPEEncoder instance.
    #[new]
    fn new(merge_history: Vec<((usize, usize), usize)>) -> Self {
        RustBPEEncoder {
            encoder: BPEEncoder::new(merge_history),
        }
    }

    /// Encodes a token sequence using learned BPE merge rules.
    ///
    /// Args:
    ///     tokens: Input sequence of tokens to encode.
    ///
    /// Returns:
    ///     Encoded token sequence with merges applied.
    fn encode(&self, tokens: Vec<usize>) -> Vec<usize> {
        self.encoder.encode(tokens)
    }

    /// Checks if a token pair has a learned merge rule.
    ///
    /// Args:
    ///     left: Left token in the pair.
    ///     right: Right token in the pair.
    ///
    /// Returns:
    ///     True if a merge rule exists for this pair, False otherwise.
    fn can_merge(&self, left: usize, right: usize) -> bool {
        self.encoder.can_merge(left, right)
    }

    /// Returns the total number of merge rules.
    ///
    /// Returns:
    ///     The number of learned merge rules in this encoder.
    fn num_merges(&self) -> usize {
        self.encoder.num_merges()
    }
}

/// PyO3 module definition for BPE implementations.
///
/// This module exposes Rust-based BPE training and encoding functionality to Python.
/// The function name must match the library name specified in Cargo.toml.
///
/// # Exports
///
/// * `RustBPETrainer` - Fast BPE training implementation.
/// * `RustBPEEncoder` - Efficient BPE encoding using learned rules.
#[pymodule]
fn _bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPETrainer>()?;
    m.add_class::<RustBPEEncoder>()?;
    Ok(())
}
