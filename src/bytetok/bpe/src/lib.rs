//! Fast BPE (Byte-Pair Encoding) trainer using Algorithm 2
//! from "A Formal Perspective on Byte-Pair Encoding"
//!
//! This is a PyO3 extension module providing Python bindings.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

use pyo3::prelude::*;
use rayon::prelude::*;

mod encoder;
mod tokenizer;
mod trainer;
mod types;
use trainer::BPETrainer;

use crate::encoder::BPEEncoder;
use crate::tokenizer::Tokenizer;

/// Python wrapper for full BPE tokenizer (regex splitting + BPE encoding).
///
/// Unlike `RustBPEEncoder` which only applies merge rules to pre-tokenized
/// integer sequences, this class accepts raw text strings and performs the
/// entire pipeline in Rust: regex splitting → byte conversion → BPE merge.
///
/// This eliminates per-chunk Python/Rust FFI overhead and enables true
/// parallel encoding via Rayon (with the GIL fully released).
///
/// # Example (Python)
///
/// ```python
/// from bytetok.bpe import RustBPETokenizer
///
/// merge_history = [((97, 98), 256)]  # 'a','b' -> 256
/// pattern = r"\S+"
/// tok = RustBPETokenizer(merge_history, pattern)
///
/// tokens = tok.encode_text("ab cd")       # [256, 99, 100]
/// batch  = tok.encode_texts(["ab", "cd"]) # [[256], [99, 100]]
/// ```
#[pyclass]
pub struct RustBPETokenizer {
    tokenizer: Tokenizer,
}

#[pymethods]
impl RustBPETokenizer {
    /// Creates a new tokenizer from merge history and a regex split pattern.
    ///
    /// Args:
    ///     merge_history: List of merge rules as ((left_token, right_token), merged_token).
    ///                    Order determines merge priority (earlier = higher priority).
    ///     pattern: Regex pattern string for splitting text into chunks.
    ///              Must be compatible with fancy-regex (supports lookaheads
    ///              but NOT possessive quantifiers like ?+ or ++).
    ///
    /// Returns:
    ///     A new RustBPETokenizer instance.
    ///
    /// Raises:
    ///     ValueError: If the regex pattern fails to compile.
    #[new]
    fn new(merge_history: Vec<((usize, usize), usize)>, pattern: &str) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;

        let tokenizer = Tokenizer::new(merge_history, pattern)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid regex pattern: {e}")))?;

        Ok(Self { tokenizer })
    }
    /// Encode a single text string entirely in Rust.
    ///
    /// Pipeline: regex split → UTF-8 bytes → BPE merge.
    /// The GIL is released for the entire computation.
    ///
    /// Args:
    ///     text: Input text to encode.
    ///
    /// Returns:
    ///     Encoded token sequence.
    fn encode_text(&self, py: Python<'_>, text: &str) {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_text(text));
    }
    /// Encode multiple texts in parallel entirely in Rust.
    ///
    /// Each text is independently split and BPE-encoded on a Rayon worker
    /// thread. The GIL is released for the entire computation.
    ///
    /// Args:
    ///     texts: List of text strings to encode.
    ///
    /// Returns:
    ///     List of encoded token sequences in input order.
    fn encode_texts(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<usize>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_texts(&texts))
    }
    /// Encode a single text as raw bytes → BPE (no regex splitting).
    ///
    /// Useful for tokenizers that operate on the full byte stream.
    /// The GIL is released for the entire computation.
    ///
    /// Args:
    ///     text: Input text to encode as a single byte chunk.
    ///
    /// Returns:
    ///     Encoded token sequence.
    fn encode_bytes(&self, py: Python<'_>, text: &str) -> Vec<usize> {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_bytes(text))
    }
    /// Encode multiple texts as raw bytes → BPE in parallel (no regex splitting).
    ///
    /// The GIL is released for the entire computation.
    ///
    /// Args:
    ///     texts: List of text strings to encode.
    ///
    /// Returns:
    ///     List of encoded token sequences in input order.
    fn encode_bytes_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<usize>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_bytes_batch(&texts))
    }
}

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
    ///     This method releases the Python GIL while the Rust training loop runs.
    ///     Training will stop early if no more pairs can be merged.
    fn train(&mut self, py: Python<'_>, num_merges: usize) {
        let trainer = &mut self.trainer;
        // allow rust code to run without the GIL
        py.detach(move || {
            trainer.train(num_merges);
        })
    }

    /// Performs a single merge operation on the most frequent token pair.
    ///
    /// Returns:
    ///     True if a merge was performed, False if no pairs remain to merge.
    ///
    /// Note:
    ///     This method releases the Python GIL while the merge step runs.
    fn merge_step(&mut self, py: Python<'_>) -> bool {
        let trainer = &mut self.trainer;
        py.detach(move || trainer.merge_step())
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
    ///
    /// Note:
    ///     This method releases the Python GIL while the Rust encoder executes.
    fn encode(&self, py: Python<'_>, tokens: Vec<usize>) -> Vec<usize> {
        let encoder = &self.encoder;
        py.detach(move || encoder.encode(tokens))
    }

    /// Encodes many token sequences in parallel using Rayon.
    ///
    /// Args:
    ///     inputs: List of token sequences to encode.
    ///
    /// Returns:
    ///     List of encoded token sequences in input order.
    ///
    /// Note:
    ///     This method releases the Python GIL while Rayon workers execute.
    fn encode_many(&self, py: Python<'_>, inputs: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let encoder = &self.encoder;
        py.detach(move || {
            inputs
                .into_par_iter()
                .map(|tokens| encoder.encode(tokens))
                .collect()
        })
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
    m.add_class::<RustBPETokenizer>()?;
    m.add_class::<RustBPETrainer>()?;
    m.add_class::<RustBPEEncoder>()?;
    Ok(())
}
