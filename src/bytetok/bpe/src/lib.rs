//! This is a PyO3 extension module providing Python bindings
//! for BPE tokenizer and trainer.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(unused_must_use)]

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod converter;
mod error;
mod tokenizer;
mod trainer;
mod types;

use crate::error::{DecodeError, EncodeError, TokenizerInitError};
use crate::tokenizer::BPETokenizer;
use crate::trainer::BPETrainer;

/// Parses the Python `errors` string into an `ErrorMode`.
///
/// Defaults to `"replace"` when `None` is passed.
fn parse_error_mode(errors: Option<&str>) -> PyResult<error::ErrorMode> {
    let mode_str = errors.unwrap_or("replace");
    mode_str
        .parse::<error::ErrorMode>()
        .map_err(PyErr::new::<PyValueError, _>)
}

/// Converts a `DecodeError` into a Python `ValueError`.
fn decode_err_to_pyerr(e: DecodeError) -> PyErr {
    PyErr::new::<PyValueError, _>(e.to_string())
}

/// Converts an `EncodeError` into a Python `ValueError`.
fn encode_err_to_pyerr(e: EncodeError) -> PyErr {
    PyErr::new::<PyValueError, _>(e.to_string())
}

/// Python wrapper for full BPE tokenizer (regex splitting + BPE encoding).
///
/// The tokenizer provides two encoding modes:
/// - **Pattern-based**: Uses regex to split text before encoding (e.g., GPT-style).
/// - **Raw bytes**: Treats entire text as one chunk (e.g., BasicTokenizer).
///
/// # Example (Python)
///
/// ```python
/// from bytetok.bpe import RustBPETokenizer
///
/// merge_history = [((97, 98), 256)]  # 'a','b' -> 256
/// pattern = r"\S+"
/// tok = RustBPETokenizer(merge_history, pattern, [("<|endoftext|>", 100257)])
///
/// # Pattern-based encoding (with regex splitting).
/// tokens = tok.encode_text("ab cd")       # [256, 99, 100]
/// batch  = tok.encode_texts(["ab", "cd"]) # [[256], [99, 100]]
///
/// # Raw byte encoding (no regex splitting).
/// tokens = tok.encode_bytes("ab cd")       # Different result without splitting.
/// batch  = tok.encode_bytes_batch(["ab", "cd"])
/// ```
#[pyclass]
pub struct RustBPETokenizer {
    tokenizer: BPETokenizer,
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
    ///     special_tokens: Optional list of (token_text, token_id) pairs.
    ///                     Token IDs must not overlap with existing vocab IDs.
    ///
    /// Returns:
    ///     A new RustBPETokenizer instance.
    ///
    /// Raises:
    ///     ValueError: If the regex pattern fails to compile or a special token is invalid.
    #[new]
    #[pyo3(signature = (merge_history, pattern, special_tokens = HashMap::new()))]
    fn new(
        merge_history: Vec<((usize, usize), usize)>,
        pattern: &str,
        special_tokens: HashMap<String, usize>,
    ) -> PyResult<Self> {
        let tokenizer =
            BPETokenizer::new(merge_history, pattern, special_tokens).map_err(|e| match e {
                TokenizerInitError::InvalidPattern(err) => {
                    PyErr::new::<PyValueError, _>(format!("invalid regex pattern: {err}"))
                }
                TokenizerInitError::InvalidSpecialToken(err) => {
                    PyErr::new::<PyValueError, _>(format!("invalid special token: {err}"))
                }
            })?;

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
    ///     Encoded token sequence as a list of token IDs.
    ///
    /// Raises:
    ///     ValueError: If the regex engine fails during text splitting.
    fn encode_text(&self, py: Python<'_>, text: &str) -> PyResult<Vec<usize>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_text(text))
            .map_err(encode_err_to_pyerr)
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
    ///
    /// Raises:
    ///     ValueError: If the regex engine fails during text splitting.
    fn encode_texts(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<usize>>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer.encode_texts(&text_refs)
        })
        .map_err(encode_err_to_pyerr)
    }

    /// Encode a single text string with special token handling.
    ///
    /// Pipeline: special-token split → regex split → UTF-8 bytes → BPE merge.
    /// Special tokens in `allowed_special` are matched literally and emitted
    /// as single token IDs; surrounding text is encoded normally.
    /// The GIL is released for the entire computation.
    ///
    /// Args:
    ///     text: Input text to encode.
    ///     allowed_special: Mapping of special token strings to their token IDs.
    ///
    /// Returns:
    ///     Encoded token sequence as a list of token IDs.
    ///
    /// Raises:
    ///     ValueError: If the regex engine fails during text splitting.
    fn encode_text_with_special(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: HashMap<String, usize>,
    ) -> PyResult<Vec<usize>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_text_with_special(text, allowed_special))
            .map_err(encode_err_to_pyerr)
    }

    /// Encode multiple texts in parallel with special token handling.
    ///
    /// Special tokens in `allowed_special` are matched literally and emitted
    /// as single token IDs; surrounding text is encoded normally via the
    /// regex split → BPE pipeline. Normal segments across all texts are
    /// encoded in a single Rayon batch.
    /// The GIL is released for the entire computation.
    ///
    /// Args:
    ///     texts: List of text strings to encode.
    ///     allowed_special: Mapping of special token strings to their token IDs.
    ///
    /// Returns:
    ///     List of encoded token sequences in input order.
    ///
    /// Raises:
    ///     ValueError: If the regex engine fails during text splitting.
    fn encode_texts_with_special(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        allowed_special: HashMap<String, usize>,
    ) -> PyResult<Vec<Vec<usize>>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer.encode_texts_with_special(&text_refs, allowed_special)
        })
        .map_err(encode_err_to_pyerr)
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
        py.detach(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer.encode_bytes_batch(&text_refs)
        })
    }

    /// Decode a token sequence back into a UTF-8 string.
    ///
    /// The GIL is released for the entire computation.
    ///
    /// Args:
    ///     tokens: List of token IDs to decode.
    ///     errors: How to handle invalid UTF-8 — "strict" raises ValueError,
    ///             "replace" substitutes U+FFFD (default: "replace").
    ///
    /// Returns:
    ///     Decoded string.
    ///
    /// Raises:
    ///     ValueError: If a token ID is unknown, or UTF-8 is invalid under "strict" mode,
    ///                 or `errors` is not a recognised mode.
    #[pyo3(signature = (tokens, errors=None))]
    fn decode_tokens(
        &self,
        py: Python<'_>,
        tokens: Vec<usize>,
        errors: Option<&str>,
    ) -> PyResult<String> {
        let mode = parse_error_mode(errors)?;
        let tokenizer = &self.tokenizer;

        let result = py.detach(move || tokenizer.decode_tokens(&tokens, mode));
        result.map_err(decode_err_to_pyerr)
    }

    /// Decode multiple token sequences in parallel.
    ///
    /// The GIL is released for the entire computation.
    ///
    /// Args:
    ///     token_seqs: List of token sequences to decode.
    ///     errors: How to handle invalid UTF-8 — "strict" or "replace" (default: "replace").
    ///
    /// Returns:
    ///     List of decoded strings in input order.
    ///
    /// Raises:
    ///     ValueError: If any token ID is unknown, or UTF-8 is invalid under "strict" mode.
    #[pyo3(signature = (token_seqs, errors = None))]
    fn decode_tokens_batch(
        &self,
        py: Python<'_>,
        token_seqs: Vec<Vec<usize>>,
        errors: Option<&str>,
    ) -> PyResult<Vec<String>> {
        let mode = parse_error_mode(errors)?;
        let tokenizer = &self.tokenizer;
        let refs: Vec<&[usize]> = token_seqs.iter().map(|v| v.as_slice()).collect();
        let result = py.detach(move || tokenizer.decode_tokens_batch(&refs, mode));
        result.map_err(decode_err_to_pyerr)
    }

    /// Returns the vocabulary size (number of tokens).
    ///
    /// Returns:
    ///     Total number of tokens in the vocabulary.
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
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
        self.trainer.encodings()
    }

    /// Returns the complete history of merge operations.
    ///
    /// Returns:
    ///     A list of tuples: ((left_token, right_token), merged_token).
    ///     The order represents the sequence in which merges were learned.
    fn get_merge_history(&self) -> Vec<((usize, usize), usize)> {
        self.trainer.merge_history()
    }

    /// Prints the current state of the trainer for debugging.
    ///
    /// Outputs the current token sequence and the top 5 most frequent pairs.
    fn print_state(&self) {
        self.trainer.print_state();
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
/// * `RustBPETokenizer` - Full encoding/decoding pipeline (regex splitting + BPE merge + special tokens).
#[pymodule]
fn _bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPETokenizer>()?;
    m.add_class::<RustBPETrainer>()?;
    Ok(())
}
