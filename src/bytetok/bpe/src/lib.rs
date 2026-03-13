//! PyO3 extension module providing Python bindings for the BPE tokenizer and trainer.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(unused_must_use)]

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

mod converter;
mod error;
mod tokenizer;
mod trainer;
mod types;

use crate::error::{DecodeError, EncodeError, TokenizerInitError};
use crate::tokenizer::BPETokenizer;
use crate::trainer::BPETrainer;
use crate::types::{Count, Token};

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
///
/// - **Pattern-based**: Uses regex to split text before encoding (e.g., GPT-style).
/// - **Raw bytes**: Treats entire text as one chunk (e.g., BasicTokenizer).
///
/// # Example
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
    /// Creates a tokenizer from merge history and a regex split pattern.
    ///
    /// `merge_history` is a list of merge rules in `((left_token, right_token),
    /// merged_token)` form. Earlier entries have higher merge priority.
    ///
    /// `pattern` must be compatible with `fancy-regex`, which supports features
    /// such as lookaheads but not possessive quantifiers like `?+` or `++`.
    ///
    /// `special_tokens` maps literal token text to token IDs. Special token IDs
    /// must not overlap with existing vocabulary IDs.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if the regex pattern fails to compile or a special
    /// token definition is invalid.
    #[new]
    #[pyo3(signature = (merge_history, pattern, special_tokens = HashMap::new()))]
    fn new(
        merge_history: Vec<((u32, u32), u32)>,
        pattern: &str,
        special_tokens: HashMap<String, u32>,
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
    /// Encodes one string entirely in Rust.
    ///
    /// The encoding pipeline is regex split, UTF-8 byte conversion, and then
    /// BPE merging. The GIL is released for the entire computation.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if the regex engine fails during text splitting.
    fn encode_text(&self, py: Python<'_>, text: &str) -> PyResult<Vec<Token>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_text(text))
            .map_err(encode_err_to_pyerr)
    }
    /// Encodes multiple strings in parallel entirely in Rust.
    ///
    /// Each input is split and BPE-encoded on a Rayon worker thread. Results
    /// are returned in input order. The GIL is released for the entire
    /// computation.
    ///
    /// If `show_progress` is `true`, batch encoding displays a progress bar.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if the regex engine fails during text splitting.
    #[pyo3(signature = (texts, show_progress = true))]
    fn encode_texts(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        show_progress: bool,
    ) -> PyResult<Vec<Vec<Token>>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer.encode_texts(&text_refs, show_progress)
        })
        .map_err(encode_err_to_pyerr)
    }

    /// Encodes one string with special token handling.
    ///
    /// The encoding pipeline first matches literal special tokens from
    /// `allowed_special`, then applies regex splitting, UTF-8 byte conversion,
    /// and BPE merging to the remaining text. The GIL is released for the
    /// entire computation.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if the regex engine fails during text splitting.
    fn encode_text_with_special(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: HashMap<String, Token>,
    ) -> PyResult<Vec<Token>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_text_with_special(text, allowed_special))
            .map_err(encode_err_to_pyerr)
    }

    /// Encodes multiple strings in parallel with special token handling.
    ///
    /// Special tokens in `allowed_special` are matched literally and emitted as
    /// single token IDs. Remaining text is encoded through the normal regex and
    /// BPE pipeline. Normal segments across all texts are processed in a single
    /// Rayon batch, and results are returned in input order. The GIL is
    /// released for the entire computation.
    ///
    /// If `show_progress` is `true`, encoding the normal segments displays a
    /// progress bar.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if the regex engine fails during text splitting.
    #[pyo3(signature = (texts, allowed_special, show_progress = true))]
    fn encode_texts_with_special(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        allowed_special: HashMap<String, Token>,
        show_progress: bool,
    ) -> PyResult<Vec<Vec<Token>>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer.encode_texts_with_special(&text_refs, allowed_special, show_progress)
        })
        .map_err(encode_err_to_pyerr)
    }
    /// Encodes one string as raw bytes followed by BPE merging.
    ///
    /// This bypasses regex splitting and is useful for tokenizers that operate
    /// on the full byte stream. The GIL is released for the entire computation.
    fn encode_bytes(&self, py: Python<'_>, text: &str) -> Vec<Token> {
        let tokenizer = &self.tokenizer;
        py.detach(move || tokenizer.encode_bytes(text))
    }
    /// Encodes multiple strings as raw bytes followed by BPE merging in parallel.
    ///
    /// This bypasses regex splitting for every input. Results are returned in
    /// input order, and the GIL is released for the entire computation.
    ///
    /// If `show_progress` is `true`, batch encoding displays a progress bar.
    #[pyo3(signature = (texts, show_progress = true))]
    fn encode_bytes_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        show_progress: bool,
    ) -> PyResult<Vec<Vec<Token>>> {
        let tokenizer = &self.tokenizer;
        py.detach(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenizer.encode_bytes_batch(&text_refs, show_progress)
        })
        .map_err(encode_err_to_pyerr)
    }

    /// Decodes a token sequence into a UTF-8 string.
    ///
    /// `errors` controls invalid UTF-8 handling. `"strict"` raises
    /// `ValueError`, while `"replace"` substitutes `U+FFFD`. The default mode
    /// is `"replace"`. The GIL is released for the entire computation.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if a token ID is unknown, UTF-8 is invalid in
    /// `"strict"` mode, or `errors` does not name a recognised mode.
    #[pyo3(signature = (tokens, errors=None))]
    fn decode_tokens(
        &self,
        py: Python<'_>,
        tokens: Vec<Token>,
        errors: Option<&str>,
    ) -> PyResult<String> {
        let mode = parse_error_mode(errors)?;
        let tokenizer = &self.tokenizer;

        let result = py.detach(move || tokenizer.decode_tokens(&tokens, mode));
        result.map_err(decode_err_to_pyerr)
    }

    /// Decodes multiple token sequences in parallel.
    ///
    /// `errors` uses the same handling modes as [`Self::decode_tokens`]. If
    /// `show_progress` is `true`, batch decoding displays a progress bar.
    /// Results are returned in input order, and the GIL is released for the
    /// entire computation.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if any token ID is unknown, UTF-8 is invalid in
    /// `"strict"` mode, or `errors` does not name a recognised mode.
    #[pyo3(signature = (token_seqs, errors = None, show_progress = true))]
    fn decode_tokens_batch(
        &self,
        py: Python<'_>,
        token_seqs: Vec<Vec<Token>>,
        errors: Option<&str>,
        show_progress: bool,
    ) -> PyResult<Vec<String>> {
        let mode = parse_error_mode(errors)?;
        let tokenizer = &self.tokenizer;
        let refs: Vec<&[Token]> = token_seqs.iter().map(|v| v.as_slice()).collect();
        let result = py.detach(move || tokenizer.decode_tokens_batch(&refs, mode, show_progress));
        result.map_err(decode_err_to_pyerr)
    }

    /// Returns the number of tokens in the vocabulary.
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

/// Python wrapper for BPE trainer.
///
/// Provides an efficient Rust-based implementation of Byte Pair Encoding training
/// that can be used from Python via PyO3 bindings.
///
/// # Example
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
    initial_tokens: Vec<Token>,
    has_initial_tokens: bool,
}

#[pymethods]
impl RustBPETrainer {
    /// Creates a trainer from an initial token sequence.
    ///
    /// `tokens` is the starting sequence, such as byte values `0..=255`.
    /// `next_token_id` is the token ID assigned to the first learned merge.
    #[new]
    fn new(tokens: Vec<Token>, next_token_id: Token) -> Self {
        let trainer = BPETrainer::from_pieces(vec![(tokens.clone(), 1)], next_token_id, 1);
        RustBPETrainer {
            trainer,
            initial_tokens: tokens,
            has_initial_tokens: true,
        }
    }

    /// Creates a trainer from raw corpus text.
    ///
    /// The corpus is pretokenized in Rust, optionally using a regex pattern,
    /// and aggregated into weighted pieces before training begins.
    ///
    /// When `pattern` is omitted or empty, whitespace splitting is used.
    /// `next_token_id` is assigned to the first learned merge, and `min_count`
    /// sets the minimum pair frequency required for a merge candidate.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if `pattern` fails to compile.
    #[staticmethod]
    #[pyo3(signature = (corpus, pattern = None, next_token_id = 256, min_count = 1))]
    fn from_corpus(
        corpus: &str,
        pattern: Option<&str>,
        next_token_id: Token,
        min_count: Count,
    ) -> PyResult<Self> {
        let trainer = BPETrainer::from_corpus(corpus, pattern, next_token_id, min_count)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

        Ok(Self {
            trainer,
            initial_tokens: Vec::new(),
            has_initial_tokens: false,
        })
    }

    /// Trains the model by performing up to `num_merges` merge operations.
    ///
    /// If `show_progress` is `true`, training displays a progress bar. The GIL
    /// is released while the Rust training loop runs, and training stops early
    /// if no more pairs can be merged.
    #[pyo3(signature = (num_merges, show_progress = true))]
    fn train(&mut self, py: Python<'_>, num_merges: usize, show_progress: bool) -> PyResult<()> {
        let trainer = &mut self.trainer;
        // allow rust code to run without the GIL
        py.detach(move || {
            trainer
                .train(num_merges, show_progress)
                .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))
        })
    }

    /// Performs one merge on the most frequent token pair.
    ///
    /// Returns `true` if a merge was performed, or `false` if no pairs remain.
    /// The GIL is released while the merge step runs.
    fn merge_step(&mut self, py: Python<'_>) -> bool {
        let trainer = &mut self.trainer;
        let before = trainer.merge_history().len();
        py.detach(move || {
            let _ = trainer.train(1, false);
            trainer.merge_history().len() > before
        })
    }

    /// Returns the current token sequence after all learned merges.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if the trainer was created with
    /// [`Self::from_corpus`], which does not preserve a single reconstructable
    /// token stream.
    fn get_tokens(&self) -> PyResult<Vec<Token>> {
        if !self.has_initial_tokens {
            return Err(PyErr::new::<PyValueError, _>(
                "token sequence is unavailable for corpus-based trainers",
            ));
        }

        let mut tokens = self.initial_tokens.clone();

        for ((left, right), merged) in self.trainer.merge_history() {
            if tokens.len() < 2 {
                break;
            }

            let mut next = Vec::with_capacity(tokens.len());
            let mut i = 0usize;

            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == left && tokens[i + 1] == right {
                    next.push(merged);
                    i += 2;
                } else {
                    next.push(tokens[i]);
                    i += 1;
                }
            }

            tokens = next;
        }

        Ok(tokens)
    }

    /// Returns the full merge history.
    ///
    /// Each entry is `((left_token, right_token), merged_token)`. The order
    /// matches the sequence in which merges were learned.
    fn get_merge_history(&self) -> Vec<((Token, Token), Token)> {
        self.trainer.merge_history()
    }

    /// Returns the learned merges and reconstructed vocabulary.
    ///
    /// The returned `merges` mapping has `(left_token, right_token)` keys and
    /// merged token IDs as values. The returned `vocab` mapping contains token
    /// IDs to their byte representation.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if the merge history references a token that is not
    /// yet present in the reconstructed vocabulary.
    fn get_merges_and_vocab(&self, py: Python<'_>) -> PyResult<(Py<PyDict>, Py<PyDict>)> {
        let merge_history = self.get_merge_history();

        // transfer Rust `HashMap` contents into Python dictionaries
        let merges = PyDict::new(py);
        let vocab = PyDict::new(py);

        let mut vocab_rs: HashMap<Token, Vec<u8>> =
            HashMap::with_capacity(256 + merge_history.len());

        // build the base vocabulary
        for tok in 0u32..256 {
            vocab_rs.insert(tok, vec![tok as u8]);
        }

        for ((left, right), merged) in merge_history {
            // construct the merged byte sequence for this learned token
            let left_bytes = vocab_rs.get(&left).ok_or_else(|| {
                PyErr::new::<PyValueError, _>(format!("missing token {left} in vocabulary"))
            })?;

            let right_bytes = vocab_rs.get(&right).ok_or_else(|| {
                PyErr::new::<PyValueError, _>(format!("missing token {right} in vocabulary"))
            })?;

            let mut merged_bytes = Vec::with_capacity(left_bytes.len() + right_bytes.len());

            merged_bytes.extend_from_slice(left_bytes);
            merged_bytes.extend_from_slice(right_bytes);

            merges.set_item((left, right), merged)?;
            vocab_rs.insert(merged, merged_bytes);
        }

        for (tok, bytes) in vocab_rs {
            vocab.set_item(tok, PyBytes::new(py, &bytes))?;
        }

        // `unbind()` converts the bound dictionaries into owned Python objects
        // that lets py03 return them safely after this `Python` context ends
        Ok((merges.unbind(), vocab.unbind()))
    }

    /// Prints the current trainer state for debugging.
    fn print_state(&self) -> PyResult<()> {
        if self.has_initial_tokens {
            println!("tokens: {:?}", self.get_tokens()?);
        } else {
            println!("tokens: <unavailable for corpus-based trainer>");
        }
        println!("merge_history: {:?}", self.trainer.merge_history());
        println!("num_pieces: {}", self.trainer.num_pieces());
        println!("num_live_pairs: {}", self.trainer.num_live_pairs());
        Ok(())
    }
}

/// PyO3 module definition for BPE implementations.
///
/// This module exposes Rust-based BPE training and encoding functionality to Python.
/// The function name must match the library name specified in ``Cargo.toml``.
///
/// **Exports:**
///
/// - ``RustBPETrainer`` — Fast BPE training implementation.
/// - ``RustBPETokenizer`` — Full encoding/decoding pipeline (regex splitting + BPE merge + special tokens).
#[pymodule]
fn _bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPETokenizer>()?;
    m.add_class::<RustBPETrainer>()?;
    Ok(())
}
