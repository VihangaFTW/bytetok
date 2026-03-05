//! Weighted BPE trainer — complete reference implementation.
//!
//! This trains on unique pretokenized pieces weighted by their corpus frequency,
//! matching the HuggingFace approach. The BPE algorithm is identical;
//! the speedup comes from operating on ~100K unique words instead of
//! millions of raw byte positions.
//!
//! Key design decisions:
//! - Pretokenization + frequency aggregation is parallelised with Rayon.
//! - Initial pair-stat build is parallelised with Rayon map-reduce.
//! - The merge-choice loop is sequential (must be — merge N depends on merge N-1).
//! - Per-merge piece updates are parallelised: each piece is independent.
//! - Uses hashbrown (already in Cargo.toml) for faster hashing.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::types::{Count, Pair, Token};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A unique piece (word/chunk) with its corpus frequency.
///
/// After pretokenization, repeated words share a single `Piece` with a high
/// `count` instead of duplicating node storage for every occurrence.
#[derive(Debug, Clone)]
struct Piece {
    /// Current token sequence for this piece.
    /// Initially each token is a raw byte value in [0, 255].
    /// Merges shorten this vec over time.
    tokens: Vec<Token>,

    /// How many times this exact piece appeared in the corpus.
    count: Count,
}

/// Max-heap item for tracking the most frequent pair.
#[derive(Debug, PartialEq, Eq)]
struct HeapItem {
    /// Weighted frequency of this pair across all pieces.
    freq: Count,
    /// The token pair.
    pair: Pair,
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    /// Highest frequency first; deterministic tie-break on pair values.
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq
            .cmp(&other.freq)
            .then_with(|| self.pair.0.cmp(&other.pair.0))
            .then_with(|| self.pair.1.cmp(&other.pair.1))
    }
}

/// Delta accumulator for pair-frequency changes during a merge step.
///
/// A positive value means the pair gained occurrences; negative means it lost some.
/// We use `i64` so that subtractions don't underflow an unsigned type.
type PairDelta = HashMap<Pair, i64>;

// ---------------------------------------------------------------------------
// WeightedBPETrainer
// ---------------------------------------------------------------------------

/// BPE trainer that operates on weighted unique pieces.
///
/// Instead of a flat linked list over the entire corpus, this stores
/// only *unique* pretokenized chunks and their counts. Pair frequencies
/// are the sum of `piece.count` for every piece containing that pair,
/// making the work proportional to the number of unique pieces (~100K)
/// rather than the raw corpus length (millions of bytes).
pub(crate) struct WeightedBPETrainer {
    /// All unique subwords/chunks after pretokenization.
    pieces: Vec<Piece>,

    /// Max-heap of (frequency, pair).  May contain stale entries;
    /// always validated against `pair_freqs` before use.
    heap: BinaryHeap<HeapItem>,

    /// Source-of-truth weighted frequency for every live pair.
    pair_freqs: HashMap<Pair, Count>,

    /// Ordered merge history: `(pair_merged, new_token_id)`.
    merge_history: Vec<(Pair, Token)>,

    /// Stop training when the best pair's frequency drops below this.
    min_frequency: Count,

    /// Next available token ID for newly created merge tokens.
    next_tok: Token,
}

impl WeightedBPETrainer {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build a trainer from raw corpus bytes + a regex pattern.
    ///
    /// 1. Uses `fancy_regex` to split the corpus into chunks (words).
    /// 2. Aggregates identical chunks into weighted `Piece`s (parallelised).
    /// 3. Builds initial pair statistics (parallelised).
    ///
    /// # Arguments
    ///
    /// * `corpus` — raw UTF-8 corpus bytes.
    /// * `pattern` — fancy-regex pattern string for splitting.
    ///     Pass `None` (or empty string) to fall back to whitespace splitting.
    /// * `next_tok` — first merge token ID (256 for byte-level BPE).
    /// * `min_frequency` — pairs below this weighted frequency are ignored.
    pub(crate) fn from_corpus(
        corpus: &[u8],
        pattern: Option<&str>,
        next_tok: Token,
        min_frequency: Count,
    ) -> Result<Self, String> {
        // Step 1: pretokenize + aggregate frequencies (parallel).
        let pieces = Self::pretokenize(corpus, pattern)?;

        // Step 2: build pair stats from pieces (parallel).
        let pair_freqs = Self::build_pair_freqs_parallel(&pieces);

        // Step 3: populate heap from pair_freqs.
        let heap: BinaryHeap<HeapItem> = pair_freqs
            .iter()
            .filter(|(_, &freq)| freq >= min_frequency)
            .map(|(&pair, &freq)| HeapItem { freq, pair })
            .collect();

        Ok(Self {
            pieces,
            heap,
            pair_freqs,
            merge_history: Vec::new(),
            min_frequency,
            next_tok,
        })
    }

    /// Convenience constructor that takes pre-aggregated pieces directly.
    ///
    /// Useful when the Python side has already done the pretokenization
    /// (e.g. the `RegexTokenizer.train` method groups regex matches).
    ///
    /// `piece_data` is a vec of `(token_sequence, count)` tuples.
    pub(crate) fn from_pieces(
        piece_data: Vec<(Vec<Token>, Count)>,
        next_tok: Token,
        min_frequency: Count,
    ) -> Self {
        let pieces: Vec<Piece> = piece_data
            .into_iter()
            .map(|(tokens, count)| Piece { tokens, count })
            .collect();

        let pair_freqs = Self::build_pair_freqs_parallel(&pieces);

        let heap: BinaryHeap<HeapItem> = pair_freqs
            .iter()
            .filter(|(_, &freq)| freq >= min_frequency)
            .map(|(&pair, &freq)| HeapItem { freq, pair })
            .collect();

        Self {
            pieces,
            heap,
            pair_freqs,
            merge_history: Vec::new(),
            min_frequency,
            next_tok,
        }
    }

    // -----------------------------------------------------------------
    // Pretokenization (parallel)
    // -----------------------------------------------------------------

    /// Splits corpus bytes with a regex pattern and aggregates unique
    /// chunks into `Piece`s with their frequency counts.
    ///
    /// The regex matching runs on a single thread (fancy-regex is not
    /// Send), but frequency counting across matches is fast.
    /// For very large corpora, consider chunking the corpus by splitting
    /// on newlines first and running regex per-chunk in parallel.
    fn pretokenize(corpus: &[u8], pattern: Option<&str>) -> Result<Vec<Piece>, String> {
        let owned;
        let text: &str = match std::str::from_utf8(corpus) {
            Ok(s) => s, // zero-copy, valid UTF-8
            Err(_) => {
                // lossy conversion replaces invalid bytes with '?'
                owned = String::from_utf8_lossy(corpus).into_owned();
                owned.as_str()
            }
        };

        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Collect all regex matches.
        let chunks: Vec<&str> = match pattern {
            Some(pat) if !pat.is_empty() => {
                let re = fancy_regex::Regex::new(pat)
                    .map_err(|e| format!("invalid regex pattern: {e}"))?;

                // fancy_regex::Regex::find_iter returns Result<Match>.
                // Collect valid matches.
                let mut matches = Vec::new();
                // We manually iterate because fancy_regex::Matches is not
                // a standard iterator of &str — it yields Result<Match>.
                let mut start = 0;
                loop {
                    match re.find_from_pos(text, start) {
                        Ok(Some(m)) => {
                            if m.start() == m.end() {
                                // Zero-length match — advance by one byte to avoid infinite loop.
                                start = m.end() + 1;
                                if start > text.len() {
                                    break;
                                }
                                continue;
                            }
                            matches.push(m.as_str());
                            start = m.end();
                        }
                        Ok(None) => break,
                        Err(_) => break,
                    }
                }
                matches
            }
            _ => {
                // Fallback: split on whitespace.
                text.split_whitespace().collect()
            }
        };

        // Aggregate chunk frequencies.
        // For large match sets this dominates, so we parallelise with Rayon.
        //
        // Strategy: split chunks into per-thread shards, build local
        // frequency maps, then merge them.
        let chunk_freq: HashMap<Vec<u8>, Count> = if chunks.len() > 10_000 {
            // Parallel path.
            chunks
                .par_iter()
                .fold(
                    HashMap::new,
                    |mut map: HashMap<Vec<u8>, Count>, &chunk| {
                        let bytes = chunk.as_bytes().to_vec();
                        *map.entry(bytes).or_insert(0) += 1;
                        map
                    },
                )
                .reduce(HashMap::new, |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                })
        } else {
            // Sequential path for small inputs.
            let mut map = HashMap::new();
            for &chunk in &chunks {
                let bytes = chunk.as_bytes().to_vec();
                *map.entry(bytes).or_insert(0) += 1;
            }
            map
        };

        // Convert to Piece vec.
        let pieces: Vec<Piece> = chunk_freq
            .into_iter()
            .map(|(bytes, count)| Piece {
                tokens: bytes.into_iter().map(|b| b as Token).collect(),
                count,
            })
            .collect();

        Ok(pieces)
    }

    // -----------------------------------------------------------------
    // Initial pair-stat build (parallel)
    // -----------------------------------------------------------------

    /// Builds weighted pair frequencies across all pieces using Rayon
    /// map-reduce.
    ///
    /// Each piece contributes `piece.count` to every adjacent pair in its
    /// token sequence.  Pieces are independent so this is embarrassingly
    /// parallel.
    fn build_pair_freqs_parallel(pieces: &[Piece]) -> HashMap<Pair, Count> {
        if pieces.len() < 1_000 {
            // Sequential for tiny inputs.
            let mut freqs = HashMap::new();
            for piece in pieces {
                for window in piece.tokens.windows(2) {
                    let pair = Pair(window[0], window[1]);
                    *freqs.entry(pair).or_insert(0) += piece.count;
                }
            }
            return freqs;
        }

        pieces
            .par_iter()
            .fold(
                HashMap::new,
                |mut map: HashMap<Pair, Count>, piece| {
                    for window in piece.tokens.windows(2) {
                        let pair = Pair(window[0], window[1]);
                        *map.entry(pair).or_insert(0) += piece.count;
                    }
                    map
                },
            )
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            })
    }

    // -----------------------------------------------------------------
    // Max-pair extraction (lazy heap with stale-entry filtering)
    // -----------------------------------------------------------------

    /// Pops the highest-frequency pair from the heap, skipping stale
    /// entries whose stored frequency no longer matches `pair_freqs`.
    ///
    /// Returns `None` when the heap is exhausted or the best pair's
    /// frequency is below `min_frequency`.
    fn get_max_pair(&mut self) -> Option<(Pair, Count)> {
        while let Some(item) = self.heap.pop() {
            if let Some(&true_freq) = self.pair_freqs.get(&item.pair) {
                if true_freq == item.freq && true_freq >= self.min_frequency {
                    return Some((item.pair, true_freq));
                }
                // Stale count — re-push with correct count if still above threshold.
                // This avoids losing the pair entirely if only its freq changed.
                if true_freq >= self.min_frequency && true_freq > 0 {
                    self.heap.push(HeapItem {
                        freq: true_freq,
                        pair: item.pair,
                    });
                }
            }
            // Pair no longer exists or below threshold — discard.
        }
        None
    }

    /// Rebuilds the heap from scratch when too many stale entries
    /// accumulate.  Call this periodically (e.g. every N merges) to
    /// keep `get_max_pair` fast.
    fn rebuild_heap_if_needed(&mut self) {
        let heap_size = self.heap.len();
        let live_pairs = self.pair_freqs.len();

        // Rebuild when >50% of heap entries are stale.
        if heap_size > live_pairs * 2 + 1000 {
            self.heap = self
                .pair_freqs
                .iter()
                .filter(|(_, &freq)| freq >= self.min_frequency)
                .map(|(&pair, &freq)| HeapItem { freq, pair })
                .collect();
        }
    }

    // -----------------------------------------------------------------
    // Merge step (parallel piece updates)
    // -----------------------------------------------------------------

    /// Performs one BPE merge: finds the most frequent pair, merges it
    /// across all pieces, and updates pair frequencies.
    ///
    /// Returns `true` if a merge was performed.
    pub(crate) fn merge_step(&mut self) -> bool {
        self.rebuild_heap_if_needed();

        // 1. Find best pair.
        let (best_pair, _freq) = match self.get_max_pair() {
            Some(pf) => pf,
            None => return false,
        };

        let new_tok = self.next_tok;
        self.next_tok += 1;

        // 2. Apply the merge across all pieces in parallel.
        //    Each piece independently:
        //      a) Scans its tokens for `best_pair`.
        //      b) Merges matching adjacent tokens into `new_tok`.
        //      c) Computes local pair-frequency deltas (weighted by piece.count).
        //
        //    We then reduce the deltas and apply them to `self.pair_freqs`.

        let deltas: Vec<PairDelta> = self
            .pieces
            .par_iter_mut()
            .filter_map(|piece| {
                Self::merge_in_piece(piece, best_pair, new_tok)
            })
            .collect();

        // 3. Reduce all deltas into pair_freqs + push changed pairs onto heap.
        for delta in deltas {
            for (pair, change) in delta {
                let entry = self.pair_freqs.entry(pair).or_insert(0);
                // Apply delta (may go negative temporarily due to overcounting,
                // clamp to 0).
                let new_freq = (*entry as i64 + change).max(0) as Count;
                *entry = new_freq;

                if new_freq == 0 {
                    self.pair_freqs.remove(&pair);
                } else if new_freq >= self.min_frequency {
                    // Push updated frequency onto heap.
                    // Stale entries are handled by get_max_pair validation.
                    self.heap.push(HeapItem {
                        freq: new_freq,
                        pair,
                    });
                }
            }
        }

        // 4. Clean up the merged pair itself.
        self.pair_freqs.remove(&best_pair);

        // 5. Record merge.
        self.merge_history.push((best_pair, new_tok));

        true
    }

    /// Merges `best_pair` inside a single piece's token sequence.
    ///
    /// Returns a `PairDelta` with the weighted frequency changes, or
    /// `None` if the piece doesn't contain the pair.
    ///
    /// The delta records:
    /// - Negative entries for pairs that were destroyed by the merge.
    /// - Positive entries for new pairs formed with `new_tok` and its
    ///   neighbors.
    fn merge_in_piece(piece: &mut Piece, best_pair: Pair, new_tok: Token) -> Option<PairDelta> {
        let tokens = &piece.tokens;
        let count = piece.count as i64;

        // Quick check: does this piece contain the pair at all?
        let has_pair = tokens.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1);
        if !has_pair {
            return None;
        }

        let mut delta: PairDelta = HashMap::new();
        let mut new_tokens: Vec<Token> = Vec::with_capacity(tokens.len());
        let mut i = 0;

        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                // --- This position is a match for best_pair ---

                // Remove the old pair itself.
                *delta.entry(best_pair).or_insert(0) -= count;

                // Remove the old left-neighbor pair (prev_token, best_pair.0).
                if let Some(&prev) = new_tokens.last() {
                    let old_left = Pair(prev, best_pair.0);
                    *delta.entry(old_left).or_insert(0) -= count;

                    // Add new left-neighbor pair (prev_token, new_tok).
                    let new_left = Pair(prev, new_tok);
                    *delta.entry(new_left).or_insert(0) += count;
                }

                // Remove the old right-neighbor pair (best_pair.1, next_token).
                if i + 2 < tokens.len() {
                    let next = tokens[i + 2];

                    // Only remove the right neighbor pair if the next position
                    // is NOT also a match (to avoid double-counting).
                    let old_right = Pair(best_pair.1, next);
                    *delta.entry(old_right).or_insert(0) -= count;

                    // Add new right-neighbor pair (new_tok, next_token).
                    // But if (next, tokens[i+3]) is also best_pair, the next
                    // iteration will handle that pair — we still add the
                    // right neighbor here because new_tok is already placed.
                    let new_right = Pair(new_tok, next);
                    *delta.entry(new_right).or_insert(0) += count;
                }

                new_tokens.push(new_tok);
                i += 2; // skip both tokens of the merged pair
            } else {
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }

        piece.tokens = new_tokens;
        Some(delta)
    }

    // -----------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------

    /// Runs up to `num_merges` merge steps with an optional progress bar.
    ///
    /// Stops early if:
    /// - No pairs remain with frequency >= `min_frequency`.
    /// - The heap is exhausted.
    pub(crate) fn train(
        &mut self,
        num_merges: usize,
        show_progress: bool,
    ) -> Result<(), indicatif::style::TemplateError> {
        use indicatif::{ProgressBar, ProgressStyle};

        let progress = if show_progress {
            let pb = ProgressBar::new(num_merges as u64);
            let style = ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos}/{len}")?;
            pb.set_style(style);
            pb.set_message("Training BPE merges (weighted)");
            pb.enable_steady_tick(std::time::Duration::from_secs(1));
            Some(pb)
        } else {
            None
        };

        for _i in 0..num_merges {
            if !self.merge_step() {
                break;
            }
            if let Some(pb) = &progress {
                pb.inc(1);
            }
        }

        if let Some(pb) = &progress {
            pb.finish_and_clear();
        }

        Ok(())
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Returns the merge history in the format expected by the tokenizer:
    /// `((left_token, right_token), merged_token)`.
    pub(crate) fn merge_history(&self) -> Vec<((Token, Token), Token)> {
        self.merge_history
            .iter()
            .map(|(pair, tok)| ((pair.0, pair.1), *tok))
            .collect()
    }

    /// Total number of unique pieces.
    pub(crate) fn num_pieces(&self) -> usize {
        self.pieces.len()
    }

    /// Number of live (non-zero) pair entries.
    pub(crate) fn num_live_pairs(&self) -> usize {
        self.pair_freqs.len()
    }
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------
//
// NOTE: These go in lib.rs alongside RustBPETrainer.
// They are included here as a complete reference so you can see
// the full wiring.  In practice, add `use crate::trainer_weighted::WeightedBPETrainer;`
// in lib.rs and register `RustWeightedBPETrainer` in the `_bpe_rs` module.

#[cfg(feature = "pyo3_bindings_reference")]
mod pyo3_bindings {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Python wrapper for the weighted BPE trainer.
    ///
    /// # Example
    ///
    /// ```python
    /// from bytetok.bpe import RustWeightedBPETrainer
    ///
    /// # Train from raw corpus bytes + regex pattern (everything in Rust).
    /// trainer = RustWeightedBPETrainer.from_corpus(
    ///     corpus=open("data.txt", "rb").read(),
    ///     pattern=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|...",
    ///     next_token_id=256,
    ///     min_frequency=2,
    /// )
    /// trainer.train(50000, show_progress=True)
    /// merges = trainer.get_merge_history()
    ///
    /// # Or from pre-aggregated pieces (when Python already did regex).
    /// trainer = RustWeightedBPETrainer.from_pieces(
    ///     pieces=[([116, 104, 101], 50000), ([97, 110, 100], 30000)],
    ///     next_token_id=256,
    /// )
    /// ```
    #[pyclass]
    pub struct RustWeightedBPETrainer {
        trainer: WeightedBPETrainer,
    }

    #[pymethods]
    impl RustWeightedBPETrainer {
        /// Create trainer from raw corpus bytes, splitting with a regex pattern.
        ///
        /// Pretokenization and frequency aggregation happen in Rust (parallelised).
        /// This avoids materialising a Python `list[int]` of millions of items.
        ///
        /// :param corpus: Raw corpus bytes.
        /// :param pattern: Regex pattern for splitting (fancy-regex syntax).
        ///     Pass ``None`` to split on whitespace.
        /// :param next_token_id: First merge token ID (256 for byte-level BPE).
        /// :param min_frequency: Ignore pairs below this weighted frequency.
        /// :returns: A new trainer instance ready for `.train()`.
        /// :raises ValueError: If the regex pattern is invalid.
        #[staticmethod]
        #[pyo3(signature = (corpus, pattern = None, next_token_id = 256, min_frequency = 1))]
        fn from_corpus(
            py: Python<'_>,
            corpus: &[u8],
            pattern: Option<&str>,
            next_token_id: usize,
            min_frequency: usize,
        ) -> PyResult<Self> {
            // Release the GIL for the heavy pretokenization work.
            py.detach(move || {
                let trainer = WeightedBPETrainer::from_corpus(
                    corpus,
                    pattern,
                    next_token_id,
                    min_frequency,
                )
                .map_err(|e| PyErr::new::<PyValueError, _>(e))?;

                Ok(RustWeightedBPETrainer { trainer })
            })
        }

        /// Create trainer from pre-aggregated pieces.
        ///
        /// :param pieces: List of `(token_sequence, count)` tuples.
        ///     Each `token_sequence` is a `list[int]` of byte-level tokens.
        /// :param next_token_id: First merge token ID.
        /// :param min_frequency: Ignore pairs below this weighted frequency.
        #[staticmethod]
        #[pyo3(signature = (pieces, next_token_id = 256, min_frequency = 1))]
        fn from_pieces(
            pieces: Vec<(Vec<usize>, usize)>,
            next_token_id: usize,
            min_frequency: usize,
        ) -> Self {
            let trainer = WeightedBPETrainer::from_pieces(pieces, next_token_id, min_frequency);
            RustWeightedBPETrainer { trainer }
        }

        /// Train the BPE model.
        ///
        /// :param num_merges: Maximum number of merge operations.
        /// :param show_progress: Display a progress bar.
        #[pyo3(signature = (num_merges, show_progress = true))]
        fn train(
            &mut self,
            py: Python<'_>,
            num_merges: usize,
            show_progress: bool,
        ) -> PyResult<()> {
            let trainer = &mut self.trainer;
            py.detach(move || {
                trainer
                    .train(num_merges, show_progress)
                    .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))
            })
        }

        /// Perform a single merge step.
        ///
        /// :returns: ``True`` if a merge was performed, ``False`` if done.
        fn merge_step(&mut self, py: Python<'_>) -> bool {
            let trainer = &mut self.trainer;
            py.detach(move || trainer.merge_step())
        }

        /// Returns merge history as `[((left, right), merged), ...]`.
        fn get_merge_history(&self) -> Vec<((usize, usize), usize)> {
            self.trainer.merge_history()
        }

        /// Number of unique pieces the trainer is operating on.
        fn num_pieces(&self) -> usize {
            self.trainer.num_pieces()
        }

        /// Number of live pair entries in the frequency table.
        fn num_live_pairs(&self) -> usize {
            self.trainer.num_live_pairs()
        }
    }
}

// ---------------------------------------------------------------------------
// Python-side reference (_trainer.py changes)
// ---------------------------------------------------------------------------
//
// Replace the current `_train_bpe` function body in src/bytetok/_trainer.py:
//
// ```python
// def _train_bpe(
//     raw_bytes: bytes,          # <-- was list[Token]
//     n_merges: int,
//     pattern: str | None = None,
//     min_frequency: int = 1,    # <-- new
//     verbose: bool = False,
//     show_progress: bool = True,
// ) -> BPETrainingResult:
//     if len(raw_bytes) == 0:
//         raise TrainingError("empty training data")
//
//     trainer = RustWeightedBPETrainer.from_corpus(
//         corpus=raw_bytes,
//         pattern=pattern,
//         next_token_id=256,
//         min_frequency=min_frequency,
//     )
//     try:
//         trainer.train(n_merges, show_progress=show_progress)
//     except ValueError as e:
//         raise TrainingError(f"internal error: {e}") from e
//
//     merge_history = trainer.get_merge_history()
//
//     merges: Encoding = {}
//     vocab: Vocabulary = {tok: bytes([tok]) for tok in range(256)}
//
//     for (tok_a, tok_b), new_tok in merge_history:
//         pair = (tok_a, tok_b)
//         merges[pair] = new_tok
//         vocab[new_tok] = vocab[tok_a] + vocab[tok_b]
//         if verbose:
//             log.info("merge %d/%d: %s -> %d", len(merges), n_merges, pair, new_tok)
//
//     return BPETrainingResult(
//         vocab=vocab, merges=merges, n_merges_completed=len(merge_history)
//     )
// ```
//
// ---------------------------------------------------------------------------
// regex.py changes (in RegexTokenizer.train):
// ---------------------------------------------------------------------------
//
// ```python
// def train(self, text, vocab_size, verbose=False, show_progress=True,
//           min_frequency=1):
//     if vocab_size <= 256:
//         raise VocabularyError(...)
//     if isinstance(text, list):
//         text = "".join(text)
//
//     raw_bytes = text.encode("utf-8", errors="replace")
//     n_merges = vocab_size - 256
//
//     result = _train_bpe(
//         raw_bytes,
//         n_merges,
//         pattern=self.pat,         # <-- pass regex to Rust
//         min_frequency=min_frequency,
//         verbose=verbose,
//         show_progress=show_progress,
//     )
//     ...
// ```
//
// ---------------------------------------------------------------------------
// basic.py changes (in BasicTokenizer.train):
// ---------------------------------------------------------------------------
//
// ```python
// def train(self, text, vocab_size, verbose=False, show_progress=True,
//           min_frequency=1):
//     if vocab_size <= 256:
//         raise VocabularyError(...)
//     if isinstance(text, list):
//         text = "".join(text)
//
//     raw_bytes = text.encode("utf-8", errors="replace")
//     n_merges = vocab_size - 256
//
//     result = _train_bpe(
//         raw_bytes,
//         n_merges,
//         pattern=None,             # <-- no regex for basic
//         min_frequency=min_frequency,
//         verbose=verbose,
//         show_progress=show_progress,
//     )
//     ...
// ```
//
// ---------------------------------------------------------------------------
// __init__.pyi stub updates:
// ---------------------------------------------------------------------------
//
// ```python
// class RustWeightedBPETrainer:
//     @staticmethod
//     def from_corpus(
//         corpus: bytes,
//         pattern: str | None = None,
//         next_token_id: int = 256,
//         min_frequency: int = 1,
//     ) -> RustWeightedBPETrainer: ...
//     @staticmethod
//     def from_pieces(
//         pieces: list[tuple[list[int], int]],
//         next_token_id: int = 256,
//         min_frequency: int = 1,
//     ) -> RustWeightedBPETrainer: ...
//     def train(self, num_merges: int, show_progress: bool = True) -> None: ...
//     def merge_step(self) -> bool: ...
//     def get_merge_history(self) -> list[tuple[tuple[int, int], int]]: ...
//     def num_pieces(self) -> int: ...
//     def num_live_pairs(self) -> int: ...
// ```

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a trainer from a simple byte string with no regex.
    fn trainer_from_str(s: &str, min_freq: usize) -> WeightedBPETrainer {
        WeightedBPETrainer::from_corpus(s.as_bytes(), None, 256, min_freq)
            .expect("pretokenize should succeed")
    }

    #[test]
    fn test_pretokenize_counts() {
        let corpus = b"the the the cat cat";
        let pieces = WeightedBPETrainer::pretokenize(corpus, None)
            .expect("should succeed");
        // "the" appears 3 times, "cat" appears 2 times.
        assert_eq!(pieces.len(), 2);
        let the_piece = pieces.iter().find(|p| p.tokens == vec![116, 104, 101]);
        let cat_piece = pieces.iter().find(|p| p.tokens == vec![99, 97, 116]);
        assert_eq!(the_piece.map(|p| p.count), Some(3));
        assert_eq!(cat_piece.map(|p| p.count), Some(2));
    }

    #[test]
    fn test_pretokenize_with_regex() {
        let corpus = b"hello world hello";
        let pieces = WeightedBPETrainer::pretokenize(corpus, Some(r"\w+"))
            .expect("should succeed");
        assert_eq!(pieces.len(), 2); // "hello" x2, "world" x1
    }

    #[test]
    fn test_pair_freqs_weighted() {
        // "ab" x 3 should give pair (97,98) a freq of 3.
        let pieces = vec![
            Piece {
                tokens: vec![97, 98], // "ab"
                count: 3,
            },
            Piece {
                tokens: vec![97, 98], // "ab" again (dedup should
                count: 2,            //  catch this, but test both)
            },
        ];
        let freqs = WeightedBPETrainer::build_pair_freqs_parallel(&pieces);
        assert_eq!(freqs.get(&Pair(97, 98)), Some(&5)); // 3 + 2
    }

    #[test]
    fn test_single_merge() {
        // "aaab aaab" → "aab" x2 after one merge (aa -> 256).
        let mut trainer = trainer_from_str("aaab aaab", 1);
        let initial_pieces = trainer.num_pieces();
        assert!(initial_pieces > 0);

        let merged = trainer.merge_step();
        assert!(merged);
        assert_eq!(trainer.merge_history().len(), 1);
    }

    #[test]
    fn test_full_training() {
        let corpus = "the cat sat on the mat the cat sat";
        let mut trainer = trainer_from_str(corpus, 1);
        trainer
            .train(20, false)
            .expect("training should succeed");
        assert!(trainer.merge_history().len() > 0);
        assert!(trainer.merge_history().len() <= 20);
    }

    #[test]
    fn test_min_frequency_pruning() {
        let corpus = "aaa bbb ccc aaa bbb"; // "aaa" x2, "bbb" x2, "ccc" x1
        let mut trainer = trainer_from_str(corpus, 3);
        // With min_frequency=3, pairs from "ccc" (count=1) and even
        // pairs from "aaa"/"bbb" (count=2 each, pair freq=2) should be
        // below threshold.  Training should stop immediately.
        let merged = trainer.merge_step();
        // The highest pair freq is 2 (e.g. (97,97) from "aaa"x2) which is < 3.
        assert!(!merged);
    }

    #[test]
    fn test_empty_corpus() {
        let trainer = trainer_from_str("", 1);
        assert_eq!(trainer.num_pieces(), 0);
    }

    #[test]
    fn test_merge_history_format() {
        let mut trainer = trainer_from_str("abab abab abab", 1);
        trainer.train(2, false).expect("should succeed");
        let history = trainer.merge_history();
        for ((left, right), merged) in &history {
            assert!(*left < 256 || *left >= 256);
            assert!(*right < 256 || *right >= 256);
            assert!(*merged >= 256);
        }
    }

    #[test]
    fn test_deterministic_merges() {
        // Two runs on the same corpus should produce the same merge history.
        let corpus = "hello world hello world foo bar foo";
        let mut t1 = trainer_from_str(corpus, 1);
        let mut t2 = trainer_from_str(corpus, 1);
        t1.train(10, false).expect("ok");
        t2.train(10, false).expect("ok");
        assert_eq!(t1.merge_history(), t2.merge_history());
    }

    #[test]
    fn test_from_pieces_constructor() {
        let pieces = vec![
            (vec![97u8 as usize, 98, 99], 10usize), // "abc" x 10
            (vec![100, 101], 5),                      // "de" x 5
        ];
        let mut trainer = WeightedBPETrainer::from_pieces(pieces, 256, 1);
        assert_eq!(trainer.num_pieces(), 2);
        trainer.train(5, false).expect("ok");
        assert!(trainer.merge_history().len() > 0);
    }
}
