//! Weighted BPE trainer reference implementation.
//!
//! This version branches from the current `src/bytetok/bpe/src/trainer_weighted.rs`
//! pretokenization design, then layers in the remaining weighted-training
//! pieces: initial pair counting, lazy heap maintenance, and parallel
//! piece-local merge application.

use std::{cmp::Ordering, collections::BinaryHeap};

use fancy_regex::Regex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    error::TokenizerInitError,
    types::{Count, Pair, Token},
};

const LOCAL_MAP_CAPACITY: usize = 8192;
const PAIR_MAP_CAPACITY: usize = 4096;
const CHUNK_BYTES: usize = 8 * 1024 * 1024;
const HEAP_REBUILD_MARGIN: usize = 1000;

/// A unique piece representing a subword with its corpus frequency.
#[derive(Debug, Clone)]
struct Piece {
    /// Current token sequence for this piece.
    ///
    /// Initially each token is a raw byte value in `[0, 255]`.
    tokens: Vec<Token>,
    /// Number of times this piece appears in the corpus.
    count: Count,
}

#[derive(Debug, PartialEq, Eq)]
struct HeapItem {
    pair: Pair,
    count: Count,
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.pair.0.cmp(&other.pair.0))
            .then_with(|| self.pair.1.cmp(&other.pair.1))
    }
}

type PairDelta = FxHashMap<Pair, i64>;

pub(crate) struct WeightedBPETrainer {
    /// All unique subwords or chunks in the corpus after pretokenization.
    pieces: Vec<Piece>,
    /// Max-heap tracking the most frequent token pair.
    heap: BinaryHeap<HeapItem>,
    /// Weighted frequency per token pair.
    ///
    /// Heap entries may be stale and must be validated against this map.
    pair_freqs: FxHashMap<Pair, Count>,
    /// Ordered merge history: `(pair_merged, new_token_id)`.
    merge_history: Vec<(Pair, Token)>,
    /// Stop training early when the best pair frequency drops below this.
    min_frequency: Count,
    /// Next available token ID for new merged tokens.
    next_tok: Token,
}

impl WeightedBPETrainer {
    /// Builds a trainer from raw text and an optional regex pattern.
    pub(crate) fn from_corpus(
        corpus: &str,
        pattern: Option<&str>,
        next_tok: Token,
        min_frequency: Count,
    ) -> Result<Self, TokenizerInitError> {
        let pieces = Self::pretokenize(corpus, pattern)?;
        let pair_freqs = Self::build_pair_freqs_parallel(&pieces);
        let heap = Self::build_heap(&pair_freqs, min_frequency);

        Ok(Self {
            pieces,
            heap,
            pair_freqs,
            merge_history: Vec::new(),
            min_frequency,
            next_tok,
        })
    }

    /// Builds a trainer from pre-aggregated pieces.
    pub(crate) fn from_pieces(
        pieces: Vec<(Vec<Token>, Count)>,
        next_tok: Token,
        min_frequency: Count,
    ) -> Self {
        let pieces: Vec<Piece> = pieces
            .into_iter()
            .map(|(tokens, count)| Piece { tokens, count })
            .collect();
        let pair_freqs = Self::build_pair_freqs_parallel(&pieces);
        let heap = Self::build_heap(&pair_freqs, min_frequency);

        Self {
            pieces,
            heap,
            pair_freqs,
            merge_history: Vec::new(),
            min_frequency,
            next_tok,
        }
    }

    /// Splits corpus bytes against a regex pattern and counts unique chunks.
    ///
    /// Each chunk becomes a `Piece` whose initial tokens are raw bytes in
    /// `[0, 255]`.
    fn pretokenize(corpus: &str, pattern: Option<&str>) -> Result<Vec<Piece>, TokenizerInitError> {
        if corpus.is_empty() {
            return Ok(Vec::new());
        }

        let chunks = Self::pretok_newline_chunks(corpus, CHUNK_BYTES);

        let chunk_freq: FxHashMap<&[u8], Count> = match pattern {
            Some(pat) if !pat.is_empty() => {
                let _ = Regex::new(pat).map_err(TokenizerInitError::InvalidPattern)?;

                chunks
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                Regex::new(pat).expect("pattern was validated"),
                                FxHashMap::with_capacity_and_hasher(
                                    LOCAL_MAP_CAPACITY,
                                    Default::default(),
                                ),
                            )
                        },
                        |(re, mut map), chunk| {
                            Self::pretok_count_matches(&re, chunk, &mut map);
                            (re, map)
                        },
                    )
                    .map(|(_, map)| map)
                    .reduce(
                        || {
                            FxHashMap::with_capacity_and_hasher(
                                LOCAL_MAP_CAPACITY,
                                Default::default(),
                            )
                        },
                        |mut left_map, right_map| {
                            for (chunk, count) in right_map {
                                *left_map.entry(chunk).or_insert(0) += count;
                            }
                            left_map
                        },
                    )
            }
            _ => chunks
                .into_par_iter()
                .fold(
                    || FxHashMap::with_capacity_and_hasher(LOCAL_MAP_CAPACITY, Default::default()),
                    |mut map, chunk| {
                        Self::pretok_count_whitespace(chunk, &mut map);
                        map
                    },
                )
                .reduce(
                    || FxHashMap::with_capacity_and_hasher(LOCAL_MAP_CAPACITY, Default::default()),
                    |mut left_map, right_map| {
                        for (chunk, count) in right_map {
                            *left_map.entry(chunk).or_insert(0) += count;
                        }
                        left_map
                    },
                ),
        };

        Ok(Self::pretok_map_to_pieces(chunk_freq))
    }

    /// Counts regex matches directly into a local shard map.
    fn pretok_count_matches<'a>(re: &Regex, text: &'a str, map: &mut FxHashMap<&'a [u8], Count>) {
        let mut start = 0;

        while start <= text.len() {
            match re.find_from_pos(text, start) {
                Ok(Some(m)) => {
                    if m.start() == m.end() {
                        start = m.end() + 1;
                        continue;
                    }

                    *map.entry(m.as_str().as_bytes()).or_insert(0) += 1;
                    start = m.end();
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
    }

    /// Counts whitespace chunks directly into a local shard map.
    fn pretok_count_whitespace<'a>(text: &'a str, map: &mut FxHashMap<&'a [u8], Count>) {
        for chunk in text.split_whitespace() {
            *map.entry(chunk.as_bytes()).or_insert(0) += 1;
        }
    }

    /// Builds parallel chunks while preserving newline tokens.
    ///
    /// This keeps the exact newline bytes inside the chunk so regex branches
    /// still see the original text.
    fn pretok_newline_chunks(corpus: &str, target_bytes: usize) -> Vec<&str> {
        if corpus.len() <= target_bytes {
            return vec![corpus];
        }

        let mut chunks = Vec::with_capacity(corpus.len().div_ceil(target_bytes));
        let bytes = corpus.as_bytes();
        let mut start = 0usize;
        let mut last_nl_end = 0usize;

        for (i, &byte) in bytes.iter().enumerate() {
            if byte == b'\n' {
                last_nl_end = i + 1;
            }

            if i + 1 - start >= target_bytes && start < last_nl_end {
                chunks.push(&corpus[start..last_nl_end]);
                start = last_nl_end;
            }
        }

        if start < corpus.len() {
            chunks.push(&corpus[start..]);
        }

        if chunks.is_empty() {
            chunks.push(corpus);
        }

        chunks
    }

    /// Converts borrowed chunk keys into owned pieces exactly once.
    fn pretok_map_to_pieces(chunk_freq: FxHashMap<&[u8], Count>) -> Vec<Piece> {
        chunk_freq
            .into_iter()
            .map(|(bytes, count)| Piece {
                tokens: bytes.iter().map(|&b| b as Token).collect(),
                count,
            })
            .collect()
    }

    /// Builds weighted pair frequencies across all pieces.
    fn build_pair_freqs_parallel(pieces: &[Piece]) -> FxHashMap<Pair, Count> {
        if pieces.len() < 1_000 {
            let mut pair_freqs =
                FxHashMap::with_capacity_and_hasher(PAIR_MAP_CAPACITY, Default::default());

            for piece in pieces {
                for window in piece.tokens.windows(2) {
                    let pair = Pair(window[0], window[1]);
                    *pair_freqs.entry(pair).or_insert(0) += piece.count;
                }
            }

            return pair_freqs;
        }

        pieces
            .par_iter()
            .fold(
                || FxHashMap::with_capacity_and_hasher(PAIR_MAP_CAPACITY, Default::default()),
                |mut local_map, piece| {
                    for window in piece.tokens.windows(2) {
                        let pair = Pair(window[0], window[1]);
                        *local_map.entry(pair).or_insert(0) += piece.count;
                    }
                    local_map
                },
            )
            .reduce(
                || FxHashMap::with_capacity_and_hasher(PAIR_MAP_CAPACITY, Default::default()),
                |mut left_map, right_map| {
                    for (pair, count) in right_map {
                        *left_map.entry(pair).or_insert(0) += count;
                    }
                    left_map
                },
            )
    }

    /// Builds the initial max-heap from the live pair frequencies.
    fn build_heap(
        pair_freqs: &FxHashMap<Pair, Count>,
        min_frequency: Count,
    ) -> BinaryHeap<HeapItem> {
        pair_freqs
            .iter()
            .filter(|(_, &count)| count >= min_frequency)
            .map(|(&pair, &count)| HeapItem { pair, count })
            .collect()
    }

    /// Pops the highest-frequency live pair from the heap.
    fn get_max_pair(&mut self) -> Option<(Pair, Count)> {
        while let Some(item) = self.heap.pop() {
            if let Some(&true_count) = self.pair_freqs.get(&item.pair) {
                if true_count == item.count && true_count >= self.min_frequency {
                    return Some((item.pair, true_count));
                }

                if true_count >= self.min_frequency && true_count > 0 {
                    self.heap.push(HeapItem {
                        pair: item.pair,
                        count: true_count,
                    });
                }
            }
        }

        None
    }

    /// Rebuilds the heap when stale entries start to dominate.
    fn rebuild_heap_if_needed(&mut self) {
        let heap_size = self.heap.len();
        let live_pairs = self.pair_freqs.len();

        if heap_size > live_pairs.saturating_mul(2) + HEAP_REBUILD_MARGIN {
            self.heap = Self::build_heap(&self.pair_freqs, self.min_frequency);
        }
    }

    /// Performs one weighted merge step.
    pub(crate) fn merge_step(&mut self) -> bool {
        self.rebuild_heap_if_needed();

        let (best_pair, _) = match self.get_max_pair() {
            Some(value) => value,
            None => return false,
        };

        let new_tok = self.next_tok;
        self.next_tok += 1;

        let deltas: Vec<PairDelta> = self
            .pieces
            .par_iter_mut()
            .filter_map(|piece| Self::merge_in_piece(piece, best_pair, new_tok))
            .collect();

        for delta in deltas {
            self.apply_delta(delta);
        }

        self.pair_freqs.remove(&best_pair);
        self.merge_history.push((best_pair, new_tok));

        true
    }

    /// Applies a reduced delta map back into the global pair table.
    fn apply_delta(&mut self, delta: PairDelta) {
        for (pair, change) in delta {
            let current = self.pair_freqs.get(&pair).copied().unwrap_or(0) as i64;
            let new_count = (current + change).max(0) as Count;

            if new_count == 0 {
                self.pair_freqs.remove(&pair);
                continue;
            }

            self.pair_freqs.insert(pair, new_count);

            if new_count >= self.min_frequency {
                self.heap.push(HeapItem {
                    pair,
                    count: new_count,
                });
            }
        }
    }

    /// Merges `best_pair` inside a single piece.
    ///
    /// The returned delta map contains weighted pair-frequency updates.
    fn merge_in_piece(piece: &mut Piece, best_pair: Pair, new_tok: Token) -> Option<PairDelta> {
        if piece.tokens.len() < 2 {
            return None;
        }

        let tokens = &piece.tokens;
        if !tokens
            .windows(2)
            .any(|window| window[0] == best_pair.0 && window[1] == best_pair.1)
        {
            return None;
        }

        let mut delta = FxHashMap::with_capacity_and_hasher(PAIR_MAP_CAPACITY, Default::default());
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let piece_count = piece.count as i64;
        let mut i = 0usize;

        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                *delta.entry(best_pair).or_insert(0) -= piece_count;

                if let Some(&prev_tok) = new_tokens.last() {
                    let old_left = Pair(prev_tok, best_pair.0);
                    *delta.entry(old_left).or_insert(0) -= piece_count;

                    let new_left = Pair(prev_tok, new_tok);
                    *delta.entry(new_left).or_insert(0) += piece_count;
                }

                if i + 2 < tokens.len() {
                    let next_tok = tokens[i + 2];
                    let old_right = Pair(best_pair.1, next_tok);
                    *delta.entry(old_right).or_insert(0) -= piece_count;

                    let new_right = Pair(new_tok, next_tok);
                    *delta.entry(new_right).or_insert(0) += piece_count;
                }

                new_tokens.push(new_tok);
                i += 2;
            } else {
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }

        piece.tokens = new_tokens;
        Some(delta)
    }

    /// Runs up to `num_merges` merge steps with an optional progress bar.
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
            pb.set_message("Training weighted BPE");
            pb.enable_steady_tick(std::time::Duration::from_secs(1));
            Some(pb)
        } else {
            None
        };

        for _ in 0..num_merges {
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

    /// Returns merge history in converter-compatible format.
    pub(crate) fn merge_history(&self) -> Vec<((Token, Token), Token)> {
        self.merge_history
            .iter()
            .map(|(pair, tok)| ((pair.0, pair.1), *tok))
            .collect()
    }

    /// Returns the number of unique pieces tracked by the trainer.
    pub(crate) fn num_pieces(&self) -> usize {
        self.pieces.len()
    }

    /// Returns the number of currently live pairs.
    pub(crate) fn num_live_pairs(&self) -> usize {
        self.pair_freqs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trainer_from_str(s: &str, min_freq: Count) -> WeightedBPETrainer {
        WeightedBPETrainer::from_corpus(s, None, 256, min_freq)
            .expect("reference trainer should build")
    }

    #[test]
    fn test_pretokenize_counts() {
        let pieces = WeightedBPETrainer::pretokenize("the the the cat cat", None)
            .expect("pretokenization should succeed");

        assert_eq!(pieces.len(), 2);
        let the_piece = pieces
            .iter()
            .find(|piece| piece.tokens == vec![116, 104, 101]);
        let cat_piece = pieces
            .iter()
            .find(|piece| piece.tokens == vec![99, 97, 116]);
        assert_eq!(the_piece.map(|piece| piece.count), Some(3));
        assert_eq!(cat_piece.map(|piece| piece.count), Some(2));
    }

    #[test]
    fn test_pretokenize_with_regex() {
        let pieces = WeightedBPETrainer::pretokenize("hello world hello", Some(r"\w+"))
            .expect("regex pretokenization should succeed");

        assert_eq!(pieces.len(), 2);
    }

    #[test]
    fn test_pretokenize_preserves_newlines() {
        let pieces = WeightedBPETrainer::pretokenize("a\n\nb", Some(r"\s*[\r\n]+|\S+"))
            .expect("regex pretokenization should succeed");

        assert!(pieces.iter().any(|piece| piece.tokens == vec![97]));
        assert!(pieces.iter().any(|piece| piece.tokens == vec![10, 10]));
        assert!(pieces.iter().any(|piece| piece.tokens == vec![98]));
    }

    #[test]
    fn test_pair_freqs_weighted() {
        let pieces = vec![
            Piece {
                tokens: vec![97, 98],
                count: 3,
            },
            Piece {
                tokens: vec![97, 98],
                count: 2,
            },
        ];

        let freqs = WeightedBPETrainer::build_pair_freqs_parallel(&pieces);
        assert_eq!(freqs.get(&Pair(97, 98)), Some(&5));
    }

    #[test]
    fn test_single_merge() {
        let mut trainer = trainer_from_str("aaab aaab", 1);
        assert!(trainer.num_pieces() > 0);

        let merged = trainer.merge_step();
        assert!(merged);
        assert_eq!(trainer.merge_history().len(), 1);
    }

    #[test]
    fn test_full_training() {
        let mut trainer = trainer_from_str("the cat sat on the mat the cat sat", 1);
        trainer.train(20, false).expect("training should succeed");

        assert!(!trainer.merge_history().is_empty());
        assert!(trainer.merge_history().len() <= 20);
    }

    #[test]
    fn test_min_frequency_pruning() {
        let mut trainer = trainer_from_str("aaa bbb ccc aaa bbb", 3);
        assert!(!trainer.merge_step());
    }

    #[test]
    fn test_empty_corpus() {
        let trainer = trainer_from_str("", 1);
        assert_eq!(trainer.num_pieces(), 0);
    }

    #[test]
    fn test_merge_history_format() {
        let mut trainer = trainer_from_str("abab abab abab", 1);
        trainer.train(2, false).expect("training should succeed");

        for ((left, right), merged) in trainer.merge_history() {
            assert!(left <= merged);
            assert!(right <= merged || right < 256);
            assert!(merged >= 256);
        }
    }

    #[test]
    fn test_deterministic_merges() {
        let mut t1 = trainer_from_str("hello world hello world foo bar foo", 1);
        let mut t2 = trainer_from_str("hello world hello world foo bar foo", 1);

        t1.train(10, false).expect("training should succeed");
        t2.train(10, false).expect("training should succeed");

        assert_eq!(t1.merge_history(), t2.merge_history());
    }

    #[test]
    fn test_from_pieces_constructor() {
        let pieces = vec![(vec![97, 98, 99], 10), (vec![100, 101], 5)];
        let mut trainer = WeightedBPETrainer::from_pieces(pieces, 256, 1);

        assert_eq!(trainer.num_pieces(), 2);
        trainer.train(5, false).expect("training should succeed");
        assert!(!trainer.merge_history().is_empty());
    }
}
