//! Trains a weighted byte-pair encoding model over deduplicated corpus pieces.
//!
//! This module implements a BPE trainer that first pretokenizes the input
//! corpus into unique byte sequences and tracks how often each sequence
//! appears. Merge steps then operate on those weighted pieces instead of the
//! full corpus, which reduces repeated work while preserving frequency counts.
//!
//! The trainer maintains live pair counts, a heap of candidate merges, and an
//! ordered merge history that can be consumed by the converter layer.
use std::{cmp::Ordering, collections::BinaryHeap};

use fancy_regex::Regex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    error::TrainerError,
    types::{Count, Pair, Token},
};

use indicatif::{ProgressBar, ProgressStyle};

// all heuristic; source: gpt 5.4
const LOCAL_MAP_CAPACITY: usize = 8192;
const PAIR_MAP_CAPACITY: usize = 4096;
const CHUNK_BYTES: usize = 8 * 1024 * 1024;
const HEAP_REBUILD_MARGIN: usize = 1000;

/// Represents a unique tokenized piece and its corpus frequency.
///
/// A `Piece` stores the current token sequence for one distinct chunk produced
/// during pretokenization together with the number of times that exact chunk
/// occurs in the corpus.
struct Piece {
    // Current token sequence for this piece.
    // Intially, tokens represented by their byte representation in [0-255].
    tokens: Vec<Token>,
    // Number of times this piece appears in the corpus.
    count: Count,
}

#[derive(Debug, PartialEq, Eq)]
/// Represents a candidate pair stored in the max-heap.
///
/// Heap items are ordered primarily by pair frequency and secondarily by the
/// pair itself so merge selection remains deterministic across runs.
struct HeapItem {
    pair: Pair,
    count: Count,
}

/// Records weighted pair-count updates produced by merging within one piece.
///
/// Each entry stores the signed change that should be applied to the global
/// live pair counts after replacing all occurrences of a chosen pair inside a
/// single [`Piece`].
type PairDelta = FxHashMap<Pair, i64>;

impl PartialOrd for HeapItem {
    /// Compares two heap items using the total ordering defined by [`Ord`].
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    /// Orders heap items by descending count with deterministic pair tie-breaks.
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            // not required; ensures deterministic tie-breaking among runs.
            .then_with(|| self.pair.0.cmp(&other.pair.0))
            .then_with(|| self.pair.1.cmp(&other.pair.1))
    }
}

/// Trains weighted BPE merges from a pretokenized corpus.
///
/// The trainer keeps deduplicated pieces, weighted adjacent-pair counts, and a
/// max-heap of merge candidates. Each successful merge updates the tracked
/// pieces in place and appends an entry to the merge history.
pub(crate) struct BPETrainer {
    /// All unique subwords/chunks in corpus after pretokenization.
    pieces: Vec<Piece>,
    /// Max-heap tracking the most common token pair.
    heap: BinaryHeap<HeapItem>,
    /// Weighted count per token pair. Source of truth.
    /// Heap entries may be stale; always validate against this map.
    live_pairs: FxHashMap<Pair, Count>,
    /// Ordered merge history: (pair_merged, new_token_id).
    merge_history: Vec<(Pair, Token)>,
    /// Stop training early when best pair count drops below this.
    min_count: Count,
    /// Next available token ID for new merged tokens.
    next_tok: Token,
}

impl BPETrainer {
    /// Builds a trainer from raw corpus text.
    ///
    /// The corpus is first pretokenized into unique weighted pieces. The
    /// trainer then computes initial pair frequencies and seeds the heap with
    /// pairs whose counts satisfy `min_count`.
    ///
    /// # Arguments
    ///
    /// - `corpus` - The input text used for training.
    /// - `pattern` - An optional regex used for pretokenization. When `None`
    ///   or empty, whitespace splitting is used instead.
    /// - `next_tok` - The first token ID available for learned merges.
    /// - `min_count` - The minimum weighted pair count required for a merge to
    ///   be considered.
    ///
    /// # Returns
    ///
    /// A newly initialized [`BPETrainer`].
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] if `pattern` is provided but fails to compile.
    pub(crate) fn from_corpus(
        corpus: &str,
        pattern: Option<&str>,
        next_tok: Token,
        min_count: Count,
    ) -> Result<Self, TrainerError> {
        let pieces = Self::pretokenize(corpus, pattern)?;
        let pair_freqs = Self::build_pair_counts(&pieces);
        let heap = Self::build_heap(&pair_freqs, min_count);

        Ok(Self {
            pieces,
            heap,
            live_pairs: pair_freqs,
            merge_history: Vec::new(),
            min_count,
            next_tok,
        })
    }

    /// Builds a trainer from pre-aggregated token pieces.
    ///
    /// This constructor skips pretokenization and is primarily useful when
    /// tests or upstream code already have weighted token sequences available.
    ///
    /// # Arguments
    ///
    /// - `pieces` - Pairs of token sequences and their frequencies.
    /// - `next_tok` - The first token ID available for learned merges.
    /// - `min_count` - The minimum weighted pair count required for a merge to
    ///   be considered.
    ///
    /// # Returns
    ///
    /// A newly initialized [`BPETrainer`].
    pub(crate) fn from_pieces(
        pieces: Vec<(Vec<Token>, Count)>,
        next_tok: Token,
        min_count: Count,
    ) -> Self {
        // convert to Pieces
        let pieces: Vec<Piece> = pieces
            .into_iter()
            .map(|(tokens, count)| Piece { tokens, count })
            .collect();

        let pair_freqs = Self::build_pair_counts(&pieces);
        let heap = Self::build_heap(&pair_freqs, min_count);

        Self {
            pieces,
            heap,
            live_pairs: pair_freqs,
            merge_history: Vec::new(),
            min_count,
            next_tok,
        }
    }

    /// Runs up to `num_merges` weighted merge steps.
    ///
    /// Training stops early when no live pair remains whose weighted count is
    /// at least `self.min_count`. When `show_progress` is `true`, a progress
    /// bar is created for the requested merge budget.
    ///
    /// # Arguments
    ///
    /// - `num_merges` - The maximum number of merge steps to perform.
    /// - `show_progress` - Whether to display a progress bar during training.
    ///
    /// # Returns
    ///
    /// `Ok(())` after training completes or stops early.
    ///
    /// # Errors
    ///
    /// Returns an error if the progress bar template cannot be constructed.
    pub(crate) fn train(
        &mut self,
        num_merges: usize,
        show_progress: bool,
    ) -> Result<(), indicatif::style::TemplateError> {
        // create progress bar
        let progress = if show_progress {
            let pb = ProgressBar::new(num_merges as u64);
            let style = ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos}/{len}")?;

            pb.set_style(style);
            pb.set_message("Training BPE merges");
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

        // clean up indicatif resources
        if let Some(pb) = &progress {
            pb.finish_and_clear();
        }

        Ok(())
    }

    /// Returns the learned merge history in converter-compatible form.
    ///
    /// Each entry contains the merged pair and the token ID assigned to the
    /// result, in the order the merges were learned.
    ///
    /// # Returns
    ///
    /// A vector of `((left, right), merged)` tuples describing the learned
    /// merge sequence.
    pub(crate) fn merge_history(&self) -> Vec<((Token, Token), Token)> {
        self.merge_history
            .iter()
            .map(|(pair, tok)| ((pair.0, pair.1), *tok))
            .collect()
    }

    /// Returns the number of unique weighted pieces tracked by the trainer.
    ///
    /// # Returns
    ///
    /// The number of distinct pieces produced during pretokenization or passed
    /// to [`Self::from_pieces`].
    pub(crate) fn num_pieces(&self) -> usize {
        self.pieces.len()
    }

    /// Returns the number of currently live adjacent pairs.
    ///
    /// # Returns
    ///
    /// The number of pairs that still have a positive tracked count.
    pub(crate) fn num_live_pairs(&self) -> usize {
        self.live_pairs.len()
    }

    /// Performs a single weighted merge step.
    ///
    /// This selects the best currently live pair, merges it across all pieces,
    /// applies the resulting pair-count deltas, and records the new token in
    /// the merge history.
    ///
    /// # Returns
    ///
    /// `true` if a merge was performed, or `false` if no eligible pair
    /// remained.
    fn merge_step(&mut self) -> bool {
        self.rebuild_heap_if_needed();

        // get max pair
        let (best_pair, _) = match self.max_pair() {
            Some(pair) => pair,
            _ => return false,
        };

        let new_token = self.next_tok;

        // merge best pairs across all pieces in parallel
        let deltas: Vec<PairDelta> = self
            .pieces
            .par_iter_mut()
            // discard None outputs while mapping
            .filter_map(|piece| Self::merge_in_piece(piece, best_pair, new_token))
            .collect();

        for delta in deltas {
            self.apply_delta(delta);
        }
        // this pair no longer exists
        self.live_pairs.remove(&best_pair);
        // track new merge
        self.merge_history.push((best_pair, new_token));

        self.next_tok = new_token + 1;

        true
    }

    /// Removes and returns the highest-priority live pair from the heap.
    ///
    /// Heap entries may be stale because pair counts are updated lazily. This
    /// method validates each popped item against `self.live_pairs` and
    /// requeues refreshed counts when needed.
    ///
    /// # Returns
    ///
    /// `Some((pair, count))` for the best currently valid pair, or `None` if
    /// no live pair satisfies `self.min_count`.
    fn max_pair(&mut self) -> Option<(Pair, Count)> {
        // heap can contain items with stale counts
        while let Some(item) = self.heap.pop() {
            if let Some(&true_count) = self.live_pairs.get(&item.pair) {
                if true_count == item.count && true_count >= self.min_count {
                    return Some((item.pair, item.count));
                }
                // update pair's count if its above min freq
                if true_count >= self.min_count && true_count > 0 {
                    self.heap.push(HeapItem {
                        pair: item.pair,
                        count: true_count,
                    });
                }
                // stale entry with true count below min_freq; continue popping
            }

            // pair doesnt exist in source of truth; ignore
        }

        // heap empty
        None
    }

    /// Rebuilds the heap when stale entries become too numerous.
    ///
    /// This keeps heap operations efficient by discarding outdated items once
    /// the heap grows substantially larger than the source-of-truth pair map.
    fn rebuild_heap_if_needed(&mut self) {
        let heap_size = self.heap.len();
        let live_size = self.live_pairs.len();

        // allow some lee-way because rebuilding heap is expensive: O(|live_pairs|)
        if heap_size > live_size.saturating_mul(2) + HEAP_REBUILD_MARGIN {
            self.heap = Self::build_heap(&self.live_pairs, self.min_count);
        }
    }

    /// Applies weighted pair-count deltas to the global live-pair map.
    ///
    /// Updated pairs are reinserted into the heap when their new counts remain
    /// at or above `self.min_count`, and pairs whose counts drop to zero are
    /// removed entirely.
    ///
    /// # Arguments
    ///
    /// - `deltas` - The signed count adjustments produced by a piece-level
    ///   merge.
    fn apply_delta(&mut self, deltas: PairDelta) {
        for (pair, change) in deltas {
            // copying here so we can set a default
            // without side effects on truth source
            let target_count = self.live_pairs.get(&pair).copied().unwrap_or(0) as i64;

            let new_count = (target_count + change).max(0) as Count;

            // remove dead pair; continue onto next pair
            if new_count == 0 {
                self.live_pairs.remove(&pair);
                continue;
            }

            // update live pair count
            self.live_pairs.insert(pair, new_count);

            // track updated pair in heap
            if new_count >= self.min_count {
                self.heap.push(HeapItem {
                    pair,
                    count: new_count,
                });
            }
        }
    }

    /// Merges every occurrence of `best_pair` within a single piece.
    ///
    /// This helper rewrites `piece.tokens` by replacing each non-overlapping
    /// occurrence of `best_pair` with `new_token`. It also computes the
    /// corresponding weighted adjustments to global pair counts so the caller
    /// can update training state without rebuilding counts from scratch.
    ///
    /// If the piece has fewer than two tokens, or if `best_pair` does not
    /// occur in the piece, this function returns `None` and leaves the piece
    /// unchanged.
    ///
    /// # Arguments
    ///
    /// - `piece` - The piece to update in place.
    /// - `best_pair` - The adjacent token pair to merge.
    /// - `new_token` - The token that replaces each merged occurrence.
    ///
    /// # Returns
    ///
    /// `Some(PairDelta)` containing weighted pair-count changes when at least
    /// one merge is applied, or `None` if the piece does not contain
    /// `best_pair`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut piece = Piece {
    ///     tokens: vec![1, 2, 3],
    ///     count: 2,
    /// };
    ///
    /// let delta = WeightedBPETrainer::merge_in_piece(&mut piece, Pair(1, 2), 256);
    ///
    /// assert!(delta.is_some());
    /// assert_eq!(piece.tokens, vec![256, 3]);
    /// ```
    fn merge_in_piece(piece: &mut Piece, best_pair: Pair, new_token: Token) -> Option<PairDelta> {
        let tokens = &piece.tokens;

        if tokens.len() < 2 {
            return None;
        }

        // check for at least one best pair match
        if !tokens
            .windows(2)
            .any(|pair| pair[0] == best_pair.0 && pair[1] == best_pair.1)
        {
            return None;
        }
        // stores updated counts of all pairs
        // that have their counts affected
        // this map is used to update live pairs in caller
        let mut delta: PairDelta =
            FxHashMap::with_capacity_and_hasher(PAIR_MAP_CAPACITY, Default::default());

        // updated token sequence of piece after merges
        let mut new_tokens: Vec<Token> = Vec::with_capacity(tokens.len());

        // casted for the subtraction operation
        // with hashmap elements
        let piece_count = piece.count as i64;

        let mut i = 0usize;

        while i < tokens.len() {
            // best pairs are merged; pair count
            // decrements proportional to piece count
            if i + 1 < tokens.len() && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                *delta.entry(best_pair).or_insert(0) -= piece_count;

                // decrement count for old left pair
                if let Some(&prev_tok) = new_tokens.last() {
                    let left_pair = Pair(prev_tok, tokens[i]);
                    *delta.entry(left_pair).or_insert(0) -= piece_count;

                    // create new pair with the merged token
                    // count proportional to piece count
                    let new_left_pair = Pair(prev_tok, new_token);
                    *delta.entry(new_left_pair).or_insert(0) += piece_count;
                }

                // decrement count for old right pair
                if let Some(&next_tok) = tokens.get(i + 2) {
                    let right_pair = Pair(tokens[i], next_tok);
                    *delta.entry(right_pair).or_insert(0) -= piece_count;

                    // add new right pair with merged token
                    // again, its count proportional to piece count
                    let new_right_pair = Pair(new_token, next_tok);
                    *delta.entry(new_right_pair).or_insert(0) += piece_count;
                }

                // build new token sequence
                new_tokens.push(new_token);
                // i+1 token got merged
                i += 2;
            } else {
                // no match; advance pointer
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }
        piece.tokens = new_tokens;
        Some(delta)
    }

    /// Pretokenizes the corpus into unique weighted pieces.
    ///
    /// When a regex pattern is provided, all matches are counted directly.
    /// Otherwise each chunk is split with `split_whitespace`. The resulting
    /// byte slices are converted into [`Piece`] values whose initial tokens are
    /// raw byte values in the range `0..=255`.
    ///
    /// # Arguments
    ///
    /// - `corpus` - The input text to split into pieces.
    /// - `pattern` - An optional regex used to extract pieces.
    ///
    /// # Returns
    ///
    /// A vector of unique weighted pieces.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] if `pattern` is provided but fails to compile.
    fn pretokenize(corpus: &str, pattern: Option<&str>) -> Result<Vec<Piece>, TrainerError> {
        if corpus.is_empty() {
            return Ok(Vec::new());
        }

        let chunks = Self::pretok_newline_chunks(corpus, CHUNK_BYTES);

        // collect all regex matches and accumulate chunk frequencies
        let chunk_count: FxHashMap<&[u8], Count> = match pattern {
            Some(pat) if !pat.is_empty() => {
                // ensure pattern compiles upfront
                let _ =
                    fancy_regex::Regex::new(pat).map_err(|e| TrainerError::InvalidPattern(e))?;

                chunks
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                fancy_regex::Regex::new(pat).expect("pattern was validated"),
                                // thousands of unique chunks to process on average
                                // start with buffer to avoid expensive rehash cycles
                                FxHashMap::with_capacity_and_hasher(
                                    LOCAL_MAP_CAPACITY,
                                    Default::default(),
                                ),
                            )
                        },
                        |(re, mut map), line| {
                            Self::pretok_count_matches(&re, line, &mut map);
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
                        |mut a_map, l_map| {
                            for (chunk, count) in l_map {
                                *a_map.entry(chunk).or_insert(0) += count;
                            }
                            a_map
                        },
                    )
            }
            // no pattern provided; split each line by whitespace
            _ => chunks
                .into_par_iter()
                .fold(
                    || FxHashMap::with_capacity_and_hasher(LOCAL_MAP_CAPACITY, Default::default()),
                    |mut map, line| {
                        for chunk in line.split_whitespace() {
                            *map.entry(chunk.as_bytes()).or_insert(0) += 1;
                        }
                        map
                    },
                )
                .reduce(
                    || FxHashMap::with_capacity_and_hasher(LOCAL_MAP_CAPACITY, Default::default()),
                    |mut a_map, l_map| {
                        for (chunk, count) in l_map {
                            *a_map.entry(chunk).or_insert(0) += count;
                        }
                        a_map
                    },
                ),
        };

        Ok(Self::pretok_map_to_pieces(chunk_count))
    }

    /// Counts regex matches from one text chunk into a local frequency map.
    ///
    /// Empty matches are skipped by advancing one byte to avoid infinite
    /// loops.
    ///
    /// # Arguments
    ///
    /// - `re` - The compiled regex used for matching.
    /// - `text` - The chunk of text to scan.
    /// - `map` - The local map that accumulates byte-slice counts.
    fn pretok_count_matches<'a>(re: &Regex, text: &'a str, map: &mut FxHashMap<&'a [u8], Count>) {
        // fancy regex returns result for matches
        // manual loop required
        let mut start = 0;

        while start <= text.len() {
            match re.find_from_pos(text, start) {
                Ok(Some(m)) => {
                    // handle empty string matches
                    if m.start() == m.end() {
                        // advance pointer to prevent infinite loop
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

    /// Splits the corpus into newline-aligned chunks for parallel processing.
    ///
    /// This keeps the exact bytes of each newline inside its chunk so regex
    /// matching still sees the original text layout while avoiding splits in
    /// the middle of a line whenever possible.
    ///
    /// # Arguments
    ///
    /// - `corpus` - The full input corpus.
    /// - `target_bytes` - The approximate target size for each chunk.
    ///
    /// # Returns
    ///
    /// A vector of borrowed string slices covering the full corpus.
    fn pretok_newline_chunks(corpus: &str, target_bytes: usize) -> Vec<&str> {
        if corpus.len() <= target_bytes {
            return vec![corpus];
        }

        let mut chunks = Vec::with_capacity(corpus.len().div_ceil(target_bytes));

        let bytes = corpus.as_bytes();
        // start of line
        let mut start = 0usize;
        // end of line; pos after actual newline byte(s)
        let mut last_nl_end = 0usize;

        for (i, &byte) in bytes.iter().enumerate() {
            if byte == b'\n' {
                // preserve the newline byte(s) inside chunk
                last_nl_end = i + 1;
            }
            // avoid splitting middle of line to ensure correct regex behavior
            // ensures chunk is split after at least one newline encountered
            if i + 1 - start >= target_bytes && start < last_nl_end {
                //  split chunk ending in newline
                chunks.push(&corpus[start..last_nl_end]);
                start = last_nl_end;
            }
        }

        // last chunk might not end with newline
        // manually add trailing bytes to avoid missing corpus end
        if start < corpus.len() {
            chunks.push(&corpus[start..]);
        }
        // no newline in corpus
        if chunks.is_empty() {
            chunks.push(corpus)
        }

        chunks
    }

    /// Converts counted byte slices into owned [`Piece`] values.
    ///
    /// # Arguments
    ///
    /// - `chunk_count` - Weighted byte-slice counts collected during
    ///   pretokenization.
    ///
    /// # Returns
    ///
    /// A vector of owned pieces whose tokens are initialized from raw bytes.
    fn pretok_map_to_pieces(chunk_count: FxHashMap<&[u8], Count>) -> Vec<Piece> {
        chunk_count
            .into_iter()
            .map(|(bytes, count)| Piece {
                tokens: bytes.iter().map(|&b| b as Token).collect(),
                count,
            })
            .collect()
    }

    /// Builds weighted adjacent-pair counts across all pieces.
    ///
    /// Small inputs are counted serially, while larger inputs use Rayon to
    /// build local maps and reduce them into a single result.
    ///
    /// # Arguments
    ///
    /// - `pieces` - The pieces whose adjacent token pairs should be counted.
    ///
    /// # Returns
    ///
    /// A map from token pair to weighted occurrence count.
    fn build_pair_counts(pieces: &[Piece]) -> FxHashMap<Pair, Count> {
        // serial processing for smaller batch
        if pieces.len() < 1_000 {
            let mut pair_counts =
                FxHashMap::with_capacity_and_hasher(LOCAL_MAP_CAPACITY, Default::default());

            for piece in pieces {
                for window in piece.tokens.windows(2) {
                    let pair = Pair(window[0], window[1]);
                    *pair_counts.entry(pair).or_insert(0) += piece.count;
                }
            }

            return pair_counts;
        }

        // parallel processing for bigger batches
        pieces
            .par_iter()
            .fold(
                || FxHashMap::with_capacity_and_hasher(LOCAL_MAP_CAPACITY, Default::default()),
                |mut local_map, piece| {
                    for window in piece.tokens.windows(2) {
                        let pair = Pair(window[0], window[1]);
                        *local_map.entry(pair).or_insert(0) += piece.count;
                    }
                    local_map
                },
            )
            .reduce(
                || FxHashMap::with_capacity_and_hasher(LOCAL_MAP_CAPACITY, Default::default()),
                |mut acum_map, local_map| {
                    for (pair, count) in local_map {
                        *acum_map.entry(pair).or_insert(0) += count;
                    }
                    acum_map
                },
            )
    }

    /// Builds a max-heap of candidate merges from pair counts.
    ///
    /// Only pairs whose counts are at least `min_count` are included.
    ///
    /// # Arguments
    ///
    /// - `pair_counts` - The source-of-truth weighted pair counts.
    /// - `min_count` - The minimum count a pair must have to be inserted.
    ///
    /// # Returns
    ///
    /// A heap ordered by pair frequency with deterministic tie-breaking.
    fn build_heap(pair_counts: &FxHashMap<Pair, Count>, min_count: Count) -> BinaryHeap<HeapItem> {
        pair_counts
            .iter()
            .filter(|(_, count)| **count >= min_count)
            .map(|(pair, count)| HeapItem {
                pair: *pair,
                count: *count,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trainer_from_str(s: &str, min_freq: Count) -> BPETrainer {
        BPETrainer::from_corpus(s, None, 256, min_freq)
            .expect("reference trainer should build")
    }

    #[test]
    fn test_pretokenize_counts() {
        let pieces = BPETrainer::pretokenize("the the the cat cat", None)
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
        let pieces = BPETrainer::pretokenize("hello world hello", Some(r"\w+"))
            .expect("regex pretokenization should succeed");

        assert_eq!(pieces.len(), 2);
    }

    #[test]
    fn test_pretokenize_preserves_newlines() {
        let pieces = BPETrainer::pretokenize("a\n\nb", Some(r"\s*[\r\n]+|\S+"))
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

        let freqs = BPETrainer::build_pair_counts(&pieces);
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
        let mut trainer = trainer_from_str("ab cd ef", 2);
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
        let mut trainer = BPETrainer::from_pieces(pieces, 256, 1);

        assert_eq!(trainer.num_pieces(), 2);
        trainer.train(5, false).expect("training should succeed");
        assert!(!trainer.merge_history().is_empty());
    }
}
