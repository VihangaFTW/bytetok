/// Weighted BPE training on unique pieces
use std::{cmp::Ordering, collections::BinaryHeap};

use fancy_regex::Regex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    error::TokenizerInitError,
    types::{Count, Pair, Token},
};

const LOCAL_MAP_CAPACITY: usize = 8192;
const CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// A unique piece represeting a subword with its corpus frequency.
struct Piece {
    // Current token sequence for this piece.
    // Intially, tokens represented by their byte representation in [0-255].
    tokens: Vec<Token>,
    // Number of times this piece appears in the corpus.
    count: Count,
}

#[derive(Debug, PartialEq, Eq)]
struct HeapItem {
    pair: Pair,
    count: Count,
}

impl PartialOrd for HeapItem {
    // Ord implementation ensures all heap items are comparable.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            // not required; ensures deterministic tie-breaking among runs.
            .then_with(|| self.pair.0.cmp(&other.pair.0))
            .then_with(|| self.pair.1.cmp(&other.pair.1))
    }
}

pub(crate) struct WeightedBPETrainer {
    /// All unique subwords/chunks in corpus after pretokenization.
    pieces: Vec<Piece>,
    /// Max-heap tracking the most frequent token pair.
    heap: BinaryHeap<HeapItem>,
    /// Weighted frequency per token pair. Source of truth.
    /// Heap entries may be stale; always validate against this map.
    pair_freqs: FxHashMap<Pair, Count>,
    /// Ordered merge history: (pair_merged, new_token_id).
    merge_history: Vec<(Pair, Token)>,
    /// Stop training early when best pair freq drops below this.
    min_frequency: Count,
    /// Next available token ID for new merged tokens.
    next_tok: Token,
}

impl WeightedBPETrainer {
    /// Splits corpus bytes against a regex pattern and
    /// counts unique chunks.
    ///
    /// Each chunk becomes a `Piece` whose initial tokens
    /// are raw bytes [0,255].
    fn pretokenize(corpus: &str, pattern: Option<&str>) -> Result<Vec<Piece>, TokenizerInitError> {
        if corpus.is_empty() {
            return Ok(Vec::new());
        }

        let chunks = Self::pretok_newline_chunks(corpus, CHUNK_BYTES);

        // collect all regex matches and accumulate chunk frequencies
        let chunk_freq: FxHashMap<&[u8], Count> = match pattern {
            Some(pat) if !pat.is_empty() => {
                // ensure pattern compiles upfront
                let _ = fancy_regex::Regex::new(pat)
                    .map_err(|e| TokenizerInitError::InvalidPattern(e))?;

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

        Ok(Self::pretok_map_to_pieces(chunk_freq))
    }

    /// Counts regex matches directly into a locally shared map.
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

    /// Builds parallel chunks while preserving newline tokens.
    ///
    /// This keeps the exact bytes of each newline in the chunk so regex
    /// branches still see the original text.
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

    /// Converts borrowed chunk keys into owned `Piece`s.
    fn pretok_map_to_pieces(chunk_freq: FxHashMap<&[u8], Count>) -> Vec<Piece> {
        chunk_freq
            .into_iter()
            .map(|(bytes, count)| Piece {
                tokens: bytes.iter().map(|&b| b as Token).collect(),
                count,
            })
            .collect()
    }



}
