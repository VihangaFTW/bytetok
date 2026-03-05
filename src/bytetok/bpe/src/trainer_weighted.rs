/// Weighted BPE training on unique pieces
use std::{cmp::Ordering, collections::BinaryHeap};

use hashbrown::HashMap;

use crate::types::{Count, Pair, Token};

/// A unique piece (word/chunk) with its corpus frequency.
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
    pair_freqs: HashMap<Pair, Count>,
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
    fn pretokenize(corpus: &str, pattern: Option<&str>) -> Result<Vec<Piece>, String> {
        // first, chunk the corpus by splitting on newlines
        // then, then run regex on each chunk in parallel
        let lines = corpus
            .lines()
            .filter(|chunk| !chunk.is_empty());

        

        Ok()
    }
}
