//! Core BPE training algorithm (Algorithm 2).
//!
//! Optimized implementation from "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
//! https://aclanthology.org/2023.findings-acl.38.pdf
//!
//! Time complexity: O(N log V) vs O(NV) for naive implementation.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    ops::ControlFlow,
};

use crate::types::{TextIdx, Token, TokenFreq};

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
struct TokenPair(Token, Token);

/// Node in doubly-linked list.
/// It is used to represent training token sequence.
#[derive(Debug)]
struct Node {
    token: Token,
    /// Index of neighbouring nodes in token seq.
    prev_idx: Option<TextIdx>,
    next_idx: Option<TextIdx>,
}

/// Item in max heap.
#[derive(Debug, PartialEq, Eq)]
struct HeapItem {
    freq: TokenFreq,
    pair: TokenPair,
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Ord implementation ensures all heap items are comparable.
        Some(self.cmp(other))
    }
}

/// Implement Ord for heap item: highest freq pair at top.
/// Needed for heap to compare items.
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by freq between two pairs.
        // Break tie by comparing pair values.
        self.freq
            .cmp(&other.freq)
            .then_with(|| self.pair.0.cmp(&other.pair.0))
            .then_with(|| self.pair.1.cmp(&other.pair.1))
    }
}

/// BPE training structure.
///
/// This trainer uses a *Vec-as-arena* pattern to represent a
/// linked list of nodes. Direct `Node` ↔ `Node` references are
/// avoided due to Rust's ownership and borrowing rules.
///
/// Nodes are stored in a `Vec<Option<Node>>`, where:
/// - The `Vec` provides stable indices for nodes
/// - Deletions are O(1) by setting entries to `None`
/// - Traversal is done via index-based left/right links inside `Node`
#[derive(Debug, Default)]
pub(crate) struct BPETrainer {
    /// Storage arena for nodes.
    ///
    /// `None` represents a deleted node.
    nodes: Vec<Option<Node>>,

    /// Vec index pointing linked list's head.
    head_idx: Option<usize>,

    /// Max heap of (frequency, pair) sorted by frequency.
    ///
    /// Contains stale entries that need to be guarded against.
    heap: BinaryHeap<HeapItem>,

    /// Positions where each token pair occurs.
    /// pair -> set of positions [index of first token of every pair].
    ///
    /// Source of truth for pair positions.
    pair_positions: HashMap<TokenPair, HashSet<usize>>,

    /// Current frequencies of each pair.
    ///
    /// Source of truth for pair frequencies.
    pair_freqs: HashMap<TokenPair, usize>,

    /// Next available merge token ID.
    next_tok: Token,

    /// History of merges: (token_a, token_b) -> merged_token.
    merge_history: Vec<((Token, Token), Token)>,
}

impl BPETrainer {
    /// Create a new BPE trainer from an initial token sequence.
    ///
    /// # Arguments
    /// * `tokens` - Initial sequence of tokens (e.g., bytes 0-255)
    /// * `next_tok` - Next available token ID (e.g., 256 for bytes)
    ///
    /// # Example
    /// ```ignore
    /// let tokens = vec![0, 1, 0, 1, 2];
    /// let mut trainer = BPETrainer::new(tokens, 256);
    /// ```
    pub(crate) fn new(tokens: &[Token], next_tok: Token) -> Self {
        let n = tokens.len();
        // Create linked list storage.
        let mut nodes = Vec::with_capacity(n);

        for (i, &token) in tokens.iter().enumerate() {
            let prev = if i > 0 { Some(i - 1) } else { None };
            let next = if i < n - 1 { Some(i + 1) } else { None };
            nodes.push(Some(Node {
                token,
                prev_idx: prev,
                next_idx: next,
            }));
        }

        // Handle case: tokens vec might be empty.
        let head = if nodes.is_empty() { None } else { Some(0) };

        let mut trainer = BPETrainer {
            nodes,
            head_idx: head,
            heap: BinaryHeap::new(),
            pair_positions: HashMap::new(),
            pair_freqs: HashMap::new(),
            next_tok,
            merge_history: Vec::new(),
        };

        // Initialize fields with token pair counts.
        trainer.build_initial_pairs();

        trainer
    }

    /// Perform one merge operation.
    /// Refer to algorithm 2: https://aclanthology.org/2023.findings-acl.38.pdf
    ///
    /// Returns true if a merge was performed, false if no pairs remain.
    ///
    /// # Time Complexity
    /// Worst case: `O(N*log V)` where `N` is the token sequence length and `V` refers to vocab size.
    ///
    /// For reference, a naive BPE implementation has worst case `O(N*V)`.
    pub(crate) fn merge_step(&mut self) -> bool {
        // Get the most frequent pair.
        let (merge_pair, _merge_freq) = match self.get_max_pair() {
            Some(pair) => pair,
            // No pairs to merge.
            None => return false,
        };

        #[cfg(debug_assertions)]
        println!(
            "Merging pair ({}, {}) -> token {}",
            merge_pair.0, merge_pair.1, self.next_tok
        );

        // Get all positions where this token occurs.
        // Note that we need to modify the positions so copy values.
        // Explicit type Vec required here to let compiler know what to collect into.
        let positions: Vec<TextIdx> = self
            .pair_positions
            .get(&merge_pair)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();

        // Update token id for next merge.
        let new_tok_id = self.next_tok;
        self.next_tok += 1;

        // Process each occurrence of target pairs and perform a merge.
        for &pos in &positions {
            // Retrieve two node indices.
            let (idx1, idx2) = match self.get_merge_idxs(merge_pair, pos) {
                ControlFlow::Continue(idxs) => idxs,
                ControlFlow::Break(_) => continue,
            };

            let cur_prev_idx = self.nodes[idx1].as_ref().and_then(|n| n.prev_idx);
            let new_next_idx = self.nodes[idx2].as_ref().and_then(|n| n.next_idx);

            // Remove old neighbours.
            self.remove_neighbours(merge_pair, idx1, idx2);

            // Perform merge in linked list.
            self.merge_pair_in_list(new_next_idx, new_tok_id, idx1, idx2);

            // Add new neighbours.
            self.add_neighbours(new_tok_id, idx1, cur_prev_idx, new_next_idx);
        }

        // Record merge in history.
        self.merge_history
            .push(((merge_pair.0, merge_pair.1), new_tok_id));

        // Un-track merged pair.
        self.pair_freqs.remove(&merge_pair);
        self.pair_positions.remove(&merge_pair);

        true
    }

    /// Train BPE with M merges.
    ///
    /// # Arguments
    /// * `num_merges` - Number of merge operations to perform.
    pub(crate) fn train(&mut self, num_merges: usize) {
        for _i in 0..num_merges {
            if !self.merge_step() {
                #[cfg(debug_assertions)]
                println!("No more pairs to merge after {} merges", _i);
                break;
            }
        }
    }

    /// Get the current token sequence as a vector.
    pub(crate) fn get_encodings(&self) -> Vec<Token> {
        let mut result = Vec::new();

        let mut current = self.head_idx;

        while let Some(idx) = current {
            if let Some(node) = &self.nodes[idx] {
                result.push(node.token);
                current = node.next_idx;
            } else {
                break;
            }
        }

        result
    }

    /// Get the merge history as a vector of ((token_a, token_b), merged_token).
    pub(crate) fn get_merge_history(&self) -> Vec<((Token, Token), Token)> {
        self.merge_history.clone()
    }

    /// Build initial pair frequencies from the linked list.
    ///
    /// # Time Complexity
    /// This scans the sequence once: O(N).
    fn build_initial_pairs(&mut self) {
        let mut cur_idx = self.head_idx;

        // Record pair counts and positions.
        while let Some(idx) = cur_idx {
            // Pattern match to get node (initially to get head node).
            if let Some(node) = &self.nodes[idx] {
                // Pattern match next field to an index that maps to a node.
                if let Some(next_idx) = node.next_idx
                    && let Some(next_node) = &self.nodes[next_idx]
                {
                    let pair = TokenPair(node.token, next_node.token);
                    // Increment pair frequency.
                    *self.pair_freqs.entry(pair).or_insert(0) += 1;
                    // Record pair position.
                    self.pair_positions.entry(pair).or_default().insert(idx);
                }
                // Move onto the start of next pair.
                cur_idx = node.next_idx;
            } else {
                break;
            }
        }

        // Populate max heap with pair counts.
        for (&pair, &freq) in &self.pair_freqs {
            self.heap.push(HeapItem { freq, pair })
        }
    }

    /// Remove and get most frequent (pair, freq) from max heap.
    /// Return None if no pair exists.
    fn get_max_pair(&mut self) -> Option<(TokenPair, TokenFreq)> {
        // After merges, some entries in the heap will have stale counts.
        // So we need to keep popping from the heap until we find
        // a pair that exists in current token sequence and validate its count.
        while let Some(entry) = self.heap.pop() {
            if let Some(&true_freq) = self.pair_freqs.get(&entry.pair)
                && true_freq == entry.freq
            {
                return Some((entry.pair, entry.freq));
            }
            // Stale entry, discard it and keep on popping until we get a valid pair.
        }

        None
    }

    /// Update bookkeeping data regarding neighbouring pairs
    /// such as their start position and counts.
    /// Called when left and right neighbor pairs are removed
    /// after a merge because their counts need to decrease.
    /// The actual nodes in the linked list are NOT modified.
    fn remove_pair_at(&mut self, idx: TextIdx, pair: TokenPair) {
        // Decrement count.
        if let Some(freq) = self.pair_freqs.get_mut(&pair)
            && *freq > 0
        {
            *freq -= 1;
        }

        // Remove start position.
        if let Some(pos_set) = self.pair_positions.get_mut(&pair) {
            pos_set.remove(&idx);
        }
    }

    /// Add a pair occurrence at a specific position.
    /// The actual nodes in the linked list are NOT modified.
    fn add_pair_at(&mut self, idx: usize, pair: TokenPair) {
        // Add position.
        self.pair_positions.entry(pair).or_default().insert(idx);
        // Increment freq.
        let freq = self.pair_freqs.entry(pair).or_insert(0);
        *freq += 1;

        // Add new pair in heap.
        self.heap.push(HeapItem { freq: *freq, pair });
    }

    fn add_neighbours(
        &mut self,
        new_tok_id: TextIdx,
        idx1: TextIdx,
        cur_prev_idx: Option<usize>,
        new_next_idx: Option<TextIdx>,
    ) {
        // Increment freq for new left pair and track new position.
        if let Some(prev_idx) = cur_prev_idx
            && let Some(prev_node) = &self.nodes[prev_idx]
        {
            let new_pair = TokenPair(prev_node.token, new_tok_id);
            self.add_pair_at(prev_idx, new_pair);
        }

        // Increment freq for new right pair and track new position.
        if let Some(next_idx) = new_next_idx
            && let Some(next_node) = &self.nodes[next_idx]
        {
            let new_pair = TokenPair(new_tok_id, next_node.token);
            self.add_pair_at(idx1, new_pair);
        }
    }

    fn merge_pair_in_list(
        &mut self,
        next_idx: Option<TextIdx>,
        tok_id: Token,
        idx1: TextIdx,
        idx2: TextIdx,
    ) {
        // Update first token of pair to contain merged token.
        if let Some(node) = &mut self.nodes[idx1] {
            node.token = tok_id;
            node.next_idx = next_idx;
        }

        // Update new right's prev pointer to merged token.
        if let Some(new_right_idx) = next_idx
            && let Some(new_right_node) = &mut self.nodes[new_right_idx]
        {
            new_right_node.prev_idx = Some(idx1);
        }

        // Mark second token of merge pair as deleted.
        self.nodes[idx2] = None;
    }

    /// Decrement neighbour pair counts and update global pair positions
    /// during a merge step.
    fn remove_neighbours(&mut self, merge_pair: TokenPair, idx1: TextIdx, idx2: TextIdx) {
        // Remove old left neighbour if it exists.
        if let Some(prev_idx) = self.nodes[idx1].as_ref().and_then(|n| n.prev_idx)
            // Check if prev node (token) exists.
            && let Some(prev_node) = &self.nodes[prev_idx]
        {
            let old_pair = TokenPair(prev_node.token, merge_pair.0);
            self.remove_pair_at(prev_idx, old_pair);
        }

        // Remove old right neighbour if it exists.
        if let Some(next_idx) = self.nodes[idx2].as_ref().and_then(|n| n.next_idx)
            && let Some(next_node) = &self.nodes[next_idx]
        {
            let old_pair = TokenPair(merge_pair.1, next_node.token);
            self.remove_pair_at(idx2, old_pair);
        }
    }

    /// Extracts and verifies the indices of two nodes
    /// representing given `pair` at `self.nodes[pos]`.
    fn get_merge_idxs(
        &mut self,
        pair: TokenPair,
        pos: TextIdx,
    ) -> ControlFlow<(), (TextIdx, TextIdx)> {
        let idx1 = pos;
        // Retrieve pair's remaining token index.
        let idx2 = match &self.nodes[idx1] {
            Some(node1) => match node1.next_idx {
                Some(idx2) => idx2,
                // Invalid next token (node deleted by previous merge).
                None => return ControlFlow::Break(()),
            },
            // Invalid start token (node deleted by a previous merge).
            None => return ControlFlow::Break(()),
        };

        let is_target = match (&self.nodes[idx1], &self.nodes[idx2]) {
            (Some(node1), Some(node2)) => node1.token == pair.0 && node2.token == pair.1,
            _ => false,
        };

        // Verify this is still the target pair.
        if !is_target {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue((idx1, idx2))
    }

    /// Print the current state for debugging.
    pub(crate) fn print_state(&self) {
        print!("Tokens: ");
        for token in self.get_encodings() {
            print!("{} ", token);
        }
        println!();

        println!("Top pairs:");
        let mut pairs: Vec<_> = self.pair_freqs.iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(a.1));
        for (pair, freq) in pairs.iter().take(5) {
            println!("  ({}, {}) : {}", pair.0, pair.1, freq);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_merge() {
        let tokens = [0, 1, 0, 0, 1, 1, 0, 0];
        let mut trainer = BPETrainer::new(&tokens, 2);
        trainer.train(3);
        let final_tokens = trainer.get_encodings();
        // Should have fewer tokens than we started with.
        assert!(final_tokens.len() < 8);
    }

    #[test]
    fn test_empty_sequence() {
        let tokens = [];
        let trainer = BPETrainer::new(&tokens, 0);
        assert_eq!(trainer.get_encodings(), Vec::<usize>::new());
    }

    #[test]
    fn test_single_token() {
        let tokens = [0];
        let trainer = BPETrainer::new(&tokens, 1);
        assert_eq!(trainer.get_encodings(), vec![0]);
    }

    #[test]
    fn test_no_merges_needed() {
        let tokens = [0, 1, 2, 3];
        let mut trainer = BPETrainer::new(&tokens, 4);
        // Try to merge but no pairs repeat.
        let merged = trainer.merge_step();
        // Should successfully merge once (any adjacent pair).
        assert!(merged);
    }
}
