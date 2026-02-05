//! BPE Encoder - Efficient token encoding using learned merge rules.
//!
//! This implementation applies BPE merges to input token sequences using
//! a priority queue approach inspired by the training algorithm from:
//! "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
//! https://aclanthology.org/2023.findings-acl.38.pdf
//!
//! The encoder applies merges in the order they were learned during training
//! to ensure deterministic and correct application of merge rules.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::types::Token;

/// Represents a token pair that can be merged.
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
struct TokenPair(Token, Token);

/// Item in the priority queue for merge ordering.
///
/// Merges learned earlier during training are applied first.
/// This ensures we apply merges in the correct order.
#[derive(Debug, PartialEq, Eq)]
struct MergeCandidate {
    /// Merge order from training (0 = first merge, 1 = second merge, etc.).
    /// Lower values are applied first during encoding.
    merge_order: usize,
    /// The pair to merge.
    pair: TokenPair,
    /// Position in the token sequence where this pair starts.
    position: usize,
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Apply merges with lower merge_order first (earlier in training).
        // We reverse the comparison (other vs self) to create min-heap behavior
        // from Rust's max-heap BinaryHeap.
        // Break ties by position (earlier positions first).
        other
            .merge_order
            .cmp(&self.merge_order)
            .then_with(|| self.position.cmp(&other.position))
    }
}

/// Alternative implementation using a more optimized approach with position tracking.
///
/// This implementation is closer to the training algorithm and uses a priority queue
/// with dynamic position updates for better performance on longer sequences.
pub struct BPEEncoderOptimized {
    merges: HashMap<TokenPair, (usize, usize)>,
}

impl BPEEncoderOptimized {
    /// Create from merge history (same as BPEEncoder).
    pub fn from_merge_history(merge_history: Vec<((Token, Token), Token)>) -> Self {
        let mut merges = HashMap::new();

        for (merge_order, ((tok_a, tok_b), merged_tok)) in merge_history.into_iter().enumerate() {
            let pair = TokenPair(tok_a, tok_b);
            merges.insert(pair, (merged_tok, merge_order));
        }

        Self { merges }
    }

    /// Encode using a priority queue approach with position tracking.
    ///
    /// Uses a skip-based approach to avoid expensive Vec::remove() operations.
    /// Positions are marked as consumed/skipped rather than mutating the vector.
    ///
    /// # Time Complexity
    ///
    /// **O(M log M + N)** where:
    /// - M is the number of merge operations performed
    /// - N is the original sequence length
    ///
    /// Breakdown:
    /// - Building initial heap: O(N)
    /// - Processing merges: O(M log M) for heap operations
    /// - Building final output: O(N)
    ///
    /// This is much better than a naive O(N²) implementation that uses Vec::remove().
    pub fn encode(&self, tokens: Vec<Token>) -> Vec<Token> {
        if tokens.len() <= 1 {
            return tokens;
        }

        let original_tokens = tokens;
        let mut heap = BinaryHeap::new();

        // Track the token at each position: Some(token) = output this, None = skip (consumed by merge).
        // Initialize with all original tokens.
        let mut results: Vec<Option<Token>> = original_tokens.iter().map(|&t| Some(t)).collect();

        // Initialize heap with all mergeable pairs.
        self.populate_heap(&original_tokens, &mut heap);

        // Process merges in training order (lowest merge_order first).
        while let Some(candidate) = heap.pop() {
            let pos = candidate.position;

            // Skip if either position is out of bounds or already consumed.
            // Using `.get()` keeps this robust even if position generation changes in the future.
            let (Some(Some(left_token)), Some(Some(right_token))) =
                (results.get(pos), results.get(pos + 1))
            else {
                continue;
            };

            let pair = TokenPair(*left_token, *right_token);
            if pair != candidate.pair {
                continue;
            }

            // Apply merge: update position pos with merged token, mark pos+1 as consumed.
            let (merged_token, _) = self.merges[&candidate.pair];
            results[pos] = Some(merged_token);
            results[pos + 1] = None;

            // Add new mergeable pairs involving the merged token.
            self.add_new_pairs_optimized(pos, merged_token, &results, &mut heap);
        }

        // Build final output by filtering out consumed positions (None values).
        results.into_iter().flatten().collect()
    }

    /// Populate the priority queue with all initial mergeable pairs.
    fn populate_heap(&self, tokens: &[Token], heap: &mut BinaryHeap<MergeCandidate>) {
        for i in 0..tokens.len().saturating_sub(1) {
            let pair = TokenPair(tokens[i], tokens[i + 1]);

            if let Some(&(_, merge_order)) = self.merges.get(&pair) {
                heap.push(MergeCandidate {
                    merge_order,
                    pair,
                    position: i,
                });
            }
        }
    }

    /// Add new mergeable pairs after a merge operation.
    ///
    /// Checks pairs involving the newly merged token at position `pos`.
    fn add_new_pairs_optimized(
        &self,
        pos: usize,
        merged_token: Token,
        result: &[Option<Token>],
        heap: &mut BinaryHeap<MergeCandidate>,
    ) {
        // Check left neighbor pair: (left_pos, pos).
        // Find the nearest non-consumed position to the left of `pos`.
        if pos > 0 {
            let mut left_pos = pos - 1;

            // Walk left over consumed slots (`None`) until we either find a live token
            // or exhaust the left side of the sequence.
            while left_pos > 0 && result.get(left_pos) == Some(&None) {
                left_pos -= 1;
            }

            if let Some(left_token) = result.get(left_pos).and_then(|&t| t) {
                let pair = TokenPair(left_token, merged_token);
                if let Some(&(_, merge_order)) = self.merges.get(&pair) {
                    heap.push(MergeCandidate {
                        merge_order,
                        pair,
                        position: left_pos,
                    });
                }
            }
        }

        // Check right neighbor pair: (pos, right_pos).
        // Find the nearest non-consumed position to the right of `pos`.
        let mut right_pos = pos + 1;
        while let Some(None) = result.get(right_pos) {
            right_pos += 1;
        }

        if let Some(right_token) = result.get(right_pos).and_then(|&t| t) {
            let pair = TokenPair(merged_token, right_token);
            if let Some(&(_, merge_order)) = self.merges.get(&pair) {
                heap.push(MergeCandidate {
                    merge_order,
                    pair,
                    position: pos,
                });
            }
        }
    }

    /// Check if a token pair can be merged.
    pub fn can_merge(&self, left: Token, right: Token) -> bool {
        self.merges.contains_key(&TokenPair(left, right))
    }

    /// Get the number of merge rules.
    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encoding() {
        // Create simple merge rules.
        let history = vec![
            ((0, 1), 2), // Merge (0,1) -> 2 first.
            ((2, 0), 3), // Then merge (2,0) -> 3.
        ];

        let encoder = BPEEncoderOptimized::from_merge_history(history);

        // Test encoding.
        let tokens = vec![0, 1, 0];
        let encoded = encoder.encode(tokens);

        // Should apply both merges: [0,1,0] -> [2,0] -> [3].
        assert_eq!(encoded, vec![3]);
    }

    #[test]
    fn test_no_merges_possible() {
        let history = vec![((0, 1), 2)];
        let encoder = BPEEncoderOptimized::from_merge_history(history);

        // Tokens that don't form the merge pair.
        let tokens = vec![0, 2, 1];
        let encoded = encoder.encode(tokens);

        // Should remain unchanged.
        assert_eq!(encoded, vec![0, 2, 1]);
    }

    #[test]
    fn test_single_token() {
        let history = vec![((0, 1), 2)];
        let encoder = BPEEncoderOptimized::from_merge_history(history);

        let tokens = vec![0];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![0]);
    }

    #[test]
    fn test_empty_sequence() {
        let history = vec![((0, 1), 2)];
        let encoder = BPEEncoderOptimized::from_merge_history(history);

        let tokens: Vec<Token> = vec![];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, Vec::<Token>::new());
    }

    #[test]
    fn test_merge_priority() {
        // Create merges with explicit merge orders.
        let history = vec![
            ((0, 0), 2), // Merge order 0 (applied first).
            ((0, 1), 3), // Merge order 1 (applied second).
        ];

        let encoder = BPEEncoderOptimized::from_merge_history(history);

        // Sequence where both pairs exist.
        let tokens = vec![0, 0, 1];
        let encoded = encoder.encode(tokens);

        // Should apply (0,0) first: [0,0,1] -> [2,1].
        // Then no more merges possible.
        assert_eq!(encoded, vec![2, 1]);
    }

    #[test]
    fn test_optimized_encoder() {
        let history = vec![((0, 1), 2), ((2, 0), 3)];

        let encoder = BPEEncoderOptimized::from_merge_history(history);

        let tokens = vec![0, 1, 0];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![3]);
    }

    #[test]
    fn test_can_merge() {
        let history = vec![((0, 1), 2), ((1, 2), 3)];
        let encoder = BPEEncoderOptimized::from_merge_history(history);

        assert!(encoder.can_merge(0, 1));
        assert!(encoder.can_merge(1, 2));
        assert!(!encoder.can_merge(0, 2));
        assert!(!encoder.can_merge(2, 3));
    }

    #[test]
    fn test_num_merges() {
        let history = vec![((0, 1), 2), ((1, 2), 3), ((2, 3), 4)];
        let encoder = BPEEncoderOptimized::from_merge_history(history);

        assert_eq!(encoder.num_merges(), 3);
    }
}
