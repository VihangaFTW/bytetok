//! BPE Encoder - Efficient token encoding using learned merge rules.
//!
//! This implementation applies BPE merges to input token sequences using
//! a priority queue approach inspired by the training algorithm from:
//! "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
//! https://aclanthology.org/2023.findings-acl.38.pdf
//!
//! The encoder applies merges in the order they were learned during training
//! to ensure deterministic and correct application of merge rules.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

use crate::types::Token;

/// Merge order indicates when a merge rule was learned during training.
///
/// Lower values represent earlier merges (e.g., 0 = first merge, 1 = second merge).
type MergeOrder = usize;

/// A pair of adjacent tokens that can potentially be merged.
///
/// Used as a key for looking up merge rules learned during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TokenPair(Token, Token);

/// Item in the priority queue for merge ordering.
///
/// Candidates are ordered by merge_order (earliest first) with position
/// as a tiebreaker. This ensures we apply merges in the correct training order.
#[derive(Debug, PartialEq, Eq)]
struct MergeCandidate {
    /// Merge order from training (0 = first merge, 1 = second merge, etc.).
    ///
    /// Lower values have higher priority and will be applied first.
    merge_order: MergeOrder,
    
    /// The token pair to be merged.
    pair: TokenPair,
    
    /// Position in the token sequence where this pair starts.
    ///
    /// Used as a tiebreaker when multiple pairs have the same merge_order.
    position: usize,
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // We reverse the comparison (other vs self) to create min-heap behavior
        // from Rust's max-heap BinaryHeap.
        // Break ties by position (earlier positions first).
        other
            .merge_order
            .cmp(&self.merge_order)
            .then_with(|| other.position.cmp(&self.position))
    }
}

/// Efficient BPE encoder that applies learned merge rules to token sequences.
///
/// Uses a priority queue approach to ensure merges are applied in the correct
/// order, following the sequence they were learned during training.
///
/// # Time Complexity
///
/// Encoding is O(N log N) where N is the input token sequence length,
/// significantly faster than naive O(N²) implementations.
///
/// # Example
///
/// ```ignore
/// let merge_history = vec![((0, 1), 2), ((2, 0), 3)];
/// let encoder = BPEEncoder::new(merge_history);
/// let tokens = vec![0, 1, 0];
/// let encoded = encoder.encode(tokens);
/// assert_eq!(encoded, vec![3]);
/// ```
pub(crate) struct BPEEncoder {
    /// Maps token pairs to (merged_token, merge_order).
    ///
    /// The merge_order indicates when this merge was learned during training,
    /// with lower values representing earlier merges.
    merges: HashMap<TokenPair, (Token, MergeOrder)>,
}

impl BPEEncoder {
    /// Creates a new BPE encoder from merge history.
    ///
    /// # Arguments
    ///
    /// * `merge_history` - Iterator of merge rules as `((left_token, right_token), merged_token)`.
    ///   The order of iteration determines merge priority (earlier = higher priority).
    ///
    /// # Important
    ///
    /// The ordering in `merge_history` is the source of truth for the encoding algorithm.
    /// Verify the integrity of this ordering before calling `encode()`; incorrect ordering
    /// will produce incorrect encodings.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let merge_history = vec![
    ///     ((0, 1), 256),  // First merge: pair (0,1) -> token 256
    ///     ((256, 0), 257), // Second merge: pair (256,0) -> token 257
    /// ];
    /// let encoder = BPEEncoder::new(merge_history);
    /// ```
    pub(crate) fn new(merge_history: impl IntoIterator<Item = ((Token, Token), Token)>) -> Self {
        let mut merges = HashMap::new();
        for (merge_order, (pair, tok)) in merge_history.into_iter().enumerate() {
            merges.insert(TokenPair(pair.0, pair.1), (tok, merge_order));
        }
        Self { merges }
    }

    /// Encodes a token sequence by applying learned BPE merge rules.
    ///
    /// Merges are applied in the order they were learned during training,
    /// using a priority queue to efficiently find and apply the next merge.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input sequence of tokens to encode.
    ///
    /// # Returns
    ///
    /// A new token sequence with all applicable merge rules applied.
    /// The output will have the same or fewer tokens than the input.
    ///
    /// # Time Complexity
    ///
    /// O(N log N) where N is the input sequence length.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let encoder = BPEEncoder::new(vec![((0, 1), 2)]);
    /// let encoded = encoder.encode(vec![0, 1, 0, 1]);
    /// assert_eq!(encoded, vec![2, 2]);
    /// ```
    pub(crate) fn encode(&self, tokens: Vec<Token>) -> Vec<Token> {
        if tokens.len() <= 1 {
            return tokens;
        }

        let mut heap = BinaryHeap::new();

        // track which positions are consumed by merges
        // history[pos] = Some(token) | None
        // None indicates consumed by prev merge
        let mut results: Vec<Option<Token>> = tokens.iter().map(|&t| Some(t)).collect();

        // populate heap with all mergeable pairs
        self.initialize_minheap(&tokens, &mut heap);

        // process merges in training order (lowest merge_order first)
        while let Some(candidate) = heap.pop() {
            let pos = candidate.position;

            // verify pair at pos is a valid merge.
            // retrieve first token of possible merge pair
            // this token could be a normal token or a merged token
            let Some(left) = results.get(pos).copied().flatten() else {
                continue;
            };
            // find the closest next normal/merges token to the right
            // this handles the case where the first token was a merged token
            // so the token at pos+1 is always None. Need to search further
            let mut right_idx = pos + 1;

            // skip all None entries to the right until a token is found
            while right_idx < results.len() && matches!(results.get(right_idx), Some(None)) {
                right_idx += 1;
            }
            let Some(right) = results.get(right_idx).copied().flatten() else {
                continue;
            };

            // validate live pair
            if candidate.pair != TokenPair(left, right) {
                continue;
            }

            // merge pair using merge rule
            let Some(&(merge_tok, _order)) = self.merges.get(&candidate.pair) else {
                continue;
            };

            // track merge
            results[pos] = Some(merge_tok);
            results[right_idx] = None;

            // new merge candidates might appear after merge
            // check left
            self.track_new_merge_candidate(&mut heap, &results, pos, merge_tok, true);
            // check right
            self.track_new_merge_candidate(&mut heap, &results, pos, merge_tok, false);
        }

        // build final output list: filter None values
        let encoding: Vec<Token> = results.into_iter().flatten().collect();
        encoding
    }
    /// Adds new merge candidates to the priority queue after a successful merge.
    ///
    /// After merging two tokens, the new merged token may form mergeable pairs
    /// with its left and right neighbors. This method checks for such pairs and
    /// adds them to the heap if merge rules exist.
    ///
    /// # Arguments
    ///
    /// * `heap` - The priority queue to add candidates to.
    /// * `results` - Current token sequence with merged positions marked as None.
    /// * `pos` - Position of the newly merged token.
    /// * `merged_tok` - The token ID of the newly merged token.
    /// * `check_left` - If true, checks left neighbor; otherwise checks right neighbor.
    fn track_new_merge_candidate(
        &self,
        heap: &mut BinaryHeap<MergeCandidate>,
        results: &[Option<Token>],
        pos: usize,
        merged_tok: usize,
        check_left: bool,
    ) {
        let mut idx;
        let n = results.len();

        // check left
        if check_left {
            // nothing to the left of position 0
            if pos == 0 {
                return;
            }

            idx = pos - 1;
            // discard all None entries to the left of pos
            while idx > 0 && matches!(results.get(idx), Some(None)) {
                idx -= 1;
            }
        } else {
            if pos + 1 >= n {
                return;
            }
            // check right
            idx = pos + 1;
            // discard all None entries to the right of pos
            while idx < n && matches!(results.get(idx), Some(None)) {
                idx += 1;
            }
        };

        // extract closest token
        let Some(&Some(tok)) = results.get(idx) else {
            return;
        };

        let pair = if check_left {
            TokenPair(tok, merged_tok)
        } else {
            TokenPair(merged_tok, tok)
        };

        // push new pair as candidate if merge rule exists
        if let Some(&(_merge_tok, merge_order)) = self.merges.get(&pair) {
            let position = if check_left { idx } else { pos };
            // track new merge candidate
            heap.push(MergeCandidate {
                merge_order,
                pair,
                position,
            });
        };
    }

    /// Populates the priority queue with all initial mergeable pairs from a token sequence.
    ///
    /// Scans through the token sequence and adds all adjacent pairs that have
    /// learned merge rules to the priority queue.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The input token sequence.
    /// * `heap` - The priority queue to populate.
    fn initialize_minheap(&self, tokens: &[Token], heap: &mut BinaryHeap<MergeCandidate>) {
        for i in 0..tokens.len().saturating_sub(1) {
            let pair = TokenPair(tokens[i], tokens[i + 1]);
            if let Some(&(_, merge_order)) = self.merges.get(&pair) {
                let entry = MergeCandidate {
                    pair,
                    position: i,
                    merge_order,
                };
                heap.push(entry);
            }
        }
    }

    /// Checks if a token pair can be merged according to learned rules.
    ///
    /// # Arguments
    ///
    /// * `left` - The left token in the pair.
    /// * `right` - The right token in the pair.
    ///
    /// # Returns
    ///
    /// `true` if a merge rule exists for this pair, `false` otherwise.
    pub(crate) fn can_merge(&self, left: Token, right: Token) -> bool {
        self.merges.contains_key(&TokenPair(left, right))
    }

    /// Returns the total number of merge rules in this encoder.
    ///
    /// # Returns
    ///
    /// The number of learned merge rules available for encoding.
    pub(crate) fn num_merges(&self) -> usize {
        self.merges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encoding() {
        let history = vec![((0, 1), 2), ((2, 0), 3)];

        let encoder = BPEEncoder::new(history);

        // Basic two-step merge.
        let tokens = vec![0, 1, 0];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![3]);
    }

    #[test]
    fn test_single_token_no_change() {
        let history = vec![((0, 1), 2)];
        let encoder = BPEEncoder::new(history);

        let tokens = vec![7];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![7]);
    }

    #[test]
    fn test_no_merge_rules_apply() {
        let history = vec![((5, 6), 7)];
        let encoder = BPEEncoder::new(history);

        let tokens = vec![0, 1, 2, 3];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_merge_skips_consumed_right() {
        let history = vec![((0, 1), 2), ((2, 0), 3)];
        let encoder = BPEEncoder::new(history);

        let tokens = vec![0, 1, 0, 9];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![3, 9]);
    }

    #[test]
    fn test_merge_skips_consumed_left() {
        let history = vec![((0, 1), 4), ((2, 3), 5), ((4, 5), 6)];
        let encoder = BPEEncoder::new(history);

        let tokens = vec![0, 1, 2, 3];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![6]);
    }

    #[test]
    fn test_tie_break_by_position() {
        let history = vec![((0, 1), 2), ((2, 1), 3)];
        let encoder = BPEEncoder::new(history);

        let tokens = vec![0, 1, 1];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![3]);
    }

    #[test]
    fn test_multiple_disjoint_merges() {
        let history = vec![((0, 0), 2), ((1, 1), 3)];
        let encoder = BPEEncoder::new(history);

        let tokens = vec![0, 0, 1, 1];
        let encoded = encoder.encode(tokens);

        assert_eq!(encoded, vec![2, 3]);
    }
}
