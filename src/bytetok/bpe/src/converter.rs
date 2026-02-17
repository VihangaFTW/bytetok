//! BPE Converter - Efficient token encoding using learned merge rules.
//!
//! This implementation applies BPE merges to input token sequences using
//! a priority queue approach inspired by the training algorithm from:
//! "A Formal Perspective on Byte-Pair Encoding"
//! https://aclanthology.org/2023.findings-acl.38.pdf
//!
//! The converter applies merges in the order they were learned during training
//! to ensure deterministic and correct application of merge rules.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

use crate::{error::{DecodeError, SpecialTokenError}, types::{ByteSeq, MergeOrder, Token}};



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

/// Efficient BPE converter that applies learned merge rules to token sequences.
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
/// let converter = BPEConverter::new(merge_history);
/// let tokens = vec![0, 1, 0];
/// let encoded = converter.encode(tokens);
/// assert_eq!(encoded, vec![3]);
/// ```
pub(crate) struct BPEConverter {
    /// Maps token pairs to (merged_token, merge_order).
    ///
    /// The merge_order indicates when this merge was learned during training,
    /// with lower values representing earlier merges.
    merges: HashMap<TokenPair, (Token, MergeOrder)>,
    
    /// Maps token IDs to their byte sequences.
    ///
    /// - vocab[0..256]: Base vocabulary (single bytes)
    /// - vocab[256..]: Merged tokens (concatenated byte sequences)
    vocab: Vec<ByteSeq>,

    special_tokens: HashMap<Token, ByteSeq>
}

impl BPEConverter {
    /// Creates a new BPE converter from merge history.
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
    /// will produce incorrect encodings AND degrade performance.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let merge_history = vec![
    ///     ((0, 1), 256),  // First merge: pair (0,1) -> token 256
    ///     ((256, 0), 257), // Second merge: pair (256,0) -> token 257
    /// ];
    /// let converter = BPEConverter::new(merge_history);
    /// ```
    pub(crate) fn new(merge_history: impl IntoIterator<Item = ((Token, Token), Token)>, special_tokens: &[(&str, Token)]) -> Result<Self, SpecialTokenError> {
        let mut merges = HashMap::new();
        let mut vocab: Vec<ByteSeq>  = Vec::new();
        
        let mut max_token_id = 0;

        // initialize base vocabulary (0-255 → single bytes).
        for i in 0..256 {
            vocab.push(vec![i as u8])
        }
        
        for (merge_order, (pair, tok)) in merge_history.into_iter().enumerate() {
            merges.insert(TokenPair(pair.0, pair.1), (tok, merge_order));
            
            // track maximum token id across all tokens
            max_token_id = max_token_id.max(pair.0).max(pair.1).max(tok);
            
            // build vocabulary entry for merged token by concatenating constituent byte sequences
            // ensure vector can be safely indexed at position tok without panicking; O(1) time
            // ensure merge history is in order; otherwise O(N) time
            while vocab.len() <= tok {
                vocab.push(Vec::new());
            }
            // we cant reuse the vector pushed above due to borrow checker
            // not allowing simultaneous mutable and immutable borrow of vocab
            // solution is to use a temporary vector to accumulate byte sequence
            let mut merged_bytes = Vec::new();
            if let Some(left_bytes) = vocab.get(pair.0) {
                merged_bytes.extend_from_slice(left_bytes);
            }
            if let Some(right_bytes) = vocab.get(pair.1) {
                merged_bytes.extend_from_slice(right_bytes);
            }
            vocab[tok] = merged_bytes;
        }
        // add special tokens to vocab
        let mut special_map: HashMap<Token, ByteSeq> = HashMap::new();
        
        for (s, tok) in special_tokens.iter(){
            // 
            if *tok < vocab.len(){
                return Err(SpecialTokenError::IllegalToken(*tok))
            }
            special_map.insert(*tok, s.as_bytes().to_vec());
        }

        Ok(Self {
            merges,
            vocab,
            special_tokens: special_map
        })
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
    /// let converter = BPEConverter::new(vec![((0, 1), 2)]);
    /// let encoded = converter.encode(vec![0, 1, 0, 1]);
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

    /// Returns the total number of merge rules in this converter.
    ///
    /// # Returns
    ///
    /// The number of learned merge rules available for encoding.
    pub(crate) fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// Returns a reference to the vocabulary.
    ///
    /// # Returns
    ///
    /// Reference to the vocabulary mapping token IDs to byte sequences.
    pub(crate) fn vocab(&self) -> &[ByteSeq] {
        &self.vocab
    }

    /// Decodes a token sequence back into bytes.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Sequence of token IDs to decode.
    ///
    /// # Returns
    ///
    /// The concatenated byte sequence representing the decoded tokens.
    /// Returns an empty vector if any token ID is out of bounds.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let converter = BPEConverter::new(vec![((97, 98), 256)]);
    /// let bytes = converter.decode(&[256, 99]);
    /// assert_eq!(bytes, vec![97, 98, 99]); // "abc"
    /// ```
    pub(crate) fn decode(&self, tokens: &[Token]) -> Result<ByteSeq, DecodeError> {
        let mut result = Vec::new();
        for &token in tokens {
            if let Some(bytes) = self.vocab.get(token) {
                result.extend_from_slice(bytes);
                continue;
            }

            if let Some(bytes) = self.special_tokens.get(&token) {
                result.extend_from_slice(bytes);
                continue;
            }

            return Err(DecodeError::UnknownToken(token));
        };
        Ok(result)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encoding() {
        let history = vec![((0, 1), 2), ((2, 0), 3)];

        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        // Basic two-step merge.
        let tokens = vec![0, 1, 0];
        let encoded = converter.encode(tokens);

        assert_eq!(encoded, vec![3]);
    }

    #[test]
    fn test_single_token_no_change() {
        let history = vec![((0, 1), 2)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        let tokens = vec![7];
        let encoded = converter.encode(tokens);

        assert_eq!(encoded, vec![7]);
    }

    #[test]
    fn test_no_merge_rules_apply() {
        let history = vec![((5, 6), 7)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        let tokens = vec![0, 1, 2, 3];
        let encoded = converter.encode(tokens);

        assert_eq!(encoded, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_merge_skips_consumed_right() {
        let history = vec![((0, 1), 2), ((2, 0), 3)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        let tokens = vec![0, 1, 0, 9];
        let encoded = converter.encode(tokens);

        assert_eq!(encoded, vec![3, 9]);
    }

    #[test]
    fn test_merge_skips_consumed_left() {
        let history = vec![((0, 1), 4), ((2, 3), 5), ((4, 5), 6)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        let tokens = vec![0, 1, 2, 3];
        let encoded = converter.encode(tokens);

        assert_eq!(encoded, vec![6]);
    }

    #[test]
    fn test_tie_break_by_position() {
        let history = vec![((0, 1), 2), ((2, 1), 3)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        let tokens = vec![0, 1, 1];
        let encoded = converter.encode(tokens);

        assert_eq!(encoded, vec![3]);
    }

    #[test]
    fn test_multiple_disjoint_merges() {
        let history = vec![((0, 0), 2), ((1, 1), 3)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        let tokens = vec![0, 0, 1, 1];
        let encoded = converter.encode(tokens);

        assert_eq!(encoded, vec![2, 3]);
    }

    #[test]
    fn test_decode_base_tokens() {
        let history = vec![];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        // Decode "abc" (UTF-8 bytes: [97, 98, 99]).
        let decoded = converter.decode(&[97, 98, 99]).expect("decoding failed");
        assert_eq!(decoded, vec![97, 98, 99]);
    }

    #[test]
    fn test_decode_merged_tokens() {
        let history = vec![((97, 98), 256)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        // Token 256 should decode to [97, 98].
        let decoded = converter.decode(&[256, 99]).expect("decoding failed");
        assert_eq!(decoded, vec![97, 98, 99]);
    }

    #[test]
    fn test_decode_nested_merges() {
        let history = vec![((97, 98), 256), ((256, 99), 257)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        // Token 257 should decode to [97, 98, 99].
        let decoded = converter.decode(&[257]).expect("decoding failed");
        assert_eq!(decoded, vec![97, 98, 99]);
    }

    #[test]
    fn test_decode_invalid_token() {
        let history = vec![((97, 98), 256)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        // Token 999 doesn't exist - should return an error.
        let result = converter.decode(&[97, 999]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_special_token() {
        let converter = BPEConverter::new(vec![], &[("<|eot|>", 1000)]).expect("converter init failed");
        let decoded = converter.decode(&[1000]).expect("decoding failed");
        assert_eq!(decoded, b"<|eot|>".to_vec());
    }

    #[test]
    fn test_decode_special_token_mixed_with_vocab() {
        let history = vec![((97, 98), 256)];
        let converter = BPEConverter::new(history, &[("<|eot|>", 1000)]).expect("converter init failed");
        let decoded = converter.decode(&[256, 1000, 99]).expect("decoding failed");
        assert_eq!(decoded, b"ab<|eot|>c".to_vec());
    }

    #[test]
    fn test_special_token_overlaps_base_vocab() {
        let result = BPEConverter::new(vec![], &[("<|bad|>", 255)]);
        assert!(matches!(result, Err(SpecialTokenError::IllegalToken(255))));
    }

    #[test]
    fn test_special_token_overlaps_merged_vocab() {
        let history = vec![((97, 98), 256)];
        let result = BPEConverter::new(history, &[("<|bad|>", 256)]);
        assert!(matches!(result, Err(SpecialTokenError::IllegalToken(256))));
    }

    #[test]
    fn test_vocab_size() {
        let history = vec![((97, 98), 256), ((256, 99), 257)];
        let converter = BPEConverter::new(history, &[]).expect("converter init failed");

        // Should have at least 258 entries (0-255 base + 256, 257).
        assert!(converter.vocab().len() >= 258);
        assert_eq!(converter.vocab()[97], vec![97]);
        assert_eq!(converter.vocab()[256], vec![97, 98]);
        assert_eq!(converter.vocab()[257], vec![97, 98, 99]);
    }
}
