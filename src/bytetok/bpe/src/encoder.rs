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

type MergeOrder = usize;

#[derive(Debug)]
pub(crate) enum EncodeError {
    CandidateOutOfBounds { pos: usize, len: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TokenPair(Token, Token);

/// Item in the priority queue for merge ordering.
/// This ensures we apply merges in the correct order.
#[derive(Debug, PartialEq, Eq)]
struct MergeCandidate {
    /// Merge order from training (0 = first merge, 1 = second merge etc)
    merge_order: MergeOrder,
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
        // We reverse the comparison (other vs self) to create min-heap behavior
        // from Rust's max-heap BinaryHeap.
        // Break ties by position (earlier positions first).
        other
            .merge_order
            .cmp(&self.merge_order)
            .then_with(|| other.position.cmp(&self.position))
    }
}

/// This struct wraps an efficient BPE implementation for encoding text. It uses a priority queue with dynamic position updates for better performance on longer sequences.
///
/// # Time Complexity

pub(crate) struct BPEEncoder {
    /// Maps token pairs to (merged_token, merge_order).
    /// merge_order indicates when this merge was learned during training.
    merges: HashMap<TokenPair, (Token, MergeOrder)>,
}

impl BPEEncoder {
    /// Create a new instance of BPEEncoder.
    /// # Note
    /// The ordering in `merge_history` is taken as the source of truth for the merge rules in the encoding algorithm. Verify integirty of this ordering before calling `encode()`; otherwise it might be the source of nasty bugs.
    pub(crate) fn new(merge_history: impl IntoIterator<Item = (TokenPair, Token)>) -> Self {
        let mut merges = HashMap::new();
        for (merge_order, (pair, tok)) in merge_history.into_iter().enumerate() {
            merges.insert(pair, (tok, merge_order));
        }
        Self { merges }
    }

    pub(crate) fn encode(&self, tokens: Vec<Token>) -> Result<Vec<Token>, EncodeError> {
        if tokens.len() <= 1 {
            return Ok(tokens);
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

            // verify pair at pos is a valid merge
            // note: result[pos] = Some does not
            // guarantee result[pos+1] = Some
            let (Some(left), Some(right)) = (
                results.get(pos).copied().flatten(),
                results.get(pos + 1).copied().flatten(),
            ) else {
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
            results[pos + 1] = None;

            // new merge candidates might appear after merge
            // check right
            self.track_new_merge_candidate(&mut heap, &results, pos, merge_tok, true)?;
            // check left
            self.track_new_merge_candidate(&mut heap, &results, pos, merge_tok, false)?;
        }

        // build final output list: filter None values
        let encoding: Vec<Token> = results.into_iter().flatten().collect();
        Ok(encoding)
    }
    /// Add new merge candidates to the min heap after a successful merge.
    /// # Note
    ///  Caller should check whether `pos` is out of bounds to avoid panic.
    fn track_new_merge_candidate(
        &self,
        heap: &mut BinaryHeap<MergeCandidate>,
        results: &[Option<Token>],
        pos: usize,
        merged_tok: usize,
        check_right: bool,
    ) -> Result<(), EncodeError> {
        // todo REFACTOR
        if pos > 0 {
            let mut idx;
            let n = results.len();

            if check_right {
                idx = pos + 1;
                // discard all None entries to the right of pos
                while idx < n && results.get(idx).is_none() {
                    idx += 1;
                }
            } else {
                idx = pos - 1;
                // discard all None entries to the left of pos
                while idx > 0 && results.get(idx).is_none() {
                    idx -= 1;
                }
            };

            // extract closest token
            let Some(&Some(tok)) = results.get(idx) else {
                return Err(EncodeError::CandidateOutOfBounds {
                    pos,
                    len: results.len(),
                });
            };

            // push new pair as candidate if merge rule exists
            let pair = TokenPair(tok, merged_tok);
            if let Some(&(_merge_tok, order)) = self.merges.get(&pair) {
                let left_candidate = MergeCandidate {
                    merge_order: order,
                    pair,
                    position: idx,
                };
                heap.push(left_candidate);
            };
        };

        Err(EncodeError::CandidateOutOfBounds {
            pos,
            len: results.len(),
        })
    }

    /// Populate the priority queue with all initial mergeable pairs from a token sequence.
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

    /// Check if a token pair can be merged.
    pub(crate) fn can_merge(&self, left: Token, right: Token) -> bool {
        self.merges.contains_key(&TokenPair(left, right))
    }

    /// Get the number of merge rules.
    pub(crate) fn num_merges(&self) -> usize {
        self.merges.len()
    }
}
