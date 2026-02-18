//! This module provides a complete encoding pipeline:
//! 1. Regex pattern matching to split text into chunks.
//! 2. UTF-8 byte conversion for each chunk.
//! 3. BPE merge application on byte sequences.
//!
//! The tokenizer supports both single-text and parallel batch encoding
//! via Rayon, with options for regex-based splitting or raw byte encoding.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{
    converter::BPEConverter,
    error::{DecodeError, EncodeError, ErrorMode, TokenizerInitError},
    types::Token,
};
use fancy_regex::Regex;

/// BPE Tokenizer that performs regex splitting, encoding and decoding.
///
/// This struct encapsulates the complete encoding pipeline by combining:
/// - A compiled regex pattern for text splitting.
/// - A BPE converter with learned merge rules.
///
/// It provides methods for both pattern-based and raw byte encoding,
/// with support for parallel batch processing via Rayon.
pub(crate) struct BPETokenizer {
    pattern: Regex,
    converter: BPEConverter,
}

impl BPETokenizer {
    /// Creates a new tokenizer from merge history and a regex split pattern.
    ///
    /// # Arguments
    ///
    /// * `merge_history` - BPE merge rules as ((left, right), merged_token).
    ///   Order determines merge priority (earlier = higher priority).
    /// * `pattern` - Regex pattern string used to split text into chunks.
    ///   Must be compatible with `fancy_regex` (supports lookaheads/lookbehinds
    ///   but NOT possessive quantifiers like `?+` or `++`).
    ///
    /// # Errors
    ///
    /// Returns an error if the regex pattern fails to compile.
    ///
    /// # Compatibility note
    ///
    /// Patterns using possessive quantifiers such as GPT-4 with its `?+` and `++`
    /// must be converted to atomic groups before passing here:
    ///   - `X?+` → `(?>X?)`
    ///   - `X++` → `(?>X+)`
    ///   - `X*+` → `(?>X*)`
    ///
    pub(crate) fn new(
        merge_history: impl IntoIterator<Item = ((Token, Token), Token)>,
        pattern: &str,
        special_tokens: HashMap<String, Token>,
    ) -> Result<Self, TokenizerInitError> {
        let converter = BPEConverter::new(merge_history, &special_tokens)?;
        let pattern = Regex::new(pattern)?;

        Ok(Self {
            converter,
            pattern,
        })
    }

    /// Encode a full text string: regex split → bytes → BPE.
    ///
    /// This is the main encoding pipeline that combines regex-based
    /// text splitting with BPE merge rules.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text string to encode.
    ///
    /// # Returns
    ///
    /// Complete token sequence obtained by:
    /// 1. Splitting text using the regex pattern.
    /// 2. Converting each chunk to UTF-8 bytes.
    /// 3. Applying BPE merges to each chunk.
    /// 4. Concatenating all chunk results.
    ///
    /// # Errors
    ///
    /// Returns `EncodeError::RegexMatch` if the regex engine fails during
    /// text splitting (e.g. backtracking limit exceeded).
    pub(crate) fn encode_text(&self, text: &str) -> Result<Vec<Token>, EncodeError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // pre-allocate: on average, BPE compresses text by 30-40%
        let mut all_tokens = Vec::with_capacity(text.len() / 3);

        for mat in self.pattern.find_iter(text) {
            let m = mat.map_err(|e| EncodeError::RegexMatch(e.to_string()))?;
            let chunk = m.as_str();

            if chunk.is_empty() {
                continue;
            }

            let encoded = self.encode_chunk(chunk);
            all_tokens.extend_from_slice(&encoded);
        }

        Ok(all_tokens)
    }

    /// Encode many texts in parallel using Rayon.
    ///
    /// Each text is independently split and BPE-encoded on a Rayon worker.
    /// This method is highly efficient for batch processing as it parallelizes
    /// the entire encoding pipeline for each text.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to encode.
    ///
    /// # Returns
    ///
    /// Vector of token sequences in the same order as input texts.
    ///
    /// # Errors
    ///
    /// Returns the first `EncodeError` encountered during parallel encoding.
    pub(crate) fn encode_texts(&self, texts: &[&str]) -> Result<Vec<Vec<Token>>, EncodeError> {
        texts
            .par_iter()
            .map(|text| self.encode_text(text))
            .collect()
    }

    /// Encode raw text as bytes → BPE without regex splitting.
    ///
    /// Treats the entire input text as a single chunk, bypassing
    /// the regex splitting step. Useful for tokenizers that operate
    /// on the full byte stream (e.g. BasicTokenizer) without
    /// pre-splitting by pattern.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode as a single continuous byte stream.
    ///
    /// # Returns
    ///
    /// Token sequence after BPE merges on the complete byte representation.
    pub(crate) fn encode_bytes(&self, text: &str) -> Vec<Token> {
        let byte_tokens: Vec<Token> = text.bytes().map(|b| b as Token).collect();

        if byte_tokens.len() <= 1 {
            return byte_tokens;
        }

        self.converter.encode(byte_tokens)
    }

    /// Encode many texts as raw bytes to BPE in parallel without regex splitting.
    ///
    /// Each text is treated as a single byte stream and encoded independently
    /// on a Rayon worker. Useful for tokenizers that operate on the full
    /// byte stream (e.g. BasicTokenizer) without pre-splitting by pattern.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to encode without regex splitting.
    ///
    /// # Returns
    ///
    /// Vector of token sequences in the same order as input texts.
    pub(crate) fn encode_bytes_batch(&self, texts: &[&str]) -> Vec<Vec<Token>> {
        texts
            .par_iter()
            .map(|text| self.encode_bytes(text))
            .collect()
    }
    /// Encode a single text string with special token handling.
    ///
    /// Special tokens in `allowed_special` are matched literally and emitted
    /// as single token IDs; surrounding normal text passes through the
    /// regex split → bytes → BPE pipeline.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode.
    /// * `allowed_special` - Mapping of special token strings to their token IDs.
    ///
    /// # Returns
    ///
    /// Complete token sequence with special tokens inserted at their positions.
    ///
    /// # Errors
    ///
    /// Returns `EncodeError::RegexMatch` if the regex engine fails during
    /// text splitting of a normal segment.
    pub(crate) fn encode_text_with_special(
        &self,
        text: &str,
        allowed_special: HashMap<String, Token>,
    ) -> Result<Vec<Token>, EncodeError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        if allowed_special.is_empty() {
            return self.encode_text(text);
        }

        let mut all_tokens: Vec<Token> = Vec::with_capacity(text.len() / 3);

        let segments = self.split_on_special_tokens(text, &allowed_special)?;

        for (segment, special_id) in &segments {
            match special_id {
                Some(id) => all_tokens.push(*id),
                None => all_tokens.extend(self.encode_text(segment)?),
            }
        }

        Ok(all_tokens)
    }

    /// Encode multiple texts in parallel with special token handling.
    ///
    /// Each text is first split on special tokens; the normal segments across
    /// all texts are then flattened into a single Rayon batch for parallel
    /// BPE encoding. Results are reassembled into per-text token sequences
    /// with special token IDs inserted at their original positions.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to encode.
    /// * `allowed_special` - Mapping of special token strings to their token IDs.
    ///
    /// # Returns
    ///
    /// Vector of token sequences in the same order as input texts.
    ///
    /// # Errors
    ///
    /// Returns the first `EncodeError` encountered during parallel encoding.
    pub(crate) fn encode_texts_with_special(
        &self,
        texts: &[&str],
        allowed_special: HashMap<String, Token>,
    ) -> Result<Vec<Vec<Token>>, EncodeError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if allowed_special.is_empty() {
            return self.encode_texts(texts);
        }

        let split_segments: Vec<Vec<(String, Option<Token>)>> = texts
            .iter()
            .map(|t| self.split_on_special_tokens(t, &allowed_special))
            .collect::<Result<_, _>>()?;

        // flatten all normal segments across all texts into one list for Rayon.
        let flattened_normal_segs: Vec<&str> = split_segments
            .iter()
            .flat_map(|split| split.iter())
            .filter_map(|(seg, id)| if id.is_none() { Some(seg.as_str()) } else { None })
            .collect();

        // encode every normal segment in one parallel Rayon batch.
        // collect() on a ParallelIterator preserves input order.
        let encoded_normals: Vec<Vec<Token>> = flattened_normal_segs
            .par_iter()
            .map(|t| self.encode_text(t))
            .collect::<Result<_, _>>()?;

        // reassemble per-text token sequences.
        let mut normal_idx = 0;
        let mut results = Vec::with_capacity(texts.len());

        for text_split in &split_segments {
            let mut tokens: Vec<Token> = Vec::new();
            for (_, id) in text_split {
                match id {
                    Some(tok) => tokens.push(*tok),
                    None => {
                        tokens.extend_from_slice(&encoded_normals[normal_idx]);
                        normal_idx += 1;
                    }
                }
            }
            results.push(tokens);
        }

        Ok(results)
    }

    /// Decodes a token sequence back into a UTF-8 string.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Sequence of token IDs to decode.
    ///
    /// # Returns
    ///
    /// `Ok(String)` containing the decoded text, or `Err(DecodeError)` if:
    /// - A token ID is not found in the vocabulary (`DecodeError::UnknownToken`)
    /// - The decoded bytes are not valid UTF-8 (`DecodeError::InvalidUtf8`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tokenizer = BPETokenizer::new(...)?;
    /// let tokens = vec![256, 99];
    /// let text = tokenizer.decode_tokens(&tokens)?;
    /// ```
    pub(crate) fn decode_tokens(
        &self,
        tokens: &[Token],
        errors: ErrorMode,
    ) -> Result<String, DecodeError> {
        let bytes = self.converter.decode(tokens)?;

        match errors {
            ErrorMode::Strict => String::from_utf8(bytes).map_err(DecodeError::InvalidUtf8),
            ErrorMode::Replace => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        }
    }

    /// Decodes multiple token sequences in parallel.
    ///
    /// Each token sequence is independently decoded on a Rayon worker.
    ///
    /// # Arguments
    ///
    /// * `token_seqs` - Slice of token sequences to decode.
    /// * `errors` - How to handle invalid UTF-8 in decoded bytes.
    ///
    /// # Returns
    ///
    /// `Ok(Vec<String>)` containing decoded strings in the same order as input, or
    /// `Err(DecodeError)` if any sequence fails to decode (e.g., unknown token or invalid UTF-8).
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during parallel decoding:
    /// - `DecodeError::UnknownToken` if a token ID is not in the vocabulary
    /// - `DecodeError::InvalidUtf8` if decoded bytes are not valid UTF-8
    pub(crate) fn decode_tokens_batch(
        &self,
        token_seqs: &[&[Token]],
        errors: ErrorMode,
    ) -> Result<Vec<String>, DecodeError> {
        token_seqs
            .par_iter()
            .map(|tokens| self.decode_tokens(tokens, errors))
            .collect()
    }

    /// Returns the vocabulary size (number of tokens).
    ///
    /// # Returns
    ///
    /// The total number of tokens in the vocabulary.
    pub(crate) fn vocab_size(&self) -> usize {
        self.converter.vocab().len()
    }

    /// Encode an already split regex chunk into BPE tokens.
    ///
    /// Converts the chunk into UTF-8 bytes first and then
    /// applies the pre-learnt BPE merges from training.
    ///
    /// This method is inlined because it is called in a tight
    /// loop per regex match.
    ///
    /// # Arguments
    ///
    /// * `chunk` - A text substring obtained from regex splitting.
    ///
    /// # Returns
    ///
    /// Token sequence after applying BPE merges to the byte representation.
    #[inline]
    fn encode_chunk(&self, chunk: &str) -> Vec<Token> {
        // String → UTF-8 bytes → Token integers
        let byte_tokens: Vec<Token> = chunk.bytes().map(|b| b as Token).collect();
        if byte_tokens.len() <= 1 {
            return byte_tokens;
        }
        // Apply BPE encoding to the byte sequence
        self.converter.encode(byte_tokens)
    }

    /// Segments text into alternating normal and special-token spans.
    ///
    /// Compiles a regex from the `allowed_special` keys, then scans `text`
    /// for literal matches. Each matched special token is emitted with its
    /// token ID; non-matching spans are emitted as normal segments.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to segment.
    /// * `allowed_special` - Mapping of special token strings to their token IDs.
    ///
    /// # Returns
    ///
    /// A list of `(chunk, Option<token_id>)` pairs where the second element
    /// is `None` for normal text and `Some(id)` for a special token.
    /// Chunks are owned strings so they can be sent across Rayon threads.
    ///
    /// # Errors
    ///
    /// Returns `EncodeError::RegexMatch` if the special-token pattern fails
    /// to compile or a match error occurs during scanning.
    fn split_on_special_tokens(
        &self,
        text: &str,
        allowed_special: &HashMap<String, Token>,
    ) -> Result<Vec<(String, Option<Token>)>, EncodeError> {
        let pattern = allowed_special
            .keys()
            .map(|s| fancy_regex::escape(s))
            .collect::<Vec<_>>()
            .join("|");

        let re = Regex::new(&pattern)
            .map_err(|e| EncodeError::RegexMatch(e.to_string()))?;

        let mut segments: Vec<(String, Option<Token>)> = Vec::new();
        let mut segment_start = 0;

        for mat in re.find_iter(text) {
            let mat = mat.map_err(|e| EncodeError::RegexMatch(e.to_string()))?;

            // guard against empty normal segments when text starts with a
            // special token or contains two consecutive special tokens.
            if mat.start() > segment_start {
                segments.push((text[segment_start..mat.start()].to_string(), None));
            }

            let special_str = mat.as_str();
            segments.push((special_str.to_string(), Some(allowed_special[special_str])));
            segment_start = mat.end();
        }

        if segment_start < text.len() {
            segments.push((text[segment_start..].to_string(), None));
        }

        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SpecialTokenError;

    fn make_tokenizer(merges: Vec<((Token, Token), Token)>, pat: &str) -> BPETokenizer {
        match BPETokenizer::new(merges, pat, HashMap::new()) {
            Ok(tokenizer) => tokenizer,
            Err(_) => panic!("pattern failed to compile"),
        }
    }

    #[test]
    fn test_encode_text_no_merges() {
        // Pattern splits on whitespace-separated words
        let tok = make_tokenizer(vec![], r"\S+");
        let result = tok.encode_text("ab cd").expect("encoding failed");
        // No merges → raw UTF-8 bytes
        assert_eq!(result, vec![97, 98, 99, 100]);
    }

    #[test]
    fn test_encode_text_with_merges() {
        // Merge (97, 98) → 256 i.e. 'a','b' → 256
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let result = tok.encode_text("ab cd").expect("encoding failed");
        // "ab" → [256], "cd" → [99, 100]
        assert_eq!(result, vec![256, 99, 100]);
    }

    #[test]
    fn test_encode_texts_parallel() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let texts = &["ab", "cd", "ab"];
        let results = tok.encode_texts(texts).expect("encoding failed");
        assert_eq!(results, vec![vec![256], vec![99, 100], vec![256]]);
    }

    #[test]
    fn test_encode_bytes_ignores_pattern() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        // encode_bytes treats the entire string as one chunk
        let result = tok.encode_bytes("ab");
        assert_eq!(result, vec![256]);
    }

    #[test]
    fn test_empty_text() {
        let tok = make_tokenizer(vec![], r"\S+");
        assert_eq!(
            tok.encode_text("").expect("encoding failed"),
            Vec::<Token>::new()
        );
    }

    #[test]
    fn test_single_byte_chunk() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r".");
        // Each character is its own chunk; no pairs to merge within a chunk
        let result = tok.encode_text("ab").expect("encoding failed");
        assert_eq!(result, vec![97, 98]);
    }

    #[test]
    fn test_unicode_bytes() {
        // 'é' is U+00E9, encoded as [0xC3, 0xA9] in UTF-8
        let tok = make_tokenizer(vec![((0xC3, 0xA9), 256)], r"\S+");
        let result = tok.encode_text("é").expect("encoding failed");
        assert_eq!(result, vec![256]);
    }

    #[test]
    fn test_lookahead_pattern() {
        // Pattern with negative lookahead.
        let tok = make_tokenizer(vec![], r"\s+(?!\S)|\S+|\s+");
        let result = tok.encode_text("hello world").expect("encoding failed");
        // "hello" → [104,101,108,108,111], " " → [32], "world" → [119,111,114,108,100]
        assert_eq!(
            result,
            vec![104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
        );
    }

    #[test]
    fn test_encode_bytes_batch_parallel() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let texts = &["ab", "cd"];
        let results = tok.encode_bytes_batch(texts);
        assert_eq!(results, vec![vec![256], vec![99, 100]]);
    }

    #[test]
    fn test_decode_tokens_strict() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        // Token 256 = "ab"
        let decoded = tok
            .decode_tokens(&[256, 99], ErrorMode::Strict)
            .expect("decoding failed");
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_decode_base_tokens_strict() {
        let tok = make_tokenizer(vec![], r"\S+");
        // Raw UTF-8 bytes for "hello"
        let decoded = tok
            .decode_tokens(&[104, 101, 108, 108, 111], ErrorMode::Strict)
            .expect("decoding failed");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_decode_tokens_batch_strict() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let seq1 = vec![256];
        let seq2 = vec![99, 100];
        let token_seqs: Vec<&[Token]> = vec![&seq1, &seq2];
        let decoded = tok
            .decode_tokens_batch(&token_seqs, ErrorMode::Strict)
            .expect("batch decoding failed");
        assert_eq!(decoded, vec!["ab", "cd"]);
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let tok = make_tokenizer(vec![((97, 98), 256), ((256, 99), 257)], r"\S+");
        let original = "abc def";
        let encoded = tok.encode_text(original).expect("encoding failed");
        let decoded = tok
            .decode_tokens(&encoded, ErrorMode::Strict)
            .expect("decoding failed");
        assert_eq!(decoded, "abcdef"); // Note: spaces are removed by \S+ pattern
    }

    #[test]
    fn test_decode_unknown_token_errors() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        // Token 9999 is not in vocabulary
        let result = tok.decode_tokens(&[97, 9999], ErrorMode::Replace);
        assert!(result.is_err());
    }

    #[test]
    fn test_vocab_size() {
        let tok = make_tokenizer(vec![((97, 98), 256), ((256, 99), 257)], r"\S+");
        assert!(tok.vocab_size() >= 258);
    }

    #[test]
    fn test_new_rejects_special_token_id_overlap() {
        let mut special_tokens = HashMap::new();
        special_tokens.insert(String::from("<|bad|>"), 256);
        let result = BPETokenizer::new(vec![((97, 98), 256)], r"\S+", special_tokens);
        match result {
            Err(TokenizerInitError::InvalidSpecialToken(SpecialTokenError::IllegalToken(tok))) => {
                assert_eq!(tok, 256);
            }
            _ => panic!("expected InvalidSpecialToken error"),
        }
    }
}
