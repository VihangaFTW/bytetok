//! This module provides a complete encoding pipeline:
//! 1. Regex pattern matching to split text into chunks.
//! 2. UTF-8 byte conversion for each chunk.
//! 3. BPE merge application on byte sequences.
//!
//! The tokenizer supports both single-text and parallel batch encoding
//! via Rayon, with options for regex-based splitting or raw byte encoding.



use rayon::prelude::*;

use crate::{converter::BPEConverter, error::{DecodeError, EncodeError, ErrorMode, TokenizerInitError}, types::Token};
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
        special_tokens: &[(&str, Token)],
    ) -> Result<Self, TokenizerInitError> {
        let converter = BPEConverter::new(merge_history, special_tokens)?;
        let pattern = Regex::new(pattern)?;

        // TODO: Handle special_tokens.
        let _ = special_tokens;

        Ok(Self { converter, pattern })
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
        // String → UTF-8 bytes → Token integers.
        let byte_tokens: Vec<Token> = chunk.bytes().map(|b| b as Token).collect();
        if byte_tokens.len() <= 1 {
            return byte_tokens;
        }
        // Apply BPE encoding to the byte sequence.
        self.converter.encode(byte_tokens)
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

        // pre-allocate: on average, BPE compresses text by 30-40%.
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
    pub(crate) fn decode_tokens(&self, tokens: &[Token], errors: ErrorMode) -> Result<String, DecodeError> {
        // unknown errors propagate regardless of error mode
        let bytes = self.converter.decode(tokens)?;

        match errors {
            
            ErrorMode::Strict => return String::from_utf8(bytes).map_err(|e|DecodeError::InvalidUtf8(e)),
            ErrorMode::Replace => return Ok(String::from_utf8_lossy(&bytes).into_owned())
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
    pub(crate) fn decode_tokens_batch(&self, token_seqs: &[&[Token]], errors: ErrorMode) -> Result<Vec<String>, DecodeError >{
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


}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SpecialTokenError;

    fn make_tokenizer(merges: Vec<((Token, Token), Token)>, pat: &str) -> BPETokenizer {
        match BPETokenizer::new(merges, pat, &[]) {
            Ok(tokenizer) => tokenizer,
            Err(_) => panic!("pattern failed to compile"),
        }
    }

    #[test]
    fn test_encode_text_no_merges() {
        // Pattern splits on whitespace-separated words.
        let tok = make_tokenizer(vec![], r"\S+");
        let result = tok.encode_text("ab cd").expect("encoding failed");
        // No merges → raw UTF-8 bytes.
        assert_eq!(result, vec![97, 98, 99, 100]);
    }

    #[test]
    fn test_encode_text_with_merges() {
        // Merge (97, 98) → 256 i.e. 'a','b' → 256.
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let result = tok.encode_text("ab cd").expect("encoding failed");
        // "ab" → [256], "cd" → [99, 100].
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
        // encode_bytes treats the entire string as one chunk.
        let result = tok.encode_bytes("ab");
        assert_eq!(result, vec![256]);
    }

    #[test]
    fn test_empty_text() {
        let tok = make_tokenizer(vec![], r"\S+");
        assert_eq!(tok.encode_text("").expect("encoding failed"), Vec::<Token>::new());
    }

    #[test]
    fn test_single_byte_chunk() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r".");
        // Each character is its own chunk — no pairs to merge within a chunk.
        let result = tok.encode_text("ab").expect("encoding failed");
        assert_eq!(result, vec![97, 98]);
    }

    #[test]
    fn test_unicode_bytes() {
        // 'é' is U+00E9, encoded as [0xC3, 0xA9] in UTF-8.
        let tok = make_tokenizer(vec![((0xC3, 0xA9), 256)], r"\S+");
        let result = tok.encode_text("é").expect("encoding failed");
        assert_eq!(result, vec![256]);
    }

    #[test]
    fn test_lookahead_pattern() {
        // Pattern with negative lookahead.
        let tok = make_tokenizer(vec![], r"\s+(?!\S)|\S+|\s+");
        let result = tok.encode_text("hello world").expect("encoding failed");
        // "hello" → [104,101,108,108,111], " " → [32], "world" → [119,111,114,108,100].
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
        // Token 256 = "ab".
        let decoded = tok.decode_tokens(&[256, 99], ErrorMode::Strict).expect("decoding failed");
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_decode_base_tokens_strict() {
        let tok = make_tokenizer(vec![], r"\S+");
        // Raw UTF-8 bytes for "hello".
        let decoded = tok.decode_tokens(&[104, 101, 108, 108, 111], ErrorMode::Strict).expect("decoding failed");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_decode_tokens_batch_strict() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let seq1 = vec![256];
        let seq2 = vec![99, 100];
        let token_seqs: Vec<&[Token]> = vec![&seq1, &seq2];
        let decoded = tok.decode_tokens_batch(&token_seqs, ErrorMode::Strict).expect("batch decoding failed");
        assert_eq!(decoded, vec!["ab", "cd"]);
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let tok = make_tokenizer(vec![((97, 98), 256), ((256, 99), 257)], r"\S+");
        let original = "abc def";
        let encoded = tok.encode_text(original).expect("encoding failed");
        let decoded = tok.decode_tokens(&encoded, ErrorMode::Strict).expect("decoding failed");
        assert_eq!(decoded, "abcdef"); // Note: spaces are removed by \S+ pattern.
    }

    #[test]
    fn test_decode_unknown_token_errors() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        // Token 9999 is not in vocabulary.
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
        let result = BPETokenizer::new(vec![((97, 98), 256)], r"\S+", &[("<|bad|>", 256)]);
        match result {
            Err(TokenizerInitError::InvalidSpecialToken(SpecialTokenError::IllegalToken(tok))) => {
                assert_eq!(tok, 256);
            }
            _ => panic!("expected InvalidSpecialToken error"),
        }
    }
}
