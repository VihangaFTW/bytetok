//! Full tokenizer implementation that combines regex splitting
//! with BPE encoding.

use rayon::prelude::*;

use crate::{encoder::BPEEncoder, types::Token};
use fancy_regex::Regex;

/// Tokenizer that performs regex splitting and BPE.
pub(crate) struct Tokenizer {
    pattern: Regex,
    encoder: BPEEncoder,
}

impl Tokenizer {
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
    ) -> Result<Self, fancy_regex::Error> {
        let encoder = BPEEncoder::new(merge_history);
        let pattern = Regex::new(pattern)?;
        Ok(Self { encoder, pattern })
    }

    /// Encode an already split regex chunk into BPE tokens.
    ///
    /// Converts the chunk into UTF-8 bytes first and then
    /// applies the pre-learnt BPE merges from training.
    ///
    /// This method is inlined because it is called in a tight
    /// loop per regex match.
    #[inline]
    fn encode_chunk(&self, chunk: &str) -> Vec<Token> {
        //  string -> bytes -> int
        let byte_tokens: Vec<Token> = chunk.bytes().map(|b| b as Token).collect();
        if byte_tokens.len() <= 1 {
            return byte_tokens;
        }
        // bpe encode
        self.encoder.encode(byte_tokens)
    }

    /// Encode a full text string: regex split → bytes → BPE.
    pub(crate) fn encode_text(&self, text: &str) -> Vec<Token> {
        if text.is_empty() {
            return Vec::new();
        }

        // on average, bpe compresses text by 30-40%
        let mut all_tokens = Vec::with_capacity(text.len() / 3);

        for mat in self.pattern.find_iter(text) {
            let Ok(m) = mat else { continue };
            let chunk = m.as_str();

            if chunk.is_empty() {
                continue;
            }

            let encoded = self.encode_chunk(chunk);
            all_tokens.extend_from_slice(&encoded);
        }

        all_tokens
    }

    /// Encode many texts in parallel using Rayon.
    ///
    /// Each text is independently split and BPE-encoded on a Rayon worker.
    pub(crate) fn encode_texts(&self, texts: &[String]) -> Vec<Vec<Token>> {
        texts
            .par_iter()
            .map(|text| self.encode_text(text))
            .collect()
    }

    /// Encode raw text as bytes → BPE without regex splitting.
    ///
    /// Useful for tokenizers that operate on the full byte stream
    /// (e.g. BasicTokenizer) without pre-splitting by pattern.
    pub(crate) fn encode_bytes(&self, text: &str) -> Vec<Token> {
        let byte_tokens: Vec<Token> = text.bytes().map(|b| b as Token).collect();

        if byte_tokens.len() <= 1 {
            return byte_tokens;
        }

        self.encoder.encode(byte_tokens)
    }

    /// Encode many texts as raw bytes to BPE in parallel without regex splitting.
    ///
    /// Useful for tokenizers that operate on the full byte stream
    /// (e.g. BasicTokenizer) without pre-splitting by pattern.
    pub(crate) fn encode_bytes_batch(&self, texts: &[String]) -> Vec<Vec<Token>> {
        texts
            .par_iter()
            .map(|text| self.encode_bytes(text))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokenizer(merges: Vec<((Token, Token), Token)>, pat: &str) -> Tokenizer {
        match Tokenizer::new(merges, pat) {
            Ok(tokenizer) => tokenizer,
            Err(_) => panic!("pattern failed to compile"),
        }
    }

    #[test]
    fn test_encode_text_no_merges() {
        // Pattern splits on whitespace-separated words
        let tok = make_tokenizer(vec![], r"\S+");
        let result = tok.encode_text("ab cd");
        // No merges → raw UTF-8 bytes.
        assert_eq!(result, vec![97, 98, 99, 100]);
    }

    #[test]
    fn test_encode_text_with_merges() {
        // Merge (97, 98) → 256 i.e. 'a','b' → 256
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let result = tok.encode_text("ab cd");
        // "ab" → [256], "cd" → [99, 100]
        assert_eq!(result, vec![256, 99, 100]);
    }

    #[test]
    fn test_encode_texts_parallel() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let texts: Vec<String> = vec!["ab".into(), "cd".into(), "ab".into()];
        let results = tok.encode_texts(&texts);
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
        assert_eq!(tok.encode_text(""), Vec::<Token>::new());
    }

    #[test]
    fn test_single_byte_chunk() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r".");
        // each character is its own chunk — no pairs to merge within a chunk
        let result = tok.encode_text("ab");
        assert_eq!(result, vec![97, 98]);
    }

    #[test]
    fn test_unicode_bytes() {
        // 'é' is U+00E9, encoded as [0xC3, 0xA9] in UTF-8.
        let tok = make_tokenizer(vec![((0xC3, 0xA9), 256)], r"\S+");
        let result = tok.encode_text("é");
        assert_eq!(result, vec![256]);
    }

    #[test]
    fn test_lookahead_pattern() {
        // pattern with negative lookahead
        let tok = make_tokenizer(vec![], r"\s+(?!\S)|\S+|\s+");
        let result = tok.encode_text("hello world");
        // "hello" → [104,101,108,108,111], " " → [32], "world" → [119,111,114,108,100].
        assert_eq!(
            result,
            vec![104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
        );
    }

    #[test]
    fn test_encode_bytes_batch_parallel() {
        let tok = make_tokenizer(vec![((97, 98), 256)], r"\S+");
        let texts: Vec<String> = vec!["ab".into(), "cd".into()];
        let results = tok.encode_bytes_batch(&texts);
        assert_eq!(results, vec![vec![256], vec![99, 100]]);
    }
}
