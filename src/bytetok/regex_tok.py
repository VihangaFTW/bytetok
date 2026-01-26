"""Regex-based byte-level tokenizer implementation."""

from typing import override

from exceptions import PatternError, VocabularyError
from strategy import SpecialTokenStrategy

from .base_tok import Tokenizer
from ._bpe import Token, bpe_merge, update_bpe_freqs
import regex as re
from collections import Counter
import logging


log = logging.getLogger(__name__)


class RegexTokenizer(Tokenizer):
    """Tokenizer that splits text using regex patterns before applying BPE."""

    def __init__(self, pattern: str | None = None) -> None:
        super().__init__()
        if pattern is not None:
            self.pat = pattern
        self.compiled_pat: re.Pattern[str] = _compile_pattern(self.pat)
        self.special_toks: dict[str, Token] = {}

    @override
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
        """Train tokenizer by learning byte pair merges on regex-split chunks."""
        if vocab_size <= 256:
            raise VocabularyError(
                "vocab size must be greater than 256", vocab_size=vocab_size
            )

        # handle list input and convert text to bytes
        if isinstance(text, list):
            text = "".join(text)

        # split text into chunks defined by pattern
        chunks = re.findall(self.compiled_pat, text)
        # convert each chunk to byte sequence
        tokens: list[list[int]] = [
            list(chunk.encode("utf-8", errors="replace")) for chunk in chunks
        ]

        n_merges = vocab_size - 256
        merges = {}
        vocab = {tok: bytes([tok]) for tok in range(256)}
        # byte-pair frequency counter
        for i in range(n_merges):
            new_token = 256 + i
            bp_freqs = Counter()
            # collect global frequency for each byte-pair
            for chunk_toks in tokens:
                update_bpe_freqs(chunk_toks, bp_freqs)
            # find most common token pair
            rank0 = bp_freqs.most_common(1)[0][0]
            # merge pair within each chunk with new token
            tokens = [bpe_merge(chunk_toks, rank0, new_token) for chunk_toks in tokens]
            # save merge info and update vocabulary with new token's mapping
            merges[rank0] = new_token
            vocab[new_token] = vocab[rank0[0]] + vocab[rank0[1]]
            # debugging: log new merge info
            if verbose:
                log.info(f"merge {i + 1}/{n_merges}: {rank0} -> {new_token}")

        self.enc_merges = merges
        self.dec_vocab = vocab

    @override
    def encode(
        self, text: str, strategy: SpecialTokenStrategy | None = None
    ) -> list[Token]:
        """
        Encode text into a sequence of tokens.

        Args:
            text: Text to encode.
            strategy: Strategy to handle special tokens in `text`.
        """

        # no strategy is inferred as no special token handling
        if strategy is None:
            return self._apply_bpe_text(text)

        # retrieve special tokens as defined by chosen strategy
        special_toks = strategy.handle(text, self.special_toks)

        # escape regex metachars like "+" in special tokens to avoid unwanted effects
        esc_special_toks = [re.escape(seq) for seq in special_toks]
        # The capturing group parentheses make re.split() include
        # the matched delimiters in result
        # text is split on special tokens while keeping them as separate chunks
        # otherwise we lose the special tokens after split (leads to lossy decoding)
        special_pat = "(" + "|".join(esc_special_toks) + ")"
        chunks = re.split(special_pat, text)
        tokens = []
        for chunk in chunks:
            if chunk in special_toks:
                # special token sequence already have a unique token id
                tokens.append(special_toks[chunk])
            else:
                tokens.extend(
                    self._apply_bpe_chunk(list(chunk.encode("utf-8", errors="replace")))
                )

        return tokens

    @override
    def decode(self, tokens: list[Token]):
        """Decode a sequence of tokens back into text."""
        txt_bytes = []
        for tok in tokens:
            if tok in self.dec_vocab:
                txt_bytes.append(self.dec_vocab[tok])
            elif tok in self.inverted_special_tokens:
                txt_bytes.append(
                    self.inverted_special_tokens[tok].encode("utf-8", errors="replace")
                )
            else:
                raise VocabularyError("token not found in vocabulary", invalid_tok=tok)

        text = b"".join(txt_bytes).decode("utf-8", errors="replace")
        return text

    @classmethod
    def from_pretrained(cls, model_path: str) -> "RegexTokenizer":
        tokenizer = cls()
        tokenizer.load(model_path)
        return tokenizer

    def register_special_tokens(self, special_toks: dict[str, Token]) -> None:
        """
        This method should be called in __init__ for subclasses that implement
        special token handling.
        """
        self.special_toks = special_toks
        self.inverted_special_tokens = {
            token: seq for seq, token in self.special_toks.items()
        }

    def _apply_bpe_text(self, text: str) -> list[Token]:
        # split text into chunks as defined by pattern
        chunks = re.findall(self.compiled_pat, text)

        tokens: list[Token] = []
        # compress each text chunk into tokens via bpe
        for chunk in chunks:
            # aggregate local tokens to global compressed token sequence
            tokens.extend(
                self._apply_bpe_chunk(list(chunk.encode("utf-8", errors="replace")))
            )

        return tokens

    def _special_tokens_exists(self, text: str) -> bool:
        if self.special_toks:
            return not all(seq not in text for seq in self.special_toks)
        return False


def _compile_pattern(pattern: str) -> re.Pattern:
    """
    Compile and validate a regex pattern.

    Args:
        pattern: Regex pattern string to compile.

    Returns:
        Compiled regex pattern.

    Raises:
        ValueError: If pattern is invalid.
    """
    try:
        return re.compile(pattern)
    except re.error as e:
        raise PatternError("invalid regex pattern", pattern=pattern, regex_err=e)
