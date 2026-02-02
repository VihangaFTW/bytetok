"""Regex-based byte-level tokenizer implementation."""

from typing import override

from ..errors import PatternError, SpecialTokenError, TrainingError, VocabularyError
from ..pattern import TokenPattern
from ..strategy import SpecialTokenStrategy

from .base import Tokenizer
from .._decorators import measure_time
from .._bpe import (
    Encoding,
    Token,
    Vocabulary,
)
from ..bpe import RustBPETrainer

import regex as re
import logging


log = logging.getLogger(__name__)


class RegexTokenizer(Tokenizer):
    """Tokenizer that splits text using regex patterns before applying BPE."""

    TOKENIZER_TYPE = "regex"

    def __init__(self, pattern: str | None = None) -> None:
        super().__init__()
        if pattern is None:
            self.pat = TokenPattern.get("gpt4o")
        else:
            self.pat = pattern
        self.compiled_pat: re.Pattern[str] = _compile_pattern(self.pat)
        self.special_toks: dict[str, Token] = {}
        self.inverted_special_tokens: dict[Token, str] = {}

    @override
    @measure_time
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
        chunks = [m.group(0) for m in re.finditer(self.compiled_pat, text)]
        # convert each chunk to byte sequence and flatten into single sequence
        tokens: list[int] = []
        for chunk in chunks:
            tokens.extend(list(chunk.encode("utf-8", errors="replace")))

        n_merges = vocab_size - 256

        self._train_rust(tokens, n_merges, verbose)

    @override
    def encode(
        self, text: str, strategy: SpecialTokenStrategy | None = None
    ) -> list[Token]:
        """
        Encode text into a sequence of tokens.

        :param text: Text to encode.
        :param strategy: Strategy to handle special tokens in `text`.
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
    def decode(self, tokens: list[Token]) -> str:
        """Decode a sequence of tokens back into text."""
        txt_bytes = []
        for tok in tokens:
            if tok in self.vocab:
                txt_bytes.append(self.vocab[tok])
            elif tok in self.inverted_special_tokens:
                txt_bytes.append(
                    self.inverted_special_tokens[tok].encode("utf-8", errors="replace")
                )
            else:
                raise VocabularyError("token not found in vocabulary", invalid_tok=tok)

        text = b"".join(txt_bytes).decode("utf-8", errors="replace")
        return text

    def register_special_tokens(self, special_toks: list[str]) -> None:
        """
        Register special tokens with auto-assigned IDs.

        Special token IDs are assigned sequentially starting from vocab_size.
        Must be called after training the tokenizer.

        :param special_toks: List of special token strings to register.
        :raises SpecialTokenError: If tokenizer hasn't been trained yet.
        """
        if not hasattr(self, "merges"):
            raise SpecialTokenError(
                f"{self.__class__.__name__} must be trained before registering special tokens"
            )
        # special tokens have auto assigned ids
        vocab_size = 256 + len(self.merges)

        # special tokens are appended at the end of trained vocab
        for idx, seq in enumerate(special_toks):
            self.special_toks[seq] = vocab_size + idx

        self.inverted_special_tokens = {
            token: seq for seq, token in self.special_toks.items()
        }

        # rebuild vocab to include special tokens for decoding
        for seq, tok in self.special_toks.items():
            self.vocab[tok] = seq.encode("utf-8")

    def _apply_bpe_text(self, text: str) -> list[Token]:
        # split text into chunks as defined by pattern
        chunks = [m.group(0) for m in re.finditer(self.compiled_pat, text)]

        tokens: list[Token] = []
        # compress each text chunk into tokens via bpe
        for chunk in chunks:
            # aggregate local tokens to global compressed token sequence
            tokens.extend(
                self._apply_bpe_chunk(list(chunk.encode("utf-8", errors="replace")))
            )

        return tokens

    def _train_rust(self, tokens: list[int], n_merges: int, verbose: bool) -> None:
        """Train using fast Rust implementation."""

        if len(tokens) == 0:
            raise TrainingError("empty token sequence, no training performed")

        trainer = RustBPETrainer(tokens, 256)

        # train for n_merges
        trainer.train(n_merges)

        # get merge history
        merge_history = trainer.get_merge_history()

        # build merges dictionary and vocabulary
        merges = {}
        vocab = {tok: bytes([tok]) for tok in range(256)}

        for (tok_a, tok_b), new_tok in merge_history:
            pair = (tok_a, tok_b)
            merges[pair] = new_tok
            vocab[new_tok] = vocab[tok_a] + vocab[tok_b]

            if verbose:
                log.info(f"merge {len(merges)}/{n_merges}: {pair} -> {new_tok}")

        if len(merge_history) < n_merges:
            log.warning(
                f"no more byte pairs to merge after {len(merge_history)} merges "
                f"(requested {n_merges}). stopping early."
            )

        self.merges: Encoding = merges
        self.vocab: Vocabulary = vocab


def _compile_pattern(pattern: str) -> re.Pattern:
    """
    Compile and validate a regex pattern.

    :param pattern: Regex pattern string to compile.
    :return: Compiled regex pattern.
    :raises ValueError: If pattern is invalid.
    """
    try:
        return re.compile(pattern)
    except re.error as e:
        raise PatternError("invalid regex pattern", pattern=pattern, regex_err=e)
