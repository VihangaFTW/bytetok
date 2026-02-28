"""Regex-based byte-level tokenizer implementation."""

import regex as re
import logging
from typing import override


from ..errors import TokenizationError, VocabularyError
from ..pattern import TokenPattern
from ..strategy import SpecialTokenStrategy

from ..types import (
    Token,
)
from .._trainer import _train_bpe


from .base import Tokenizer


log = logging.getLogger(__name__)


class RegexTokenizer(Tokenizer):
    """Tokenizer that splits text using regex patterns before applying BPE."""

    TOKENIZER_TYPE = "regex"

    def __init__(self, pattern: str | None = None) -> None:
        """Initialize tokenizer with a provided or default split pattern."""
        super().__init__()
        if pattern is None:
            self.pat = TokenPattern.get("gpt4o")
        else:
            self.pat = pattern
        self.special_toks: dict[str, Token] = {}

    @override
    def load(self, model_filename: str) -> None:
        """Load tokenizer state; delegates to base and updates internal state."""
        super().load(model_filename)

    @override
    def train(
        self,
        text: str | list[str],
        vocab_size: int,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> None:
        """
        Train on regex-split text chunks.

        Input is segmented by the configured pattern, then flattened into
        UTF-8 byte tokens for BPE merge learning.

        :param verbose: Log each learned merge when ``True``.
        :raises VocabularyError: If ``vocab_size`` is less than or equal to 256.
        :raises TrainingError: If no trainable byte tokens are produced.
        """
        if vocab_size <= 256:
            raise VocabularyError(
                "vocab size must be greater than 256", vocab_size=vocab_size
            )

        # handle list input and convert text to bytes
        if isinstance(text, list):
            text = "".join(text)
        # split text into chunks defined by pattern
        chunks = [m.group(0) for m in re.finditer(self.pat, text)]
        # convert each chunk to byte sequence and flatten into single sequence
        tokens: list[int] = []
        for chunk in chunks:
            tokens.extend(list(chunk.encode("utf-8", errors="replace")))

        n_merges = vocab_size - 256

        result = _train_bpe(
            tokens, n_merges, verbose=verbose, show_progress=show_progress
        )

        if result.n_merges_completed < n_merges:
            log.warning(
                f"no more byte pairs to merge after {result.n_merges_completed} merges "
                f"(requested {n_merges}) stopping early"
            )

        self.merges = result.merges
        self.vocab = result.vocab
        # Invalidate Rust cache since merges changed.
        self._tokenizer = None

    @override
    def _encode_impl(
        self,
        text: str,
        strategy: SpecialTokenStrategy | None = None,
    ) -> list[Token]:
        """
        Encode text into a sequence of tokens.

        If ``strategy`` is ``None``, applies regex chunking and BPE only.
        Otherwise, matched special tokens are kept atomic; non-special spans
        are encoded with BPE.

        :param strategy: Strategy to select allowed special tokens.
        :raises TokenizationError: If encoding fails.
        """
        tokenizer = self._get_rust_tokenizer(pattern=self.pat)

        if strategy is None:
            try:
                return tokenizer.encode_text(text)
            except ValueError as e:
                raise TokenizationError("failed to encode text") from e

        allowed_special = strategy.handle(text, self.special_toks)

        try:
            if not allowed_special:
                return tokenizer.encode_text(text)
            return tokenizer.encode_text_with_special(text, allowed_special)
        except ValueError as e:
            raise TokenizationError("failed to encode text") from e

    @override
    def _encode_batch_impl(
        self,
        texts: list[str],
        strategy: "SpecialTokenStrategy | None" = None,
        show_progress: bool = True,
    ) -> list[list[Token]]:
        """
        Encode many texts in parallel via Rust/Rayon.

        :param strategy: Optional special token handling strategy.
        :raises TrainingError: If the tokenizer has not been trained yet.
        :raises TokenizationError: If encoding fails.
        """
        tokenizer = self._get_rust_tokenizer(pattern=self.pat)

        if strategy is None:
            try:
                return tokenizer.encode_texts(texts, show_progress=show_progress)
            except ValueError as e:
                raise TokenizationError("failed to encode texts") from e

        # Run strategy per text to trigger validation
        for t in texts:
            # Raises SpecialTokenError if strat is allow-none-raise
            strategy.handle(t, self.special_toks)

        # Resolve allowed specials once; the result is text-independent.
        allowed_special = strategy.handle("", self.special_toks)

        try:
            if not allowed_special:
                return tokenizer.encode_texts(texts, show_progress=show_progress)
            return tokenizer.encode_texts_with_special(
                texts,
                allowed_special,
                show_progress=show_progress,
            )
        except ValueError as e:
            raise TokenizationError("failed to encode texts") from e
