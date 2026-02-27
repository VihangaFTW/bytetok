"""Basic byte-level tokenizer implementation."""

from typing import override, TYPE_CHECKING
from .base import Tokenizer
import logging

from ..errors import VocabularyError

from ..types import Token
from .._trainer import _train_bpe

# need only classname for type annotation
if TYPE_CHECKING:
    from ..strategy import SpecialTokenStrategy

log = logging.getLogger(__name__)


class BasicTokenizer(Tokenizer):
    """Tokenizer that operates directly on byte sequences without regex splitting."""

    TOKENIZER_TYPE = "basic"

    def __init__(self) -> None:
        """Initialize a basic byte-level tokenizer."""
        super().__init__()

    @override
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
        """
        Train on raw text using byte-level BPE.

        Concatenates list inputs, encodes as UTF-8 bytes, and learns
        ``vocab_size - 256`` merges on top of the base byte vocabulary.

        :param verbose: Log each learned merge when ``True``.
        :raises VocabularyError: If ``vocab_size`` is less than or equal to 256.
        :raises TrainingError: If the encoded training sequence is empty.
        """
        if vocab_size <= 256:
            raise VocabularyError(
                "vocab size must be greater than 256", vocab_size=vocab_size
            )

        # handle list input and convert text to bytes
        if isinstance(text, list):
            text = "".join(text)

        tokens = list(text.encode("utf-8"))

        # merges beyond base byte vocabulary
        n_merges = vocab_size - 256

        result = _train_bpe(tokens, n_merges, verbose=verbose)

        if result.n_merges_completed < n_merges:
            log.warning(
                f"no more byte pairs to merge after {result.n_merges_completed} merges "
                f"(requested {n_merges}) stopping early"
            )

        self.merges = result.merges
        self.vocab = result.vocab
        self._tokenizer = None

    @override
    def _encode_impl(
        self,
        text: str,
        strategy: "SpecialTokenStrategy | None" = None,
    ) -> list[Token]:
        """
        Encode text into tokens using byte-level BPE.

        The ``strategy`` argument is ignored; this tokenizer does not support
        special token handling.
        """
        _ = strategy
        tokenizer = self._get_rust_tokenizer(pattern=r".+")
        return tokenizer.encode_bytes(text)

    @override
    def _encode_batch_impl(
        self,
        texts: list[str],
        strategy: "SpecialTokenStrategy | None" = None,
    ) -> list[list[Token]]:
        """
        Encode multiple texts in parallel via Rust/Rayon.

        The ``strategy`` argument is ignored; this tokenizer does not support
        special token handling.
        """
        _ = strategy
        tokenizer = self._get_rust_tokenizer(pattern=r".+")
        return tokenizer.encode_bytes_batch(texts)
