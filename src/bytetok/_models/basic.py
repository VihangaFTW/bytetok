"""Basic byte-level tokenizer implementation."""

from typing import override, TYPE_CHECKING
from .base import Tokenizer
from .._decorators import measure_time
import logging

from ..errors import VocabularyError, TrainingError

from ..types import Token
from ..trainer import _train_bpe

# need only classname for type annotation
if TYPE_CHECKING:
    from ..strategy import SpecialTokenStrategy

log = logging.getLogger(__name__)


class BasicTokenizer(Tokenizer):
    """
    Tokenizer that operates directly on byte sequences without regex splitting
    """

    TOKENIZER_TYPE = "basic"

    def __init__(self) -> None:
        """Initialize a basic byte-level tokenizer."""
        super().__init__()

    @override
    @measure_time
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
        """
        Train the tokenizer on raw text using byte-level BPE.

        This implementation concatenates list inputs, encodes text as UTF-8
        bytes, and learns ``vocab_size - 256`` merges on top of the base byte
        vocabulary.

        :param text: Training text as a single string or list of strings.
        :param vocab_size: Target vocabulary size including the base 256 bytes.
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
    def encode(
        self,
        text: str,
        strategy: "SpecialTokenStrategy | None" = None,
    ) -> list[Token]:
        """
        Encode text into tokens using byte-level BPE.

        The ``strategy`` argument is ignored because this tokenizer does not
        support special token handling.

        :param text: Input text to encode.
        :param strategy: Ignored for this tokenizer.
        :returns: Encoded token sequence.
        """
        _ = strategy
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before encoding"
            )
        tokenizer = self._get_rust_tokenizer(pattern=r".+")
        return tokenizer.encode_bytes(text)

    @override
    def encode_batch(
        self,
        texts: list[str],
        strategy: "SpecialTokenStrategy | None" = None,
    ) -> list[list[Token]]:
        """
        Encode multiple texts in parallel via Rust/Rayon.

        The ``strategy`` argument is ignored because this tokenizer does not
        support special token handling.

        :param texts: Text inputs to encode.
        :param strategy: Ignored for this tokenizer.
        :returns: Encoded token sequences in input order.
        """
        _ = strategy
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before encoding"
            )
        if not texts:
            return []
        tokenizer = self._get_rust_tokenizer(pattern=r".+")
        return tokenizer.encode_bytes_batch(texts)

    @override
    def decode(self, tokens: list[Token]) -> str:
        """
        Decode tokens into UTF-8 text.

        :param tokens: Token sequence to decode.
        :returns: Decoded text where invalid UTF-8 is replaced.
        :raises TrainingError: If the tokenizer has not been trained yet.
        """
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before decoding"
            )
        tokenizer = self._get_rust_tokenizer(pattern=r".+")
        try:
            return tokenizer.decode_tokens(tokens, errors="replace")
        except ValueError as e:
            raise VocabularyError("token not found in vocabulary") from e
