"""Basic byte-level tokenizer implementation."""

from typing import override, TYPE_CHECKING
from .base import ParallelMode, Tokenizer
from .._decorators import measure_time
import logging

from ..errors import VocabularyError

from ..types import Token
from ..trainer import train_bpe

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

        result = train_bpe(tokens, n_merges, verbose=verbose)

        if result.n_merges_completed < n_merges:
            log.warning(
                f"no more byte pairs to merge after {result.n_merges_completed} merges "
                f"(requested {n_merges}) stopping early"
            )

        self.merges = result.merges  # used for encoding text -> tokens
        self.vocab = result.vocab  # used for decoding tokens -> text
        # Invalidate Rust caches since merges changed.
        self._converter = None
        self._tokenizer = None

    @override
    def encode(
        self,
        text: str,
        strategy: "SpecialTokenStrategy | None" = None,
        num_workers: int | None = None,
    ) -> list[Token]:
        """
        Encode text into tokens using byte-level BPE.

        The ``strategy`` argument is ignored because this tokenizer does not
        support special token handling.

        :param text: Input text to encode.
        :param strategy: Ignored for this tokenizer.
        :param num_workers: Ignored for single-text encoding.
        :returns: Encoded token sequence.
        """
        # BasicTokenizer does not support special token strategies.
        _ = strategy
        # BasicTokenizer does not use worker hints for single-text encoding.
        _ = num_workers
        if not self.merges:
            return list(text.encode("utf-8", errors="replace"))
        tokenizer = self._get_rust_tokenizer(pattern=r".+")
        return tokenizer.encode_bytes(text)

    @override
    def encode_batch(
        self,
        texts: list[str],
        strategy: "SpecialTokenStrategy | None" = None,
        num_workers: int | None = None,
        parallel_mode: ParallelMode = ParallelMode.AUTO,
    ) -> list[list[Token]]:
        """
        Encode multiple texts with optional batch-level parallel processing.

        The ``strategy`` argument is ignored because this tokenizer does not
        support special token handling. Parallelization happens across texts,
        not within a single text because merges can span arbitrary byte boundaries
        inside a single text. Hence, "chunk" mode falls back to "off".

        :param texts: Text inputs to encode.
        :param strategy: Ignored for this tokenizer.
        :param num_workers: Worker count for batch-level parallel mode.
        :param parallel_mode: Batch parallelization mode selection.
        :returns: Encoded token sequences in input order.
        """
        # BasicTokenizer does not support special token strategies.
        _ = strategy

        # Worker hints are ignored because Rust batch methods handle parallelism internally.
        _ = num_workers
        if not texts:
            return []
        if not self.merges:
            return [list(text.encode("utf-8", errors="replace")) for text in texts]
        tokenizer = self._get_rust_tokenizer(pattern=r".+")

        match parallel_mode:
            case ParallelMode.OFF:
                return [tokenizer.encode_bytes(text) for text in texts]
            case ParallelMode.BATCH:
                return tokenizer.encode_bytes_batch(texts)
            case ParallelMode.CHUNK:
                # Chunk-mode is equivalent to serial full-text encoding for BasicTokenizer.
                return [tokenizer.encode_bytes(text) for text in texts]
            case ParallelMode.AUTO:
                if len(texts) <= 1:
                    return [tokenizer.encode_bytes(texts[0])]
                return tokenizer.encode_bytes_batch(texts)

    @override
    def decode(self, tokens: list[Token]) -> str:
        """
        Decode tokens into UTF-8 text.

        :param tokens: Token sequence to decode.
        :returns: Decoded text where invalid UTF-8 is replaced.
        """
        tokenizer = self._get_rust_tokenizer(pattern=r".+")
        try:
            return tokenizer.decode_tokens(tokens, errors="replace")
        except ValueError as e:
            raise VocabularyError("token not found in vocabulary") from e

    def _encode_one(self, text: str) -> list[Token]:
        """Encode one text string through byte conversion and BPE merges."""
        return self.encode(text)

