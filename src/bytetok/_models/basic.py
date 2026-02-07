"""Basic byte-level tokenizer implementation."""

from typing import override, TYPE_CHECKING
from bytetok.bpe import RustBPETrainer
from .base import Tokenizer
from .._decorators import measure_time
import logging

from ..errors import TrainingError, VocabularyError

from .._bpe import Encoding, Token

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
        super().__init__()

    @override
    @measure_time
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
        """Train tokenizer by learning byte pair merges from the input sequence."""
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

        # train tokenizer
        self._train_rust(tokens, n_merges, verbose)

    @override
    def encode(
        self, text: str, strategy: "SpecialTokenStrategy | None" = None
    ) -> list[Token]:
        """Encode text into a sequence of tokens."""
        # BasicTokenizer does not support special token strategies
        _ = strategy
        # encode Unicode text into bytes
        txt_bytes = text.encode("utf-8", errors="replace")
        # convert each byte to [0-255] token range
        tokens = list(txt_bytes)
        # return bpe of tokens
        return self._apply_fast_bpe_chunk(tokens)

    @override
    def decode(self, tokens: list[Token]) -> str:
        """Decode a sequence of tokens back into text."""
        # token stream -> byte stream
        txt_bytes = b"".join(self.vocab[tok] for tok in tokens)
        # byte stream -> python string
        return txt_bytes.decode("utf-8", errors="replace")

    def _train_rust(self, tokens: list[int], n_merges: int, verbose: bool) -> None:
        """Train using fast Rust implementation."""

        if len(tokens) == 0:
            raise TrainingError("empty token sequence, no training performed")

        trainer = RustBPETrainer(tokens, 256)
        trainer.train(n_merges)

        # get merge history
        merge_history = trainer.get_merge_history()

        # build merges dictionary and vocabulary
        merges: Encoding = {}
        vocab = {tok: bytes([tok]) for tok in range(256)}

        for pair, tok in merge_history:
            merges[pair] = tok
            vocab[tok] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                log.info(f"merge {len(merges)}/{n_merges}: {pair} -> {tok}")

        if len(merge_history) < n_merges:
            log.warning(
                f"no more byte pairs to merge after {len(merge_history)} merges "
                f"(requested {n_merges}). stopping early."
            )

        self.merges = merges  # used for encoding text -> tokens
        self.vocab = vocab  # used for decoding tokens -> text
        # invalidate encoder cache since merges changed
        self._encoder = None
