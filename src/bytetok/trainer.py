"""Standalone BPE training module."""

from dataclasses import dataclass
import logging

from bytetok.errors import TrainingError

from ._bpe import (
    Token,
    TokenPair,
    Encoding,
    Vocabulary,
)
from .bpe import RustBPETrainer

log = logging.getLogger(__name__)


@dataclass
class BPETrainingResult:
    """Results from one BPE training run."""

    vocab: Vocabulary
    merges: Encoding
    n_merges_completed: int


class BPETrainer:
    """
    BPE trainer that learns merge operations from token sequences.

    Trained using a fast Rust implementation.

    Example:
       >>> tokens = [72, 101, 108, 108, 111]  # "Hello" as bytes
       >>> trainer = BPETrainer()
       >>> result = trainer.train(tokens, n_merges=10)
       >>> print(f"Learned {result.n_merges_completed} merges")
       >>> print(f"Vocabulary size: {len(result.vocab)}")
    """

    def __init__(self) -> None:
        self.trainer: RustBPETrainer | None = None

    def train(
        self, tokens: list[Token], n_merges: int, verbose: bool = False
    ) -> BPETrainingResult:
        """
        Train BPE on a sequence of tokens.

        :param tokens: Sequence of token IDs (typically bytes 0-255).
        :param n_merges: Number of merge operations to learn.
        :param verbose: Whether to log merge operations.
        :return: Training results containing merges and vocabulary.
        """
        if len(tokens) < 10:
            raise TrainingError("text should have at least 10 characters")

        # Initialize Rust trainer.
        self.trainer = RustBPETrainer(tokens, 256)

        # Train for requested number of merges.
        self.trainer.train(n_merges)

        merge_history = self.trainer.get_merge_history()

        merges: Encoding = {}
        vocab: Vocabulary = {tok: bytes([tok]) for tok in range(256)}

        for (tok_a, tok_b), new_tok in merge_history:
            pair: TokenPair = (tok_a, tok_b)
            merges[pair] = new_tok
            vocab[new_tok] = vocab[tok_a] + vocab[tok_b]

            if verbose:
                log.info(
                    "merge %d/%d: %s -> %d",
                    len(merges),
                    n_merges,
                    pair,
                    new_tok,
                )

        return BPETrainingResult(
            vocab=vocab,
            merges=merges,
            n_merges_completed=len(merge_history),
        )
