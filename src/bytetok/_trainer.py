"""Standalone BPE training module."""

from dataclasses import dataclass
import logging

from bytetok.errors import TrainingError

from .types import (
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


def _train_bpe(
    tokens: list[Token], n_merges: int, verbose: bool = False
) -> BPETrainingResult:
    """
    Train BPE on a sequence of tokens via the Rust implementation.

    :param verbose: Log each merge operation when ``True``.
    """
    if len(tokens) == 0:
        raise TrainingError("empty token sequence, no training performed")

    trainer = RustBPETrainer(tokens, 256)
    trainer.train(n_merges)

    merge_history = trainer.get_merge_history()

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


__all__ = ["_train_bpe"]
