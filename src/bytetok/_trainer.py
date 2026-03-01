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
    tokens: list[Token],
    n_merges: int,
    verbose: bool = False,
    show_progress: bool = True,
) -> BPETrainingResult:
    """
    Train a BPE model from a token sequence using the Rust trainer.

    :param tokens: Input token sequence used for training.
    :param n_merges: Maximum number of merge operations to perform.
    :param verbose: Log each learned merge when ``True``.
    :param show_progress: Display a Rust-side progress bar during training when ``True``.
    :returns: Training output containing vocab, merge rules, and completed merge count.
    :raises TrainingError: If ``tokens`` is empty or if the Rust trainer fails.
    """
    if len(tokens) == 0:
        raise TrainingError("empty token sequence, no training performed")

    trainer = RustBPETrainer(tokens, 256)
    try:
        trainer.train(n_merges, show_progress=show_progress)
    except ValueError as e:
        raise TrainingError("internal error") from e

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
