"""Helpers for training BPE models with the Rust backend."""

from dataclasses import dataclass
import logging

from bytetok.errors import TrainingError

from .types import (
    Token,
    Encoding,
    Vocabulary,
)
from .bpe import RustBPETrainer

log = logging.getLogger(__name__)


@dataclass
class BPETrainingResult:
    """Container for the artifacts produced by one training run."""

    vocab: Vocabulary
    merges: Encoding
    n_merges_completed: int


def _train_bpe_from_tokens(
    tokens: list[Token],
    *,
    n_merges: int,
    verbose: bool = False,
    show_progress: bool = True,
) -> BPETrainingResult:
    """
    Train a BPE model from a pre-tokenized sequence.

    This entry point is useful when the caller already has the initial token
    stream. For raw text corpora, prefer `_train_bpe_from_corpus()`, which lets
    Rust handle pretokenization as well as training.

    Args:
        tokens: Initial token sequence used for training.
        n_merges: Maximum number of merge operations to perform.
        verbose: Whether to log each learned merge.
        show_progress: Whether to show the Rust-side training progress bar.

    Returns:
        The learned vocabulary, merge table, and completed merge count.

    Raises:
        TrainingError: If `tokens` is empty or the Rust trainer reports an error.
    """
    if len(tokens) == 0:
        raise TrainingError("empty token sequence, no training performed")

    trainer = RustBPETrainer(tokens, 256)
    try:
        trainer.train(n_merges, show_progress=show_progress)
    except ValueError as e:
        raise TrainingError(f"internal error: {e}") from e

    merge_history = trainer.get_merge_history()
    merges, vocab = trainer.get_merges_and_vocab()

    if verbose and merge_history:
        for i, (pair, new_tok) in enumerate(merge_history, start=1):
            log.info(
                "merge %d/%d: %s -> %d",
                i,
                n_merges,
                pair,
                new_tok,
            )

    return BPETrainingResult(
        vocab=vocab,
        merges=merges,
        n_merges_completed=len(merges),
    )


def _train_bpe_from_corpus(
    corpus: str,
    pattern: str,
    *,
    n_merges: int,
    verbose: bool = True,
    show_progress: bool = True,
) -> BPETrainingResult:
    """
    Train a BPE model directly from a text corpus.

    The corpus is pretokenized on the Rust side before training starts. This is
    usually the faster path when your input data is still raw text.

    Args:
        corpus: Input text used to initialize the trainer.
        pattern: Regex pattern used for Rust-side pretokenization.
        n_merges: Maximum number of merge operations to perform.
        verbose: Whether to log each learned merge.
        show_progress: Whether to show the Rust-side training progress bar.

    Returns:
        The learned vocabulary, merge table, and completed merge count.

    Raises:
        TrainingError: If `corpus` is empty or the Rust trainer reports an error.
    """
    if len(corpus) == 0:
        raise TrainingError("empty corpus, no training performed")

    trainer = RustBPETrainer.from_corpus(corpus, pattern, min_count=1)
    try:
        trainer.train(n_merges, show_progress=show_progress)
    except ValueError as e:
        raise TrainingError(f"internal error: {e}") from e

    merge_history = trainer.get_merge_history()
    merges, vocab = trainer.get_merges_and_vocab()

    if verbose and merge_history:
        for i, (pair, new_tok) in enumerate(merge_history, start=1):
            log.info(
                "merge %d/%d: %s -> %d",
                i,
                n_merges,
                pair,
                new_tok,
            )

    return BPETrainingResult(
        vocab=vocab,
        merges=merges,
        n_merges_completed=len(merges),
    )


__all__ = ["_train_bpe_from_tokens", "_train_bpe_from_corpus", "BPETrainingResult"]
