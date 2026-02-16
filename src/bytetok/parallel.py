"""Parallel processing mode helpers for batch encoding."""

from enum import Enum
from typing import TYPE_CHECKING, Literal

from .errors import StrategyError
from ._bpe import Token

if TYPE_CHECKING:
    from ._models.base import Tokenizer
    from .strategy import SpecialTokenStrategy

ParallelStrategy = Literal["auto", "batch", "chunk", "off"]


class ParallelMode(str, Enum):
    """Named parallelization modes for batch encoding."""

    AUTO = "auto"
    BATCH = "batch"
    CHUNK = "chunk"
    OFF = "off"

    @classmethod
    def get(cls, name: str) -> "ParallelMode":
        """Get parallel mode by name (case-insensitive)."""
        try:
            return cls[name.upper()]
        except KeyError:
            raise StrategyError(
                "unknown mode",
                invalid_name=name,
                available_strats=[mode.value for mode in cls],
            )


def list_parallel_modes() -> list[str]:
    """Return available parallel mode names."""
    return [mode.value for mode in ParallelMode]


def encode_batch(
    tokenizer: "Tokenizer",
    texts: list[str],
    strategy: "SpecialTokenStrategy | None" = None,
    num_workers: int | None = None,
    parallel_mode: ParallelStrategy = "auto",
) -> list[list[Token]]:
    """Encode many texts with optional parallel processing mode."""
    return tokenizer.encode_batch(
        texts,
        strategy=strategy,
        num_workers=num_workers,
        parallel_mode=ParallelMode.get(parallel_mode),
    )


__all__ = [
    "ParallelStrategy",
    "ParallelMode",
    "list_parallel_modes",
    "encode_batch",
]
