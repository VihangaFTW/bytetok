"""Special token handling for tokenization."""

from typing import Final, Literal, overload, override
from abc import ABC, abstractmethod
import logging
from .types import Token

from .errors import SpecialTokenError, StrategyError

log = logging.getLogger(__name__)

# =========================================================================================

# special token handling strategies


class SpecialTokenStrategy(ABC):
    """Base strategy for handling special tokens during encoding."""

    @abstractmethod
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
        """Return the special tokens to use for encoding."""


class AllowAllStrategy(SpecialTokenStrategy):
    """Strategy that allows all registered special tokens."""

    @override
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
        """Return all registered special tokens unchanged."""
        if not special_toks:
            log.warning("no special tokens registered")
        return special_toks


class AllowNoneRaiseStrategy(SpecialTokenStrategy):
    """Strategy that raises if special tokens are found in text to be encoded."""

    @override
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
        """Raise when text contains disallowed special tokens."""
        if special_toks:
            found = {seq for seq in special_toks if seq in text}
            if found:
                raise SpecialTokenError(
                    "special tokens found in text but not allowed", found_tokens=found
                )
        return {}


class AllowNoneStrategy(SpecialTokenStrategy):
    """Strategy that silently ignores all special tokens."""

    @override
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
        """Ignore special tokens and encode text as normal content."""
        if special_toks and not all(seq not in text for seq in special_toks):
            log.warning("special tokens found in text but not allowed")
        return {}


class AllowCustomStrategy(SpecialTokenStrategy):
    """Strategy that allows only specified special tokens."""

    def __init__(self, allowed_subset: set[str]) -> None:
        """Store the special token subset allowed during encoding."""
        super().__init__()
        self.allowed_subset = allowed_subset

    @override
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
        """Return only special tokens present in the allowed subset."""
        return {
            seq: tok for seq, tok in special_toks.items() if seq in self.allowed_subset
        }


StrategyName = Literal["all", "none", "none-raise", "custom"]

_SPECIAL_TOKEN_STRATEGIES: Final[dict[str, type[SpecialTokenStrategy]]] = {
    "all": AllowAllStrategy,
    "none": AllowNoneStrategy,
    "none-raise": AllowNoneRaiseStrategy,
    "custom": AllowCustomStrategy,
}


def list_strategies() -> list[str]:
    """Return available special token strategy names."""
    return list(_SPECIAL_TOKEN_STRATEGIES.keys())


@overload
def get_strategy(
    name: Literal["all", "none", "none-raise"],
) -> SpecialTokenStrategy:
    """Return a built-in strategy that does not need extra arguments."""
    ...


@overload
def get_strategy(
    name: Literal["custom"], allowed_subset: set[str]
) -> AllowCustomStrategy:
    """Return a custom strategy limited to ``allowed_subset``."""
    ...


def get_strategy(
    name: StrategyName = "none-raise", allowed_subset: set[str] | None = None
) -> SpecialTokenStrategy:
    """
    Create a special token strategy by name.

    :param name: Strategy identifier — "all", "none", "none-raise", or "custom".
    :param allowed_subset: Required for "custom"; tokens allowed during encoding.
    :raises StrategyError: If name is unknown or allowed_subset is missing for custom.
    """
    if name not in _SPECIAL_TOKEN_STRATEGIES:
        raise StrategyError(
            "unknown strategy name",
            invalid_name=name,
            available_strats=list(_SPECIAL_TOKEN_STRATEGIES.keys()),
        )

    if name == "custom":
        if allowed_subset is None:
            raise StrategyError("allowed_subset is required for custom strategy")
        return AllowCustomStrategy(allowed_subset)

    return _SPECIAL_TOKEN_STRATEGIES[name]()


__all__ = [
    "StrategyName",
    "SpecialTokenStrategy",
    "AllowAllStrategy",
    "AllowNoneStrategy",
    "AllowNoneRaiseStrategy",
    "AllowCustomStrategy",
    "list_strategies",
    "get_strategy",
]
