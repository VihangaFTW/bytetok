"""Special token handling strategies for tokenization."""

from typing import override
from abc import ABC, abstractmethod
import logging
from ._bpe import Token

from .exceptions import SpecialTokenError

log = logging.getLogger(__name__)


class SpecialTokenStrategy(ABC):
    """Base strategy for handling special tokens during encoding."""

    @abstractmethod
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
        """Return the special tokens to use for encoding."""


class AllowAllStrategy(SpecialTokenStrategy):
    """Strategy that allows all registered special tokens."""

    @override
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
        if not special_toks:
            log.warning("no special tokens registered..")
        return special_toks


class AllowNoneRaiseStrategy(SpecialTokenStrategy):
    """Strategy that raises if special tokens are found in text to be encoded."""

    @override
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
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
        if special_toks and not all(seq not in text for seq in special_toks):
            log.warning("special tokens found in text but not allowed")
        return {}


class AllowCustomStrategy(SpecialTokenStrategy):
    """Strategy that allows only specified special tokens."""

    def __init__(self, allowed_subset: set[str]) -> None:
        super().__init__()
        self.allowed_subset = allowed_subset

    @override
    def handle(self, text: str, special_toks: dict[str, Token]) -> dict[str, Token]:
        return {
            seq: tok for seq, tok in special_toks.items() if seq in self.allowed_subset
        }
