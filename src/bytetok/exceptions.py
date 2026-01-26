"""Custom exception hierarchy for bytetok tokenization errors."""

import regex as re

from _bpe import Token


class ByteTokError(Exception):
    """Base exception for all bytetok errors."""


class SpecialTokenError(ByteTokError):
    """Raised when special token handling fails."""

    def __init__(self, message: str, *, found_tokens: set[str] | None = None) -> None:
        """Initialize with optional found_tokens that get appended to the message."""
        if found_tokens:
            message = f"{message} (found: {', '.join(sorted(found_tokens))})"
        super().__init__(message)
        self.found_tokens = found_tokens


class TokenizationError(ByteTokError):
    """Raised when tokenization fails."""

    def __init__(
        self,
        message: str,
        *,
        position: int | None = None,
        input_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.position = position
        self.input_text = input_text


class VocabularyError(ByteTokError):
    """Raised when vocabulary operations fail."""

    def __init__(
        self,
        message: str,
        *,
        vocab_size: int | None = None,
        invalid_tok: Token | None = None,
    ) -> None:
        """Initialize with optional token and vocab_size that get appended to the message."""
        extra = " "
        # training: vocab size <= 256
        if vocab_size:
            extra += f"(vocab size: {vocab_size}) "
        # decoding: token not in model vocab
        if invalid_tok:
            extra += f"(invalid token: {invalid_tok}) "
        super().__init__(message + extra)
        self.vocab_size = vocab_size
        self.invalid_tok = invalid_tok


class TrainingError(ByteTokError):
    """Raised when tokenizer training fails."""

    def __init__(self, message: str, *, vocab_size: int) -> None:
        super().__init__(message)
        self.vocab_size = vocab_size


class ModelLoadError(ByteTokError):
    """Raised when loading a tokenizer model fails."""

    def __init__(
        self,
        message: str,
        *,
        model_path: str | None = None,
        version_mismatch: tuple[str, str] | None = None,
    ) -> None:
        extra = " "
        if model_path:
            extra += f"(path: {model_path}) "
        if version_mismatch is not None:
            extra += f"(expected: {version_mismatch[1]}) (got {version_mismatch[0]}) "
        super().__init__(message + extra)
        self.model_path = model_path
        self.version_mismatch = version_mismatch


class PatternError(ByteTokError):
    """Raised when compiling and/or validating regex patterns."""

    def __init__(
        self,
        message: str,
        *,
        pattern: str | None = None,
        regex_err: re.error | None = None,
    ) -> None:
        """
        Initialize PatternError with pattern details.

        Args:
            message: Error message.
            pattern: The regex pattern that failed.
            regex_err: The underlying regex error from the regex library.
        """
        extra = " "
        if pattern:
            extra += f"(pattern: {pattern!r}) "
        if regex_err:
            extra += f"(reason: {regex_err}) "
        super().__init__(message + extra)
        self.pattern = pattern
        self.regex_err = regex_err


class StrategyError(ByteTokError):
    """Raised when strategy operations fail."""

    def __init__(
        self,
        message: str,
        *,
        invalid_name: str | None = None,
        available_strats: list[str] | None = None,
    ) -> None:
        """ """
        extra = " "
        if invalid_name:
            extra += f"(available: {available_strats}) (got {invalid_name}) "
        super().__init__(message + extra)
        self.invalid_name = invalid_name
        self.available_strats = available_strats
