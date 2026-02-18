"""Factory functions for creating tokenizers."""

from ._models.basic import BasicTokenizer
from .errors import ModelLoadError
from ._models.regex import RegexTokenizer
from ._models.base import MODEL_SUFFIX, Tokenizer
from .pattern import Pattern, get_pattern

from typing import Final, overload
from pathlib import Path


# Tokenizer factory
# ===================================================================================

_TOKENIZER_REGISTRY: Final[dict[str, type[Tokenizer]]] = {
    "regex": RegexTokenizer,
    "basic": BasicTokenizer,
}


@overload
def get_tokenizer(pattern: Pattern) -> Tokenizer:
    """Create a tokenizer from a named built-in pattern."""
    ...


@overload
def get_tokenizer(*, custom_pattern: str) -> Tokenizer:
    """Create a tokenizer from a custom regex pattern."""
    ...


def get_tokenizer(
    pattern: Pattern = "gpt4o", *, custom_pattern: str | None = None
) -> Tokenizer:
    """
    Create a tokenizer with a built-in or custom regex pattern.

    :param pattern: Built-in pattern name (e.g., "gpt2", "gpt4o", "llama3").
                    Ignored if custom_pattern is provided.
    :param custom_pattern: Custom regex pattern string. Overrides pattern parameter.
    :return: Configured tokenizer instance.
    :raises PatternError: If custom_pattern is invalid regex.

    .. code-block:: python

        # Use built-in pattern
        tokenizer = get_tokenizer("gpt4o")

        # Use custom pattern
        tokenizer = get_tokenizer(custom_pattern=r"'s|'t|'re|'ve|'m|'ll|'d")
    """
    # regex class initializer handles invalid custom patterns
    if custom_pattern is not None:
        return RegexTokenizer(custom_pattern)

    return RegexTokenizer(get_pattern(pattern))


def _detect_tokenizer_type(model_path: str) -> str:
    """Read tokenizer type from model file header."""
    path = Path(model_path)

    if not path.exists():
        raise ModelLoadError("model filepath does not exist", model_path=str(path))

    if path.suffix != MODEL_SUFFIX:
        raise ModelLoadError("expected .model file", model_path=str(path))

    with path.open("r", encoding="utf-8") as f:
        # skip version to get tokenizer type
        _ = f.readline().strip()

        tok_type = f.readline().strip()
        if tok_type.startswith("type "):
            return tok_type[5:]

        raise ModelLoadError(f"expected tokenizer type got {tok_type}")


def from_pretrained(model_path: str) -> Tokenizer:
    """
    Load a pre-trained tokenizer from disk.

    Automatically detects tokenizer type from
    the model file header and loads the appropriate implementation.

    :param model_path: Path to the .model file.
    :return: Loaded tokenizer instance with vocabulary and configuration.
    :raises ModelLoadError: If file doesn't exist, has wrong extension, or contains
                            unknown tokenizer type.

    .. code-block:: python

        tokenizer = from_pretrained("path/to/model.model")
        tokens = tokenizer.encode("Hello world")
    """
    # extract model type from file
    tok_type = _detect_tokenizer_type(model_path)

    # look up class from registry
    if tok_type not in _TOKENIZER_REGISTRY:
        raise ModelLoadError(
            "unknown tokenizer type in model file",
            type_mismatch=(tok_type, list(_TOKENIZER_REGISTRY.keys())),
        )

    # instantiate and load
    tokenizer = _TOKENIZER_REGISTRY[tok_type]()

    tokenizer.load(model_path)

    return tokenizer


__all__ = [
    "Pattern",
    "get_pattern",
    "get_tokenizer",
    "from_pretrained",
]
