"""Factory functions for creating tokenizers."""

from .tokenizers.basic import BasicTokenizer
from bytetok.src.bytetok.exceptions import ModelLoadError, StrategyError
from bytetok.src.bytetok.strategy import (
    AllowAllStrategy,
    AllowCustomStrategy,
    AllowNoneRaiseStrategy,
    AllowNoneStrategy,
    SpecialTokenStrategy,
)
from .tokenizers.regex import RegexTokenizer
from .tokenizers.base import MODEL_SUFFIX, Tokenizer
from pattern import TokenPattern

from typing import Final, Literal, overload
from pathlib import Path


# Strategy factory
# ===================================================================================

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
) -> SpecialTokenStrategy: ...


@overload
def get_strategy(
    name: Literal["custom"], allowed_subset: set[str]
) -> AllowCustomStrategy: ...


def get_strategy(
    name: StrategyName = "none-raise", allowed_subset: set[str] | None = None
) -> SpecialTokenStrategy:
    """
    Create a special token handling strategy.

    :param name: Strategy name: "all" allows all special tokens, "none" skips them,
                 "none-raise" raises on special tokens, "custom" allows only a subset.
    :param allowed_subset: Required when name="custom". Set of allowed special tokens.
    :return: Configured strategy instance.
    :raises StrategyError: If strategy name is unknown or custom strategy lacks allowed_subset.

    .. code-block:: python

        strategy = get_strategy("all")
        strategy = get_strategy("none-raise")
        strategy = get_strategy("custom", allowed_subset={"<|endoftext|>"})
    """

    # handle invalid strat names
    if name not in _SPECIAL_TOKEN_STRATEGIES:
        raise StrategyError(
            "unknown strategy name",
            invalid_name=name,
            available_strats=list(_SPECIAL_TOKEN_STRATEGIES.keys()),
        )

    # custom strat requires subset
    if name == "custom":
        if allowed_subset is None:
            raise StrategyError("allowed_subset is required for custom strategy")
        return AllowCustomStrategy(allowed_subset)

    return _SPECIAL_TOKEN_STRATEGIES[name]()


# ===================================================================================


# Tokenizer factory
# ===================================================================================

Pattern = Literal[
    "gpt2",
    "gpt4",
    "gpt4o",
    "llama3",
    "qwen2",
    "deepseek-coder",
    "deepseek-llm",
    "starcoder",
    "falcon",
    "bloom",
    "chatglm4",
]

_TOKENIZER_REGISTRY: Final[dict[str, type[Tokenizer]]] = {
    "regex": RegexTokenizer,
    "basic": BasicTokenizer,
}


def list_patterns() -> list[str]:
    """Return names of all available built-in tokenization patterns."""
    return [pat.name for pat in TokenPattern]


def get_pattern(name: Pattern) -> str:
    return TokenPattern.get(name)


@overload
def get_tokenizer(pattern: Pattern) -> Tokenizer: ...


@overload
def get_tokenizer(*, custom_pattern: str) -> Tokenizer: ...


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

    # verify given pattern exists
    # get() handles invalid pattern names
    pat_str = TokenPattern.get(pattern)
    return RegexTokenizer(pat_str)


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


# ===================================================================================
