"""Factory functions for creating tokenizers."""

from bytetok.src.bytetok.exceptions import StrategyError
from bytetok.src.bytetok.strategy import (
    AllowAllStrategy,
    AllowCustomStrategy,
    AllowNoneRaiseStrategy,
    AllowNoneStrategy,
    SpecialTokenStrategy,
)
from .regex_tok import RegexTokenizer
from .base_tok import Tokenizer
from pattern import TokenPattern

from typing import Final, Literal, overload


# Strategy factory
# ===================================================================================

StrategyName = Literal["all", "none", "none_raise", "custom"]

_strategies: Final[dict[str, type[SpecialTokenStrategy]]] = {
    "all": AllowAllStrategy,
    "none": AllowNoneStrategy,
    "none_raise": AllowNoneRaiseStrategy,
    "custom": AllowCustomStrategy,
}


def list_strategies() -> list[str]:
    return list(_strategies.keys())


@overload
def get_strategy(
    name: Literal["all", "none", "none_raise"],
) -> SpecialTokenStrategy: ...


@overload
def get_strategy(
    name: Literal["custom"], allowed_subset: set[str]
) -> AllowCustomStrategy: ...


def get_strategy(
    name: StrategyName = "none_raise", allowed_subset: set[str] | None = None
) -> SpecialTokenStrategy:
    """
    Get a special token handling strategy.

    Args:
        name: Strategy name ("all", "none", "none_raise", "custom").
        **kwargs: Strategy-specific arguments (e.g., allowed_subset for custom).

    Returns:
        A SpecialTokenStrategy instance.

    Example:
        strategy = get_strategy("all")
        strategy = get_strategy("none_raise")
        strategy = get_strategy("custom", allowed_subset={"<|endoftext|>"})
    """

    # handle invalid strat names
    if name not in _strategies:
        raise StrategyError(
            "unknown strategy name",
            invalid_name=name,
            available_strats=list(_strategies.keys()),
        )

    # custom strat requires subset
    if name == "custom":
        if allowed_subset is None:
            raise StrategyError("allowed_subset is required for custom strategy")
        return AllowCustomStrategy(allowed_subset)

    return _strategies[name]()


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


def list_patterns() -> list[str]:
    """
    List all available built-in patterns.

    Returns:
        List of pattern names.

    Example:
        patterns = list_patterns()  # ["GPT2", "GPT4"]
    """
    return [pat.name for pat in TokenPattern]


@overload
def get_tokenizer(pattern: Pattern) -> Tokenizer: ...


@overload
def get_tokenizer(*, custom_pattern: str) -> Tokenizer: ...


def get_tokenizer(
    pattern: Pattern = "gpt4o", *, custom_pattern: str | None = None
) -> Tokenizer:
    """ """
    # regex class initializer handles invalid custom patterns
    if custom_pattern is not None:
        return RegexTokenizer(custom_pattern)

    # verify given pattern exists
    # get() handles invalid pattern names
    pat_str = TokenPattern.get(pattern)
    return RegexTokenizer(pat_str)


# todo method reads the model type
def from_pretrained(model_path: str) -> Tokenizer:
    """
    Load a pre-trained tokenizer from disk.

    Args:
        model_path: Path to the .model file.

    Returns:
        A loaded tokenizer instance.

    Example:
        tokenizer = from_pretrained("path/to/model.model")
    """
    return RegexTokenizer.from_pretrained(model_path)


# ===================================================================================
