"""Factory functions for creating tokenizers."""

from .regex_tok import RegexTokenizer
from .base_tok import Tokenizer
from pattern import TokenPattern


def get_tokenizer(pattern: str = "gpt4") -> RegexTokenizer:
    """
    Get a tokenizer with a specific pattern.

    Args:
        pattern: Pattern name ("gpt2", "gpt4") or custom regex pattern.

    Returns:
        A new RegexTokenizer instance.

    Example:
        tokenizer = get_tokenizer("gpt4")
        tokenizer = get_tokenizer("gpt2")
    """
    return RegexTokenizer.from_pattern(pattern)


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


def list_patterns() -> list[str]:
    """
    List all available built-in patterns.

    Returns:
        List of pattern names.

    Example:
        patterns = list_patterns()  # ["GPT2", "GPT4"]
    """
    return [pat.name for pat in TokenPattern]
