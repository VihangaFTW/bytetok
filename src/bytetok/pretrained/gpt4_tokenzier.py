"""
A fun reproduction of GPT-4's tokenization using its pre-trained merges via tiktoken library.
"""

from ..tokenizers.regex import RegexTokenizer
from factory import get_pattern
import tiktoken


class GPT4Tokenizer(RegexTokenizer):
    """Tokenizer that reproduces GPT-4 tokenization using pre-trained merges."""

    def __init__(self) -> None:
        """Initialize with GPT-4 pattern and load official merges from tiktoken."""
        super().__init__(get_pattern("gpt4"))
        # get official gpt4 merges
        _ = tiktoken.get_encoding("cl100k_base")
        # todo: complete class
        pass
