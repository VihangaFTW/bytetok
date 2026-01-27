"""
A fun reproduction of GPT-4's tokenization using its pre-trained merges via tiktoken library.
"""

from ..tokenizers.regex import RegexTokenizer
from factory import get_pattern
import tiktoken


class GPT4Tokenizer(RegexTokenizer):
    """ """

    def __init__(self) -> None:
        super().__init__(get_pattern("gpt4"))
        # get official gpt4 merges
        _ = tiktoken.get_encoding("cl100k_base")
        # todo: complete class
        pass
