"""Regex-based byte-level tokenizer implementation."""

from base_tok import Tokenizer
from bpe import Token


class RegexTokenizer(Tokenizer):
    """Tokenizer that splits text using regex patterns before applying BPE."""
    def __init__(self) -> None:
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        """Train tokenizer by learning byte pair merges on regex-split chunks."""
        pass

    def encode(self, text):
        """Encode text into a sequence of tokens using regex splitting."""
        pass

    def decode(self, tokens: list[Token]):
        """Decode a sequence of tokens back into text."""
        pass
