"""Basic byte-level tokenizer implementation."""

from typing import Final
from datasets import load_dataset
from bpe import BytePair, Token
from base_tok import Tokenizer


max_byte_size: Final[int] = 256


class BasicTokenizer(Tokenizer):
    """Tokenizer that operates directly on byte sequences without regex splitting."""
    def __init__(self) -> None:
        super().__init__()

    def train(self, text: list[int], vocab_size: int, verbose=False):
        """Train tokenizer by learning byte pair merges from the input sequence."""
        iterations = vocab_size - max_byte_size

    def encode(self, text: str) -> list[Token]:
        """Encode text into a sequence of tokens."""
        pass

    def decode(self, tokens: list[Token]):
        """Decode a sequence of tokens back into text."""
        pass


def main() -> None:
    """Load and preprocess sci-fi books dataset for tokenizer training."""
    # preprocessing
    ds = load_dataset("stevez80/Sci-Fi-Books-gutenberg", split="train")
    tokens = list("".join(ds[:]["text"]).encode("utf-8"))
    vocab_size = 280


if __name__ == "__main__":
    main()
