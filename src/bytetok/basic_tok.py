from typing import Final
from datasets import load_dataset
from bpe import BytePair, Token
from base_tok import Tokenizer


max_byte_size: Final[int] = 256


class BasicTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text: list[int], vocab_size: int, verbose=False):
        iterations = vocab_size - max_byte_size

    def encode(self, text):
        pass

    def decode(self, tokens: list[Token]):
        pass


def main() -> None:
    # preprocessing
    ds = load_dataset("stevez80/Sci-Fi-Books-gutenberg", split="train")
    tokens = list("".join(ds[:]["text"]).encode("utf-8"))
    vocab_size = 280


if __name__ == "__main__":
    main()
