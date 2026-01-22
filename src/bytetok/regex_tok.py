from tok import Tokenizer
from bpe import Token


class RegexTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        pass

    def encode(self, text):
        pass

    def decode(self, tokens: list[Token]):
        pass
