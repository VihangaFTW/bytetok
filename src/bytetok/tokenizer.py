from abc import ABC, abstractmethod
from datasets import load_dataset



class Tokenizer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, text, vocab_size, verbose=False): ...

    @abstractmethod
    def encode(self, text): ...

    @abstractmethod
    def decode(self, tokens: list[int]): ...


class BasicTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        pass

    def encode(self, text):
        pass

    def decode(self, tokens: list[int]):
        pass
    
    
class RegexTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        pass

    def encode(self, text):
        pass

    def decode(self, tokens: list[int]):
        pass


def main() -> None:
    pass

if __name__ ==  "__main__":
    main()