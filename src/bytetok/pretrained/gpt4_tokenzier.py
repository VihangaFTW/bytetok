from bytetok.src.bytetok.regex_tok import RegexTokenizer
from factory import get_pattern
import tiktoken


class GPT4Tokenizer(RegexTokenizer):
    """ """

    def __init__(self) -> None:
        super().__init__(get_pattern("gpt4"))
        # get official gpt4 merges
        enc = tiktoken.get_encoding("cl100k_base")
        # todo: complete class
        pass