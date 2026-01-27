"""Basic byte-level tokenizer implementation."""

from typing import Counter, override, TYPE_CHECKING
from datasets import load_dataset
from .._bpe import Token, bpe_merge_with_freq_update
from .base import Tokenizer
import logging

if TYPE_CHECKING:
    from ..strategy import SpecialTokenStrategy

log = logging.getLogger(__name__)


class BasicTokenizer(Tokenizer):
    """
    Tokenizer that operates directly on byte sequences without regex splitting
    """

    TOKENIZER_TYPE = "basic"

    def __init__(self) -> None:
        super().__init__()

    @override
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
        """Train tokenizer by learning byte pair merges from the input sequence."""
        if vocab_size <= 256:
            raise ValueError("Vocab size must be greater than 256")

        # handle list input and convert text to bytes
        if isinstance(text, list):
            text = "".join(text)

        txt_bytes = list(text.encode("utf-8"))

        # merges beyond base byte vocabulary
        n_merges = vocab_size - 256
        merges = {}
        vocab = {tok: bytes([tok]) for tok in range(256)}

        # compute initial byte-pair frequencies once
        bp_freqs: Counter = Counter()
        bp_freqs.update(zip(txt_bytes, txt_bytes[1:]))

        # BPE algorithm with incremental frequency updates
        for i in range(n_merges):
            new_token = 256 + i
            # check if any valid pairs remain
            # 1. text compressed to single token
            # 2. very short input text such that enough pairs cannot form
            # before vocab size met
            if not bp_freqs:
                log.warning(
                    f"no more byte pairs to merge after {i} merges "
                    f"(requested {n_merges}). stopping early."
                )
                break
            # find most common token pair.
            rank0 = bp_freqs.most_common(1)[0][0]
            # merge pair with new token and update frequencies incrementally
            txt_bytes = bpe_merge_with_freq_update(
                txt_bytes, rank0, new_token, bp_freqs
            )
            # save merge info and update vocabulary with new token's mapping
            merges[rank0] = new_token
            vocab[new_token] = vocab[rank0[0]] + vocab[rank0[1]]
            # debugging: log new merge info.
            if verbose:
                log.info(f"Merge {i + 1}/{n_merges}: {rank0} -> {new_token}")

        self.merges = merges  # used for encoding text -> tokens
        self.vocab = vocab  # used for decoding tokens -> text

    @override
    def encode(
        self, text: str, strategy: "SpecialTokenStrategy | None" = None
    ) -> list[Token]:
        """Encode text into a sequence of tokens."""
        # BasicTokenizer does not support special token strategies
        _ = strategy
        # encode Unicode text into bytes
        txt_bytes = text.encode("utf-8", errors="replace")
        # convert each byte to [0-255] token range
        tokens = list(txt_bytes)
        # return bpe of tokens
        return self._apply_bpe_chunk(tokens)

    @override
    def decode(self, tokens: list[Token]) -> str:
        """Decode a sequence of tokens back into text."""
        # token stream -> byte stream
        txt_bytes = b"".join(self.vocab[tok] for tok in tokens)
        # byte stream -> python string
        return txt_bytes.decode("utf-8", errors="replace")


def main() -> None:
    """Load and preprocess sci-fi books dataset for tokenizer training."""
    # preprocessing
    ds = load_dataset("stevez80/Sci-Fi-Books-gutenberg", split="train")
    text = "".join(ds[:100]["text"])
    vocab_size = 280

    # train tokenizer on dataset
    btok = BasicTokenizer()
    btok.train(text, vocab_size, verbose=True)

    # save vocabulary
    btok.save("token-map")

    # test
    # pay attention to how the emojis are rendered
    # emojis are 4 byte representations
    # regex pattern required pre-tokenization to ensure
    # multi bytes are not split
    tc = """
    Café naïve résumé coöperate — ﬁancée; São Paulo vs. München.
    Ελληνικά: αλφάβητο, μαθηματικά ∑∫√ ≈ ≠ ≤ ≥.
    Русский текст: съешь ещё этих мягких французских булок.
    العربية: اللغة العربية جميلةٌ، والتشكيلُ مهمٌّ.
    עברית: שלום עולם.
    हिन्दी: यह एक परीक्षण वाक्य है।
    বাংলা: এটি একটি পরীক্ষা বাক্য।
    한국어: 이것은 테스트 문장입니다.
    日本語: 日本語の文章です。かなカナ漢字。
    中文: 简体中文和繁體中文測試。
    Emoji: 😀😃😄👩🏽‍💻👨‍👩‍👧‍👦🇦🇺🇧🇩❤️‍🔥
    Combining: á é ï ō ů (NFD-like) vs á é ï ō ů (NFC).
    Zero-width: A​B​C (ZWSP), word⁠joiner⁠test.
    Spaces: space NBSP EM EN THIN HAIR IDEOGRAPHIC　END
    """
    tokens = btok.encode(tc)
    print(f"Total characters: {len(tc)}")
    print(f"Total tokens: {len(tokens)}")
    print(f"Tokens: {tokens}")

    dec_txt = btok.decode(tokens)
    print(f"Decoded text: {dec_txt}")


if __name__ == "__main__":
    main()
