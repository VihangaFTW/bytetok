"""Basic byte-level tokenizer implementation."""

from datasets import load_dataset
from ._bpe import Token, bpe_freqs, bpe_merge
from .base_tok import Tokenizer
import logging

log = logging.getLogger(__name__)


class BasicTokenizer(Tokenizer):
    """Tokenizer that operates directly on byte sequences without regex splitting."""

    def __init__(self) -> None:
        super().__init__()

    def train(self, text: list[int], vocab_size: int, verbose=False):
        """Train tokenizer by learning byte pair merges from the input sequence."""
        if vocab_size <= 256:
            raise ValueError("Vocab size must be greater than 256")
        # merges beyond base byte vocabulary
        n_merges = vocab_size - 256
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        # BPE algorithm
        for i in range(n_merges):
            # find most common token pair
            pairs = bpe_freqs(text)
            rank0 = pairs.most_common(1)[0][0]
            new_token = 256 + i
            # merge pair with new token
            text = bpe_merge(text, rank0, new_token)
            # save merge info and update vocabulary with new token's mapping
            merges[rank0] = new_token
            vocab[new_token] = vocab[rank0[0]] + vocab[rank0[1]]
            # debugging: log new merge info
            if verbose:
                log.info(f"Merge {i + 1}/{n_merges}: {rank0} -> {new_token}")

            self.enc_merges = merges  # used for encoding text -> tokens
            self.dec_vocab = vocab  # usef for decoding tokens -> text

    def encode(self, text: str) -> list[Token]:
        """Encode text into a sequence of tokens."""
        # encode Unicode text into bytes
        txt_bytes = text.encode("utf-8", errors="replace")
        # convert each byte to [0-255] token range
        tokens = list(txt_bytes)
        # loop text compression using BPE algorithm
        while len(tokens) >= 2:
            bp_freqs = bpe_freqs(tokens)
            # retrieve the byte pair with the lowest merge index
            # because higher index tokens might depend on lower index merged tokens
            pair = min(bp_freqs, key=lambda bp: self.enc_merges.get(bp, float("inf")))
            # no merge mapping for current target bp
            if pair not in self.enc_merges:
                break
            # merge target pair
            tokens = bpe_merge(tokens, pair, self.enc_merges[pair])

        return tokens

    def decode(self, tokens: list[Token]) -> str:
        """Decode a sequence of tokens back into text."""
        # token stream -> byte stream
        txt_bytes = b"".join(self.dec_vocab[tok] for tok in tokens)
        # byte stream -> python string
        return txt_bytes.decode("utf-8", errors="replace")


def main() -> None:
    """Load and preprocess sci-fi books dataset for tokenizer training."""
    # preprocessing
    ds = load_dataset("stevez80/Sci-Fi-Books-gutenberg", split="train")
    tokens = list("".join(ds[:100]["text"]).encode("utf-8"))
    vocab_size = 280

    # train tokenizer on dataset
    btok = BasicTokenizer()
    btok.train(tokens, vocab_size, verbose=True)

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
