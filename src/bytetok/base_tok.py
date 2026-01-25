"""
Base tokenizer interface for byte-level tokenization implementations.
"""

from abc import ABC, abstractmethod
from _bpe import BytePair, Token
from _sanitise import render_bytes
from pathlib import Path
from typing import Final

VERSION: Final[str] = "bytetok v1"
MODEL_SUFFIX: Final[str] = ".model"
VOCAB_SUFFIX: Final[str] = ".vocab"


class Tokenizer(ABC):
    """
    Abstract base class for byte-level tokenizers.

    Manages vocabulary, byte pair merges, and provides serialization methods.
    """

    def __init__(self) -> None:
        """Initialize tokenizer with base 256 vocabulary."""
        super().__init__()
        # byte pair -> merge token
        self.enc_merges: dict[BytePair, Token] = {}
        # tokens -> bytes
        self.dec_vocab = self._build_vocab()
        # regex pattern for splitting train data
        self.pattern: str = ""

    @abstractmethod
    def train(self, text: list[int], vocab_size: int, verbose=False):
        """Train tokenizer on byte sequence to learn merges up to target vocab size."""
        ...

    @abstractmethod
    def encode(self, text: str) -> list[Token]:
        """Encode text into a sequence of tokens."""
        ...

    @abstractmethod
    def decode(self, tokens: list[Token]) -> str:
        """Decode a sequence of tokens back into text."""
        ...

    def save(self, file_prefix: str, reg_pat: str = "") -> None:
        """
        Save tokenizer state to disk.

        Creates two files: a .model file with merge mappings and a .vocab file
        with human-readable token representations.

        Args:
            file_prefix: Path prefix for output files.
            reg_pat: Optional regex pattern for text splitting.
        """
        # write merges to machine-readable file: used to load()
        model_path = Path(file_prefix).with_suffix(MODEL_SUFFIX)
        # create directory if does not exist
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with model_path.open("w", newline="\n") as f:
            # header: version and regex pattern if exists
            f.write(f"{VERSION}\n")
            if reg_pat:
                f.write(f"re {reg_pat}\n")
            # re header for consistency
            else:
                f.write("re \n")
            # body: mapping for all merged tokens
            for pair, mtok in self.enc_merges.items():
                f.write(f"{pair[0]} {pair[1]} {mtok}\n")

        # write token -> text vocabulary for human readability
        vocab_path = Path(file_prefix).with_suffix(VOCAB_SUFFIX)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        inverted_merges = {mtok: pair for pair, mtok in self.enc_merges.items()}

        with vocab_path.open("w", encoding="utf-8", newline="\n") as f:
            for tok, b in self.dec_vocab.items():
                subword = render_bytes(b)
                # token arises from merging: show derivation from child tokens
                if tok in inverted_merges:
                    # extract child tokens and convert to bytes
                    ctok0, ctok1 = inverted_merges[tok]
                    raw_subword0, raw_subword1 = (
                        self.dec_vocab[ctok0],
                        self.dec_vocab[ctok1],
                    )
                    # raw bytes -> utf-8 while handling utf fragments and
                    # escaping control characters
                    subword0, subword1 = (
                        render_bytes(raw_subword0),
                        render_bytes(raw_subword1),
                    )
                    f.write(f"[{tok}] [{subword0}][{subword1}] -> {subword}\n")
                else:
                    # one of base 256 tokens: no merging
                    f.write(f"[{tok}] {subword}\n")

    def load(self, model_filename: str) -> None:
        """
        Load tokenizer state from a .model file.

        Restores merge mappings and rebuilds vocabulary.

        Args:
            model_filename: Path to the .model file.

        Raises:
            ValueError: If file extension is not .model or version mismatch occurs.
        """
        path = Path(model_filename)

        if path.suffix != MODEL_SUFFIX:
            raise ValueError("Model file must have a .model extension")

        merges = {}

        with path.open("r", encoding="utf-8") as f:
            # verify version match
            model_ver = f.readline().strip()
            if model_ver != VERSION:
                raise ValueError(
                    f"Version mismatch: expected {VERSION}, got {model_ver}"
                )

            # store split pattern if it exists
            regex = f.readline().strip()
            if regex.startswith("re ") and len(regex) > 3:
                self.pattern = regex[3:]

            # read and load merges
            for line in f:
                # tokens are stored as strings in file
                ctok0, ctok1, mtok = map(int, line.split())
                merges[(ctok0, ctok1)] = mtok

        # atomically update tokenizer state after successful read
        self.enc_merges = merges
        self.dec_vocab = self._build_vocab()

    def _build_vocab(self) -> dict[Token, bytes]:
        """
        Build token-to-bytes vocabulary mapping.

        Creates base 256 byte tokens and expands with merged tokens in order.
        """
        # mapping for base 256 tokens
        vocab = {btok: bytes([btok]) for btok in range(256)}
        # mapping for new merged tokens
        for (tok0, tok1), mtok in self.enc_merges.items():
            vocab[mtok] = vocab[tok0] + vocab[tok1]

        return vocab
