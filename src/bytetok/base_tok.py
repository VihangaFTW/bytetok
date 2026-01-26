"""
Base tokenizer interface for byte-level tokenization implementations.
"""

from abc import ABC, abstractmethod
from _bpe import BytePair, Token, update_bpe_freqs, bpe_merge
from _sanitise import render_bytes
from pathlib import Path
from typing import Final
from collections import Counter
from exceptions import ModelLoadError

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
        self.pat: str = ""
        self.special_toks: dict[str, Token] = {}

    @abstractmethod
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
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

        :param file_prefix: Path prefix for output files.
        :param reg_pat: Optional regex pattern for text splitting.
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

        :param model_filename: Path to the .model file.
        :raises ModelLoadError: If file does not exist, extension is not .model, or version mismatch occurs.
        """
        path = Path(model_filename)

        if not path.exists():
            raise ModelLoadError("model filepath does not exist", model_path=str(path))

        if not path.suffix == MODEL_SUFFIX:
            raise ModelLoadError("expected .model file", model_path=str(path))

        merges = {}

        with path.open("r", encoding="utf-8") as f:
            # verify version match
            model_ver = f.readline().strip()
            if model_ver != VERSION:
                raise ModelLoadError(
                    "model version mismatch",
                    version_mismatch=(model_ver, VERSION),
                )

            # store split pattern if it exists
            regex = f.readline().strip()
            if regex.startswith("re ") and len(regex) > 3:
                self.pat = regex[3:]

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

    def _apply_bpe_chunk(self, tokens: list[Token]) -> list[Token]:
        """
        Apply BPE merges to a token sequence.

        :param tokens: List of tokens (initially bytes 0-255).
        :returns: Compressed token sequence after applying learned merges.
        """
        # loop text compression using BPE algorithm
        while len(tokens) >= 2:
            bp_freqs = Counter()
            update_bpe_freqs(tokens, bp_freqs)
            # retrieve the byte pair with the lowest merge index
            # because higher index tokens might depend on lower index merged tokens
            pair: BytePair = min(
                bp_freqs,
                key=lambda bp: self.enc_merges[bp]
                if bp in self.enc_merges
                else float("inf"),
            )
            # no pair to merge
            if pair not in self.enc_merges:
                break
            # merge target pair
            tokens = bpe_merge(tokens, pair, self.enc_merges[pair])

        return tokens
