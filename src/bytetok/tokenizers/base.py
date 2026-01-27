"""
Base tokenizer interface for byte-level tokenization implementations.
"""

from abc import ABC, abstractmethod
from .._bpe import BytePair, Encoding, Token, Vocabulary, bpe_merge
from .._sanitise import render_bytes
from pathlib import Path
from typing import Final, TYPE_CHECKING
from ..exceptions import ModelLoadError
import logging

if TYPE_CHECKING:
    from ..strategy import SpecialTokenStrategy

PREFIX: Final[str] = "ByteTok"
VERSION: Final[str] = "0.1.0"
MODEL_SUFFIX: Final[str] = ".model"
VOCAB_SUFFIX: Final[str] = ".vocab"

log = logging.getLogger(__name__)


class Tokenizer(ABC):
    """
    Abstract base class for byte-level tokenizers.

    Manages vocabulary, byte pair merges, and provides serialization methods.
    """

    TOKENIZER_TYPE: str = "base"

    def __init__(self) -> None:
        """Initialize tokenizer with base 256 vocabulary."""
        super().__init__()
        # byte pair -> merge token
        self.merges: Encoding = {}
        # regex pattern for splitting train data
        self.pat: str = ""
        self.special_toks: dict[str, Token] = {}
        # tokens -> bytes
        self.vocab: Vocabulary = self._build_vocab()

    @abstractmethod
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
        """Train tokenizer on byte sequence to learn merges up to target vocab size."""
        ...

    @abstractmethod
    def encode(
        self, text: str, strategy: "SpecialTokenStrategy | None" = None
    ) -> list[Token]:
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
        log.info(f"saving tokenizer to {file_prefix}")

        # write merge pair -> merge token mappings for loading model in future
        self._save_model(file_prefix, reg_pat)

        # write token -> text vocabulary for human readability
        self._save_vocab(file_prefix)

        log.info("tokenizer saved successfully")

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

        log.info(f"loading model from {path}")

        merges: Encoding = {}
        special_toks: dict[str, Token] = {}

        with path.open("r", encoding="utf-8") as f:
            # verify version match
            model_ver = f.readline().strip().split(" ")[1]
            if model_ver != VERSION:
                raise ModelLoadError(
                    "model version mismatch",
                    version_mismatch=(model_ver, [VERSION]),
                )
            # read tokenizer type
            tok_type = f.readline().strip()
            if tok_type.startswith("type "):
                tok_type = tok_type[5:]
                # validate that loaded type matches current instance type
                if tok_type != self.TOKENIZER_TYPE:
                    raise ModelLoadError(
                        "tokenizer type mismatch",
                        type_mismatch=(tok_type, [self.TOKENIZER_TYPE]),
                    )

            # store split pattern if it exists
            model_re = f.readline().strip()
            if model_re.startswith("re ") and len(model_re) > 3:
                self.pat = model_re[3:]

            # read and load special tokens
            start_marker = f.readline().strip()
            if start_marker != "---":
                raise ModelLoadError(
                    f"start sequence marker missing: (expected ---) (got {start_marker})"
                )

            # parse special token count
            n_special_tokens = f.readline().strip()
            try:
                n_special_tokens = int(n_special_tokens)
                if n_special_tokens < 0:
                    raise ValueError()
            except ValueError:
                raise ModelLoadError(f"invalid special token count: {n_special_tokens}")

            log.debug(f"loading {n_special_tokens} special tokens...")

            count = 0
            while count < n_special_tokens:
                # split from the right as the token sequence might contain whitespace
                # we dont want the string to be split inside the token sequence
                sp_tok: list[str] = f.readline().strip().rsplit(maxsplit=1)
                if len(sp_tok) != 2:
                    raise ModelLoadError(
                        f"special token mapping must be delimited by a whitespace: {sp_tok}"
                    )
                try:
                    # load special token data into tokenizer
                    special_toks[sp_tok[0]] = int(sp_tok[1])
                except ValueError:
                    raise ModelLoadError(f"token is not a number: {sp_tok[1]}")

                count += 1

            end_marker = f.readline().strip()
            if end_marker != "---":
                raise ModelLoadError(
                    f"end sequence marker missing: (expected ---) (got {end_marker})"
                )

            log.debug(f"{n_special_tokens} special tokens loaded")

            # read and load merges
            log.debug("loading merge tokens...")
            for line in f:
                try:
                    # tokens are stored as strings in file
                    ctok0, ctok1, mtok = map(int, line.split())
                    merges[(ctok0, ctok1)] = mtok
                except ValueError:
                    raise ModelLoadError(
                        f"invalid merge format at line: {line.strip()}"
                    )

            log.debug(f"loaded {len(merges)} merge rules")

        # atomically update tokenizer state after successful read
        self.special_toks = special_toks
        self.merges = merges
        self.vocab = self._build_vocab()

        log.info(
            f"model loaded successfully: {len(self.special_toks)} special tokens, {len(self.merges)} merge rules, {len(self.vocab)} total tokens"
        )

    def _build_vocab(self) -> Vocabulary:
        """
        Build token-to-bytes vocabulary mapping.

        Creates base 256 byte tokens, adds special tokens as UTF-8 bytes,
        and expands with merged tokens in order.
        """
        # mapping for base 256 tokens
        vocab = {btok: bytes([btok]) for btok in range(256)}
        # mapping for special tokens: encode as UTF-8 bytes
        for seq, tok in self.special_toks.items():
            vocab[tok] = seq.encode("utf-8")
        # mapping for new merged tokens
        for (tok0, tok1), mtok in self.merges.items():
            vocab[mtok] = vocab[tok0] + vocab[tok1]

        log.debug(f"built vocabulary with {len(vocab)} tokens")
        return vocab

    def _save_model(self, file_prefix: str, reg_pat: str = "") -> None:
        """Save merge mappings and special tokens to .model file."""

        model_path = Path(file_prefix).with_suffix(MODEL_SUFFIX)
        # create directory if does not exist
        model_path.parent.mkdir(parents=True, exist_ok=True)

        log.debug(f"saving model to {model_path}")
        log.debug(
            f"saving {len(self.special_toks)} special tokens and {len(self.merges)} merge rules"
        )

        with model_path.open("w", newline="\n") as f:
            # header: version, tokenizer type, regex pattern if exists
            f.write(f"{PREFIX} {VERSION}\n")
            f.write(f"type {self.TOKENIZER_TYPE}\n")
            if reg_pat:
                f.write(f"re {reg_pat}\n")
            else:
                f.write("re \n")
            # start of special tokens marker
            f.write("---\n")
            # special token metadata: total number
            f.write(f"{len(self.special_toks)}\n")
            # body 1: mapping for all special tokens
            for seq, tok in self.special_toks.items():
                f.write(f"{seq} {tok}\n")
            # end of special tokens marker
            f.write("---\n")
            # body 2: mapping for all merged tokens
            for pair, mtok in self.merges.items():
                f.write(f"{pair[0]} {pair[1]} {mtok}\n")

    def _save_vocab(self, file_prefix: str) -> None:
        """Save human readable token representations to .vocab file."""
        vocab_path = Path(file_prefix).with_suffix(VOCAB_SUFFIX)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        log.debug(f"saving vocab to {vocab_path}")

        inverted_merges = {mtok: pair for pair, mtok in self.merges.items()}

        with vocab_path.open("w", encoding="utf-8", newline="\n") as f:
            # save special token sequence -> token
            for seq, tok in self.special_toks.items():
                f.write(f"ST [{tok}] {seq}\n")
            # save base tokens + merge toke pairs -> token ids
            for tok, b in self.vocab.items():
                subword = render_bytes(b)
                # token arises from merging: show derivation from child tokens
                if tok in inverted_merges:
                    # extract child tokens and convert to bytes
                    ctok0, ctok1 = inverted_merges[tok]
                    raw_subword0, raw_subword1 = (
                        self.vocab[ctok0],
                        self.vocab[ctok1],
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

    def _apply_bpe_chunk(self, tokens: list[Token]) -> list[Token]:
        """
        Apply BPE merges to a token sequence.

        :param tokens: List of tokens (initially bytes 0-255).
        :returns: Compressed token sequence after applying learned merges.
        """
        # loop text compression using BPE algorithm.
        while len(tokens) >= 2:
            # get all unique bigram pairs.
            # we dont need to count frequencies to find min token.
            # see: https://github.com/karpathy/minbpe/issues/87#issuecomment-2273349030
            bigrams = set(zip(tokens, tokens[1:]))
            # retrieve the byte pair with the lowest merge index.
            # because higher index tokens might depend on lower index merged tokens.
            # use dict.get() to avoid double lookup (membership check + retrieval).
            pair: BytePair = min(
                bigrams,
                key=lambda bp: self.merges.get(bp, float("inf")),
            )
            # no pair to merge.
            if pair not in self.merges:
                break
            # merge target pair.
            tokens = bpe_merge(tokens, pair, self.merges[pair])

        return tokens
