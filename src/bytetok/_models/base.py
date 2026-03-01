"""
Base tokenizer interface for byte-level tokenization implementations.
"""

import logging
from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, TYPE_CHECKING

from .._sanitise import _render_bytes
from ..types import Encoding, Token, Vocabulary
from ..errors import ModelLoadError, TrainingError, VocabularyError, SpecialTokenError

if TYPE_CHECKING:
    from ..strategy import SpecialTokenStrategy
    from ..bpe import RustBPETokenizer


PREFIX: Final[str] = "ByteTok"
try:
    _version = version("bytetok")
except PackageNotFoundError:
    _version = "dev"


VERSION: Final[str] = _version
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
        # cached Rust tokenizer for full encode/decode pipeline
        self._tokenizer: "RustBPETokenizer | None" = None

    @abstractmethod
    def train(
        self,
        text: str | list[str],
        vocab_size: int,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> None:
        """Train tokenizer on byte sequence to learn merges up to target vocab size."""
        ...

    def encode(
        self,
        text: str,
        strategy: "SpecialTokenStrategy | None" = None,
    ) -> list[Token]:
        """Encode text into a sequence of tokens."""
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before encoding"
            )
        return self._encode_impl(text, strategy)

    @abstractmethod
    def _encode_impl(
        self,
        text: str,
        strategy: "SpecialTokenStrategy | None" = None,
    ) -> list[Token]:
        """Subclass-specific single-text encoding logic."""
        ...

    def decode(self, tokens: list[Token], errors: str | None = None) -> str:
        """
        Decode a sequence of tokens back into text.

        :param errors: How to handle invalid UTF-8 — "strict" or "replace" (default: "replace").
        :raises TrainingError: If the tokenizer has not been trained yet.
        :raises VocabularyError: If any token ID is not in the vocabulary.
        """
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before decoding"
            )
        tokenizer = self._get_rust_tokenizer(pattern=self._get_pattern())
        try:
            return tokenizer.decode_tokens(tokens, errors=errors)
        except ValueError as e:
            raise VocabularyError("failed to decode") from e

    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.vocab)

    def save(self, file_prefix: str) -> None:
        """
        Save tokenizer state to disk.

        Creates two files: a .model file with merge mappings and a .vocab file
        with human-readable token representations.

        :param file_prefix: Path prefix for output files.
        :raises TrainingError: If the tokenizer has not been trained yet.
        """
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before saving"
            )
        log.info(f"saving tokenizer to {file_prefix}")
        self._save_model(file_prefix)
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

            log.debug(f"loading {n_special_tokens} special tokens")

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
            log.debug("loading merge tokens")
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

        # Atomically update tokenizer state after successful read.
        self.special_toks = special_toks
        self.merges = merges
        self.vocab = self._build_vocab()
        self._tokenizer = None

        log.info(
            f"model loaded successfully: {len(self.special_toks)} special tokens, {len(self.merges)} merge rules, {len(self.vocab)} total tokens"
        )

    @abstractmethod
    def _encode_batch_impl(
        self,
        texts: list[str],
        strategy: "SpecialTokenStrategy | None" = None,
        show_progress: bool = True,
    ) -> list[list[Token]]:
        """Subclass-specific batch encoding logic."""
        ...

    def encode_batch(
        self,
        texts: list[str],
        strategy: "SpecialTokenStrategy|None" = None,
        show_progress: bool = True,
    ) -> list[list[Token]]:
        """Encode multiple texts into sequences of tokens in batch."""
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before encoding"
            )

        if not texts:
            return []

        return self._encode_batch_impl(
            texts,
            strategy,
            show_progress=show_progress,
        )

    def _get_pattern(self) -> str:
        """Return the regex pattern for encoding/decoding; default is r'.+' (match all)."""
        return self.pat or r".+"

    def decode_batch(
        self,
        token_batch: list[list[Token]],
        errors: str | None = None,
        show_progress: bool = True,
    ) -> list[str]:
        """
        Decode multiple token sequences in batch via Rust/Rayon.

        :param errors: How to handle invalid UTF-8 — "strict" or "replace" (default: "replace").
        :raises TrainingError: If the tokenizer has not been trained yet.
        :raises VocabularyError: If any token ID is not in the vocabulary.
        """
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before decoding"
            )

        if not token_batch:
            return []

        tokenizer = self._get_rust_tokenizer(pattern=self._get_pattern())
        try:
            return tokenizer.decode_tokens_batch(
                token_batch,
                errors=errors,
                show_progress=show_progress,
            )
        except ValueError as e:
            raise VocabularyError("failed to decode") from e

    def _build_vocab(self) -> Vocabulary:
        """
        Build token-to-bytes vocabulary mapping.

        Adds base 256 byte tokens, special tokens as UTF-8 bytes, then merged
        tokens in merge order so child tokens exist before their parent.
        """
        # mapping for base 256 tokens
        vocab = {btok: bytes([btok]) for btok in range(256)}
        # mapping for special tokens: encode as UTF-8 bytes
        for seq, tok in self.special_toks.items():
            vocab[tok] = seq.encode("utf-8")
        # mapping for new merged tokens — must be built in merge order so that
        # child tokens (tok0, tok1) are always present before their parent (mtok).
        for (tok0, tok1), mtok in sorted(self.merges.items(), key=lambda x: x[1]):
            vocab[mtok] = vocab[tok0] + vocab[tok1]

        log.debug(f"built vocabulary with {len(vocab)} tokens")
        return vocab

    def _save_model(self, file_prefix: str) -> None:
        """Persist merge mappings and special tokens to a .model file."""

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
            if self.pat:
                f.write(f"re {self.pat}\n")
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
        """Persist human-readable token representations to a .vocab file."""
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
                subword = _render_bytes(b)
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
                        _render_bytes(raw_subword0),
                        _render_bytes(raw_subword1),
                    )
                    f.write(f"[{tok}] [{subword0}][{subword1}] -> {subword}\n")
                else:
                    # one of base 256 tokens: no merging
                    f.write(f"[{tok}] {subword}\n")

    def set_special_tokens(self, special_toks: dict[str, Token]) -> None:
        """
        Replace the full set of special tokens with user-assigned IDs.

        Replaces any previously registered special tokens. Must be called
        after training. To extend existing tokens pass the merged dict:
        ``tok.set_special_tokens({**tok.special_toks, "<|new|>": 300})``.

        :param special_toks: Dictionary mapping token strings to integer IDs.
        :raises TrainingError: If called before training.
        :raises SpecialTokenError: If any two entries share the same ID.
        :raises VocabularyError: If any ID collides with the BPE vocabulary.
        """
        if not self.merges:
            raise TrainingError(
                f"{self.__class__.__name__} must be trained before setting special tokens"
            )

        # IDs must be unique within the incoming dict.
        ids = list(special_toks.values())
        if len(ids) != len(set(ids)):
            duplicates = {
                seq for seq, tok in special_toks.items() if ids.count(tok) > 1
            }
            raise SpecialTokenError("duplicate token ids", found_tokens=duplicates)

        # IDs must not collide with the BPE vocab (base 256 + merges).
        bpe_vocab_size = 256 + len(self.merges)
        for _, tok in special_toks.items():
            if tok < bpe_vocab_size:
                raise VocabularyError(
                    "special token id overlaps with vocabulary", invalid_tok=tok
                )

        self.special_toks = dict(special_toks)
        self.vocab = self._build_vocab()
        # Invalidate so the next encode/decode call rebuilds with new special tokens.
        self._tokenizer = None

    def _get_merge_history(self) -> list[tuple[tuple[int, int], int]]:
        """Return merge history sorted by merge token id (child pairs before parents)."""
        return sorted(self.merges.items(), key=lambda x: x[1])

    def _get_rust_tokenizer(self, pattern: str | None = None) -> "RustBPETokenizer":
        """Build or return the cached Rust tokenizer for encode/decode."""
        if self._tokenizer is None:
            from bytetok.bpe import RustBPETokenizer

            # fallback chain: use provided pattern, then self.pat, then r".+" (match entire input as one chunk)
            effective_pattern = pattern if pattern is not None else (self.pat or r".+")
            self._tokenizer = RustBPETokenizer(
                self._get_merge_history(),
                effective_pattern,
                self.special_toks,
            )
        return self._tokenizer
