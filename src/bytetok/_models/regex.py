"""Regex-based byte-level tokenizer implementation."""

from typing import override

from ..errors import (
    PatternError,
    SpecialTokenError,
    TokenizationError,
    VocabularyError,
)
from ..pattern import TokenPattern
from ..strategy import SpecialTokenStrategy

from .base import Tokenizer, ParallelMode
from .._decorators import measure_time
from ..types import (
    Token,
)
from ..trainer import train_bpe

import regex as re
import logging


log = logging.getLogger(__name__)


class RegexTokenizer(Tokenizer):
    """Tokenizer that splits text using regex patterns before applying BPE."""

    TOKENIZER_TYPE = "regex"

    def __init__(self, pattern: str | None = None) -> None:
        """Initialize tokenizer with a provided or default split pattern."""
        super().__init__()
        if pattern is None:
            self.pat = TokenPattern.get("gpt4o")
        else:
            self.pat = pattern
        self.compiled_pat: re.Pattern[str] = _compile_pattern(self.pat)
        self.special_toks: dict[str, Token] = {}
        self.inverted_special_tokens: dict[Token, str] = {}

    @override
    def load(self, model_filename: str) -> None:
        """Load tokenizer state and refresh regex/special-token caches."""
        super().load(model_filename)
        self.compiled_pat = _compile_pattern(self.pat)
        self.inverted_special_tokens = {
            token: seq for seq, token in self.special_toks.items()
        }

    @override
    @measure_time
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
        """
        Train the tokenizer on regex-split text chunks.

        Input text is first segmented by the configured regex pattern, then
        flattened into UTF-8 byte tokens and used for BPE merge learning.

        :param text: Training text as a single string or list of strings.
        :param vocab_size: Target vocabulary size including the base 256 bytes.
        :param verbose: Log each learned merge when ``True``.
        :raises VocabularyError: If ``vocab_size`` is less than or equal to 256.
        :raises TrainingError: If no trainable byte tokens are produced.
        """
        if vocab_size <= 256:
            raise VocabularyError(
                "vocab size must be greater than 256", vocab_size=vocab_size
            )

        # handle list input and convert text to bytes
        if isinstance(text, list):
            text = "".join(text)

        # split text into chunks defined by pattern
        chunks = [m.group(0) for m in re.finditer(self.compiled_pat, text)]
        # convert each chunk to byte sequence and flatten into single sequence
        tokens: list[int] = []
        for chunk in chunks:
            tokens.extend(list(chunk.encode("utf-8", errors="replace")))

        n_merges = vocab_size - 256

        result = train_bpe(tokens, n_merges, verbose=verbose)

        if result.n_merges_completed < n_merges:
            log.warning(
                f"no more byte pairs to merge after {result.n_merges_completed} merges "
                f"(requested {n_merges}) stopping early"
            )

        self.merges = result.merges
        self.vocab = result.vocab
        # Invalidate Rust caches since merges changed.
        self._converter = None
        self._tokenizer = None

    @override
    def encode(
        self,
        text: str,
        strategy: SpecialTokenStrategy | None = None,
        num_workers: int | None = None,
    ) -> list[Token]:
        """
        Encode text into a sequence of tokens.

        If ``strategy`` is ``None``, the tokenizer only applies regex chunking
        and BPE. When a strategy is provided, matched special tokens are kept as
        atomic tokens while non-special spans are encoded with BPE.

        :param text: Text to encode.
        :param strategy: Strategy used to select allowed special tokens.
        :param num_workers: Worker count for chunk-level BPE application.
        :returns: Encoded token sequence.
        :raises TokenizationError: If internal chunk assembly fails unexpectedly.
        """

        # no strategy is inferred as no special token handling
        if strategy is None:
            _ = num_workers
            tokenizer = self._get_rust_tokenizer(pattern=self.pat)
            try:
                return tokenizer.encode_text(text)
            except ValueError as e:
                raise TokenizationError("failed to encode text") from e

        # retrieve special tokens as defined by chosen strategy
        special_toks = strategy.handle(text, self.special_toks)
        # escape regex metachars like "+" in special tokens to avoid unwanted effects
        esc_special_toks = [re.escape(seq) for seq in special_toks]
        # The capturing group parentheses make re.split() include
        # the matched delimiters in result
        # text is split on special tokens while keeping them as separate chunks
        # otherwise we lose the special tokens after split (leads to lossy decoding)
        special_pat = "(" + "|".join(esc_special_toks) + ")"
        chunks = re.split(special_pat, text)

        # filter special chunks and normal chunks that require bpe
        # so that all the normal chunks can be processed in parallel

        # running accumulation of full encoding (normal + special chunks)
        # None entries will be replaced by bpe encodings later
        out_parts: list[list[Token] | None] = []

        # track normal chunks that required bpe encoding
        normal_chunks: list[str] = []
        # track indices in out_parts where normal chunks
        # should be inserted after encoding
        normal_positions: list[int] = []

        for chunk in chunks:
            if chunk in special_toks:
                # special tokens have pre-determined encodings
                out_parts.append([special_toks[chunk]])
            else:
                # normal bpe chunk requires bpe encoding later
                out_parts.append(None)
                normal_chunks.append(chunk)
                normal_positions.append(len(out_parts) - 1)

        # encode all normal chunks in parallel
        base_encodings = self._to_base_tokens(normal_chunks)
        encoded_normals = self._apply_fast_bpe_chunks(
            base_encodings, num_workers=num_workers
        )
        # insert encodings at correct positions in accumulator
        try:
            for pos, encoding in zip(normal_positions, encoded_normals, strict=True):
                out_parts[pos] = encoding
        except ValueError as e:
            raise TokenizationError("internal error, kindly report issue") from e

        tokens = []
        # flatten parts to get full text encoding
        for part in out_parts:
            if part is not None:
                tokens.extend(part)

        return tokens

    @override
    def encode_batch(
        self,
        texts: list[str],
        strategy: "SpecialTokenStrategy | None" = None,
        num_workers: int | None = None,
        parallel_mode: ParallelMode = ParallelMode.AUTO,
    ) -> list[list[Token]]:
        """
        Encode many texts using the requested parallelization mode.

        ``off`` encodes texts serially. ``chunk`` encodes each text using
        chunk-level parallelism. ``batch`` runs multiple full-text encodes in
        parallel. ``auto`` chooses chunk mode for a single input text and batch
        mode for multiple texts.

        :param texts: Text inputs to encode.
        :param strategy: Optional special token handling strategy.
        :param num_workers: Worker count for chunk or batch parallelism.
        :param parallel_mode: Parallelization policy.
        :returns: Encoded token sequences in input order.
        """
        # delegate chunks among worker threads
        if not texts:
            return []

        # Strategy mode is orchestrated in Python to preserve existing behavior.
        if strategy is not None:
            return [
                self.encode(text, strategy=strategy, num_workers=num_workers)
                for text in texts
            ]

        tokenizer = self._get_rust_tokenizer(pattern=self.pat)
        _ = num_workers

        match parallel_mode:
            case ParallelMode.OFF:
                return [tokenizer.encode_text(text) for text in texts]
            case ParallelMode.CHUNK:
                return [tokenizer.encode_text(text) for text in texts]
            case ParallelMode.BATCH:
                return tokenizer.encode_texts(texts)
            case ParallelMode.AUTO:
                if len(texts) == 1:
                    return [tokenizer.encode_text(texts[0])]
                return tokenizer.encode_texts(texts)

    @override
    def decode(self, tokens: list[Token]) -> str:
        """
        Decode tokens into text, including registered special tokens.

        Tokens are resolved first from the learned vocabulary and then from the
        inverted special-token mapping.

        :param tokens: Token sequence to decode.
        :returns: Decoded text where invalid UTF-8 is replaced.
        :raises VocabularyError: If any token is unknown to both mappings.
        """
        tokenizer = self._get_rust_tokenizer(pattern=self.pat)
        try:
            return tokenizer.decode_tokens(tokens, errors="replace")
        except ValueError as e:
            for tok in tokens:
                if tok not in self.vocab and tok not in self.inverted_special_tokens:
                    raise VocabularyError("token not found in vocabulary", invalid_tok=tok) from e
            raise VocabularyError("token not found in vocabulary") from e

    def register_special_tokens(self, special_toks: list[str]) -> None:
        """
        Register special tokens with auto-assigned IDs.

        Special token IDs are assigned sequentially starting from vocab_size.
        Must be called after training the tokenizer.

        :param special_toks: List of special token strings to register.
        :raises SpecialTokenError: If tokenizer hasn't been trained yet.
        """
        if not hasattr(self, "merges"):
            raise SpecialTokenError(
                f"{self.__class__.__name__} must be trained before registering special tokens"
            )
        # special tokens have auto assigned ids
        vocab_size = 256 + len(self.merges)

        # special tokens are appended at the end of trained vocab
        for idx, seq in enumerate(special_toks):
            self.special_toks[seq] = vocab_size + idx

        self.inverted_special_tokens = {
            token: seq for seq, token in self.special_toks.items()
        }

        # rebuild vocab to include special tokens for decoding
        for seq, tok in self.special_toks.items():
            self.vocab[tok] = seq.encode("utf-8")
        # Invalidate Rust tokenizer cache so new special tokens are visible.
        self._tokenizer = None

def _compile_pattern(pattern: str) -> re.Pattern:
    """
    Compile and validate a regex pattern.

    :param pattern: Regex pattern string to compile.
    :return: Compiled regex pattern.
    :raises ValueError: If pattern is invalid.
    """
    try:
        return re.compile(pattern)
    except re.error as e:
        raise PatternError("invalid regex pattern", pattern=pattern, regex_err=e)
