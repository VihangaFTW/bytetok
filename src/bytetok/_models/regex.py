"""Regex-based byte-level tokenizer implementation."""

from concurrent.futures import ThreadPoolExecutor
from typing import override

import os

from ..errors import (
    PatternError,
    SpecialTokenError,
    TokenizationError,
    TrainingError,
    VocabularyError,
)
from ..pattern import TokenPattern
from ..strategy import SpecialTokenStrategy

from .base import Tokenizer, ParallelMode
from .._decorators import measure_time
from .._bpe import (
    Encoding,
    Token,
    Vocabulary,
)
from ..bpe import RustBPETrainer

import regex as re
import logging


log = logging.getLogger(__name__)


class RegexTokenizer(Tokenizer):
    """Tokenizer that splits text using regex patterns before applying BPE."""

    TOKENIZER_TYPE = "regex"

    def __init__(self, pattern: str | None = None) -> None:
        super().__init__()
        if pattern is None:
            self.pat = TokenPattern.get("gpt4o")
        else:
            self.pat = pattern
        self.compiled_pat: re.Pattern[str] = _compile_pattern(self.pat)
        self.special_toks: dict[str, Token] = {}
        self.inverted_special_tokens: dict[Token, str] = {}

    @override
    @measure_time
    def train(
        self, text: str | list[str], vocab_size: int, verbose: bool = False
    ) -> None:
        """Train tokenizer by learning byte pair merges on regex-split chunks."""
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

        self._train_rust(tokens, n_merges, verbose)

    @override
    def encode(
        self,
        text: str,
        strategy: SpecialTokenStrategy | None = None,
        num_workers: int | None = None,
    ) -> list[Token]:
        """
        Encode text into a sequence of tokens.

        :param text: Text to encode.
        :param strategy: Strategy to handle special tokens in `text`.
        :param num_workers: Thread pool size for parallel encoding.
        """

        # no strategy is inferred as no special token handling
        if strategy is None:
            # split text into chunks as defined by pattern
            chunks = [m.group(0) for m in re.finditer(self.compiled_pat, text)]

            # turn chunks into base token representations
            byte_chunks = self._to_base_tokens(chunks)

            tokens: list[Token] = []

            encoded_chunks = self._apply_fast_bpe_chunks(
                byte_chunks, num_workers=num_workers
            )

            for chunk_toks in encoded_chunks:
                tokens.extend(chunk_toks)

            return tokens

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
        # delegate chunks among worker threads
        if num_workers is None:
            workers = os.cpu_count() or 1
        else:
            workers = max(1, num_workers)  # "0" interpreted as 1 worker

        def process_batch():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(
                    pool.map(
                        lambda text: self.encode(
                            text, strategy=strategy, num_workers=1
                        ),
                        texts,
                    )
                )

        match parallel_mode:
            case ParallelMode.OFF:
                return [self.encode(text, strategy, 1) for text in texts]
            case ParallelMode.CHUNK:
                return [self.encode(text, strategy, workers) for text in texts]
            case ParallelMode.BATCH:
                return process_batch()
            case ParallelMode.AUTO:
                # single text sequence parallelized on chunk level
                if len(texts) == 1:
                    return [
                        self.encode(texts[0], strategy=strategy, num_workers=workers)
                    ]
                # multiple texts processed as parallel batches
                return process_batch()

    @override
    def decode(self, tokens: list[Token]) -> str:
        """Decode a sequence of tokens back into text."""
        txt_bytes = []
        for tok in tokens:
            if tok in self.vocab:
                txt_bytes.append(self.vocab[tok])
            elif tok in self.inverted_special_tokens:
                txt_bytes.append(
                    self.inverted_special_tokens[tok].encode("utf-8", errors="replace")
                )
            else:
                raise VocabularyError("token not found in vocabulary", invalid_tok=tok)

        text = b"".join(txt_bytes).decode("utf-8", errors="replace")
        return text

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

    def _train_rust(self, tokens: list[int], n_merges: int, verbose: bool) -> None:
        """Train using fast Rust implementation."""

        if len(tokens) == 0:
            raise TrainingError("empty token sequence, no training performed")

        trainer = RustBPETrainer(tokens, 256)

        # train for n_merges
        trainer.train(n_merges)

        # get merge history
        merge_history = trainer.get_merge_history()

        # build merges dictionary and vocabulary
        merges = {}
        vocab = {tok: bytes([tok]) for tok in range(256)}

        for (tok_a, tok_b), new_tok in merge_history:
            pair = (tok_a, tok_b)
            merges[pair] = new_tok
            vocab[new_tok] = vocab[tok_a] + vocab[tok_b]

            if verbose:
                log.info(f"merge {len(merges)}/{n_merges}: {pair} -> {new_tok}")

        if len(merge_history) < n_merges:
            log.warning(
                f"no more byte pairs to merge after {len(merge_history)} merges "
                f"(requested {n_merges}) stopping early"
            )

        self.merges: Encoding = merges
        self.vocab: Vocabulary = vocab
        # invalidate encoder cache since merges changed
        self._encoder = None


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
