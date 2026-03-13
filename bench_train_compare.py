#!/usr/bin/env python3
"""ByteTok vs Hugging Face tokenizers.

Usage examples:
  PYTHONPATH=src python bench_train_compare.py --corpus README.md --vocab-size 10000 --runs 3
  PYTHONPATH=src python bench_train_compare.py --corpus README.md --repeat 200 --vocab-size 50000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import statistics
import sys
import time
from pathlib import Path


@dataclass
class BenchRun:
    train_s: float
    encode_s: float
    decode_s: float
    total_bytes: int
    total_tokens: int


def _read_corpus(path: Path, repeat: int, max_chars: int | None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if repeat > 1:
        text = text * repeat
    if max_chars is not None:
        text = text[:max_chars]
    return text


def _bytetok_pattern(mode: str) -> str | None:
    if mode == "stream":
        # Match the full corpus as one piece so Rust trains over the raw byte stream.
        return r"[\s\S]+"

    from bytetok.pattern import TokenPattern

    return TokenPattern.get("gpt4o")


def _make_chunks(corpus: str, chunk_chars: int) -> list[str]:
    return [corpus[i : i + chunk_chars] for i in range(0, len(corpus), chunk_chars)]


def _total_bytes(chunks: list[str]) -> int:
    return sum(len(chunk.encode("utf-8", errors="replace")) for chunk in chunks)


def _bench_bytetok(
    corpus: str,
    chunks: list[str],
    vocab_size: int,
    runs: int,
    mode: str,
) -> list[BenchRun]:
    from bytetok import RegexTokenizer

    pattern = _bytetok_pattern(mode)
    n_merges = vocab_size - 256
    if n_merges <= 0:
        raise ValueError("vocab_size must be > 256")
    total_bytes = _total_bytes(chunks)

    results: list[BenchRun] = []
    for _ in range(runs):
        tokenizer = RegexTokenizer(pattern=pattern)

        train_t0 = time.perf_counter()
        tokenizer.train(corpus, vocab_size=vocab_size, show_progress=False)
        train_s = time.perf_counter() - train_t0

        encode_t0 = time.perf_counter()
        encoded = tokenizer.encode_batch(chunks, show_progress=False)
        encode_s = time.perf_counter() - encode_t0

        decode_t0 = time.perf_counter()
        decoded = tokenizer.decode_batch(encoded, errors="replace", show_progress=False)
        decode_s = time.perf_counter() - decode_t0

        if decoded != chunks:
            raise RuntimeError("ByteTok decode output mismatch")

        results.append(
            BenchRun(
                train_s=train_s,
                encode_s=encode_s,
                decode_s=decode_s,
                total_bytes=total_bytes,
                total_tokens=sum(len(seq) for seq in encoded),
            )
        )
    return results


def _bench_hf(corpus: str, chunks: list[str], vocab_size: int, runs: int) -> list[BenchRun]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    total_bytes = _total_bytes(chunks)
    results: list[BenchRun] = []
    for _ in range(runs):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1, special_tokens=[])

        train_t0 = time.perf_counter()
        tok.train_from_iterator([corpus], trainer=trainer)
        train_s = time.perf_counter() - train_t0

        encode_t0 = time.perf_counter()
        encoded = tok.encode_batch(chunks)
        encode_s = time.perf_counter() - encode_t0
        token_ids = [item.ids for item in encoded]

        decode_t0 = time.perf_counter()
        decoded = tok.decode_batch(token_ids)
        decode_s = time.perf_counter() - decode_t0

        if decoded != chunks:
            raise RuntimeError("HF decode output mismatch")

        results.append(
            BenchRun(
                train_s=train_s,
                encode_s=encode_s,
                decode_s=decode_s,
                total_bytes=total_bytes,
                total_tokens=sum(len(ids) for ids in token_ids),
            )
        )
    return results


def _fmt_seconds_row(name: str, vals: list[float]) -> tuple[str, float]:
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{name:10} | {mean:9.3f}s | {sd:7.3f}s | {min(vals):8.3f}s | {max(vals):8.3f}s", mean


def _fmt_scalar_row(name: str, vals: list[float], unit: str) -> tuple[str, float]:
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{name:10} | {mean:9.2f} {unit:7} | {sd:7.2f} | {min(vals):8.2f} | {max(vals):8.2f}", mean


def _train_times(runs: list[BenchRun]) -> list[float]:
    return [run.train_s for run in runs]


def _encode_mbps(runs: list[BenchRun]) -> list[float]:
    return [run.total_bytes / run.encode_s / (1024 * 1024) for run in runs]


def _decode_mbps(runs: list[BenchRun]) -> list[float]:
    return [run.total_bytes / run.decode_s / (1024 * 1024) for run in runs]


def _decode_mtps(runs: list[BenchRun]) -> list[float]:
    return [run.total_tokens / run.decode_s / 1_000_000 for run in runs]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark BPE training, encoding, and decoding speed."
    )
    ap.add_argument("--corpus", type=Path, required=True, help="Path to corpus text file.")
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1, help="Repeat corpus text N times to scale size.")
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument(
        "--chunk-chars",
        type=int,
        default=8192,
        help="Chunk size in characters for encode and decode throughput benchmarks.",
    )
    ap.add_argument(
        "--bytetok-mode",
        choices=["regex", "stream"],
        default="regex",
        help="ByteTok Rust-side corpus splitting mode before training.",
    )
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"error: corpus file not found: {args.corpus}", file=sys.stderr)
        return 2

    if args.vocab_size <= 256:
        print("error: vocab-size must be > 256", file=sys.stderr)
        return 2
    if args.chunk_chars <= 0:
        print("error: chunk-chars must be > 0", file=sys.stderr)
        return 2

    corpus = _read_corpus(args.corpus, args.repeat, args.max_chars)
    size_mb = len(corpus.encode("utf-8", errors="replace")) / (1024 * 1024)
    chunks = _make_chunks(corpus, args.chunk_chars)

    print(f"Corpus: {args.corpus}")
    print(
        f"Size: {size_mb:.2f} MB | vocab_size={args.vocab_size:,} | "
        f"runs={args.runs} | chunks={len(chunks):,} | chunk_chars={args.chunk_chars:,}"
    )
    print()

    try:
        bt_runs = _bench_bytetok(corpus, chunks, args.vocab_size, args.runs, args.bytetok_mode)
    except Exception as e:
        print(f"ByteTok benchmark failed: {e}", file=sys.stderr)
        return 1

    hf_runs: list[BenchRun] | None = None
    hf_err: str | None = None
    try:
        hf_runs = _bench_hf(corpus, chunks, args.vocab_size, args.runs)
    except Exception as e:
        hf_err = str(e)

    print("Training")
    print("engine      | mean      | stddev  | min      | max")
    print("-" * 58)
    bt_train_line, bt_train_mean = _fmt_seconds_row("ByteTok", _train_times(bt_runs))
    print(bt_train_line)

    if hf_runs is not None:
        hf_train_line, hf_train_mean = _fmt_seconds_row("HF", _train_times(hf_runs))
        print(hf_train_line)
        if hf_train_mean > 0:
            print()
            print(f"Training ratio (HF/ByteTok): {hf_train_mean / bt_train_mean:.2f}x")
            print(f"Training ratio (ByteTok/HF): {bt_train_mean / hf_train_mean:.2f}x")
    else:
        print("HF         | unavailable (install `tokenizers`)" )
        if hf_err:
            print(f"HF error: {hf_err}")
        return 0

    print()
    print("Encoding Throughput")
    print("engine      | mean         | stddev  | min      | max")
    print("-" * 60)
    bt_encode_line, bt_encode_mean = _fmt_scalar_row("ByteTok", _encode_mbps(bt_runs), "MB/s")
    hf_encode_line, hf_encode_mean = _fmt_scalar_row("HF", _encode_mbps(hf_runs), "MB/s")
    print(bt_encode_line)
    print(hf_encode_line)

    print()
    print("Decoding Throughput")
    print("engine      | mean         | stddev  | min      | max")
    print("-" * 60)
    bt_decode_mb_line, bt_decode_mb_mean = _fmt_scalar_row("ByteTok", _decode_mbps(bt_runs), "MB/s")
    hf_decode_mb_line, hf_decode_mb_mean = _fmt_scalar_row("HF", _decode_mbps(hf_runs), "MB/s")
    print(bt_decode_mb_line)
    print(hf_decode_mb_line)

    print()
    print("Decoding Throughput (tokens)")
    print("engine      | mean         | stddev  | min      | max")
    print("-" * 60)
    bt_decode_tok_line, bt_decode_tok_mean = _fmt_scalar_row("ByteTok", _decode_mtps(bt_runs), "Mtoks/s")
    hf_decode_tok_line, hf_decode_tok_mean = _fmt_scalar_row("HF", _decode_mtps(hf_runs), "Mtoks/s")
    print(bt_decode_tok_line)
    print(hf_decode_tok_line)

    print()
    print(f"Encoding ratio (ByteTok/HF): {bt_encode_mean / hf_encode_mean:.2f}x")
    print(f"Decoding ratio MB/s (ByteTok/HF): {bt_decode_mb_mean / hf_decode_mb_mean:.2f}x")
    print(f"Decoding ratio tokens (ByteTok/HF): {bt_decode_tok_mean / hf_decode_tok_mean:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
