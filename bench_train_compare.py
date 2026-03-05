#!/usr/bin/env python3
"""Apples-to-apples BPE training benchmark: ByteTok vs Hugging Face tokenizers.

Usage examples:
  PYTHONPATH=src python bench_train_compare.py --corpus README.md --vocab-size 10000 --runs 3
  PYTHONPATH=src python bench_train_compare.py --corpus README.md --repeat 200 --vocab-size 50000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def _read_corpus(path: Path, repeat: int, max_chars: int | None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if repeat > 1:
        text = text * repeat
    if max_chars is not None:
        text = text[:max_chars]
    return text


def _bytetok_tokens(corpus: str, mode: str) -> list[int]:
    if mode == "stream":
        return list(corpus.encode("utf-8", errors="replace"))

    import regex as re
    from bytetok.pattern import TokenPattern

    pat = TokenPattern.get("gpt4o")
    buf = bytearray()
    for m in re.finditer(pat, corpus):
        buf.extend(m.group(0).encode("utf-8", errors="replace"))
    return list(buf)


def _bench_bytetok(corpus: str, vocab_size: int, runs: int, mode: str) -> list[float]:
    from bytetok.bpe import RustBPETrainer

    tokens = _bytetok_tokens(corpus, mode)
    n_merges = vocab_size - 256
    if n_merges <= 0:
        raise ValueError("vocab_size must be > 256")

    times: list[float] = []
    for _ in range(runs):
        trainer = RustBPETrainer(tokens, 256)
        t0 = time.perf_counter()
        try:
            trainer.train(n_merges, False)
        except TypeError:
            trainer.train(n_merges)
        times.append(time.perf_counter() - t0)
    return times


def _bench_hf(corpus: str, vocab_size: int, runs: int) -> list[float]:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    times: list[float] = []
    for _ in range(runs):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1, special_tokens=[])
        t0 = time.perf_counter()
        tok.train_from_iterator([corpus], trainer=trainer)
        times.append(time.perf_counter() - t0)
    return times


def _fmt_stats(name: str, vals: list[float]) -> tuple[str, float]:
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{name:10} | {mean:9.3f}s | {sd:7.3f}s | {min(vals):8.3f}s | {max(vals):8.3f}s", mean


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark BPE training speed (ByteTok vs HF tokenizers).")
    ap.add_argument("--corpus", type=Path, required=True, help="Path to corpus text file.")
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1, help="Repeat corpus text N times to scale size.")
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument(
        "--bytetok-mode",
        choices=["regex", "stream"],
        default="regex",
        help="ByteTok preprocessing mode before training.",
    )
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"error: corpus file not found: {args.corpus}", file=sys.stderr)
        return 2

    if args.vocab_size <= 256:
        print("error: vocab-size must be > 256", file=sys.stderr)
        return 2

    corpus = _read_corpus(args.corpus, args.repeat, args.max_chars)
    size_mb = len(corpus.encode("utf-8", errors="replace")) / (1024 * 1024)

    print(f"Corpus: {args.corpus}")
    print(f"Size: {size_mb:.2f} MB | vocab_size={args.vocab_size:,} | runs={args.runs}")
    print()

    try:
        bt_times = _bench_bytetok(corpus, args.vocab_size, args.runs, args.bytetok_mode)
    except Exception as e:
        print(f"ByteTok benchmark failed: {e}", file=sys.stderr)
        return 1

    hf_times: list[float] | None = None
    hf_err: str | None = None
    try:
        hf_times = _bench_hf(corpus, args.vocab_size, args.runs)
    except Exception as e:
        hf_err = str(e)

    print("engine      | mean      | stddev  | min      | max")
    print("-" * 58)
    bt_line, bt_mean = _fmt_stats("ByteTok", bt_times)
    print(bt_line)

    if hf_times is not None:
        hf_line, hf_mean = _fmt_stats("HF", hf_times)
        print(hf_line)
        if hf_mean > 0:
            print()
            print(f"Speed ratio (HF/ByteTok): {hf_mean / bt_mean:.2f}x")
            print(f"Speed ratio (ByteTok/HF): {bt_mean / hf_mean:.2f}x")
    else:
        print("HF         | unavailable (install `tokenizers`)" )
        if hf_err:
            print(f"HF error: {hf_err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
