"""Focused benchmark for ByteTok parallel encoding speed."""

import argparse
import os
import time
from pathlib import Path

from bytetok import RegexTokenizer, from_pretrained
from bytetok.parallel import encode_batch


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable units."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def make_text(target_mb: int) -> str:
    """Build deterministic synthetic text close to target size."""
    target_bytes = target_mb * 1024 * 1024
    seed = (
        "The wormhole shimmered above Titan while engines hummed in sync. "
        "Captain Rao logged coordinates and the archive AI cross-checked stellar drift. "
        "Quantum relays pulsed, translating static into maps for the next jump. "
    )
    repeat = max(1, target_bytes // len(seed.encode("utf-8")) + 1)
    text = seed * repeat
    while len(text.encode("utf-8")) > target_bytes:
        text = text[:-1]
    return text


def split_docs(text: str, docs: int) -> list[str]:
    """Split text into `docs` roughly equal parts."""
    docs = max(1, docs)
    step = max(1, len(text) // docs)
    out = [text[i : i + step] for i in range(0, len(text), step)]
    return [chunk for chunk in out if chunk]


def measure(name: str, fn, total_chars: int, total_bytes: int) -> tuple[float, float]:
    """Run one benchmark case and print throughput."""
    start = time.perf_counter()
    _ = fn()
    elapsed = time.perf_counter() - start
    mbps = total_bytes / elapsed / (1024 * 1024)
    cps = total_chars / elapsed
    print(f"{name:<30} {elapsed:>10.3f}s  {mbps:>7.2f} MB/s  {cps:>12,.0f} chars/s")
    return elapsed, mbps


def load_or_train(model_path: Path, train_text: str, vocab_size: int) -> RegexTokenizer:
    """Load model if available, otherwise train a fresh tokenizer."""
    if model_path.exists():
        tok = from_pretrained(str(model_path))
        if isinstance(tok, RegexTokenizer):
            return tok
    tok = RegexTokenizer()
    tok.train(train_text, vocab_size=vocab_size, verbose=False)
    return tok


def main() -> None:
    """Run focused parallel speed benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark ByteTok parallel speed.")
    parser.add_argument("--size-mb", type=int, default=64, help="Synthetic corpus size.")
    parser.add_argument("--docs", type=int, default=1000, help="Number of batch docs.")
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Worker count for parallel runs.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Vocab size if training is needed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="encoding.model",
        help="Existing model path to load.",
    )
    args = parser.parse_args()

    text = make_text(args.size_mb)
    docs = split_docs(text, args.docs)
    text_bytes = len(text.encode("utf-8"))
    print(f"Corpus size: {format_bytes(text_bytes)} ({len(text):,} chars)")
    print(f"Docs: {len(docs):,}")
    print(f"Workers: {args.workers}")

    model_path = Path(args.model)
    print(f"Model source: {model_path}")
    tokenizer = load_or_train(model_path, text[: min(len(text), 8 * 1024 * 1024)], args.vocab_size)

    print("\n--- Single text encode ---")
    t1, _ = measure(
        "single encode workers=1",
        lambda: tokenizer.encode(text, num_workers=1),
        len(text),
        text_bytes,
    )
    tN, _ = measure(
        f"single encode workers={args.workers}",
        lambda: tokenizer.encode(text, num_workers=args.workers),
        len(text),
        text_bytes,
    )
    print(f"single speedup: {t1 / tN:.2f}x")

    total_batch_chars = sum(len(d) for d in docs)
    total_batch_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print("\n--- Batch encode ---")
    toff, _ = measure(
        "batch mode=off workers=1",
        lambda: encode_batch(
            tokenizer,
            docs,
            num_workers=1,
            parallel_mode="off",
        ),
        total_batch_chars,
        total_batch_bytes,
    )
    tauto, auto_mbps = measure(
        f"batch mode=auto workers={args.workers}",
        lambda: encode_batch(
            tokenizer,
            docs,
            num_workers=args.workers,
            parallel_mode="auto",
        ),
        total_batch_chars,
        total_batch_bytes,
    )
    print(f"auto vs off speedup: {toff / tauto:.2f}x")
    print(f"\nParallel auto throughput: {auto_mbps:.2f} MB/s")


if __name__ == "__main__":
    main()
