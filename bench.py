"""Benchmark encode_batch() and decode_tokens_batch() on a slice of the Sci-Fi Gutenberg dataset.

Outputs a row matching the BENCHMARKS.MD table columns:
  Corpus Size | Vocab Size | Training Time | Encoding Throughput |
  Decoding Throughput | Compression Ratio | Size Reduction
"""

import argparse
import time
from pathlib import Path

from datasets import load_dataset

from bytetok import RegexTokenizer, from_pretrained

HF_DATASET = "stevez80/Sci-Fi-Books-gutenberg"


def load_corpus(num_docs: int | None) -> list[str]:
    """Load up to `num_docs` documents via dataset indexing; full dataset when None."""
    print(f"Loading {HF_DATASET} (non-streaming) …")
    ds = load_dataset(HF_DATASET, split="train")
    if num_docs is not None:
        return ds[:num_docs]["text"]
    return ds["text"]


def train_tokenizer(train_text: str, vocab_size: int) -> tuple[RegexTokenizer, float]:
    """Train a RegexTokenizer and return it along with elapsed training time in seconds."""
    tok = RegexTokenizer()
    start = time.perf_counter()
    tok.train(train_text, vocab_size=vocab_size, verbose=False)
    return tok, time.perf_counter() - start


def load_or_train(
    model_path: Path | None, train_text: str, vocab_size: int
) -> tuple[RegexTokenizer, float]:
    """Return (tokenizer, training_seconds), loading from disk when requested."""
    if model_path and model_path.exists():
        tok = from_pretrained(str(model_path))
        if isinstance(tok, RegexTokenizer):
            print(f"Loaded model from {model_path}")
            return tok, 0.0
    if model_path:
        print(f"Model not found at {model_path}; training a new one.")
    print(f"Training new model (vocab_size={vocab_size:,}) …")
    return train_tokenizer(train_text, vocab_size)


def main() -> None:
    """Run the encode/decode benchmark and print a BENCHMARKS.MD-compatible row."""
    parser = argparse.ArgumentParser(
        description="Benchmark ByteTok encode_batch() and decode_tokens_batch()."
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Number of documents to encode (default: full dataset).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Vocab size for training (default: 10,000).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional existing .model path to load; default trains a fresh model.",
    )
    args = parser.parse_args()

    docs = load_corpus(args.num_docs)
    if not docs:
        raise RuntimeError("No documents loaded from dataset.")

    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    corpus_mb = total_bytes / (1024 * 1024)

    train_text = "".join(docs)
    model_path = Path(args.model) if args.model else None
    tokenizer, train_secs = load_or_train(model_path, train_text, args.vocab_size)

    # --- Encoding ---
    t0 = time.perf_counter()
    encoded: list[list[int]] = tokenizer.encode_batch(docs)
    encode_elapsed = time.perf_counter() - t0
    encode_mbps = total_bytes / encode_elapsed / (1024 * 1024)

    # --- Decoding (uses the same Rayon-backed batch path as decode internally) ---
    t0 = time.perf_counter()
    tokenizer.decode_batch(encoded, errors="replace")
    decode_elapsed = time.perf_counter() - t0
    total_tokens = sum(len(seq) for seq in encoded)
    decode_mtps = total_tokens / decode_elapsed / 1_000_000

    # --- Compression stats ---
    compression_ratio = total_bytes / total_tokens
    size_reduction = (1 - 1 / compression_ratio) * 100

    # Training time: display as mins if >= 60 s, else as secs.
    if train_secs >= 60:
        train_str = f"{train_secs / 60:.2f} mins"
    else:
        train_str = f"{train_secs:.1f} secs"

    # --- Output ---
    print()
    header = (
        f"| {'Corpus Size':22} | {'Vocab Size':10} | {'Training Time':18} "
        f"| {'Encoding Throughput':29} | {'Decoding Throughput':19} "
        f"| {'Compression Ratio':17} | {'Size Reduction':14} |"
    )
    sep = (
        f"| {'-' * 22} | {'-' * 10} | {'-' * 18} "
        f"| {'-' * 29} | {'-' * 19} "
        f"| {'-' * 17} | {'-' * 14} |"
    )
    row = (
        f"| {f'{corpus_mb:.2f} MB':22} | {args.vocab_size:10,} | {train_str:18} "
        f"| {f'{encode_mbps:.2f} MB/sec':29} | {f'{decode_mtps:.1f}M tokens/sec':19} "
        f"| {f'{compression_ratio:.2f}x':17} | {f'{size_reduction:.1f}%':14} |"
    )
    print(header)
    print(sep)
    print(row)
    print()


if __name__ == "__main__":
    main()
