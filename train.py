"""Benchmark Rust-based BPE implementation performance."""

import logging
import time
from bytetok.models.regex import RegexTokenizer
from datasets import load_dataset

# Configure logging to show INFO level and above.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def benchmark_training():
    """Benchmark Rust-based training and encoding performance."""

    print("=" * 70)
    print("BPE PERFORMANCE BENCHMARK (Rust Implementation)")
    print("=" * 70)

    # Load dataset.
    print("\n📚 Loading dataset...")
    ds = load_dataset("stevez80/Sci-Fi-Books-gutenberg", split="train")
    text = "".join(ds[:1000]["text"])

    text_size = len(text.encode("utf-8"))
    print(f"   Text size: {format_bytes(text_size)} ({len(text):,} chars)")

    # Create tokenizer.
    tokenizer = RegexTokenizer()
    vocab_size = 10_000

    # Benchmark training.
    print(f"\n🔧 Training tokenizer (vocab_size={vocab_size})...")
    train_start = time.perf_counter()
    tokenizer.train(text, vocab_size=vocab_size, verbose=False)
    train_time = time.perf_counter() - train_start

    print(f"   ✓ Training completed in {train_time:.3f}s")
    print(f"   ✓ Merges created: {len(tokenizer.merges):,}")
    print(f"   ✓ Final vocab size: {len(tokenizer.vocab):,}")

    # Verify correctness.
    assert len(tokenizer.merges) > 0, "No merges were created"
    assert len(tokenizer.vocab) == 256 + len(tokenizer.merges), "Vocab size mismatch"

    # Benchmark encoding.
    print("\n⚡ Benchmarking encoding...")
    encode_start = time.perf_counter()
    encoded = tokenizer.encode(text)
    encode_time = time.perf_counter() - encode_start

    # Verify encoding correctness.
    assert len(encoded) > 0, "Encoding produced no tokens"
    assert all(0 <= token < len(tokenizer.vocab) for token in encoded), (
        "Invalid token IDs in encoded output"
    )
    assert len(encoded) < len(text.encode("utf-8")), "Encoding should compress the text"

    chars_per_sec = len(text) / encode_time
    mb_per_sec = text_size / encode_time / (1024 * 1024)

    print(f"   ✓ Encode time: {encode_time * 1000:.2f}ms")
    print(f"   ✓ Throughput: {chars_per_sec:,.0f} chars/sec ({mb_per_sec:.2f} MB/sec)")
    print(f"   ✓ Tokens generated: {len(encoded):,}")
    print(f"   ✓ Encoding verified: all tokens valid")

    # Benchmark decoding.
    print("\n🔄 Benchmarking decoding...")
    decode_start = time.perf_counter()
    decoded = tokenizer.decode(encoded)
    decode_time = time.perf_counter() - decode_start

    tokens_per_sec = len(encoded) / decode_time

    # Verify decoding correctness.
    assert isinstance(decoded, str), "Decoded output is not a string"
    assert len(decoded) > 0, "Decoding produced empty string"
    assert decoded == text, (
        f"Decode failed: output doesn't match input (got {len(decoded)} chars, expected {len(text)} chars)"
    )

    print(f"   ✓ Decode time: {decode_time * 1000:.2f}ms")
    print(f"   ✓ Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print("   ✓ Decoding verified: output matches input")

    # Compression stats.
    print("\n📊 Compression Statistics:")
    original_tokens = len(text.encode("utf-8"))
    compressed_tokens = len(encoded)
    compression_ratio = original_tokens / compressed_tokens
    reduction_pct = (1 - compressed_tokens / original_tokens) * 100

    print(f"   Original tokens (bytes): {original_tokens:,}")
    print(f"   Compressed tokens: {compressed_tokens:,}")
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    print(f"   Size reduction: {reduction_pct:.1f}%")

    # Save model.
    print("\n💾 Saving model...")
    save_start = time.perf_counter()
    tokenizer.save("encoding")
    save_time = time.perf_counter() - save_start
    print(f"   ✓ Model saved in {save_time * 1000:.2f}ms")

    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_training()
