"""Quick test to verify Rust-based training works."""

from bytetok.models.regex import RegexTokenizer
from datasets import load_dataset


def test_basic_training():
    """Test basic training with Rust implementation."""

    ds = load_dataset("stevez80/Sci-Fi-Books-gutenberg", split="train")
    text = "".join(ds[:1000]["text"])

    # Create tokenizer
    tokenizer = RegexTokenizer()

    # Train with small vocab
    vocab_size = 1000
    tokenizer.train(text, vocab_size=vocab_size, verbose=True)

    print("training done...")

    # Verify merges were created
    assert len(tokenizer.merges) > 0, "No merges were created"
    assert len(tokenizer.vocab) == 256 + len(tokenizer.merges), "Vocab size mismatch"

    tokenizer.save("encoding")

    # Test encoding and decoding
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text, "Decode failed: expected"

    print(f"✓ Training successful: {len(tokenizer.merges)} merges created")
    print(f"✓ Vocab size: {len(tokenizer.vocab)}")
    print(f"✓ Encoding works: {len(encoded)} tokens")
    print("✓ Decoding works: matches original text")


if __name__ == "__main__":
    test_basic_training()
