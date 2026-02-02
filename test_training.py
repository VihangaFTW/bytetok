"""Quick test to verify Rust-based training works."""

from src.bytetok.models.regex import RegexTokenizer


def test_basic_training():
    """Test basic training with Rust implementation."""
    # Sample text
    text = "hello world " * 100

    # Create tokenizer
    tokenizer = RegexTokenizer()

    # Train with small vocab
    vocab_size = 300
    tokenizer.train(text, vocab_size=vocab_size, verbose=True)

    # Verify merges were created
    assert len(tokenizer.merges) > 0, "No merges were created"
    assert len(tokenizer.vocab) == 256 + len(tokenizer.merges), "Vocab size mismatch"

    # Test encoding and decoding
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text, f"Decode failed: expected '{text}', got '{decoded}'"

    print(f"✓ Training successful: {len(tokenizer.merges)} merges created")
    print(f"✓ Vocab size: {len(tokenizer.vocab)}")
    print(f"✓ Encoding works: {len(encoded)} tokens")
    print("✓ Decoding works: matches original text")


if __name__ == "__main__":
    test_basic_training()
