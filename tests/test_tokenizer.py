"""Unit tests for ByteTok tokenizer encode/decode, edge cases, and serialization."""

import pytest
from types import SimpleNamespace

import bytetok as btok
from bytetok.errors import TrainingError


# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regex_tokenizer():
    """Return a trained RegexTokenizer."""
    tok = btok.get_tokenizer("gpt4o")
    tok.train("hello world hello world", vocab_size=500, verbose=False)
    return tok


@pytest.fixture
def basic_tokenizer():
    """Return a trained BasicTokenizer."""
    tok = btok.BasicTokenizer()
    tok.train("hello world hello world", vocab_size=500, verbose=False)
    return tok


# Encode-decode round-trip
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip_regex(regex_tokenizer):
    """Encode then decode returns original text for RegexTokenizer."""
    text = "Hello, world!"
    tokens = regex_tokenizer.encode(text)
    decoded = regex_tokenizer.decode(tokens)
    assert decoded == text


def test_encode_decode_roundtrip_basic(basic_tokenizer):
    """Encode then decode returns original text for BasicTokenizer."""
    text = "Hello, world!"
    tokens = basic_tokenizer.encode(text)
    decoded = basic_tokenizer.decode(tokens)
    assert decoded == text


def test_encode_decode_roundtrip_unicode(regex_tokenizer):
    """Round-trip preserves unicode characters."""
    text = "café naïve 日本語 🎉"
    tokens = regex_tokenizer.encode(text)
    decoded = regex_tokenizer.decode(tokens)
    assert decoded == text


# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string(regex_tokenizer):
    """Empty string encodes to empty list and decodes back."""
    tokens = regex_tokenizer.encode("")
    assert tokens == []
    decoded = regex_tokenizer.decode([])
    assert decoded == ""


def test_whitespace_only(regex_tokenizer):
    """Whitespace-only text round-trips correctly."""
    text = "   \n\t  "
    tokens = regex_tokenizer.encode(text)
    decoded = regex_tokenizer.decode(tokens)
    assert decoded == text


def test_single_character(regex_tokenizer):
    """Single character round-trips."""
    text = "x"
    tokens = regex_tokenizer.encode(text)
    decoded = regex_tokenizer.decode(tokens)
    assert decoded == text


def test_repetitive_text_creates_merges(regex_tokenizer):
    """Repetitive text produces fewer tokens due to merges."""
    text = "the the the the the"
    tokens = regex_tokenizer.encode(text)
    assert len(tokens) < len(text.encode("utf-8"))


# Decode before training raises
# ---------------------------------------------------------------------------


def test_decode_before_training_raises():
    """Decoding before training raises TrainingError."""
    tok = btok.get_tokenizer("gpt4o")
    with pytest.raises(TrainingError):
        tok.decode([0, 1, 2])


def test_encode_before_training_raises():
    """Encoding before training raises TrainingError."""
    tok = btok.get_tokenizer("gpt4o")
    with pytest.raises(TrainingError):
        tok.encode("hello")


# Save and load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(regex_tokenizer, tmp_path):
    """Save and load preserves tokenizer state."""
    prefix = str(tmp_path / "tok")
    regex_tokenizer.save(prefix)

    loaded = btok.from_pretrained(f"{prefix}.model")
    original_tokens = regex_tokenizer.encode("test string")
    loaded_tokens = loaded.encode("test string")
    assert loaded_tokens == original_tokens

    decoded = loaded.decode(loaded_tokens)
    assert decoded == "test string"


def test_basic_tokenizer_save_load_roundtrip(basic_tokenizer, tmp_path):
    """BasicTokenizer save and load preserves state."""
    prefix = str(tmp_path / "basic_tok")
    basic_tokenizer.save(prefix)

    loaded = btok.from_pretrained(f"{prefix}.model")
    assert isinstance(loaded, btok.BasicTokenizer)
    text = "hello world"
    assert loaded.decode(loaded.encode(text)) == text


# Batch encode/decode
# ---------------------------------------------------------------------------


def test_encode_batch_decode_batch(regex_tokenizer):
    """Batch encode and decode match single-text results."""
    texts = ["First.", "Second document.", "Third."]
    encoded = regex_tokenizer.encode_batch(texts)
    decoded = regex_tokenizer.decode_batch(encoded)

    for i, text in enumerate(texts):
        assert decoded[i] == text
        assert regex_tokenizer.decode(encoded[i]) == text


# Vocab size
# ---------------------------------------------------------------------------


def test_vocab_size_after_training(regex_tokenizer):
    """Vocab size is at least 256 after training."""
    assert regex_tokenizer.vocab_size() >= 256

