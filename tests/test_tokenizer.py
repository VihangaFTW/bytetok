"""Unit tests for ByteTok tokenizer encode/decode, edge cases, and serialization."""

import pytest
import regex

import bytetok as btok
from bytetok.bpe import RustBPETokenizer, RustBPETrainer
from bytetok.errors import SpecialTokenError, TokenizationError, TrainingError


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


def test_rust_trainer_from_corpus_learns_merges():
    """Corpus constructor initializes a trainer that can learn merges."""
    trainer = RustBPETrainer.from_corpus(
        "hello world hello world",
        pattern=r"\w+",
        min_count=1,
    )

    trainer.train(5, show_progress=False)

    assert trainer.get_merge_history()


def test_rust_trainer_from_corpus_has_no_flat_token_stream():
    """Corpus trainers reject flat token reconstruction."""
    trainer = RustBPETrainer.from_corpus("hello world", pattern=r"\w+")

    with pytest.raises(
        ValueError, match="token sequence is unavailable for corpus-based trainers"
    ):
        trainer.get_tokens()


def test_rust_trainer_from_corpus_rejects_invalid_pattern():
    """Corpus constructor raises on invalid regex patterns."""
    with pytest.raises(ValueError, match="invalid regex pattern"):
        RustBPETrainer.from_corpus("hello world", pattern="(")


def test_regex_training_allows_single_occurrence_merge():
    """Regex training keeps the default single-occurrence merge threshold."""
    tok = btok.get_tokenizer("gpt4o")

    tok.train("ab", vocab_size=257, verbose=False, show_progress=False)

    assert tok.vocab_size() == 257



ALL_PATTERNS = [
    "gpt2",
    "gpt4",
    "gpt4o",
    "llama3",
    "qwen2",
    "qwen35",
    "deepseek-coder",
    "deepseek-llm",
    "glm4",
    "kimi-k2",
]

DIVERSE_TEXTS = [
    "Hello, world! It's a test. I'LL DO IT, you've seen.",
    "x = a + b * c ^ d | e ~ f `g` $h <i> {j} @k #l %m &n",
    "caf\u00e9 na\u00efve \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4 \U0001f389 \U0001f469\u200d\U0001f469\u200d\U0001f467 \U0001f44d\U0001f3fd",
    "e\u0301 \u0928\u092e\u0938\u094d\u0924\u0947 \u0645\u0631\u062d\u0628\u0627 \u05e9\u05dc\u05d5\u05dd \u1f7d \u1fd3 \u212a \u2126",
    "tabs\tand\r\nCRLF\n\n\nmany   spaces   \u00a0nbsp\u3000ideographic",
    "\x00\x01\x1b[31mred\x1b[0m\x7f",
    "  leading and trailing  ",
    "12345678901234 3.14159 \u0661\u0662\u0663 \u00b2\u00b3",
]


def _train_on_diverse_texts(pattern_name):
    tok = btok.get_tokenizer(pattern_name)
    tok.train("\n".join(DIVERSE_TEXTS * 20), vocab_size=600, show_progress=False)
    return tok


def _reference_chunks(text, pattern):
    chunks = []
    last_end = 0
    flags = regex.V1 if "&&" in pattern else 0
    for m in regex.finditer(pattern, text, flags=flags):
        if m.start() == m.end():
            continue
        if m.start() > last_end:
            chunks.append(text[last_end : m.start()])
        chunks.append(m.group(0))
        last_end = m.end()
    if last_end < len(text):
        chunks.append(text[last_end:])
    return chunks


def _reference_bpe(chunk, merge_history):
    ranks = {pair: (rank, new) for rank, (pair, new) in enumerate(merge_history)}
    ids = list(chunk.encode("utf-8"))
    while len(ids) > 1:
        candidates = [(ranks[p][0], p) for p in zip(ids, ids[1:]) if p in ranks]
        if not candidates:
            break
        _, pair = min(candidates)
        new = ranks[pair][1]
        merged = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                merged.append(new)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        ids = merged
    return ids


@pytest.mark.parametrize("name", ALL_PATTERNS)
def test_builtin_pattern_source_is_ascii(name):
    assert btok.get_pattern(name).isascii()


@pytest.mark.parametrize("name", ALL_PATTERNS)
def test_builtin_pattern_roundtrip(name):
    tok = _train_on_diverse_texts(name)
    for text in DIVERSE_TEXTS:
        assert tok.decode(tok.encode(text), errors="strict") == text
    encoded = tok.encode_batch(DIVERSE_TEXTS, show_progress=False)
    assert tok.decode_batch(encoded, show_progress=False) == DIVERSE_TEXTS


@pytest.mark.parametrize("name", ALL_PATTERNS)
def test_encode_matches_reference_bpe(name):
    tok = _train_on_diverse_texts(name)
    history = tok._get_merge_history()
    for text in DIVERSE_TEXTS:
        expected = [
            t
            for chunk in _reference_chunks(text, tok.pat)
            for t in _reference_bpe(chunk, history)
        ]
        assert tok.encode(text) == expected


def test_basic_encode_matches_reference_bpe():
    tok = btok.BasicTokenizer()
    tok.train("\n".join(DIVERSE_TEXTS * 20), vocab_size=600, show_progress=False)
    history = tok._get_merge_history()
    for text in DIVERSE_TEXTS:
        assert tok.encode(text) == _reference_bpe(text, history)


def test_deepseek_llm_pattern_matches_upstream_classes():
    pat = btok.get_pattern("deepseek-llm")
    assert regex.findall(pat, "Hello, world \u1f7d\u212a") == [
        "Hello",
        ",",
        " world",
        " \u1f7d\u212a",
    ]
    assert regex.findall(pat, "abc\u2018\u201c") == ["abc", "\u2018\u201c"]


def test_custom_pattern_keeps_unmatched_text():
    tok = btok.get_tokenizer(custom_pattern=r"\w+")
    tok.train("hello, world! foo-bar", vocab_size=270, show_progress=False)
    text = "hello, world! foo-bar  \U0001f642"
    assert tok.decode(tok.encode(text), errors="strict") == text
    assert tok.encode_batch([text], show_progress=False) == [tok.encode(text)]


def test_custom_pattern_keeps_unmatched_text_with_special_tokens():
    tok = btok.get_tokenizer("deepseek-coder")
    tok.train("a = b + c " * 20, vocab_size=270, show_progress=False)
    n = tok.vocab_size()
    tok.set_special_tokens({"<|eot|>": n})
    strategy = btok.get_strategy("all")
    text = "a = b<|eot|>$c + d  "
    tokens = tok.encode(text, strategy=strategy)
    assert n in tokens
    assert tok.decode(tokens, errors="strict") == text
    batch = tok.encode_batch([text], strategy=strategy, show_progress=False)
    assert batch == [tokens]


def test_training_counts_unmatched_text():
    trainer = RustBPETrainer.from_corpus("a==b==c==d", pattern=r"[a-z]")
    trainer.train(1, show_progress=False)
    assert trainer.get_merge_history() == [((61, 61), 256)]



def test_overlapping_special_tokens_prefer_longest_match():
    tok = btok.get_tokenizer("gpt4o")
    tok.train("abc abc", vocab_size=258, show_progress=False)
    n = tok.vocab_size()
    tok.set_special_tokens(
        {"<|end|>": n, "<|end|>\n": n + 1, "<|im|>": n + 2, "<|im|><|end|>": n + 3}
    )
    strategy = btok.get_strategy("all")
    for _ in range(50):
        assert tok.encode("<|end|>\n", strategy=strategy) == [n + 1]
        assert tok.encode("<|im|><|end|>x", strategy=strategy) == [n + 3, ord("x")]
        assert tok.encode_batch(
            ["<|end|>\n", "<|im|><|end|>"], strategy=strategy, show_progress=False
        ) == [[n + 1], [n + 3]]


def test_empty_special_token_rejected(regex_tokenizer):
    with pytest.raises(SpecialTokenError):
        regex_tokenizer.set_special_tokens({"": regex_tokenizer.vocab_size()})


def test_rust_tokenizer_ignores_empty_allowed_special():
    tok = RustBPETokenizer([((97, 98), 256)], r"\S+")
    assert tok.encode_text_with_special("ab", {"": 1000}) == [256]
    assert tok.encode_texts_with_special(["ab"], {"": 1000}, show_progress=False) == [
        [256]
    ]


@pytest.mark.parametrize(
    "special",
    [
        " <sp> ",
        "<sp>\t",
        "\n<nl>",
        "<a>\r\n",
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
        "a b",
        '"quoted"',
    ],
)
def test_save_load_preserves_special_tokens(tmp_path, special):
    tok = btok.get_tokenizer("gpt4o")
    tok.train("hello world hello world", vocab_size=260, show_progress=False)
    n = tok.vocab_size()
    tok.set_special_tokens({special: n, "<|endoftext|>": n + 1})
    tok.save(str(tmp_path / "tok"))

    loaded = btok.from_pretrained(str(tmp_path / "tok.model"))
    assert loaded.special_toks == tok.special_toks

    strategy = btok.get_strategy("all")
    text = f"x{special}y<|endoftext|>"
    assert loaded.encode(text, strategy=strategy) == tok.encode(text, strategy=strategy)
    assert loaded.decode([n]) == special


def test_save_load_preserves_pattern_whitespace(tmp_path):
    tok = btok.get_tokenizer(custom_pattern="\\S+|\n| ")
    tok.train("ab cd\nab cd", vocab_size=258, show_progress=False)
    tok.save(str(tmp_path / "tok"))

    loaded = btok.from_pretrained(str(tmp_path / "tok.model"))
    assert loaded.pat == tok.pat
    assert loaded.encode("ab  cd\n") == tok.encode("ab  cd\n")



def test_training_raises_when_regex_engine_fails():
    tok = btok.get_tokenizer("gpt4o")
    with pytest.raises(TrainingError):
        tok.train(
            "hello" + " " * 3_000_000 + "x world", vocab_size=300, show_progress=False
        )


def test_encoding_raises_when_regex_engine_fails(regex_tokenizer):
    with pytest.raises(TokenizationError):
        regex_tokenizer.encode(" " * 3_000_000 + "x")
