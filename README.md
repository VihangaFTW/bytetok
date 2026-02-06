# ByteTok

A fast, modular and light-weight BPE tokenizer for NLP research and prototyping.

ByteTok implements Byte Pair Encoding (BPE) at the byte-level with a Rust-accelerated core for training and encoding. Text is first converted to raw bytes (0-255), then iteratively merged using learned pair statistics. The training algorithm is based on [Algorithm 2](https://aclanthology.org/2023.findings-acl.38.pdf) from _"A Formal Perspective on Byte-Pair Encoding"_, achieving O(N log V) training and O(N log N) encoding versus the naive O(NV) approach.

## History

This project started as a weekend experiment with BPE for text compression. I later needed a tokenizer for my custom GPT, which was bottlenecked by context length due to character-level encoding. I wanted a simple API that did four things correctly at a reasonable speed:

- Train on custom text
- Save learned encodings
- Encode text
- Decode text

Libraries like OpenAI's [tiktoken](https://github.com/openai/tiktoken) and Google's [sentencepiece](https://github.com/google/sentencepiece) exist and are probably better for production work. But ByteTok wasn't designed to compete with them or benchmaxx. I wanted a straightforward API that took a string and returned a list of integers; not something that forced me to read through documentation for 200 function arguments (looking at you, `sentencepiece`).

As my dataset requirements grew, the naive BPE implementation started struggling. So I rewrote the trainer and encoder in Rust using a much more efficient algorithm 😎.

## Features

- **Fast Rust-backed training and encoding** via PyO3/maturin for datasets larger than 100MB. ByteTok delivers _600x-1000x_ better performance when compared to a naive O(NV) implementation.
- **Built-in regex patterns** from GPT-2, GPT-4, GPT-4o, LLaMA 3, Qwen 2, DeepSeek, StarCoder, Falcon, and BLOOM.
- **Custom patterns** supported alongside the built-in presets.
- **Special token strategies** for controlling how special tokens are handled during encoding.
- **Serialization** with versioned `.model` / `.vocab` file format and `from_pretrained()` loader.

## Benchmarks

Benchmarks were run on Linux x86_64 with an Intel Core i7-12700H (20 cores @ 4.70 GHz) and 32GB DDR5 RAM.

| Dataset                                                                                     | Corpus Size            | Vocab Size | Training Time      | Encoding Throughput           | Decoding Throughput | Compression Ratio | Size Reduction |
| ------------------------------------------------------------------------------------------- | ---------------------- | ---------- | ------------------ | ----------------------------- | ------------------- | ----------------- | -------------- |
| [Sci-Fi Books (Gutenberg)](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg) | 88.85 MB (93M chars)   | 25,000     | 198s (~3.3 mins)   | 2.99M chars/sec (2.85 MB/sec) | 19.2M tokens/sec    | 1.43x             | 30.3%          |
| [Sci-Fi Books (Gutenberg)](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg) | 216.96 MB (227M chars) | 10,000     | 523s (~8.7 mins)   | 2.86M chars/sec (2.73 MB/sec) | 17.1M tokens/sec    | 1.60x             | 37.7%          |
| [Sci-Fi Books (Gutenberg)](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg) | 216.96 MB (227M chars) | 25,000     | 579s (~9.65 mins)  | 2.85M chars/sec (2.72 MB/sec) | 17.0M tokens/sec    | 1.68x             | 40.6%          |
| [Sci-Fi Books (Gutenberg)](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg) | 216.96 MB (227M chars) | 50,000     | 640s (~10.7 mins)  | 2.76M chars/sec (2.63 MB/sec) | 16.4M tokens/sec    | 1.75x             | 42.7%          |
| [Sci-Fi Books (Gutenberg)](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg) | 326.96 MB (343M chars) | 50,000     | 1048s (~17.5 mins) | 2.82M chars/sec (2.69 MB/sec) | 7.02M tokens/sec    | 1.44x             | 30.7%          |

## Requirements

- Python >= 3.13

## Installation

Install from PyPI:

```bash
# with pip
pip install bytetok

# or with uv (recommended)
uv add bytetok
```

### Building from Source

If you want to develop or build from source, you'll need a Rust toolchain ([rustup](https://rustup.rs/)):

```bash
# clone the repository
git clone https://github.com/vihanga-malaviarachchi/bytetok.git
cd bytetok

# install with uv
uv sync

# or build with maturin
pip install maturin
maturin develop
```

## Quick Start

```python
import bytetok

# create a tokenizer with a built-in pattern (default: gpt4o)
tokenizer = bytetok.get_tokenizer("gpt4o")

# train on text
tokenizer.train("your training corpus here...", vocab_size=1000)

# encode and decode
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
assert text == "Hello, world!"

# save and reload
tokenizer.save("my_tokenizer")
reloaded = bytetok.from_pretrained("my_tokenizer.model")
```

---

## API Reference

### Factory Functions

#### `bytetok.get_tokenizer(pattern="gpt4o", *, custom_pattern=None)`

Create a `RegexTokenizer` with a built-in or custom regex pattern.

- **pattern** (`str`) -- Name of a built-in pattern. Ignored when `custom_pattern` is set. Default: `"gpt4o"`.
- **custom_pattern** (`str | None`) -- A custom regex pattern string. Overrides `pattern` when provided.
- **Returns:** `Tokenizer`
- **Raises:** `PatternError` if the custom pattern is invalid regex.

```python
# built-in pattern
tokenizer = bytetok.get_tokenizer("llama3")

# custom pattern
tokenizer = bytetok.get_tokenizer(custom_pattern=r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+")
```

#### `bytetok.from_pretrained(model_path)`

Load a previously saved tokenizer from a `.model` file. The tokenizer type is auto-detected from the file header.

- **model_path** (`str`) -- Path to the `.model` file.
- **Returns:** `Tokenizer` (either `BasicTokenizer` or `RegexTokenizer` depending on what was saved).
- **Raises:** `ModelLoadError` if the file does not exist, has the wrong extension, contains an unknown tokenizer type, or has a version mismatch.

```python
tokenizer = bytetok.from_pretrained("my_tokenizer.model")
```

#### `bytetok.get_strategy(name="none-raise", allowed_subset=None)`

Create a special token handling strategy for use with `encode()`.

- **name** (`"all" | "none" | "none-raise" | "custom"`) -- Strategy name.
- **allowed_subset** (`set[str] | None`) -- Required when `name="custom"`. The set of special token strings to allow.
- **Returns:** `SpecialTokenStrategy`
- **Raises:** `StrategyError` if the name is unknown or `"custom"` is used without `allowed_subset`.

```python
strategy = bytetok.get_strategy("all")
strategy = bytetok.get_strategy("custom", allowed_subset={"<|endoftext|>"})
```

#### `bytetok.list_patterns()`

Return the names of all available built-in regex patterns.

- **Returns:** `list[str]`

```python
bytetok.list_patterns()
# ['GPT2', 'GPT4', 'GPT4O', 'LLAMA3', 'QWEN2', 'DEEPSEEK_CODER', 'DEEPSEEK_LLM',
#  'STARCODER', 'FALCON', 'BLOOM']
```

#### `bytetok.get_pattern(name)`

Get the regex pattern string for a specific built-in pattern by name.

- **name** (`str`) -- Name of the built-in pattern (case-insensitive).
- **Returns:** `str` -- The regex pattern string.
- **Raises:** `PatternError` if the pattern name is unknown.

```python
# get a specific pattern string
pattern_str = bytetok.get_pattern("llama3")

# use it to create a tokenizer
tokenizer = bytetok.RegexTokenizer(pattern=pattern_str)
```

#### `bytetok.list_strategies()`

Return the names of all available special token strategies.

- **Returns:** `list[str]`

```python
bytetok.list_strategies()
# ['all', 'none', 'none-raise', 'custom']
```

---

### Tokenizer Classes

All tokenizers inherit from the abstract base class `Tokenizer`. The two concrete implementations are `BasicTokenizer` and `RegexTokenizer`.

> The `BasicTokenizer` serves as a documentation for the simplest implementation of a BPE tokenizer. It is **not recommended** for actual use due to its lossy nature when decoding multi-byte utf-8 sequences.
>
> All ByteTok's factory methods default to `RegexTokenizer`. For custom extensions or implementations, always inherit from `RegexTokenizer`.

#### `Tokenizer` (abstract base class)

Manages vocabulary, byte pair merges, and serialization. You **do not** instantiate this directly; use `RegexTokenizer` or the factory functions instead.

##### Attributes

| Attribute      | Type                        | Description                                     |
| -------------- | --------------------------- | ----------------------------------------------- |
| `merges`       | `dict[tuple[int,int], int]` | Byte pair -> merged token ID mapping.           |
| `vocab`        | `dict[int, bytes]`          | Token ID -> byte sequence mapping.              |
| `pat`          | `str`                       | Regex pattern used for text splitting (if any). |
| `special_toks` | `dict[str, int]`            | Special token string -> token ID mapping.       |

##### `train(text, vocab_size, verbose=False)`

Train the tokenizer by learning byte pair merges from the input.

- **text** (`str | list[str]`) -- Training corpus. Lists are concatenated.
- **vocab_size** (`int`) -- Target vocabulary size. Must be > 256.
- **verbose** (`bool`) -- Log each merge operation. Default: `False`.
- **Raises:** `VocabularyError` if `vocab_size <= 256`. `TrainingError` if the input is empty.

##### `encode(text, strategy=None)`

Encode text into a list of integer token IDs.

- **text** (`str`) -- Text to encode.
- **strategy** (`SpecialTokenStrategy | None`) -- How to handle special tokens. `None` means no special token handling.
- **Returns:** `list[int]`

##### `decode(tokens)`

Decode a list of token IDs back into text.

- **tokens** (`list[int]`) -- Token IDs to decode.
- **Returns:** `str`
- **Raises:** `VocabularyError` if a token ID is not in the vocabulary (RegexTokenizer).

##### `save(file_prefix)`

Save the trained tokenizer to disk. Creates two files:

- `<file_prefix>.model` -- Binary merge mappings (used by `load()` / `from_pretrained()`).
- `<file_prefix>.vocab` -- Human-readable token representations.

Parameters:

- **file_prefix** (`str`) -- Path prefix for the output files.

```python
tokenizer.save("models/my_tok")
# creates models/my_tok.model and models/my_tok.vocab
```

##### `load(model_filename)`

Load tokenizer state from a `.model` file. Restores merges, special tokens, and rebuilds the vocabulary.

- **model_filename** (`str`) -- Path to the `.model` file.
- **Raises:** `ModelLoadError` on missing file, wrong extension, version mismatch, or type mismatch.

```python
tokenizer = bytetok.RegexTokenizer()
tokenizer.load("models/my_tok.model")
```

---

#### `BasicTokenizer()`

Tokenizer that operates directly on raw byte sequences without any regex splitting. Does not support special token strategies.

```python
tok = bytetok.BasicTokenizer()
tok.train("Hello world", vocab_size=300)
tokens = tok.encode("Hello")
text = tok.decode(tokens)
```

All methods are inherited from `Tokenizer`. The `strategy` parameter on `encode()` is accepted but ignored.

It is recommended not to use this class. Use `RegexTokenizer` instead.

---

#### `RegexTokenizer(pattern=None)`

Tokenizer that splits text with a regex pattern before applying BPE. Supports special token registration and strategies.

- **pattern** (`str | None`) -- Regex pattern for text splitting. Defaults to the `gpt4o` pattern when `None`.

```python
tok = bytetok.RegexTokenizer()                     # default gpt4o pattern
tok = bytetok.RegexTokenizer(pattern=r"\w+|\S")    # custom pattern
```

In addition to the methods inherited from `Tokenizer`, `RegexTokenizer` provides:

##### `register_special_tokens(special_toks)`

Register special tokens with auto-assigned IDs. Must be called **after** training. Token IDs are assigned sequentially starting from the current vocabulary size.

- **special_toks** (`list[str]`) -- Special token strings to register.
- **Raises:** `SpecialTokenError` if the tokenizer has not been trained yet.

```python
tok.train(text, vocab_size=1000)
tok.register_special_tokens(["<|endoftext|>", "<|pad|>", "<|start|>"])

# encode with special token awareness
strategy = bytetok.get_strategy("all")
tokens = tok.encode("Hello<|endoftext|>", strategy=strategy)
text = tok.decode(tokens)
```

---

### TokenPattern

`TokenPattern` is a `str` enum containing pre-defined regex patterns sourced from popular tokenizer implementations.

#### `TokenPattern.get(name)`

Look up a pattern by name (case-insensitive).

- **name** (`str`) -- Pattern name.
- **Returns:** `str` -- The regex pattern string.
- **Raises:** `PatternError` if the name is unknown.

```python
pattern = bytetok.TokenPattern.get("gpt4o")
```

#### Available Patterns

| Name             | Source            |
| ---------------- | ----------------- |
| `gpt2`           | OpenAI GPT-2      |
| `gpt4`           | OpenAI GPT-4      |
| `gpt4o`          | OpenAI GPT-4o     |
| `llama3`         | Meta LLaMA 3      |
| `qwen2`          | Alibaba Qwen 2    |
| `deepseek-coder` | DeepSeek Coder    |
| `deepseek-llm`   | DeepSeek LLM      |
| `starcoder`      | BigCode StarCoder |
| `falcon`         | TII Falcon        |
| `bloom`          | BigScience BLOOM  |

---

### Special Token Strategies

Strategies control how special tokens are recognised during `encode()`. Pass a strategy instance as the `strategy` parameter.

#### `SpecialTokenStrategy` (abstract base class)

Base class. Subclass this to implement custom strategies.

##### `handle(text, special_toks)`

- **text** (`str`) -- The text being encoded.
- **special_toks** (`dict[str, int]`) -- All registered special tokens.
- **Returns:** `dict[str, int]` -- The subset of special tokens to apply.

#### `AllowAllStrategy`

Allows all registered special tokens to be recognised during encoding.

#### `AllowNoneStrategy`

Silently ignores all special tokens. They are treated as regular text.

#### `AllowNoneRaiseStrategy`

Raises `SpecialTokenError` if any registered special token is found in the input text.

#### `AllowCustomStrategy(allowed_subset)`

Allows only a specified subset of special tokens.

- **allowed_subset** (`set[str]`) -- The special token strings to allow.

```python
# via factory (recommended)
strategy = bytetok.get_strategy("custom", allowed_subset={"<|endoftext|>"})

# or instantiate directly
strategy = bytetok.AllowCustomStrategy({"<|endoftext|>"})
```

---

### Exceptions

All exceptions inherit from `ByteTokError` (importable from `bytetok.errors`).

| Exception           | Raised when                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| `ByteTokError`      | Base exception for all bytetok errors.                                   |
| `VocabularyError`   | `vocab_size <= 256` during training, or unknown token ID during decode.  |
| `TrainingError`     | Training input is empty or too short.                                    |
| `ModelLoadError`    | Loading a `.model` file fails (missing, wrong format, version mismatch). |
| `PatternError`      | A regex pattern fails to compile.                                        |
| `SpecialTokenError` | Special token handling fails (e.g. `AllowNoneRaiseStrategy` finds one).  |
| `StrategyError`     | Unknown strategy name or missing `allowed_subset` for custom strategy.   |
| `TokenizationError` | General tokenization failure.                                            |

```python
from bytetok.errors import ModelLoadError

try:
    tok = bytetok.from_pretrained("missing.model")
except ModelLoadError as e:
    print(e)
```

---

### Model File Format

`save()` produces two files:

**`.model`** -- Machine-readable format used by `load()` and `from_pretrained()`:

```text
ByteTok 0.1.0
type regex
re <pattern>
---
<n_special_tokens>
<special_token_string> <token_id>
...
---
<tok_a> <tok_b> <merged_tok>
...
```

**`.vocab`** -- Human-readable vocabulary for inspection:

```text
ST [256] <|endoftext|>
[0] \u0000
...
[258] [he][llo] -> hello
```

---

## Acknowlegment

ByteTok is inspired by Andrej Kaparthy's [minbpe](https://github.com/karpathy/minbpe). A walkthrough of _minbpe_ repository is documented on his Youtube channel [here](https://youtu.be/zduSFxRajkE).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
