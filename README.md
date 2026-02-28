# ByteTok

[![CI](https://github.com/VihangaFTW/bytetok/actions/workflows/release.yaml/badge.svg)](https://github.com/VihangaFTW/bytetok/actions/workflows/release.yaml)
![PyPI - Version](https://img.shields.io/pypi/v/bytetok)
![Python versions](https://img.shields.io/pypi/pyversions/bytetok?cacheSeconds=300)
![License](https://img.shields.io/github/license/VihangaFTW/bytetok)

ByteTok implements Byte Pair Encoding (BPE) at the byte-level with a Rust-accelerated core for training and encoding. Text is first converted to raw bytes (0-255), then iteratively merged using learned pair statistics.

The training algorithm is based on an optimized [BPE algorithm](https://aclanthology.org/2023.findings-acl.38.pdf) from the paper _A Formal Perspective on Byte-Pair Encoding_. The research has enabled ByteTok to achieve O(N log V) training time and O(N log N) encoding time versus the naive O(NV) approach.

> Here, N denotes the length of the input text and V is the tokenizer's vocabulary size.

## Features

- **High-performance Rust-powered training, encoding, and decoding**: Engineered from the ground up with a parallel processing pipeline for efficient handling of large-scale NLP datasets (1GB+) with the aim of enabling rapid processing for modern LLM applications.
- **Built-in regex patterns**: Choose from a pre-tokenization regex preset that includes GPT-2, GPT-4, GPT-4o, LLaMA 3, Qwen 2 and DeepSeek.
- **Custom regex patterns**: Supported alongside the built-in presets.
- **Special token strategies**: Control how special tokens are handled during encoding.
- **Serialization**: Supports versioned `.model` / `.vocab` file formats for saving tokenizer state, as well as easy loading via a `from_pretrained()` function.

## History

This project started as a weekend experiment with BPE for text compression. I later needed a tokenizer for my custom GPT, which was bottlenecked by context length due to character-level encoding. I wanted a simple API that did four things correctly at a reasonable speed:

- Train on custom text
- Save learned encodings
- Encode text
- Decode text

Feel free to check out robust libraries such as OpenAI's [tiktoken](https://github.com/openai/tiktoken) and Google's [sentencepiece](https://github.com/google/sentencepiece) that are widely adopted in production environments. Tiktoken resembles ByteTok the most, but it should be noted that ByteTok provides a training pipeline which Tiktoken lacks.

In contrast, ByteTok was developed with a different focus. It prioritizes simplicity and usability by offering a clear API that efficiently maps strings to lists of token IDs. All this without burdening users with overly complex configuration or excessive parameters.

## Benchmarks

These benchmarks were conducted on a Linux x86_64 system equipped with an Intel Core i7-12700H processor (20 cores @ 4.70 GHz) and 32GB DDR5 RAM. Encoding and decoding throughput represent the speed of `encode_batch()` and `decode_batch()` operations, respectively.

Dataset: [Sci-Fi Books (Gutenberg)](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg)

| Corpus Size | Vocab Size | Training Time | Encoding Throughput | Decoding Throughput | Compression Ratio | Size Reduction |
| ----------- | ---------- | ------------- | ------------------- | ------------------- | ----------------- | -------------- |
| 132.36 MB   | 10,000     | 4.58 mins     | 16.12 MB/sec        | 82.4M tokens/sec    | 1.38x             | 27.5%          |
| 216.96 MB   | 10,000     | 8.75 mins     | 13.82 MB/sec        | 81.4M tokens/sec    | 1.60x             | 37.7%          |
| 216.96 MB   | 25,000     | 9.74 mins     | 14.55 MB/sec        | 70.2M tokens/sec    | 1.68x             | 40.6%          |
| 216.96 MB   | 50,000     | 10.67 mins    | 14.99 MB/sec        | 77.0M tokens/sec    | 1.75x             | 42.7%          |
| 326.96 MB   | 50,000     | 16.19 mins    | 14.61 MB/sec        | 79.3M tokens/sec    | 1.44x             | 30.7%          |

## Requirements

- Python >= 3.12

## Installation

Install from PyPI:

```bash
# with pip
pip install bytetok

# or with uv (recommended)
uv add bytetok
```

### Building from Source

If you want to develop or build from source, you will need the Rust toolchain [rustup](https://rustup.rs/).

```bash
# clone the repository
git clone https://github.com/VihangaFTW/bytetok.git

# install with uv
uv sync

# or build with maturin
uv sync --group dev
uv run maturin develop --release
```

## Quick Start

Here you will find the primary workflows for using ByteTok tokenizers. For detailed API usage and additional features, see the [full documentation in the Wiki](https://github.com/VihangaFTW/bytetok/wiki/ByteTok-Documentation).

### Basics

The API has been designed with simplicity in mind:

```python
import bytetok as btok


# Create a tokenizer with a built-in pattern (default: gpt4o).
tokenizer = btok.get_tokenizer("gpt4o")

# Train on text.
tokenizer.train("your training corpus here...", vocab_size=1000)

# Encode and decode.
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
assert text == "Hello, world!"

# Save and reload.
tokenizer.save("my_tokenizer")
reloaded = btok.from_pretrained("my_tokenizer.model")
```

Custom regex patterns can be used for pre-tokenization:

```python
import bytetok as btok


# Create a tokenizer with a custom pattern
# For example, split on whitespace and punctuation.
tokenizer = btok.get_tokenizer(custom_pattern = r"\w+|[^\w\s]")

```

For best results, it is recommended to choose from the built-in presets, which have been extensively validated.

### Parallel Encoding

ByteTok supports parallel encoding and decoding for faster processing of large batches of text.

Use `encode_batch` to perform parallel encoding to efficiently handle large collections of texts. You can then decode the resulting list of token sequences in parallel using `decode_batch`:

```python
import bytetok as btok


tokenizer = btok.get_tokenizer("gpt4o")
tokenizer.train("your training corpus here...", vocab_size=1000)

# Encode a batch of texts in parallel.
texts = ["First document...", "Second document...", "Third document..."]
encoded = tokenizer.encode_batch(texts, show_progress=False)

# Decode the batch in parallel.
decoded = tokenizer.decode_batch(encoded, errors="replace", show_progress=False)
assert decoded[0] == "First document..."
```

### Special Tokens

Register special tokens after training, then encode with a strategy to control how they are handled:

```python
import bytetok as btok


tokenizer = btok.get_tokenizer("gpt4o")
tokenizer.train("your training corpus here...", vocab_size=1000)

# Set special tokens (IDs must be >= vocab size).
tokenizer.set_special_tokens({"<|endoftext|>": 15005, "<|pad|>": 13005})

# Encode with strategy: "all" allows special tokens in text; "none" ignores them.
strategy = btok.get_strategy("all")
tokens = tokenizer.encode("Hello<|endoftext|>world", strategy=strategy)

# Batch encoding with special tokens.
encoded = tokenizer.encode_batch(
    ["Doc one.", "Doc two<|pad|>padding", "Doc three."],
    strategy=strategy,
)
```

ByteTok automatically checks for conflicts when special tokens would replace existing tokens in the vocabulary or if there are duplicates.

## Acknowledgment

ByteTok is inspired by Andrej Kaparthy's [minbpe](https://github.com/karpathy/minbpe). A walkthrough of _minbpe_ repository is documented on his Youtube channel [here](https://youtu.be/zduSFxRajkE).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
