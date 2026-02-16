# ByteTok

[![CI](https://github.com/VihangaFTW/bytetok/actions/workflows/release.yaml/badge.svg)](https://github.com/VihangaFTW/bytetok/actions/workflows/release.yaml)
![PyPI - Version](https://img.shields.io/pypi/v/bytetok)
![Python versions](https://img.shields.io/pypi/pyversions/bytetok?cacheSeconds=300)
![License](https://img.shields.io/github/license/VihangaFTW/bytetok)

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

Dataset: [Sci-Fi Books (Gutenberg)](https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg).

| Corpus Size            | Vocab Size | Training Time      | Encoding Throughput           | Decoding Throughput | Compression Ratio | Size Reduction |
| ---------------------- | ---------- | ------------------ | ----------------------------- | ------------------- | ----------------- | -------------- |
| 88.85 MB              | 25,000     | 3.3 mins           | 2.85 MB/sec                   | 19.2M tokens/sec    | 1.43x             | 30.3%          |
| 216.96 MB             | 10,000     | 8.7 mins           | 2.73 MB/sec                   | 17.1M tokens/sec    | 1.60x             | 37.7%          |
| 216.96 MB             | 25,000     | 9.65 mins          | 2.72 MB/sec                   | 17.0M tokens/sec    | 1.68x             | 40.6%          |
| 216.96 MB             | 50,000     | 10.7 mins          | 2.63 MB/sec                   | 16.4M tokens/sec    | 1.75x             | 42.7%          |
| 326.96 MB             | 50,000     | 17.5 mins          | 2.69 MB/sec                   | 7.02M tokens/sec    | 1.44x             | 30.7%          |

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
git clone https://github.com/VihangaFTW/bytetok.git
cd bytetok

# install with uv
uv sync

# or build with maturin
uv sync --group dev
uv run maturin develop
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

## Documentation

Complete API documentation is available in the [project wiki](https://github.com/VihangaFTW/bytetok/wiki/ByteTok-Documentation).


## Acknowlegment

ByteTok is inspired by Andrej Kaparthy's [minbpe](https://github.com/karpathy/minbpe). A walkthrough of _minbpe_ repository is documented on his Youtube channel [here](https://youtu.be/zduSFxRajkE).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
