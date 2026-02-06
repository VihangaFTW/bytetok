# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-06

### Added

- Byte-level BPE tokenizer with Rust-accelerated training and encoding via PyO3/maturin.
- O(N log V) training and O(N log N) encoding algorithms based on [Algorithm 2](https://aclanthology.org/2023.findings-acl.38.pdf).
- `RegexTokenizer` with built-in patterns from GPT-2, GPT-4, GPT-4o, LLaMA 3, Qwen 2, DeepSeek, StarCoder, Falcon, and BLOOM.
- `BasicTokenizer` as a minimal reference implementation.
- Custom regex pattern support via `get_tokenizer(custom_pattern=...)`.
- Special token registration and handling with four built-in strategies: `all`, `none`, `none-raise`, and `custom`.
- Versioned `.model` / `.vocab` serialization format with `save()` and `load()`.
- `from_pretrained()` factory for loading saved tokenizers with auto-detected type.
- `TokenPattern` enum for looking up pre-defined patterns by name.
- Custom exception hierarchy rooted in `ByteTokError`.
- CI/CD pipeline for building platform wheels (Linux, macOS, Windows) and publishing to PyPI.
