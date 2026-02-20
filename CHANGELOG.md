# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Ports Python-side text preprocessing in the training algorithm to Rust for faster training times.

## [0.2.0] - 2026-02-20

### Added

- Parallel encode/decode processing pipeline.
- Support for user-defined special token IDs, with validation for duplicates and overwrites.

## Fixed

- `save()` now throws a `TrainingError` when attempting to save an untrained tokenizer, preventing silent writes of empty files.

### Changed

- Ported majority of Python encode and decode logic to Rust.
- Streamlined Python `Tokenizer` internals to eliminate redundant code following migration of hot path logic to Rust.
- Added internal helper methods in `Tokenizer` ABC for wiring Rust logic to Python API.
- Moved `decode_batch()` to `Tokenizer` ABC.
- Added `encode_batch()` method in `Tokenizer` subclasses.
- Bumped package version from `v0.1.2` to `v.0.2.0`.
- Updated tokenizer version from `0.1.0` to `0.2.0` for consistency with the package version.
- Updated README quick start section with examples on parallel encoding and special token workflows.
- Improved encoding and decoding performance across single-text, batch, and special-token workloads.

![Benchmark comparison](assets/v020_bench.png)

### Removed

- Regex patterns for StarCoder, BLOOM, and Falcon were removed from the preset as remaining patterns cover most use cases.
- The `_bpe.py` module and its `slow_bpe_merge()` and `slow_bpe_merge_with_freq_update()` functions have been removed.

## [0.1.2] - 2026-02-07

### Fixed

- Fixed duplicate symbol exports in `__init__.py` and `__init__.pyi` that caused redundant auto-imports and member completions in IDEs.
- Updated `TokenPattern` documentation by turning the regex pattern table into a list and removing the Source column.
- Clarified that `BasicTokenizer` lossy decoding refers to UTF-8 reconstruction during `decode()` (uses `errors="replace"`).

## [0.1.1] - 2026-02-06

### Fixed

- Added Python version classifiers to fix PyPI badge display on GitHub.

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
