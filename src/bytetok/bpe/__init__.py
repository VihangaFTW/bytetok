"""Rust-backed BPE trainer wrapper. Re-exports RustBPETrainer and RustBPETokenizer from the compiled extension."""

from __future__ import annotations

try:
    from .._bpe_rs import RustBPETrainer, RustBPETokenizer
except Exception as e:
    raise ImportError(
        "rust extension module `bytetok._bpe_rs` is not available; "
        "this package is rust-only; build/install the extension (e.g. via maturin)"
    ) from e

__all__ = ["RustBPETrainer", "RustBPETokenizer"]
