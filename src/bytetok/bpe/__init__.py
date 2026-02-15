"""Rust-backed BPE trainer wrapper.

Maturin is configured to build a Python module named bytetok._bpe_rs.

This module is a small wrapper around the compiled extension that re-exports
classes from `bytetok._bpe_rs` so users can import them from a nicer path:

"from bytetok.bpe import RustBPETrainer"

instead of:

"from bytetok._bpe_rs import RustBPETrainer"
"""

from __future__ import annotations

try:
    from .._bpe_rs import RustBPETrainer, RustBPEEncoder
except Exception as e:
    raise ImportError(
        "rust extension module `bytetok._bpe_rs` is not available "
        "this package is rust-only; build/install the extension (e.g. via maturin)"
    ) from e

__all__ = ["RustBPETrainer", "RustBPEEncoder"]
