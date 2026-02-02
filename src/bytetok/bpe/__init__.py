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
    from .._bpe_rs import RustBPETrainer
except Exception as e:
    raise ImportError(
        "Rust extension module `bytetok._bpe_rs` is not available. "
        "This package is Rust-only; build/install the extension (e.g. via maturin)."
    ) from e

__all__ = ["RustBPETrainer"]
