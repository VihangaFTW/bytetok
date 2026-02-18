"""
Utilities for sanitizing and converting bytes to displayable strings.
"""

import unicodedata


def _escape_ctrl_chars(s: str) -> str:
    """Replace all Unicode control characters with their escape sequences."""
    cleaned = []
    for c in s:
        # control category codes vary: Cc, Cf, Cn etc.
        # so check via first character
        if unicodedata.category(c)[0] != "C":
            cleaned.append(c)
        else:
            cleaned.append(f"\\u{ord(c):04x}")
    return "".join(cleaned)


def _render_bytes(b: bytes) -> str:
    """
    Decode bytes as UTF-8 and escape control characters.

    Invalid UTF-8 sequences are replaced with the Unicode replacement character.
    """
    return _escape_ctrl_chars(b.decode("utf-8", errors="replace"))
