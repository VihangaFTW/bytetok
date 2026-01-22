"""
Utilities for sanitizing and converting bytes to displayable strings.
"""

import unicodedata


def _escape_ctrl_chars(s: str) -> str:
    """
    Replace all Unicode control characters in the input string with their Unicode escape sequences.

    Args:
        s (str): Input string.

    Returns:
        str: The input string with control characters replaced by their Unicode escapes.
    """
    cleaned = []
    for c in s:
        # control category codes vary: Cc, Cf, Cn etc.
        # so check via first character
        if unicodedata.category(c)[0] != "C":
            cleaned.append(c)
        else:
            cleaned.append(f"\\u{ord(c):04x}")
    return "".join(cleaned)


def render_bytes(b: bytes) -> str:
    """
    Decodes a bytes object using UTF-8 and escapes Unicode control characters.
    Bytes belonging to an invalid utf-8 byte sequence will be replaced by Unicode Replacement Character.

    Args:
        b (bytes): The bytes object to decode.

    Returns:
        str: The decoded string with control characters replaced by their Unicode escape sequences.
    """
    return _escape_ctrl_chars(b.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    # Test 1: Normal text
    text1 = b"Hello, World!"
    print(render_bytes(text1))  # Expected: "Hello, World!"

    # Test 2: Text with newline (control character)
    text2 = b"Line 1\nLine 2"
    print(render_bytes(text2))  # Expected: "Line 1\u000aLine 2"

    # Test 3: Text with tab (control character)
    text3 = b"Column1\tColumn2"
    print(render_bytes(text3))  # Expected: "Column1\u0009Column2"

    # Test 4: Text with carriage return
    text4 = b"Hello\rWorld"
    print(render_bytes(text4))  # Expected: "Hello\u000dWorld"

    # Test 5: Mixed control characters
    text5 = b"Name:\tJohn\nAge:\t30\n"
    print(render_bytes(text5))  # Expected: "Name:\u0009John\u000aAge:\u009030\u000a"

    # Test 6: Empty bytes
    text6 = b""
    print(render_bytes(text6))  # Expected: ""

    # Test 7: Bytes with null character
    text7 = b"Hello\x00World"
    print(render_bytes(text7))  # Expected: "Hello\u0000World"

    # Test 8: Valid UTF-8 with emojis (non-control)
    text8 = "Hello 👋 World 🌍".encode("utf-8")
    print(render_bytes(text8))  # Expected: "Hello 👋 World 🌍"
