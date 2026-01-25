"""Regex-based byte-level tokenizer implementation."""

from enum import Enum
from .base_tok import Tokenizer
from ._bpe import Token
import regex as re
from pathlib import Path


class TokenPattern(str, Enum):
    """
    Pre-defined regex patterns for different tokenizer implementations.

    Sources:
    - GPT patterns: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    - Other patterns: https://github.com/ggerganov/llama.cpp (llama-vocab.cpp)
    """

    # OpenAI models
    GPT2 = (
        r"'(?:[sdmt]|ll|ve|re)|"
        r" ?\p{L}+|"
        r" ?\p{N}+|"
        r" ?[^\s\p{L}\p{N}]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    GPT4 = (
        r"'(?i:[sdmt]|ll|ve|re)|"
        r"[^\r\n\p{L}\p{N}]?+\p{L}+|"
        r"\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]++[\r\n]*|"
        r"\s*[\r\n]|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    GPT4O = (
        r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|"
        r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|"
        r"\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    # Meta models
    LLAMA3 = (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    # Alibaba models
    QWEN2 = (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}|"  # Single digits (different from LLAMA3)
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    # DeepSeek models
    DEEPSEEK_CODER = (
        r"[\r\n]|"
        r"\s?\p{L}+|"
        r"\s?\p{P}+|"
        r"[一-龥ࠀ-一가-퟿]+|"  # CJK characters
        r"\p{N}"
    )

    DEEPSEEK_LLM = (
        r"[\r\n]|"
        r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳳⲀ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+|"
        r"\s?[!-/:-~！-／：-～'-‟　-。]+|"
        r"\s+$|"
        r"[一-龥ࠀ-一가-퟿]+|"
        r"\p{N}+"
    )

    # coding-focused models
    STARCODER = (
        r"\p{N}|"
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|"
        r"\s+(?!\S)"
    )

    FALCON = (
        r"[\p{P}\$\+<=>^\~\|`]+|"
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|"
        r"\s+(?!\S)|"
        r"[0-9][0-9][0-9]"
    )

    # multilingual models
    BLOOM = r" ?[^(\s|.,!?…。，、।۔،)]+"

    CHATGLM4 = (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    @classmethod
    def get(cls, name: str) -> str:
        """Get patterns by name (case-insensitive)."""
        try:
            return cls[name.upper()].value
        except KeyError:
            raise ValueError(
                f"Unknown pattern: {name!r}. "
                f"Valid patterns: {', '.join(pat.name for pat in cls)}"
            )


class RegexTokenizer(Tokenizer):
    """Tokenizer that splits text using regex patterns before applying BPE."""

    def __init__(self, pattern: str = TokenPattern.GPT4.value) -> None:
        super().__init__()
        self.pattern = pattern

    def train(self, text, vocab_size, verbose=False):
        """Train tokenizer by learning byte pair merges on regex-split chunks."""
        pass

    def encode(self, text):
        """Encode text into a sequence of tokens using regex splitting."""
        pass

    def decode(self, tokens: list[Token]):
        """Decode a sequence of tokens back into text."""
        pass

    @classmethod
    def from_pattern(cls, pattern: str | TokenPattern) -> "RegexTokenizer":
        """
        Create tokenizer from a pattern name or custom pattern.

        Args:
            pattern: Either a TokenPattern enum member, pattern name string ("gpt2", "gpt4"),
                    or a custom regex pattern string.

        Returns:
            A new RegexTokenizer instance.

        Example:
            .. code-block:: python
            tokenizer = RegexTokenizer.from_pattern("gpt4")
            tokenizer = RegexTokenizer.from_pattern(TokenPattern.GPT2)
            tokenizer = RegexTokenizer.from_pattern(r"custom pattern")

        """

        if isinstance(pattern, TokenPattern):
            pat = pattern
        elif isinstance(pattern, str):
            try:
                # check for valid pattern
                pat = TokenPattern.get(pattern)
            except ValueError:
                # not a valid pattern name
                # verify custom regex compiles
                if _is_valid_regex(pattern):
                    pat = pattern
                else:
                    raise ValueError(f"Invalid regex: {pattern}")
        return cls(pattern=pat)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "RegexTokenizer":
        """
        Load a pre-trained tokenizer from a .model file.

        Args:
            model_path: Path to the .model file.

        Returns:
            A loaded RegexTokenizer instance.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            ValueError: If the file format is invalid.

        Example:
            .. code-block:: python
            tokenizer = RegexTokenizer.from_pretrained("path/to/model.model")
        """
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not path.suffix == ".model":
            raise ValueError(
                f"Expected .model file, got {path.suffix}. Path: {model_path}"
            )

        tokenizer = cls()
        tokenizer.load(str(model_path))

        return tokenizer


def _is_valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False
