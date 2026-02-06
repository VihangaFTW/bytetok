from enum import Enum

from .errors import PatternError


class TokenPattern(str, Enum):
    """
    Pre-defined regex patterns for different tokenizer implementations.

    Sources:
    - GPT2 and GPT4: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    - Others: https://github.com/ggerganov/llama.cpp
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
        r"\p{N}|"  # single digits (different from LLAMA3)
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

    @classmethod
    def get(cls, name: str) -> str:
        """Get patterns by name (case-insensitive)."""
        try:
            return cls[name.upper().replace("-", "_")].value
        except KeyError:
            raise PatternError(
                f"Unknown pattern: {name!r}. "
                f"Valid patterns: {', '.join(pat.name for pat in cls)}"
            )
