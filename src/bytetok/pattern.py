"""Built-in regex pattern definitions for supported model families."""

from enum import Enum
from typing import Literal

from .errors import PatternError

Pattern = Literal[
    "gpt2",
    "gpt4",
    "gpt4o",
    "llama3",
    "qwen2",
    "qwen35",
    "deepseek-coder",
    "deepseek-llm",
    "glm4",
    "kimi-k2",
]


class TokenPattern(str, Enum):
    """
    Pre-defined regex patterns for different tokenizer implementations.

    Sources:
    - GPT2 and GPT4: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    - QWEN35 (Qwen3.5 through Qwen3.8) and GLM4 (GLM-4 through GLM-5.3): tokenizer.json
      of the upstream Hugging Face models
    - KIMI_K2 (Moonshot AI Kimi K2 through Kimi K3): tokenization_kimi.py of
      https://huggingface.co/moonshotai/Kimi-K3
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

    QWEN35 = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|"
        r"\p{N}|"
        r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    # DeepSeek models
    DEEPSEEK_CODER = (
        r"[\r\n]|"
        r"\s?\p{L}+|"
        r"\s?\p{P}+|"
        r"[\U00004E00-\U00009FA5\U00000800-\U00004E00\U0000AC00-\U0000D7FF]+|"  # CJK characters
        r"\p{N}"
    )

    DEEPSEEK_LLM = (
        r"[\r\n]|"
        r"\s?[A-Za-z\U000000B5\U000000C0-\U000000D6\U000000D8-\U000000F6"
        r"\U000000F8-\U000001BA\U000001BC-\U000001BF\U000001C4-\U00000293"
        r"\U00000295-\U000002AF\U00000370-\U00000373\U00000376\U00000377"
        r"\U0000037B-\U0000037D\U0000037F\U00000386\U00000388-\U0000038A"
        r"\U0000038C\U0000038E-\U000003A1\U000003A3-\U000003F5"
        r"\U000003F7-\U00000481\U0000048A-\U0000052F\U00000531-\U00000556"
        r"\U000010A0-\U000010C5\U000013A0-\U000013F5\U000013F8-\U000013FD"
        r"\U00001C90-\U00001CBA\U00001CBD-\U00001CBF\U00001D00-\U00001D2B"
        r"\U00001D6B-\U00001D77\U00001D79-\U00001D9A\U00001E00-\U00001F15"
        r"\U00001F18-\U00001F1D\U00001F20-\U00001F45\U00001F48-\U00001F4D"
        r"\U00001F50-\U00001F57\U00001F59\U00001F5B\U00001F5D"
        r"\U00001F5F-\U00001F7D\U00001F80-\U00001FB4\U00001FB6-\U00001FBC"
        r"\U00001FBE\U00001FC2-\U00001FC4\U00001FC6-\U00001FCC"
        r"\U00001FD0-\U00001FD3\U00001FD6-\U00001FDB\U00001FE0-\U00001FEC"
        r"\U00001FF2-\U00001FF4\U00001FF6-\U00001FFC\U00002102\U00002107"
        r"\U0000210A-\U00002113\U00002115\U00002119-\U0000211D\U00002124"
        r"\U00002126\U00002128\U0000212A-\U0000212D\U0000212F-\U00002134"
        r"\U00002139\U0000213C-\U0000213F\U00002145-\U00002149\U0000214E"
        r"\U00002183\U00002184\U00002C00-\U00002C7B\U00002C7E-\U00002CE4"
        r"\U00002CEB-\U00002CEE\U00002CF2\U00002CF3\U0000A640-\U0000A66D"
        r"\U0000A680-\U0000A69B\U0000A722-\U0000A76F\U0000A771-\U0000A787"
        r"\U0000A78B-\U0000A78E\U0000AB70-\U0000ABBF\U0000FB00-\U0000FB06"
        r"\U0000FB13-\U0000FB17\U0000FF21-\U0000FF3A\U0000FF41-\U0000FF5A"
        r"\U00010400-\U0001044F\U000104B0-\U000104D3\U000104D8-\U000104FB"
        r"\U00010C80-\U00010CB2\U00010CC0-\U00010CF2\U000118A0-\U000118DF"
        r"\U0001E900-\U0001E943]+|"
        r"\s?[!-/:-~\U0000FF01-\U0000FF0F\U0000FF1A-\U0000FF5E"
        r"\U00002018-\U0000201F\U00003000-\U00003002]+|"
        r"\s+$|"
        r"[\U00004E00-\U00009FA5\U00000800-\U00004E00\U0000AC00-\U0000D7FF]+|"
        r"\p{N}+"
    )

    GLM4 = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    KIMI_K2 = (
        r"[\p{Han}]+|"
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
        r"\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    @classmethod
    def get(cls, name: str) -> str:
        """Return the pattern string by name (case-insensitive)."""
        try:
            return cls[name.upper().replace("-", "_")].value
        except KeyError:
            raise PatternError(
                f"unknown pattern: {name!r} "
                f"valid patterns: {', '.join(pat.name for pat in cls)}"
            )


def list_patterns() -> list[str]:
    """Return names of all available built-in tokenization patterns."""
    return [pat.name for pat in TokenPattern]


def get_pattern(name: Pattern) -> str:
    """Return a built-in pattern string by name."""
    return TokenPattern.get(name)


__all__ = ["Pattern", "TokenPattern", "list_patterns", "get_pattern"]
