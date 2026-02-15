"""ByteTok: Byte-level tokenization library."""

from ._models.base import Tokenizer
from ._models.basic import BasicTokenizer
from ._models.regex import RegexTokenizer
from .factory import (
    from_pretrained,
    get_tokenizer,
)
from .pattern import TokenPattern, get_pattern, list_patterns
from .parallel import list_parallel_modes
from .strategy import (
    AllowAllStrategy,
    AllowCustomStrategy,
    AllowNoneRaiseStrategy,
    AllowNoneStrategy,
    SpecialTokenStrategy,
    get_strategy,
    list_strategies,
)

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bytetok")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = [
    "Tokenizer",
    "BasicTokenizer",
    "RegexTokenizer",
    "TokenPattern",
    "SpecialTokenStrategy",
    "AllowAllStrategy",
    "AllowNoneStrategy",
    "AllowNoneRaiseStrategy",
    "AllowCustomStrategy",
    "get_tokenizer",
    "get_strategy",
    "get_pattern",
    "from_pretrained",
    "list_patterns",
    "list_parallel_modes",
    "list_strategies",
]
