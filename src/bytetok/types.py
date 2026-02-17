"""
Core types for tokenization.
"""

type Token = int
type TokenBytes = bytes
type TokenPair = tuple[Token, Token]
type Encoding = dict[TokenPair, Token]
type Vocabulary = dict[Token, TokenBytes]
