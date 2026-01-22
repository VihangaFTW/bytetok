"""
Core Byte Pair Encoding (BPE) operations.
"""

type BytePair = tuple[int, int]
type Token = int


def bpe_freqs(tokens: list[Token]) -> dict[BytePair, Token]:
    """
    Compute the frequency of all consecutive token pairs in the token list.

    Args:
        tokens (list[Token]): List of tokens to analyze.

    Returns:
        dict[BytePair, Token]: Mapping of token pairs to their occurrence counts.
    """
    pairs: dict[BytePair, Token] = {}

    for i in range(len(tokens) - 1):
        tok0, tok1 = tokens[i], tokens[i + 1]
        pairs[(tok0, tok1)] = pairs.get((tok0, tok1), 0) + 1

    return pairs


def bpe_merge(tokens: list[Token], target: BytePair, new_tok: Token) -> list[Token]:
    """
    Merge all occurrences of a target token pair into a single new token.

    Note that some of the new tokens generated may be partial utf-8 sequences
    so they cannot be decoded into valid strings. When decoding, you should
    replace the invalid tokens with the unicode replacement character � for clarity.
    This can be done by setting errors = "replace" in python's `bytes.decode()` method.

    Args:
        tokens (list[Token]): Original list of tokens.
        target (BytePair): The consecutive pair of tokens to merge.
        new_tok (Token): The new token that replaces the target pair.

    Returns:
        list[Token]: New token list with all target pairs replaced by new_tok.
    """
    newtoks: list[Token] = []

    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == target[0] and tokens[i + 1] == target[1]:
            newtoks.append(new_tok)
            i += 2
        else:
            newtoks.append(tokens[i])
            i += 1

    return newtoks
