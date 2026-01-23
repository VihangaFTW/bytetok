"""
Core Byte Pair Encoding (BPE) operations.
"""

type BytePair = tuple[int, int]
type Token = int


def bpe_freqs(tokens: list[Token]) -> dict[BytePair, Token]:
    """Compute the frequency of all consecutive token pairs."""
    pairs: dict[BytePair, Token] = {}

    for i in range(len(tokens) - 1):
        tok0, tok1 = tokens[i], tokens[i + 1]
        pairs[(tok0, tok1)] = pairs.get((tok0, tok1), 0) + 1

    return pairs


def bpe_merge(tokens: list[Token], target: BytePair, new_tok: Token) -> list[Token]:
    """
    Merge all occurrences of a target token pair into a single new token.

    Note: Merged tokens may represent partial UTF-8 sequences. Use errors="replace"
    when decoding to handle invalid sequences gracefully.
    """
    newtoks: list[Token] = []

    i = 0
    while i < len(tokens):
        # check if we can form a pair and it matches the target
        if i < len(tokens) - 1 and tokens[i] == target[0] and tokens[i + 1] == target[1]:
            newtoks.append(new_tok)
            i += 2
        else:
            newtoks.append(tokens[i])
            i += 1

    return newtoks
