"""
Core Byte Pair Encoding (BPE) operations.
"""

from collections import Counter
from typing_extensions import deprecated

type Token = int
type TokenBytes = bytes
type TokenPair = tuple[Token, Token]
type Encoding = dict[TokenPair, Token]
type Vocabulary = dict[Token, TokenBytes]


@deprecated(
    "Reference implementation for documentation only. Use `RustBPETrainer()` for production."
)
def slow_bpe_merge(
    tokens: list[Token], target: TokenPair, new_tok: Token
) -> list[Token]:
    """
    Merge all occurrences of a target token pair into a single new token.

    Note: Merged tokens may represent partial UTF-8 sequences. Use errors="replace"
    when decoding to handle invalid sequences gracefully.
    """
    newtoks: list[Token] = []

    i = 0
    while i < len(tokens):
        # check if we can form a pair and it matches the target
        if (
            i < len(tokens) - 1
            and tokens[i] == target[0]
            and tokens[i + 1] == target[1]
        ):
            newtoks.append(new_tok)
            i += 2
        else:
            newtoks.append(tokens[i])
            i += 1

    return newtoks


@deprecated(
    "Reference implementation for documentation only. Use `RustBPETrainer()` for production."
)
def slow_bpe_merge_with_freq_update(
    tokens: list[Token],
    target: TokenPair,
    new_tok: Token,
    counter: Counter[TokenPair],
) -> list[Token]:
    """
    Merge target pair into new token and incrementally update frequency counter.

    This combines merge and frequency update in a single pass for efficiency.
    Decrements counts for pairs destroyed by the merge and increments counts
    for new pairs created.

    Naiive algorithm: O(n × M).
    Current implementation: O(n × M)

    where:

    - n = token sequence length
    - M = number of merges

    But current implementation is a bit faster per merge because it updates frequencies
    in the same pass, so we avoid an extra full scan to recompute counts each time.
    But overall training is still O(n × M) since we still do M passes over the sequence.

    Each merge requires a full O(n) scan of the token sequence.

    Note that faster algorithm(s) exist and is used in production grade
    tokenizers like tiktoken.
    See: https://github.com/karpathy/minbpe/issues/5#issue-2139918301.
    However, it is not worth the effort to implement this complex algorithm in
    Python. Maybe in future, I might refactor with a binding to this implemented in Rust.

    This implementation is adequate for training on small datasets.

    :param tokens: Current token sequence.
    :param target: The byte pair to merge.
    :param new_tok: The new token ID for the merged pair.
    :param counter: Frequency counter to update in-place.
    :return: New token sequence with merges applied.
    """
    if len(tokens) < 2:
        return tokens

    newtoks: list[Token] = []
    i = 0
    n = len(tokens)

    while i < n:
        # check if current position has the target pair
        if i < n - 1 and tokens[i] == target[0] and tokens[i + 1] == target[1]:
            # decrement count for the pair being merged
            counter[target] -= 1

            # handle left neighbor: decrement old pair, increment new pair
            if newtoks:
                left = newtoks[-1]
                old_left_pair = (left, target[0])
                counter[old_left_pair] -= 1
                new_left_pair = (left, new_tok)
                counter[new_left_pair] += 1

            # handle right neighbor: decrement old pair, increment new pair
            if i + 2 < n:
                right = tokens[i + 2]
                # only decrement if next position isn't also being merged
                if not (
                    i + 3 < n
                    and tokens[i + 2] == target[0]
                    and tokens[i + 3] == target[1]
                ):
                    old_right_pair = (target[1], right)
                    counter[old_right_pair] -= 1
                    new_right_pair = (new_tok, right)
                    counter[new_right_pair] += 1

            newtoks.append(new_tok)
            i += 2
        else:
            newtoks.append(tokens[i])
            i += 1

    # clean up zero counts to keep counter lean
    for pair in [k for k, v in counter.items() if v <= 0]:
        del counter[pair]

    return newtoks
