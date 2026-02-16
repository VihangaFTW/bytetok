"""Reusable decorators for training and tokenizer utilities."""

import time
import functools
import logging
from typing import Callable

log = logging.getLogger(__name__)


def measure_time(func: Callable) -> Callable:
    """Log execution time for the wrapped callable."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Call ``func`` and always log elapsed time."""
        start = time.perf_counter()
        try:
            # call decorated func with its normal arguments
            result = func(*args, **kwargs)
            return result
        # log execution time even if the decorated function throws error
        finally:
            elapsed = time.perf_counter() - start
            log.info(f"training completed in {elapsed:.2f} s ({elapsed / 60:.2f} mins)")

    return wrapper
