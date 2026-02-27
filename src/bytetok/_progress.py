import os

_enabled: bool = True


def enable_progress() -> None:
    """Enable progress indicators for all bytetok operations."""
    global _enabled
    _enabled = True


def disable_progess() -> None:
    """Disable progress indicators for all bytetok operations."""
    global _enabled
    _enabled = False


def _is_enabled() -> bool:
    """Check if progress is enabled (respects env var override)."""
    if os.environ.get("BYTETOK_DISABLE_PROGRESS", "").strip() == "1":
        return False
    return _enabled
