"""Environment switches read by more than one module."""

import os


def env_flag(name: str) -> bool:
    """True if the variable is set to ``1``, ``true`` or ``yes`` (any case)."""
    return os.environ.get(name, "").lower() in ("1", "true", "yes")
