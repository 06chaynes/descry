"""The failure every interface answers, in the contract's shape."""


def fail(reason: str, error: str, **more) -> dict:
    """`{ok: False, error, reason}`, and what ``more`` the reason carries:
    a shared name's `choices`."""
    return {"ok": False, "error": error, "reason": reason, **more}
