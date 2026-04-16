"""Subprocess environment sanitization.

Filters environment variables matching known credential patterns before
passing env to subprocess calls (rust-analyzer, scip-typescript, git,
ast-grep, descry.generate). Keeps tooling vars (RUSTUP_*, CARGO_*,
GITHUB_WORKSPACE, AWS_REGION, etc.) intact.
"""

import os
import re


_SECRET_PATTERNS = re.compile(
    r"(?i)("
    r"secret|token|password|passphrase|credential|"
    r"api[_-]?key|private[_-]?key|ssh[_-]?key|"
    r"aws[_-](?:access|secret|session|security)|"
    r"github[_-](?:token|pat|oauth)|"
    r"openai[_-]|anthropic[_-]|"
    r"gcp[_-]|azure[_-](?:storage|subscription|tenant|client)[_-]|"
    r"stripe[_-]|slack[_-]|"
    r"sentry[_-]|_dsn$|"
    r"(?:database|redis|mongodb?|postgres|amqp|mysql)[_-]?(?:url|uri)$"
    r")"
)


def safe_env() -> dict[str, str]:
    """Return os.environ minus variables matching credential patterns."""
    return {k: v for k, v in os.environ.items() if not _SECRET_PATTERNS.search(k)}
