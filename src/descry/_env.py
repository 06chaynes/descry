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
    r"secret|token|password|passwd|passphrase|"
    r"credential|webhook|connection[_-]?string|authorization|"
    r"api[_-]?key|private[_-]?key|ssh[_-]?key|"
    # Short roots only as a whole word, so that KEYBOARD, COMPASS, PATH,
    # XAUTHORITY, GIT_AUTHOR_NAME and the shell's own PWD do not match.
    r"(?<![a-z])(?:key|pass|pw|pat|jwt|cookies?|o?auth)(?![a-z])|"
    r"[_-]pwd(?![a-z])|pgpass|sshpass|"
    r"aws[_-](?:access|secret|session|security)|"
    r"github[_-](?:token|pat|oauth)|"
    r"openai[_-]|anthropic[_-]|"
    # AZURE_{SUBSCRIPTION,TENANT,CLIENT}_ID are identifiers, not secrets.
    r"gcp[_-]|azure[_-](?:storage|subscription|tenant|client)[_-](?!id$)|"
    r"stripe[_-]|slack[_-]|"
    r"sentry[_-]|_dsn$|"
    r"(?:database|db|redis|mongo(?:db)?|postgres|pg|mysql|mariadb|sqlite|"
    r"amqp|rabbitmq|kafka|neo4j)[_-]?(?:url|uri)"
    r")"
)

# Match the pattern without being credentials: the path of the ssh-agent
# socket, which tools that fetch over ssh need, and a thread-pool switch.
_NOT_CREDENTIALS = frozenset({"SSH_AUTH_SOCK", "TOKENIZERS_PARALLELISM"})

# Hugging Face access tokens. Stripped like any other credential unless the
# caller asks for them with safe_env(keep=HF_TOKEN_VARS); only the process
# that downloads the embedding model should.
HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def safe_env(keep: tuple[str, ...] = ()) -> dict[str, str]:
    """Return os.environ minus variables matching credential patterns.

    `keep` names variables to pass through even though they match. Only
    names in HF_TOKEN_VARS may be kept.
    """
    not_keepable = set(keep) - set(HF_TOKEN_VARS)
    if not_keepable:
        raise ValueError(f"safe_env() cannot keep {sorted(not_keepable)}")
    return {
        k: v
        for k, v in os.environ.items()
        if k in keep or k in _NOT_CREDENTIALS or not _SECRET_PATTERNS.search(k)
    }


def env_flag(name: str) -> bool:
    """True if the variable is set to ``1``, ``true`` or ``yes`` (any case)."""
    return os.environ.get(name, "").lower() in ("1", "true", "yes")
