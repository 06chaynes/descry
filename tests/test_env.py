"""Tests for descry._env — credential filtering of the subprocess environment."""

import pytest

from descry._env import HF_TOKEN_VARS, safe_env


def _filtered(monkeypatch, name: str) -> dict[str, str]:
    """Run safe_env() with `name` present in the environment."""
    monkeypatch.setenv(name, "value")
    return safe_env()


@pytest.mark.parametrize(
    "name",
    [
        "GITHUB_TOKEN",
        "GITHUB_PAT",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "OPENAI_API_KEY",
        "SSH_PRIVATE_KEY",
        "MY_PASSWORD",
        "SENTRY_DSN",
        "DATABASE_URL",
        "REDIS_URL",
        "MONGODB_URI",
        "AZURE_CLIENT_SECRET",
        "AZURE_STORAGE_ACCOUNT",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        # connection URLs
        "DB_URL",
        "PG_URL",
        "SQLITE_URL",
        "MARIADB_URL",
        "RABBITMQ_URL",
        "KAFKA_URL",
        "NEO4J_URL",
        "DATABASE_URL_2",
        "MONGO_URI_BACKUP",
        # password abbreviations
        "MYSQL_PWD",
        "PGPASS",
        "SMTP_PASS",
        "DB_PASS",
        "LDAP_BIND_PW",
        "DB_PASSWD",
        "SSHPASS",
        # personal access tokens
        "GH_PAT",
        "AZURE_DEVOPS_EXT_PAT",
        "GITLAB_CI_PAT",
        # generic keys
        "ENCRYPTION_KEY",
        "MASTER_KEY",
        "SIGNING_KEY",
        "APP_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        # JWTs, cookies and auth blobs
        "CI_JOB_JWT",
        "SESSION_COOKIE",
        "APP_COOKIES",
        "NPM_AUTH",
        "npm_config__auth",
        "DOCKER_AUTH_CONFIG",
        "COMPOSER_AUTH",
        "TWITCH_OAUTH",
        "HTTP_AUTHORIZATION",
        "AZURE_AUTH_TYPE",
        "OS_AUTH_URL",
        "SSH_AUTH_SOCK_BACKUP",
        # webhooks
        "DISCORD_WEBHOOK_URL",
        "TEAMS_WEBHOOK_URL",
        "SERVICEBUS_CONNECTION_STRING",
        # matching ignores case
        "db_url",
        "smtp_pass",
    ],
)
def test_credential_names_are_stripped(monkeypatch, name):
    assert name not in _filtered(monkeypatch, name)


@pytest.mark.parametrize(
    "name",
    [
        # tooling
        "PATH",
        "HOME",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "GITHUB_WORKSPACE",
        "AWS_REGION",
        "PYTHONPATH",
        "PWD",
        "OLDPWD",
        # identifiers
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        # "key", "pass" or "pat" inside an unrelated word
        "KEYBOARD_LAYOUT",
        "COMPASS_HOME",
        "PASSENGER_ROOT",
        "XDG_SESSION_PATH",
        "SSH_ASKPASS",
        "GIT_ASKPASS",
        # "auth" inside an unrelated word, and the ssh-agent socket
        "XAUTHORITY",
        "ICEAUTHORITY",
        "GIT_AUTHOR_NAME",
        "GOAUTH",
        "SSH_AUTH_SOCK",
        "TOKENIZERS_PARALLELISM",
        # URLs of package indexes and mirrors
        "PIP_INDEX_URL",
        "PUB_HOSTED_URL",
    ],
)
def test_ordinary_names_survive(monkeypatch, name):
    assert _filtered(monkeypatch, name)[name] == "value"


def test_hugging_face_token_vars():
    assert HF_TOKEN_VARS == ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def test_keep_passes_exactly_the_named_variables(monkeypatch):
    for name in (*HF_TOKEN_VARS, "GITHUB_TOKEN", "HF_TOKEN_BACKUP", "hf_token"):
        monkeypatch.setenv(name, "value")

    kept_one = safe_env(keep=("HF_TOKEN",))
    kept_both = safe_env(keep=HF_TOKEN_VARS)

    assert kept_one["HF_TOKEN"] == "value"
    assert "HUGGING_FACE_HUB_TOKEN" not in kept_one
    assert all(kept_both[name] == "value" for name in HF_TOKEN_VARS)
    for env in (kept_one, kept_both):
        assert not {"GITHUB_TOKEN", "HF_TOKEN_BACKUP", "hf_token"} & env.keys()


@pytest.mark.parametrize(
    "keep", [("GITHUB_TOKEN",), ("HF_TOKEN", "AWS_SECRET_ACCESS_KEY"), ("PATH",)]
)
def test_keep_rejects_names_outside_the_permitted_set(keep):
    with pytest.raises(ValueError, match="cannot keep"):
        safe_env(keep=keep)
