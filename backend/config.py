"""Fail-fast environment/config checks for security-sensitive services."""

from __future__ import annotations

import os
import secrets


class DatabaseConfigurationError(RuntimeError):
    """Raised when a relational workflow is used without its database."""


def environment() -> str:
    return os.getenv("ENVIRONMENT", "development").strip().lower()


def is_production() -> bool:
    return environment() == "production"


def _boolean_setting(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false, not {raw!r}.")


def mcp_enabled() -> bool:
    """MCP is an explicitly enabled privileged surface, never an implicit one."""
    return _boolean_setting("MCP_ENABLED", default=False)


def rate_limiting_enabled() -> bool:
    """Distributed request limits are mandatory by default in production."""
    configured = _boolean_setting("RATE_LIMITING_ENABLED", default=is_production())
    if is_production() and not configured:
        raise RuntimeError("RATE_LIMITING_ENABLED cannot be disabled in production.")
    return configured


def mcp_auth_mode() -> str:
    """Return the only currently supported MCP authentication mode.

    MCP reuses FlynnMed account JWTs so the tool layer can recover the real
    patient/clinician actor and apply the same consent boundary as the HTTP API.
    A shared API key cannot provide that identity and is intentionally rejected.
    """
    mode = os.getenv("MCP_AUTH_MODE", "jwt").strip().lower()
    if mode != "jwt":
        raise RuntimeError(
            "MCP_AUTH_MODE must be 'jwt'. Shared MCP API keys are not a safe "
            "authorization boundary for patient records."
        )
    return mode


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise DatabaseConfigurationError(
            "DATABASE_URL is required (Postgres) -- the legacy JSON-file/dual-backend "
            "local-dev store has been retired. Run `docker compose up -d db` for local dev."
        )
    return url


def psycopg_database_url(url: str | None = None) -> str:
    """Return a libpq-compatible URL from a SQLAlchemy PostgreSQL URL."""
    configured_url = (url if url is not None else database_url()).strip()
    if configured_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + configured_url.removeprefix("postgresql+psycopg://")
    return configured_url


_dev_jwt_secret: str | None = None


def jwt_secret_key() -> str:
    global _dev_jwt_secret
    secret = os.getenv("JWT_SECRET_KEY", "").strip()
    if secret:
        return secret

    if is_production():
        raise RuntimeError(
            "JWT_SECRET_KEY is required in production -- refusing to start with no secret "
            "or a shared hardcoded default."
        )

    if _dev_jwt_secret is None:
        _dev_jwt_secret = secrets.token_urlsafe(32)
        print(
            "WARNING: JWT_SECRET_KEY is not set. Using an ephemeral, randomly generated "
            "development secret -- every existing session will be invalidated on restart. "
            "Set JWT_SECRET_KEY in .env before deploying.",
        )
    return _dev_jwt_secret
