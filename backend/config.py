"""Fail-fast environment/config checks for the SQL + JWT backend (PR4+).

Deliberately not used by the legacy JSON-store/HMAC-token path (backend/
user_store.py, backend/api.py's `_token_secret`) -- those keep their own
dev-friendly fallbacks until the AUTH_BACKEND/DATA_BACKEND flags (PR4/PR5)
flip and this module becomes the only path.
"""

from __future__ import annotations

import os
import secrets


class DatabaseConfigurationError(RuntimeError):
    """Raised when a relational workflow is used without its database."""


def environment() -> str:
    return os.getenv("ENVIRONMENT", "development").strip().lower()


def is_production() -> bool:
    return environment() == "production"


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise DatabaseConfigurationError(
            "DATABASE_URL is required (Postgres) -- the legacy JSON-file/dual-backend "
            "local-dev store has been retired. Run `docker compose up -d db` for local dev."
        )
    return url


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
