import asyncio
import json

import pytest

from backend.api import database_configuration_error
from backend.api import relational_database_error
from backend.config import DatabaseConfigurationError, database_url, psycopg_database_url
from sqlalchemy.exc import SQLAlchemyError


def test_database_url_raises_specific_configuration_error(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigurationError, match="DATABASE_URL is required"):
        database_url()


def test_psycopg_database_url_removes_sqlalchemy_driver_name():
    sqlalchemy_url = "postgresql+psycopg://user:password@example.invalid/flynnmed?sslmode=require"

    assert psycopg_database_url(sqlalchemy_url) == (
        "postgresql://user:password@example.invalid/flynnmed?sslmode=require"
    )


def test_psycopg_database_url_preserves_native_postgres_url():
    native_url = "postgresql://user:password@example.invalid/flynnmed"

    assert psycopg_database_url(native_url) == native_url


def test_database_configuration_error_is_a_clear_503_response():
    response = asyncio.run(
        database_configuration_error(
            None,
            DatabaseConfigurationError("missing"),
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert "secure patient database is not connected" in payload["detail"]
    assert "Internal Server Error" not in payload["detail"]


def test_relational_database_error_explains_missing_migrations():
    response = asyncio.run(
        relational_database_error(None, SQLAlchemyError("missing table"))
    )
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert "database is not ready" in payload["detail"]
    assert "migrations" in payload["detail"]
