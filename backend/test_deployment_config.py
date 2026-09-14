from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_image_is_multistage_and_uses_exec_form_startup():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM node:20-slim AS frontend-builder" in dockerfile
    assert "FROM python:3.13-slim AS runtime" in dockerfile
    assert "USER flynnmed" in dockerfile
    assert "--chown=flynnmed:flynnmed" in dockerfile
    assert 'CMD ["sh", "/app/scripts/start.sh"]' in dockerfile


def test_startup_does_not_mutate_database_schema_or_accounts():
    startup = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")

    assert "alembic upgrade" not in startup
    assert "migrate_json_to_sql" not in startup
    assert "seed_demo_accounts" not in startup
    assert "exec uvicorn" in startup


def test_release_phase_owns_migrations_and_opt_in_data_changes():
    release = (ROOT / "scripts" / "release.sh").read_text(encoding="utf-8")
    example_env = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "python -m alembic upgrade head" in release
    assert 'MIGRATE_LEGACY_ACCOUNTS:-false' in release
    assert "DATA_BACKEND=legacy python -m backend.scripts.migrate_json_to_sql" in release
    assert 'SEED_DEMO_ACCOUNTS:-false' in release
    assert "SEED_DEMO_ACCOUNTS=false" in example_env


def test_heavy_ml_stack_is_not_in_core_runtime_requirements():
    core = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    optional = (ROOT / "requirements-ml.txt").read_text(encoding="utf-8")

    assert "\ndetoxify\n" not in f"\n{core}"
    assert "detoxify" in optional


def test_mcp_runtime_is_pinned_to_the_compatible_major_version():
    core = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    mcp_server = (ROOT / "backend" / "mcp_server.py").read_text(encoding="utf-8")

    assert "mcp>=1.28,<2" in core
    assert "stateless_http=True" in mcp_server
    assert "sys.exit(1)" not in mcp_server
    assert "raise RuntimeError(message)" in mcp_server


def test_mcp_is_opt_in_and_does_not_use_a_global_api_key():
    api = (ROOT / "backend" / "api.py").read_text(encoding="utf-8")
    example_env = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "if mcp_enabled():" in api
    assert "MCPAuthenticationMiddleware" in api
    assert "MCP_API_KEY" not in api
    assert "MCP_ENABLED=false" in example_env
    assert "MCP_AUTH_MODE=jwt" in example_env


def test_ci_boots_the_production_sql_and_jwt_path():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "DATA_BACKEND: sql" in workflow
    assert "ENVIRONMENT: production" in workflow
    assert "Boot production auth and storage path" in workflow
    assert "readiness()" in workflow


def test_docker_context_excludes_local_records_and_build_outputs():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "users.json" in ignored
    assert "data" in ignored
    assert "frontend/node_modules" in ignored
    assert "Python" in ignored
