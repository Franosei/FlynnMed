from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_image_is_multistage_and_uses_exec_form_startup():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM node:20-slim AS frontend-builder" in dockerfile
    assert "FROM python:3.13-slim AS runtime" in dockerfile
    assert 'CMD ["sh", "/app/scripts/start.sh"]' in dockerfile


def test_startup_migrates_before_starting_uvicorn():
    startup = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")

    assert startup.index("python -m alembic upgrade head") < startup.index("exec uvicorn")
    assert "--source legacy-postgres" in startup
    assert "DATA_BACKEND=legacy python -m backend.scripts.migrate_json_to_sql" in startup
    assert "export DATA_BACKEND=sql" in startup


def test_heavy_ml_stack_is_not_in_core_runtime_requirements():
    core = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    optional = (ROOT / "requirements-ml.txt").read_text(encoding="utf-8")

    assert "\ndetoxify\n" not in f"\n{core}"
    assert "detoxify" in optional


def test_mcp_runtime_is_pinned_to_the_compatible_major_version():
    core = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    mcp_server = (ROOT / "backend" / "mcp_server.py").read_text(encoding="utf-8")

    assert "mcp>=1.28,<2" in core
    assert "sys.exit(1)" not in mcp_server
    assert "raise RuntimeError(message)" in mcp_server


def test_docker_context_excludes_local_records_and_build_outputs():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "users.json" in ignored
    assert "data" in ignored
    assert "frontend/node_modules" in ignored
    assert "Python" in ignored
