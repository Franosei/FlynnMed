"""
pytest imports conftest.py before collecting any test module in this
directory, so this guarantees DATABASE_URL (and other .env vars) are set
before any test file's module-level `_db_available()` skipif check runs --
previously that depended on incidental import order (whichever test module
happened to sort alphabetically after some other module that imports
backend.api/backend.summarizer/etc., which load_dotenv() as a side effect),
which silently skipped DB-gated tests whenever collection order shifted.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Keep unit tests deterministic across developer machines. The SQL integration
# suite imports SqlUserStore/SqlCarePlanStore directly, while ordinary backend
# tests exercise the legacy dispatch used by CI when DATA_BACKEND is unset.
# A developer's local .env must not silently switch those tests to SQL.
os.environ["DATA_BACKEND"] = "legacy"

# Unit and integration tests replace OpenAI-backed calls with fakes, but a
# number of the real service objects validate their configuration while they
# are constructed.  GitHub Actions intentionally has no production OpenAI
# secret, so provide a test-only placeholder after loading any local .env.
# Preserve a real key when a developer deliberately supplies one, while also
# handling CI configurations where the variable exists but is blank. This
# does not alter application behaviour outside pytest.
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test-only-not-a-real-key"
