"""
pytest imports conftest.py before collecting any test module in this
directory, so this guarantees DATABASE_URL (and other .env vars) are set
before any test file's module-level `_db_available()` skipif check runs --
previously that depended on incidental import order (whichever test module
happened to sort alphabetically after some other module that imports
backend.api/backend.summarizer/etc., which load_dotenv() as a side effect),
which silently skipped DB-gated tests whenever collection order shifted.
"""
from dotenv import load_dotenv

load_dotenv()
