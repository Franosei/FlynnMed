"""Read-only smoke check for FlynnMed's external health-information sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.clinical_trials import _request_studies  # noqa: E402
from backend.medication_checker import MedicationInteractionChecker  # noqa: E402
from backend.official_guidance import OfficialGuidanceEngine  # noqa: E402
from backend.pubmed_search import PubMedCentralSearcher  # noqa: E402


def _run_check(check: Callable[[], object]) -> dict:
    try:
        result = check()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    count = len(result) if isinstance(result, (list, dict)) else int(bool(result))
    return {"ok": bool(result), "count": count}


def check_sources() -> dict:
    guidance = OfficialGuidanceEngine()
    checks: dict[str, Callable[[], object]] = {
        "medlineplus": lambda: guidance._search_medlineplus("asthma", 1),
        "cdc": lambda: guidance._search_cdc("seasonal influenza", 1),
        "myhealthfinder": lambda: guidance._search_myhealthfinder(
            "colorectal cancer screening", 1
        ),
        "va_dod": lambda: guidance._search_va_dod("asthma management", 1),
        "openfda": lambda: MedicationInteractionChecker().resolve_medication(
            "acetaminophen"
        ),
        "pmc_permissive_oa": lambda: PubMedCentralSearcher().search_article_records(
            "asthma", 1
        ),
        "clinicaltrials_gov": lambda: _request_studies(
            {"query.cond": "asthma", "pageSize": "1", "format": "json"}
        ),
    }
    results = {name: _run_check(check) for name, check in checks.items()}
    return {
        "ok": all(item["ok"] for item in results.values()),
        "sources": results,
    }


def main() -> int:
    report = check_sources()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
