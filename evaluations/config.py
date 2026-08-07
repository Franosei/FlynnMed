"""Environment-driven configuration for the evaluation harness.

Every value here has a sensible default and can be overridden by an
environment variable (loaded from `.env` via `python-dotenv`, matching the
rest of this codebase's convention). API keys are read only from the
environment -- never hardcoded, never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Bumped by hand when the harness's own grading prompt templates change
# materially (not FlynnMed's internal prompts, which this harness doesn't
# own or version -- see reporting.py's pipeline_version for that).
HEALTHBENCH_GRADING_PROMPT_VERSION = "healthbench-rubric-v3"
RAG_METRICS_PROMPT_VERSION = "rag-claim-audit-v4"

DATASET_URLS = {
    "healthbench": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "healthbench_hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "healthbench_consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}

EVAL_ROOT = Path(__file__).resolve().parent
DOWNLOADED_DIR = EVAL_ROOT / "datasets" / "downloaded"
NORMALIZED_DIR = EVAL_ROOT / "datasets" / "normalized"
PRIVATE_DIR = EVAL_ROOT / "datasets" / "private"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class EvalConfig:
    # Response-generation model FlynnMed's own pipeline will use for this run.
    # Pin evaluation generation independently from the application's model so
    # benchmark runs remain reproducible when OPENAI_MODEL changes.
    generator_model: str = field(
        default_factory=lambda: os.getenv("EVAL_GENERATOR_MODEL", "gpt-4o-mini")
    )
    # HealthBench rubric graders. A second call is skipped automatically when
    # primary and adjudicator resolve to the same model.
    primary_grader_model: str = field(
        default_factory=lambda: os.getenv("EVAL_PRIMARY_GRADER_MODEL", "gpt-5.6-luna")
    )
    adjudicator_model: str = field(
        default_factory=lambda: os.getenv("EVAL_ADJUDICATOR_MODEL", "gpt-5.6-luna")
    )
    rag_metrics_model: str = field(
        default_factory=lambda: os.getenv("EVAL_RAG_METRICS_MODEL", "gpt-5.6-luna")
    )
    evaluator_fallback_model: str = field(
        default_factory=lambda: os.getenv("EVAL_FALLBACK_MODEL", "gpt-5.6-luna")
    )

    generator_reasoning_effort: Optional[str] = field(
        default_factory=lambda: os.getenv("EVAL_GENERATOR_REASONING_EFFORT") or None
    )
    grader_reasoning_effort: Optional[str] = field(
        default_factory=lambda: os.getenv("EVAL_GRADER_REASONING_EFFORT") or None
    )
    adjudicator_reasoning_effort: Optional[str] = field(
        default_factory=lambda: os.getenv("EVAL_ADJUDICATOR_REASONING_EFFORT") or None
    )

    adjudication_threshold: float = field(
        default_factory=lambda: _env_float("EVAL_ADJUDICATION_THRESHOLD", 0.7)
    )
    sample_limit: Optional[int] = field(
        default_factory=lambda: _env_int("EVAL_SAMPLE_LIMIT")
    )
    output_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("EVAL_OUTPUT_PATH", str(EVAL_ROOT / "results"))
        )
    )

    max_retries: int = field(default_factory=lambda: _env_int("EVAL_MAX_RETRIES") or 5)
    request_timeout_seconds: float = field(
        default_factory=lambda: _env_float("EVAL_REQUEST_TIMEOUT_SECONDS", 120.0)
    )
    document_relevance_threshold: float = field(
        default_factory=lambda: _env_float("EVAL_DOCUMENT_RELEVANCE_THRESHOLD", 0.6)
    )
    minimum_reliable_sample_size: int = field(
        default_factory=lambda: _env_int("EVAL_MINIMUM_RELIABLE_SAMPLE_SIZE") or 5
    )
    consistency_repeats: int = field(
        default_factory=lambda: _env_int("EVAL_CONSISTENCY_REPEATS") or 0
    )
    regrade_workers: int = field(
        default_factory=lambda: max(1, _env_int("EVAL_REGRADE_WORKERS") or 4)
    )
    gold_answers_path: Optional[Path] = field(
        default_factory=lambda: (
            Path(raw)
            if (raw := os.getenv("EVAL_GOLD_ANSWERS_PATH", "").strip())
            else None
        )
    )

    def api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "No evaluation API key is configured. Set EVAL_API_KEY or "
                "OPENAI_API_KEY in the current shell or .env. Do not paste the key "
                "into logs, reports, source control, or chat."
            )
        return key

    def evaluation_api_key(self) -> str:
        """Allow evaluation judges to use a separately permissioned project key."""
        key = os.getenv("EVAL_API_KEY", "").strip() or self.api_key()
        return key

    def base_url(self) -> str:
        return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def load_config() -> EvalConfig:
    return EvalConfig()
