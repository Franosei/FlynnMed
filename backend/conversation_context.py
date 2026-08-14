"""Build consistent conversation context for every incoming query."""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ConversationContext:
    prior_messages: List[Dict]
    previous_five: List[Dict]
    full_summary: str
    patient_statement_summary: str


def _clean_messages(messages: Optional[Iterable[Dict]]) -> List[Dict]:
    cleaned: List[Dict] = []
    for raw in messages or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "").strip().lower()
        content = str(raw.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned.append({**deepcopy(raw), "role": role, "content": content})
    return cleaned


def _without_current(messages: List[Dict], current_question: str) -> List[Dict]:
    prior = list(messages)
    current = re.sub(r"\s+", " ", current_question or "").strip().casefold()
    if not current or not prior:
        return prior
    last = prior[-1]
    last_text = re.sub(r"\s+", " ", str(last.get("content") or "")).strip().casefold()
    if last.get("role") == "user" and last_text == current:
        prior.pop()
    return prior


def _compact(text: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) <= limit:
        return value
    tail = min(80, max(30, limit // 4))
    return value[: max(1, limit - tail - 3)].rstrip() + "..." + value[-tail:].lstrip()


def _summary(messages: List[Dict], *, user_only: bool = False) -> str:
    selected = [item for item in messages if not user_only or item["role"] == "user"]
    if not selected:
        return "No earlier conversation."
    per_message = max(120, min(420, 12000 // len(selected)))
    lines = []
    for index, item in enumerate(selected, start=1):
        label = "Patient stated/asked" if item["role"] == "user" else "Assistant previously replied"
        lines.append(f"{index}. {label}: {_compact(item['content'], per_message)}")
    return "\n".join(lines)


def build_conversation_context(
    chat_history: Optional[List[Dict]],
    current_question: str,
    recent_limit: int = 5,
) -> ConversationContext:
    """Return prior context, excluding one matching current message at the end.

    The five recent messages remain verbatim. The summaries are extractive and
    role-labelled so assistant prose cannot silently become a patient fact.
    """
    prior = _without_current(_clean_messages(chat_history), current_question)
    return ConversationContext(
        prior_messages=prior,
        previous_five=deepcopy(prior[-max(1, recent_limit):]),
        full_summary=_summary(prior),
        patient_statement_summary=_summary(prior, user_only=True),
    )


def render_verbatim(messages: Optional[List[Dict]]) -> str:
    if not messages:
        return "No earlier conversation."
    return "\n\n".join(
        f"{str(item.get('role') or 'user').title()}:\n{str(item.get('content') or '').strip()}"
        for item in messages
        if str(item.get("content") or "").strip()
    ) or "No earlier conversation."
