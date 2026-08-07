"""
Clinical trial search and matching.

Three-phase pipeline:
  1. Extract  -- LLM reads the full patient context and returns individual
                medical condition/symptom terms and drug names as lists.
  2. Search   -- One ClinicalTrials.gov API call per term (query.cond /
                query.intr + query.locn).  Results merged by NCT ID; each
                trial records which patient conditions found it.
  3. Score    -- Deterministic pre-scoring (coverage + location) narrows the
                candidate set, then an LLM is used to assess condition
                alignment and age/sex eligibility with clinical accuracy.

Total score components (sum to 100):
  - LLM condition alignment  : 0-50  (how well the trial addresses the patient's conditions)
  - Multi-condition coverage  : 0-30  (how many separate condition searches found this trial)
  - Location                  : 0-20  (trial site in the patient's selected country)

Reported separately (NOT in total score):
  - Age eligibility   : INCLUDED / EXCLUDED / UNKNOWN + reason
  - Sex eligibility   : INCLUDED / EXCLUDED / UNKNOWN + reason
  - Contact details   : displayed for action only

No raw patient text is ever sent to the ClinicalTrials.gov API.
No clinical content is hardcoded.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import requests
from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:
    from backend.query_expander import QueryExpander

from backend.user_store import compute_current_age
from backend.clinical_context_guard import (
    adjudicate_patient_context,
    decision_from_dict,
    source_matches_context,
)
from backend.utils import render_vital_for_prompt

load_dotenv()

API_BASE_URL = "https://clinicaltrials.gov/api/v2"
RECRUITING_STATUS = "RECRUITING"

STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because",
    "been", "before", "being", "between", "but", "can", "clinical",
    "condition", "could", "does", "from", "have", "health", "into", "list",
    "medicine", "medication", "not", "patient", "record", "saved", "should",
    "study", "symptom", "that", "the", "their", "there", "this", "trial",
    "used", "using", "with", "your",
}

_LLM_BATCH_SIZE = 5  # trials scored per LLM call
_PRE_FILTER_N = 20   # top-N trials by fast score sent to LLM scoring


@dataclass
class TrialSearchProfile:
    conditions: List[str]     # raw conditions for LLM scoring context
    symptoms: List[str]       # raw symptoms for LLM scoring context
    medications: List[str]    # raw medication names for LLM scoring context
    age: Optional[int]
    biological_sex: str
    raw_context: str          # full patient narrative → LLM extracts search terms
    clinical_context: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clean(value: object) -> str:
    return str(value or "").strip()


def _clean_markup(value: object) -> str:
    text = _clean(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _unique(values: Iterable[str]) -> List[str]:
    seen: set = set()
    result = []
    for value in values:
        cleaned = _clean(value)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            result.append(cleaned)
    return result


def _tokens(text: str) -> set:
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower())
        if t not in STOPWORDS
    }


def _age_from_api(value: str) -> Optional[int]:
    if not value or value.upper() == "N/A":
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", value.lower())
    if not m:
        return None
    amount, unit = float(m.group(1)), m.group(2)
    if unit == "year":
        return int(amount)
    if unit == "month":
        return int(amount / 12)
    return 0


def _module(study: Dict, name: str) -> Dict:
    return study.get("protocolSection", {}).get(name, {}) or {}


def _study_text(study: Dict) -> str:
    ident = _module(study, "identificationModule")
    cond = _module(study, "conditionsModule")
    desc = _module(study, "descriptionModule")
    elig = _module(study, "eligibilityModule")
    arms = _module(study, "armsInterventionsModule")
    intr_text = " ".join(
        " ".join(_clean(p) for p in (i.get("name"), i.get("type"), i.get("description")))
        for i in (arms.get("interventions") or [])
    )
    return " ".join(
        _clean_markup(part) for part in (
            ident.get("briefTitle"), ident.get("officialTitle"),
            " ".join(cond.get("conditions", []) or []),
            " ".join(cond.get("keywords", []) or []),
            desc.get("briefSummary"),
            elig.get("eligibilityCriteria"),
            intr_text,
        ) if part
    )


def _extract_locations(study: Dict) -> List[Dict]:
    cl = _module(study, "contactsLocationsModule")
    result = []
    for loc in cl.get("locations", []) or []:
        result.append({
            "facility": _clean(loc.get("facility")),
            "city": _clean(loc.get("city")),
            "state": _clean(loc.get("state")),
            "country": _clean(loc.get("country")),
            "status": _clean(loc.get("status")),
            "contacts": [
                {"name": _clean(c.get("name")), "role": _clean(c.get("role")),
                 "phone": _clean(c.get("phone")), "email": _clean(c.get("email"))}
                for c in (loc.get("contacts") or [])
            ],
        })
    return result


def _extract_contacts(study: Dict, locations: List[Dict]) -> List[Dict]:
    cl = _module(study, "contactsLocationsModule")
    contacts = []
    for c in cl.get("centralContacts", []) or []:
        contacts.append({"name": _clean(c.get("name")), "role": _clean(c.get("role")),
                         "phone": _clean(c.get("phone")), "email": _clean(c.get("email")),
                         "source": "Central contact"})
    for loc in locations:
        for c in loc.get("contacts", []):
            enriched = dict(c)
            enriched["source"] = location_label(loc)
            contacts.append(enriched)
    return [c for c in contacts if c.get("name") or c.get("phone") or c.get("email")]


def _extract_officials(study: Dict) -> List[Dict]:
    cl = _module(study, "contactsLocationsModule")
    officials = []
    for o in cl.get("overallOfficials", []) or []:
        officials.append({"name": _clean(o.get("name")), "role": _clean(o.get("role")),
                          "affiliation": _clean(o.get("affiliation"))})
    return [o for o in officials if o.get("name") or o.get("affiliation")]


def location_label(location: Dict) -> str:
    pieces = [location.get("facility", ""), location.get("city", ""),
              location.get("state", ""), location.get("country", "")]
    return ", ".join(_unique(pieces)) or "Location not listed"


# ---------------------------------------------------------------------------
# Phase 1: LLM extraction of individual search terms from patient context
# ---------------------------------------------------------------------------

def _llm_extract_search_terms(raw_context: str) -> Dict[str, List[str]]:
    """
    Ask the LLM to return individual medical condition/symptom terms and drug
    names as separate lists for the per-term API search phase.
    """
    if not raw_context.strip():
        return {"conditions": [], "medications": []}

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"conditions": [], "medications": []}

    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a clinical research coordinator preparing a multi-term ClinicalTrials.gov search.\n\n"
            "Read the patient data and extract every distinct medical condition, symptom, sign, or "
            "diagnosis -- as a JSON list of short individual terms. Also extract drug/medication names.\n\n"
            'Return ONLY: {"conditions": ["term1", ...], "medications": ["drug1", ...]}\n\n'
            "Rules:\n"
            "- Each condition entry: real medical condition/symptom/sign/diagnosis, 1-5 words.\n"
            "- List individually -- do NOT concatenate multiple conditions into one string.\n"
            "- Include the most specific terms you can (e.g. 'right iliac fossa pain', not just 'pain').\n"
            "- Do NOT include: general triage, routine, self-care, GP, normal findings, names, demographics.\n"
            "- Max 8 condition terms, max 5 medication names.\n\n"
            "- Do not extract a term from a different specialty than the confirmed clinical context. "
            "If the context says a peak urinary flow/Qmax result, never emit asthma, lung, respiratory, "
            "or peak-expiratory-flow terms.\n\n"
            f"Patient data:\n{raw_context[:3000]}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0, max_completion_tokens=250,
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        conditions = [_clean(t) for t in (parsed.get("conditions") or [])
                      if _clean(t) and len(_clean(t)) <= 80][:8]
        medications = [_clean(t) for t in (parsed.get("medications") or [])
                       if _clean(t) and len(_clean(t)) <= 60][:5]
        return {"conditions": conditions, "medications": medications}
    except Exception as exc:
        print(f"[clinical_trials] Term extraction failed: {exc}")
        return {"conditions": [], "medications": []}


# ---------------------------------------------------------------------------
# Phase 2: per-term API search helpers
# ---------------------------------------------------------------------------

def _request_studies(params: Dict[str, str], timeout: int = 20) -> List[Dict]:
    resp = requests.get(f"{API_BASE_URL}/studies", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("studies", []) or []


def _search_one_term(term: str, param_key: str, location_query: str,
                     page_size: int = 25) -> List[Dict]:
    params: Dict[str, str] = {
        "format": "json", "pageSize": str(page_size),
        "countTotal": "true", "filter.overallStatus": RECRUITING_STATUS,
        param_key: term,
    }
    if location_query.strip():
        params["query.locn"] = location_query.strip()

    studies = _request_studies(params)

    if len(studies) < 5 and location_query.strip():
        global_params = {k: v for k, v in params.items() if k != "query.locn"}
        global_params["pageSize"] = str(page_size)
        known = {_module(s, "identificationModule").get("nctId") for s in studies}
        for s in _request_studies(global_params):
            if _module(s, "identificationModule").get("nctId") not in known:
                studies.append(s)

    return studies


# ---------------------------------------------------------------------------
# Phase 3a: deterministic pre-scoring (fast, no LLM)
# ---------------------------------------------------------------------------

def _location_score(locations: List[Dict], location_query: str) -> int:
    if not location_query:
        return 10
    if not locations:
        return 0
    query_tokens = _tokens(location_query)
    if not query_tokens:
        return 10
    best_overlap, best_contains = 0, False
    lowered = location_query.lower()
    for loc in locations:
        label = location_label(loc).lower()
        best_overlap = max(best_overlap, len(query_tokens & _tokens(label)))
        if lowered in label:
            best_contains = True
    if best_contains:
        return 20
    if best_overlap:
        return min(19, 10 + best_overlap * 3)
    return 3


def _coverage_score(found_for: List[str], total_terms: int) -> int:
    if not total_terms:
        return 0
    return min(30, len(found_for) * max(1, 30 // total_terms))


def _fast_score(found_for: List[str], total_terms: int,
                locations: List[Dict], location_query: str) -> int:
    return _coverage_score(found_for, total_terms) + _location_score(locations, location_query)


# ---------------------------------------------------------------------------
# Phase 3b: deterministic age & sex eligibility
# ---------------------------------------------------------------------------

def _age_sex_eligibility(study: Dict, profile: "TrialSearchProfile") -> Dict:
    """
    Return a clear INCLUDED / EXCLUDED / UNKNOWN verdict for age and sex
    independently, with the specific reason so the user can act on it.
    """
    elig = _module(study, "eligibilityModule")

    # --- Sex ---
    trial_sex = _clean(elig.get("sex")).upper()
    psex = profile.biological_sex.strip().upper()

    if not trial_sex or trial_sex == "ALL":
        sex_status, sex_reason = "included", "Trial accepts all sexes."
    elif not psex or psex == "OTHER":
        sex_status, sex_reason = "unknown", "Biological sex not recorded in your profile."
    elif trial_sex.startswith(psex[:1]):
        sex_status = "included"
        sex_reason = f"Trial includes {trial_sex.title()}; your profile is {profile.biological_sex}."
    else:
        sex_status = "excluded"
        sex_reason = (
            f"Trial restricts to {trial_sex.title()} participants; "
            f"your profile is {profile.biological_sex}."
        )

    # --- Age ---
    min_age = _age_from_api(_clean(elig.get("minimumAge")))
    max_age = _age_from_api(_clean(elig.get("maximumAge")))

    if profile.age is None:
        age_status = "unknown"
        age_reason = "Age not recorded in your profile."
    elif min_age is None and max_age is None:
        age_status = "included"
        age_reason = "No age restriction is stated for this trial."
    elif min_age is not None and profile.age < min_age:
        age_status = "excluded"
        age_reason = (
            f"Minimum age for this trial is {elig.get('minimumAge')}; "
            f"your profile age is {profile.age} years."
        )
    elif max_age is not None and profile.age > max_age:
        age_status = "excluded"
        age_reason = (
            f"Maximum age for this trial is {elig.get('maximumAge')}; "
            f"your profile age is {profile.age} years."
        )
    else:
        min_str = elig.get("minimumAge") or "no minimum"
        max_str = elig.get("maximumAge") or "no maximum"
        age_status = "included"
        age_reason = (
            f"Your age ({profile.age} years) is within the stated range "
            f"({min_str} – {max_str})."
        )

    return {
        "age_status": age_status,
        "age_reason": age_reason,
        "sex_status": sex_status,
        "sex_reason": sex_reason,
    }


# ---------------------------------------------------------------------------
# Phase 3c: LLM condition alignment scoring (batched)
# ---------------------------------------------------------------------------

def _build_patient_summary(profile: "TrialSearchProfile") -> str:
    parts = []
    if profile.conditions:
        parts.append("Conditions/diagnoses: " + "; ".join(profile.conditions[:8]))
    if profile.symptoms:
        parts.append("Symptoms: " + "; ".join(profile.symptoms[:8]))
    if profile.medications:
        parts.append("Current medications: " + "; ".join(profile.medications[:6]))
    if profile.age is not None:
        parts.append(f"Age: {profile.age} years")
    if profile.biological_sex:
        parts.append(f"Biological sex: {profile.biological_sex}")
    if profile.raw_context:
        # Bare condition/symptom labels above can be ambiguous across specialties (e.g.
        # "clearance", "peak flow", "block") -- the full narrative is what actually
        # confirms which specialty/meaning applies, and without it the scoring LLM only
        # sees the bare ambiguous label and has no way to detect a mismatch.
        parts.append("Full clinical context (use to resolve ambiguous terms above): " + profile.raw_context[:800])
    return "\n".join(parts) if parts else "No specific clinical data recorded."


def _llm_batch_condition_match(
    trial_stubs: List[Dict],
    profile: "TrialSearchProfile",
) -> List[Dict]:
    """
    Score a batch of trials in a single LLM call.
    Each stub must have keys: index, title, conditions, summary, eligibility.
    Returns one result dict per stub in the same order.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    empty = [{"alignment_score": 0, "match_level": "unknown",
              "aligned_conditions": [], "exclusion_risks": [], "reasoning": "",
              "specialty_mismatch": False}
             for _ in trial_stubs]
    if not api_key or not trial_stubs:
        return empty

    patient_text = _build_patient_summary(profile)

    trials_block = ""
    for stub in trial_stubs:
        trials_block += (
            f"\n--- TRIAL {stub['index']} ---\n"
            f"Title: {stub['title']}\n"
            f"Conditions studied: {stub['conditions']}\n"
            f"Summary: {stub['summary'][:350]}\n"
            f"Eligibility: {stub['eligibility'][:600]}\n"
        )

    prompt = (
        "You are a clinical research coordinator assessing whether a patient may be a candidate "
        "for each of the clinical trials listed below.\n\n"
        "For EACH trial, return a JSON object inside a JSON array with these fields:\n"
        '  "index": <same integer as the TRIAL number above>,\n'
        '  "alignment_score": <integer 0-50>,\n'
        '  "match_level": "high" | "medium" | "low" | "not_relevant",\n'
        '  "aligned_conditions": [<patient conditions/symptoms that match trial inclusion criteria>],\n'
        '  "exclusion_risks": [<patient factors that may trigger exclusion criteria>],\n'
        '  "reasoning": "<1–2 sentence clinical assessment>"\n\n'
        "Scoring guide:\n"
        "  40–50: Strong -- patient's primary condition directly addressed by trial purpose.\n"
        "  25–39: Moderate -- related condition or meaningful partial overlap.\n"
        "  10–24: Weak -- tangential or indirect connection.\n"
        "   0–9:  Not relevant -- patient profile does not match trial purpose.\n\n"
        "Base the score on clinical relevance, not keyword overlap. "
        "Check both inclusion AND exclusion criteria carefully.\n\n"
        '  "specialty_mismatch": true/false -- set true if this trial\'s relevance to the '
        "patient rests on interpreting an ambiguous patient-profile term (a condition, "
        "symptom, measurement, or medication reason with a different meaning in another "
        "clinical specialty) in a specific way, and the patient profile does not confirm "
        "that specific meaning. This is a hard signal: when true, this trial must not be "
        "shown to the patient as a match regardless of the numeric score, because it may "
        "reflect the wrong meaning of the term entirely rather than genuine relevance.\n"
        "  IMPORTANT: if the patient profile does not explicitly state which specialty/meaning "
        "an ambiguous term refers to, you MUST treat it as unresolved -- do not silently pick "
        "whichever interpretation happens to make this trial relevant. In that situation set "
        "specialty_mismatch to true and cap match_level at \"low\", even if the trial would "
        "otherwise look like a strong match under the guessed interpretation.\n\n"
        f'Return ONLY a JSON object: {{"results": [...]}}\n\n'
        f"Patient profile:\n{patient_text}\n"
        f"\nTrials to assess:{trials_block}"
    )

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0, max_completion_tokens=800,
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        results_raw = parsed.get("results", []) or []

        # Map by index
        by_index: Dict[int, Dict] = {}
        for r in results_raw:
            idx = r.get("index")
            if idx is not None:
                by_index[int(idx)] = {
                    "alignment_score": max(0, min(50, int(r.get("alignment_score", 0)))),
                    "match_level": str(r.get("match_level", "unknown")),
                    "aligned_conditions": list(r.get("aligned_conditions", []) or []),
                    "exclusion_risks": list(r.get("exclusion_risks", []) or []),
                    "reasoning": _clean(r.get("reasoning", "")),
                    "specialty_mismatch": bool(r.get("specialty_mismatch", False)),
                }

        return [
            by_index.get(stub["index"], empty[i])
            for i, stub in enumerate(trial_stubs)
        ]

    except Exception as exc:
        print(f"[clinical_trials] LLM batch scoring failed: {exc}")
        return empty


def _llm_score_candidates(
    candidates: List[Dict],
    profile: "TrialSearchProfile",
) -> None:
    """
    Enrich each candidate dict in-place with LLM condition alignment scores
    and age/sex eligibility verdicts.
    candidates must have a "_study" key pointing to the raw study dict.
    """
    # Build stubs for batch LLM call
    stubs = []
    for i, cand in enumerate(candidates):
        study = cand["_study"]
        ident = _module(study, "identificationModule")
        cond_mod = _module(study, "conditionsModule")
        desc = _module(study, "descriptionModule")
        elig = _module(study, "eligibilityModule")
        stubs.append({
            "index": i,
            "title": _clean(ident.get("briefTitle") or ident.get("officialTitle")),
            "conditions": ", ".join(cond_mod.get("conditions", []) or [])[:200],
            "summary": _clean_markup(desc.get("briefSummary", ""))[:350],
            "eligibility": _clean_markup(elig.get("eligibilityCriteria", ""))[:600],
        })

    # Score in batches
    all_llm: List[Dict] = []
    for batch_start in range(0, len(stubs), _LLM_BATCH_SIZE):
        batch = stubs[batch_start: batch_start + _LLM_BATCH_SIZE]
        all_llm.extend(_llm_batch_condition_match(batch, profile))
        if batch_start + _LLM_BATCH_SIZE < len(stubs):
            time.sleep(0.2)

    # Enrich each candidate
    for i, cand in enumerate(candidates):
        study = cand["_study"]
        llm = all_llm[i]
        age_sex = _age_sex_eligibility(study, profile)

        cov_score = cand["coverage_score"]
        loc_score = cand["location_score"]
        ali_score = llm["alignment_score"]
        total = min(100, cov_score + loc_score + ali_score)

        cand.update({
            "alignment_score": ali_score,
            "match_level": llm["match_level"],
            "aligned_conditions": llm["aligned_conditions"],
            "exclusion_risks": llm["exclusion_risks"],
            "llm_reasoning": llm["reasoning"],
            "specialty_mismatch": llm.get("specialty_mismatch", False),
            "age_status": age_sex["age_status"],
            "age_reason": age_sex["age_reason"],
            "sex_status": age_sex["sex_status"],
            "sex_reason": age_sex["sex_reason"],
            "match_score": total,
        })


# ---------------------------------------------------------------------------
# Full trial record builder
# ---------------------------------------------------------------------------

def _build_trial_record(study: Dict, found_for: List[str], total_terms: int,
                        location_query: str) -> Dict:
    ident = _module(study, "identificationModule")
    status_mod = _module(study, "statusModule")
    cond_mod = _module(study, "conditionsModule")
    desc = _module(study, "descriptionModule")
    design = _module(study, "designModule")
    elig = _module(study, "eligibilityModule")
    arms = _module(study, "armsInterventionsModule")

    locations = _extract_locations(study)
    contacts = _extract_contacts(study, locations)
    officials = _extract_officials(study)

    # Best location for display
    q_tokens = _tokens(location_query)
    best_loc = max(locations, key=lambda location: len(q_tokens & _tokens(location_label(location).lower()))) \
        if locations and q_tokens else (locations[0] if locations else {})

    cov_score = _coverage_score(found_for, total_terms)
    loc_score = _location_score(locations, location_query)

    interventions = _unique([
        _clean(i.get("name"))
        for i in (arms.get("interventions") or [])
        if _clean(i.get("name"))
    ])[:6]

    nct_id = ident.get("nctId", "")
    return {
        "_study": study,               # kept for LLM scoring pass; removed before returning
        "nct_id": nct_id,
        "title": _clean(ident.get("briefTitle") or ident.get("officialTitle")),
        "official_title": _clean(ident.get("officialTitle")),
        "status": _clean(status_mod.get("overallStatus")),
        "conditions": cond_mod.get("conditions", []) or [],
        "interventions": interventions,
        "phase": ", ".join(design.get("phases", []) or []),
        "study_type": _clean(design.get("studyType")),
        "summary": _clean_markup(desc.get("briefSummary")),
        "eligibility": _clean_markup(elig.get("eligibilityCriteria")),
        "sex_criterion": _clean(elig.get("sex")),
        "minimum_age": _clean(elig.get("minimumAge")),
        "maximum_age": _clean(elig.get("maximumAge")),
        "locations": locations,
        "best_location": best_loc,
        "contacts": contacts,
        "officials": officials,
        "found_for_conditions": found_for,
        "condition_coverage": len(found_for),
        "total_conditions_searched": total_terms,
        "coverage_score": cov_score,
        "location_score": loc_score,
        "alignment_score": 0,          # filled by LLM pass
        "match_level": "pending",
        "aligned_conditions": [],
        "exclusion_risks": [],
        "llm_reasoning": "",
        "age_status": "unknown",
        "age_reason": "",
        "sex_status": "unknown",
        "sex_reason": "",
        "match_score": cov_score + loc_score,   # updated after LLM
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
    }


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_trial_search_profile(
    profile: Dict,
    memory: Dict,
    symptom_logs: List[Dict],
    medications: List[Dict],
    allergies: List[Dict],
    conditions: List[Dict],
    vitals: List[Dict],
    triage_summaries: List[Dict],
    document_summaries: Optional[List[Dict]] = None,
) -> "TrialSearchProfile":
    clinical_context = adjudicate_patient_context(
        conditions=conditions,
        medications=medications,
        vitals=vitals,
        allergies=allergies,
        triage_summaries=triage_summaries,
        document_summaries=document_summaries or [],
        longitudinal_memory=_clean(memory.get("summary", "")),
    )
    condition_terms = _unique(
        [c.get("name", "") for c in conditions if c.get("name")]
        + [m.get("reason", "") for m in medications if m.get("reason")]
    )
    symptoms = _unique(e.get("symptom", "") for e in symptom_logs)
    medication_names = _unique(m.get("name", "") for m in medications)

    parts: List[str] = []
    memory_summary = _clean(memory.get("summary", ""))
    if memory_summary:
        parts.append(f"Patient longitudinal background (not a diagnosis source):\n{memory_summary[:1200]}")

    if condition_terms:
        parts.append("Recorded conditions/history: " + "; ".join(condition_terms[:10]))

    for s in triage_summaries[:5]:
        escalation = s.get("escalation_triggers") or []
        if escalation:
            parts.append("Escalation triggers: " + "; ".join(str(e) for e in escalation[:5]))

    for e in symptom_logs[:15]:
        sym = _clean(e.get("symptom", ""))
        if sym:
            parts.append(f"Logged symptom: {sym}")

    for m in medications[:10]:
        name = _clean(m.get("name", ""))
        reason = _clean(m.get("reason", ""))
        if name:
            parts.append(f"Medication: {name}" + (f" (for {reason})" if reason else ""))

    vital_lines = [render_vital_for_prompt(v) for v in (vitals or []) if isinstance(v, dict)]
    if vital_lines:
        parts.append("Structured measurements: " + "; ".join(vital_lines[:20]))
    if clinical_context.query_terms:
        parts.append("Confirmed clinical context: " + "; ".join(clinical_context.query_terms))

    decision_dict = clinical_context.as_dict()
    decision_obj = decision_from_dict(decision_dict)
    if decision_obj and decision_obj.domain:
        condition_terms = [
            term for term in condition_terms
            if source_matches_context("", term, decision_obj)
        ]

    return TrialSearchProfile(
        conditions=condition_terms,
        symptoms=symptoms,
        medications=medication_names,
        age=compute_current_age(profile.get("date_of_birth", "")),
        biological_sex=_clean(profile.get("biological_sex")),
        raw_context="\n".join(parts),
        clinical_context=decision_dict,
    )


def _exclude_not_relevant(candidates: List[Dict]) -> List[Dict]:
    """
    Drops trials the LLM scoring pass (_llm_batch_condition_match) explicitly
    judged not relevant, and trials flagged specialty_mismatch=True -- a hard,
    explicit signal checked independently of the numeric score, so a trial whose
    apparent relevance rests on the wrong meaning of an ambiguous term (e.g. a
    measurement or condition name that reads differently in another specialty)
    can't surface just because it was scored "low"/"medium" rather than
    "not_relevant". Without this, a mismatched trial with a high coverage/
    location score could still surface in the returned results.
    """
    return [
        c for c in candidates
        if c.get("match_level") != "not_relevant" and not c.get("specialty_mismatch")
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_matching_trials(
    profile: TrialSearchProfile,
    location_query: str,
    max_results: int = 10,
    query_expander: Optional["QueryExpander"] = None,
) -> Dict:
    """
    Phase 1 -- Extract: LLM reads patient context → individual medical terms.
    Phase 2 -- Search:  one API call per term, results merged by NCT ID.
    Phase 3 -- Score:
        a) Fast pre-score all trials (coverage + location, no LLM).
        b) Take top _PRE_FILTER_N candidates.
        c) LLM scores condition alignment (batched) + deterministic age/sex eligibility.
        d) Final ranking: alignment + coverage + location (contact NOT in total).

    Phase 2 is a serial per-term loop with an explicit rate-limit sleep between
    ClinicalTrials.gov calls, so unlike NHS/PubMed retrieval, adding condition
    terms here has a direct, non-parallelizable latency cost -- each extra term
    is one more full API round trip plus the sleep. When query_expander is
    supplied, at most one extra condition term is added (the single best query
    variant), and it's generated concurrently with the existing term-extraction
    LLM call rather than after it, so that part adds no serial latency; only the
    bounded +1 search term does.
    """
    context_decision = decision_from_dict(profile.clinical_context)
    if context_decision and context_decision.requires_clarification:
        return {
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "trials": [], "condition_terms": [], "medication_terms": [],
            "location": location_query,
            "error": (
                "Trial search paused because the patient context is ambiguous or conflicting. "
                + (context_decision.clarifying_question or "Confirm the exact test or condition first.")
            ),
            "context_status": context_decision.status,
            "clinical_context": context_decision.as_dict(),
        }

    # Phase 1 -- term extraction and query expansion run concurrently; neither
    # depends on the other, and the expansion is only asked for one bounded
    # extra term so Phase 2's serial search loop doesn't grow by more than one.
    if query_expander is not None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            extract_future = executor.submit(_llm_extract_search_terms, profile.raw_context)
            expansion_future = executor.submit(query_expander.expand_with_hyde, profile.raw_context)
            extracted = extract_future.result()
            try:
                hyde_result = expansion_future.result()
            except Exception as exc:
                print(f"[clinical_trials] query expansion failed: {exc}")
                hyde_result = None
    else:
        extracted = _llm_extract_search_terms(profile.raw_context)
        hyde_result = None

    condition_terms = extracted.get("conditions", [])
    medication_terms = extracted.get("medications", [])

    if hyde_result and hyde_result.query_variants:
        best_variant = hyde_result.query_variants[0].strip()
        existing_lower = {t.strip().lower() for t in condition_terms + medication_terms}
        if best_variant and best_variant.lower() not in existing_lower:
            condition_terms = condition_terms + [best_variant]

    if context_decision and context_decision.domain:
        condition_terms = [
            term for term in condition_terms
            if source_matches_context("", term, context_decision)
        ]

    if not condition_terms and not medication_terms:
        return {
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "trials": [], "condition_terms": [], "medication_terms": [],
            "location": location_query,
            "error": (
                "No specific medical conditions or medications could be found in your saved data. "
                "Chat with FlynnMed about your health concerns -- the assistant builds a "
                "condition record the trial finder uses."
            ),
            "context_status": context_decision.status if context_decision else "insufficient",
            "clinical_context": context_decision.as_dict() if context_decision else {},
        }

    total_terms = len(condition_terms) + len(medication_terms)

    # Phase 2: merge by NCT ID
    merged: Dict[str, Dict] = {}

    for term in condition_terms:
        try:
            for study in _search_one_term(term, "query.cond", location_query):
                nct_id = _module(study, "identificationModule").get("nctId", "")
                if not nct_id:
                    continue
                if nct_id not in merged:
                    merged[nct_id] = {"study": study, "found_for": []}
                if term not in merged[nct_id]["found_for"]:
                    merged[nct_id]["found_for"].append(term)
            time.sleep(0.15)
        except Exception as exc:
            print(f"[clinical_trials] cond search failed for '{term}': {exc}")

    for drug in medication_terms:
        try:
            for study in _search_one_term(drug, "query.intr", location_query):
                nct_id = _module(study, "identificationModule").get("nctId", "")
                if not nct_id:
                    continue
                if nct_id not in merged:
                    merged[nct_id] = {"study": study, "found_for": []}
                label = f"medication: {drug}"
                if label not in merged[nct_id]["found_for"]:
                    merged[nct_id]["found_for"].append(label)
            time.sleep(0.15)
        except Exception as exc:
            print(f"[clinical_trials] intr search failed for '{drug}': {exc}")

    # Phase 3a: build records + fast pre-score
    records = [
        _build_trial_record(data["study"], data["found_for"], total_terms, location_query)
        for data in merged.values()
    ]
    records.sort(key=lambda r: (r["coverage_score"] + r["location_score"]), reverse=True)

    # Phase 3b: LLM scoring on top candidates only
    candidates = records[:_PRE_FILTER_N]
    if candidates:
        _llm_score_candidates(candidates, profile)
        candidates = _exclude_not_relevant(candidates)

    # Remove internal _study reference before returning
    for r in candidates:
        r.pop("_study", None)

    # Final ranking: LLM alignment + coverage + location
    candidates.sort(
        key=lambda r: (r["match_score"], r["alignment_score"], r["coverage_score"]),
        reverse=True,
    )

    return {
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "trials": candidates[:max_results],
        "condition_terms": condition_terms,
        "medication_terms": medication_terms,
        "location": location_query,
        "error": "",
        "context_status": context_decision.status if context_decision else "insufficient",
        "clinical_context": context_decision.as_dict() if context_decision else {},
    }
