from backend.medication_checker import check_allergy_conflicts


def _candidate(name="amoxicillin", aliases=None, pharm_classes=None):
    return {
        "query_name": name,
        "canonical_name": name.upper(),
        "aliases": aliases or [name, name.upper()],
        "pharm_classes": pharm_classes or [],
    }


def test_exact_name_match_flags_conflict():
    candidate = _candidate(name="penicillin", aliases=["penicillin", "PENICILLIN", "Pen VK"])
    allergies = [{"name": "Penicillin", "severity": "severe"}]

    flags = check_allergy_conflicts(candidate, allergies)

    assert len(flags) == 1
    assert flags[0]["match_type"] == "exact_name"
    assert flags[0]["allergy_name"] == "Penicillin"
    assert flags[0]["severity"] == "severe"


def test_drug_class_match_flags_conflict():
    candidate = _candidate(
        name="amoxicillin",
        pharm_classes=["Penicillin-class Antibacterial [EPC]"],
    )
    allergies = [{"name": "penicillin", "severity": "moderate"}]

    flags = check_allergy_conflicts(candidate, allergies)

    assert len(flags) == 1
    assert flags[0]["match_type"] == "drug_class"
    assert "Penicillin-class Antibacterial" in flags[0]["matched_text"]


def test_no_conflict_when_names_and_classes_are_unrelated():
    candidate = _candidate(
        name="metformin",
        aliases=["metformin", "METFORMIN"],
        pharm_classes=["Biguanide [EPC]"],
    )
    allergies = [{"name": "shellfish", "severity": "mild"}]

    flags = check_allergy_conflicts(candidate, allergies)

    assert flags == []


def test_empty_allergies_returns_no_flags():
    candidate = _candidate()
    assert check_allergy_conflicts(candidate, []) == []


def test_empty_candidate_returns_no_flags():
    assert check_allergy_conflicts({}, [{"name": "penicillin"}]) == []


def test_allergy_missing_name_is_skipped_not_erroring():
    candidate = _candidate(name="penicillin", aliases=["penicillin"])
    allergies = [{"name": "", "severity": "severe"}, {"severity": "severe"}]

    assert check_allergy_conflicts(candidate, allergies) == []


def test_severity_defaults_to_unknown_when_not_recorded():
    candidate = _candidate(name="penicillin", aliases=["penicillin"])
    allergies = [{"name": "penicillin"}]

    flags = check_allergy_conflicts(candidate, allergies)

    assert flags[0]["severity"] == "unknown"


def test_medication_interaction_checker_importable_from_clinical_orchestrator():
    """
    Regression lock for the bug fixed this session: clinical_orchestrator.py's
    _check_drug_interactions used to import a non-existent `MedicationChecker`
    class, which always raised ImportError (silently swallowed) so the
    check_drug_interactions tool never actually worked. Confirms the fix.
    """
    from backend.clinical_orchestrator import AgenticRetrievalLoop

    loop = AgenticRetrievalLoop(
        llm=object(), official_guidance=object(), pubmed=object(), memory=object(), user="x"
    )
    result = loop._check_drug_interactions([])
    assert result == {"summary": "No medications provided.", "sources": []}
