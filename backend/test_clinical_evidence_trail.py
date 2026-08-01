from backend.rag_system import RAGEngine


def test_clinical_evidence_trail_lists_only_sources_cited_in_answer():
    answer = "## Prioritized Decision\nUse the urgent pathway [S2]."
    sources = [
        {"source_id": "S1", "title": "Reviewed but unused", "journal": "NICE"},
        {"source_id": "S2", "title": "Stroke guidance", "journal": "NICE"},
    ]

    result = RAGEngine._append_clinical_evidence_trail(
        answer, sources, "healthcare_professional"
    )

    assert "## Evidence Used" in result
    assert "[S2] NICE: Stroke guidance" in result
    assert "Reviewed but unused" not in result


def test_patient_answer_does_not_receive_professional_evidence_trail():
    answer = "Guidance [S1]."
    sources = [{"source_id": "S1", "title": "Source"}]

    result = RAGEngine._append_clinical_evidence_trail(answer, sources, "patient")

    assert result == answer
