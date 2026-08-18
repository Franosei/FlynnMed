from backend.evidence_ranker import EvidenceRanker
from backend.intent_risk_classifier import IntentClassification
from backend.patient_history import PatientHistoryContext
from backend.role_router import RoleRouter


class _FallbackMemory:
    def _embed_text(self, _text):
        raise RuntimeError("force fallback scoring")


def test_us_government_guidance_domains_are_trusted():
    for provider, url in (
        ("cdc", "https://www.cdc.gov/flu/about/index.html"),
        ("myhealthfinder", "https://odphp.health.gov/myhealthfinder/topic"),
        ("va/dod", "https://www.healthquality.va.gov/guidelines/CD/asthma/"),
        ("openFDA", "https://open.fda.gov/apis/drug/"),
    ):
        status, score, _reason = EvidenceRanker._validate_source(
            {
                "title": "Official US guidance",
                "url": url,
                "provider": provider,
                "source_type": "official_guidance",
                "detail_snippet": "A sufficiently detailed official guidance excerpt.",
            }
        )

        assert status == "valid"
        assert score == 1.0


def test_us_source_provenance_survives_ranking():
    source = {
        "title": "CDC influenza guidance",
        "url": "https://www.cdc.gov/flu/about/index.html",
        "provider": "cdc",
        "source_type": "official_guidance",
        "query": "influenza symptoms",
        "detail_snippet": "Influenza may cause fever, cough and other respiratory symptoms.",
        "authority": "Centers for Disease Control and Prevention",
        "jurisdiction": "US",
        "licence_status": "public_domain_us",
        "licence_url": "https://www.cdc.gov/other/agencymaterials.html",
        "attribution": "Centers for Disease Control and Prevention",
        "updated_at": "2026-01-15T00:00:00Z",
    }

    ranked, _report = _rank(
        [source],
        "What are influenza symptoms?",
        PatientHistoryContext(),
    )

    assert ranked[0]["authority"] == source["authority"]
    assert ranked[0]["jurisdiction"] == "US"
    assert ranked[0]["licence_status"] == "public_domain_us"
    assert ranked[0]["updated_at"] == source["updated_at"]


def _rank(sources, question, patient_history):
    return EvidenceRanker().rank_and_tier_with_report(
        sources=sources,
        question=question,
        role_config=RoleRouter().resolve("patient"),
        intent=IntentClassification(),
        memory_store=_FallbackMemory(),
        patient_history=patient_history,
    )


def test_patient_aligned_source_is_marked_usable_for_specific_guidance():
    patient_history = PatientHistoryContext(
        age=68,
        known_conditions=["Atrial fibrillation"],
        known_medications=["Warfarin"],
    )
    sources = [
        {
            "title": "Warfarin-associated bleeding risk and headache in adults",
            "journal": "Clinical Medicine",
            "year": "2025",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/",
            "pmcid": "PMC1",
            "source_type": "pubmed_literature",
            "query": "warfarin headache bleeding risk",
            "detail_snippet": (
                "Adults taking warfarin who develop severe headache require assessment "
                "for intracranial bleeding and anticoagulation complications."
            ),
        },
        {
            "title": "General headache self-care",
            "journal": "Clinical Medicine",
            "year": "2025",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2/",
            "pmcid": "PMC2",
            "source_type": "pubmed_literature",
            "query": "headache self care",
            "detail_snippet": "Most tension headaches improve with hydration, rest, and simple analgesia.",
        },
    ]

    ranked, report = _rank(sources, "I have a severe headache; is it risky on warfarin?", patient_history)

    assert report["overall_status"] == "patient_aligned_evidence_available"
    assert ranked[0]["evidence_quality_status"] == "patient_aligned"
    assert ranked[0]["usable_for_patient_specific_guidance"] is True
    assert "Warfarin" in ranked[0]["patient_alignment_facts"]


def test_population_mismatch_is_filtered_before_answer_generation():
    patient_history = PatientHistoryContext(age=72, known_conditions=["Hypertension"])
    sources = [
        {
            "title": "Hypertension treatment in children",
            "journal": "Paediatrics Review",
            "year": "2024",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3/",
            "pmcid": "PMC3",
            "source_type": "pubmed_literature",
            "query": "hypertension treatment",
            "detail_snippet": "Children with hypertension may need paediatric specialist assessment.",
        }
    ]

    ranked, report = _rank(sources, "What does evidence say about hypertension treatment?", patient_history)

    assert ranked == []
    assert report["overall_status"] == "no_sources_passed_quality_gate"
    assert report["excluded_source_count"] == 1
    assert "children" in report["excluded_sources"][0]["mismatch_flags"][0].lower()


def test_patient_specific_query_without_text_confirmation_is_background_only():
    patient_history = PatientHistoryContext(
        age=54,
        known_conditions=["Chronic kidney disease"],
    )
    sources = [
        {
            "title": "Hypertension management in adults",
            "journal": "Clinical Medicine",
            "year": "2025",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4/",
            "pmcid": "PMC4",
            "source_type": "pubmed_literature",
            "query": "hypertension chronic kidney disease",
            "detail_snippet": "Adults with hypertension benefit from structured monitoring and follow-up.",
        }
    ]

    ranked, report = _rank(sources, "How should my hypertension be monitored?", patient_history)

    assert report["overall_status"] == "question_aligned_only"
    assert ranked[0]["evidence_quality_status"] == "background_only"
    assert ranked[0]["usable_for_patient_specific_guidance"] is False
