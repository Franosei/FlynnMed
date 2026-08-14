from types import SimpleNamespace

import backend.rag_system as rag_module
from backend.context_graph import build_context_graph
from backend.rag_system import RAGEngine
from backend.relationship_engine import derive_relationships, merge_relationships
from backend.summarizer import LLMHelper


class _FakeStore:
    medications = [
        {
            "name": "Metformin",
            "dose": "500 mg",
            "schedule": "",
            "reason": "",
            "notes": "Existing record",
        }
    ]
    allergies = []
    conditions = []
    symptoms = []
    vitals = []
    relationships = []

    @classmethod
    def reset(cls):
        cls.medications = [
            {
                "name": "Metformin",
                "dose": "500 mg",
                "schedule": "",
                "reason": "",
                "notes": "Existing record",
            }
        ]
        cls.allergies = []
        cls.conditions = []
        cls.symptoms = []
        cls.vitals = []
        cls.relationships = []

    @classmethod
    def get_medications(cls, user):
        return list(cls.medications)

    @classmethod
    def save_medication(cls, user, payload):
        cls.medications = [dict(payload)]
        return payload

    @classmethod
    def get_allergies(cls, user):
        return list(cls.allergies)

    @classmethod
    def save_allergy(cls, user, payload):
        cls.allergies.append(dict(payload))
        return payload

    @classmethod
    def get_conditions(cls, user):
        return list(cls.conditions)

    @classmethod
    def save_condition(cls, user, payload):
        cls.conditions.append(dict(payload))
        return payload

    @classmethod
    def get_symptom_logs(cls, user, limit=None):
        return list(cls.symptoms)

    @classmethod
    def add_symptom_log(cls, user, **payload):
        cls.symptoms.append(dict(payload))
        return payload

    @classmethod
    def get_vitals(cls, user, limit=None):
        return list(cls.vitals)

    @classmethod
    def save_vitals_entry(cls, user, payload):
        cls.vitals.append(dict(payload))
        return payload

    @classmethod
    def get_clinical_relationships(cls, user):
        return list(cls.relationships)

    @classmethod
    def save_clinical_relationships(cls, user, relationships):
        cls.relationships.extend(dict(item) for item in relationships)
        return list(cls.relationships)


def _engine_with_payload(payload):
    engine = RAGEngine.__new__(RAGEngine)
    engine.llm = SimpleNamespace(
        extract_explicit_chat_record_facts=lambda message: payload
    )
    return engine


def test_explicit_chat_facts_enter_structured_records_and_preserve_existing_fields(
    monkeypatch,
):
    _FakeStore.reset()
    monkeypatch.setattr(rag_module, "UserStore", _FakeStore)
    engine = _engine_with_payload(
        {
            "medications": [
                {
                    "name": "Metformin",
                    "dose": "",
                    "schedule": "twice daily",
                    "reason": "type 2 diabetes",
                }
            ],
            "allergies": [
                {
                    "name": "Penicillin",
                    "reaction": "hives",
                    "severity": "severe",
                    "allergy_type": "drug",
                }
            ],
            "conditions": [],
            "symptoms": [],
            "vitals": [],
            "relationships": [
                {
                    "source_type": "medication",
                    "source_name": "Metformin",
                    "relation": "taken_for",
                    "target_type": "condition",
                    "target_name": "type 2 diabetes",
                    "certainty": "user_reported",
                    "evidence": "I take metformin for type 2 diabetes",
                }
            ],
        }
    )

    updates = engine._capture_explicit_chat_records(
        "patient1",
        "I take metformin twice daily for type 2 diabetes and I am allergic to penicillin.",
    )

    assert _FakeStore.medications[0]["dose"] == "500 mg"
    assert _FakeStore.medications[0]["schedule"] == "twice daily"
    assert _FakeStore.allergies[0]["name"] == "Penicillin"
    assert _FakeStore.relationships[0]["relation"] == "taken_for"
    assert {item["record_type"] for item in updates} == {
        "medication",
        "allergy",
        "relationship",
    }


def test_negated_medicine_is_not_saved_even_if_extractor_returns_it(monkeypatch):
    _FakeStore.reset()
    monkeypatch.setattr(rag_module, "UserStore", _FakeStore)
    engine = _engine_with_payload(
        {
            "medications": [{"name": "Aspirin"}],
            "allergies": [],
            "conditions": [],
            "symptoms": [],
            "vitals": [],
            "relationships": [],
        }
    )

    updates = engine._capture_explicit_chat_records(
        "patient1", "I am not taking aspirin."
    )

    assert updates == []
    assert _FakeStore.medications[0]["name"] == "Metformin"


def test_causal_question_is_not_saved_as_a_patient_reported_relationship(monkeypatch):
    _FakeStore.reset()
    monkeypatch.setattr(rag_module, "UserStore", _FakeStore)
    engine = _engine_with_payload(
        {
            "medications": [],
            "allergies": [],
            "conditions": [],
            "symptoms": [],
            "vitals": [],
            "relationships": [
                {
                    "source_type": "medication",
                    "source_name": "Metformin",
                    "relation": "causes",
                    "target_type": "symptom",
                    "target_name": "nausea",
                }
            ],
        }
    )

    updates = engine._capture_explicit_chat_records(
        "patient1", "Does metformin cause nausea?"
    )

    assert updates == []
    assert _FakeStore.relationships == []


def test_context_graph_exposes_causal_edges_with_uncertainty_guard():
    graph = build_context_graph(
        question="Could the nausea be related to metformin?",
        relationships=[
            {
                "source_type": "medication",
                "source_name": "Metformin",
                "relation": "causes",
                "target_type": "symptom",
                "target_name": "nausea",
                "certainty": "user_suspected",
                "evidence": "nausea started after metformin",
            }
        ],
    )

    block = graph.relationship_prompt_block()
    assert "[association; user suspects] Metformin causes nausea" in block
    assert "not proven medical causation" in block


def test_relationship_engine_covers_every_structured_record_family():
    relationships = derive_relationships(
        medications=[{"name": "Metformin", "reason": "type 2 diabetes"}],
        allergies=[{"name": "Penicillin", "reaction": "hives"}],
        conditions=[{"name": "Asthma", "notes": "triggered by pollen"}],
        symptom_logs=[
            {"symptom": "Headache", "triggers": "stress, poor sleep"}
        ],
        vitals=[
            {
                "type": "blood_glucose",
                "value": "12.1",
                "notes": "associated with steroid treatment",
            }
        ],
        triage_summaries=[
            {"impression": "Worsening breathlessness", "next_step": "Same-day review"}
        ],
        care_plans=[
            {
                "condition": "Asthma",
                "daily_tasks": [{"action": "Use the preventer inhaler as prescribed"}],
            }
        ],
        clinical_notes=[
            {"assessment": "Likely asthma flare", "plan": "Arrange inhaler review"}
        ],
        safety_reviews=[
            {
                "category": "Medicine safety",
                "proposed_action": "Contact a pharmacist today",
                "patient_facts": [
                    {"record_type": "medicine", "value": "Warfarin and ibuprofen"}
                ],
            }
        ],
    )

    record_types = {
        item["source_type"] for item in relationships
    } | {item["target_type"] for item in relationships}
    relations = {item["relation"] for item in relationships}
    assert {
        "medication", "allergy", "condition", "symptom", "vital", "triage",
        "clinical_assessment", "care_action", "safety_action",
    } <= record_types
    assert {
        "taken_for", "allergic_reaction", "triggers", "associated_with",
        "led_to", "recommended_for",
    } <= relations
    assert all(item["relation_class"] for item in relationships)


def test_relationship_merge_deduplicates_same_edge_across_ingestion_paths():
    edge = {
        "source_type": "medication",
        "source_name": "Metformin",
        "relation": "taken_for",
        "target_type": "condition",
        "target_name": "Diabetes",
    }

    merged = merge_relationships([edge], [{**edge, "source": "document:letter.pdf"}])

    assert len(merged) == 1


def test_chat_fact_extractor_returns_only_list_sections_from_json():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"medications":[{"name":"Metformin"}],'
                        '"allergies":[],"conditions":[],"symptoms":[],"vitals":[], '
                        '"relationships":"not-a-list"}'
                    )
                )
            )
        ]
    )
    completions = SimpleNamespace(create=lambda **kwargs: response)
    helper = LLMHelper.__new__(LLMHelper)
    helper.model = "test-model"
    helper.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    extracted = helper.extract_explicit_chat_record_facts("I take metformin.")

    assert extracted["medications"] == [{"name": "Metformin"}]
    assert extracted["relationships"] == []
