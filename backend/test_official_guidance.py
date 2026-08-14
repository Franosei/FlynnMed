from backend.official_guidance import OfficialGuidanceEngine


class _MedicinePageResponse:
    status_code = 200

    def __init__(self, url: str) -> None:
        self.url = url
        self.text = """
        <html><main>
          <h1>Who can and cannot take flucloxacillin</h1>
          <p>Flucloxacillin is a prescription medicine used for bacterial infections and the exact reason should be confirmed with the prescriber.</p>
          <ul><li>Tell a doctor or pharmacist about a previous allergic reaction to flucloxacillin or another penicillin-type antibiotic before taking it.</li></ul>
        </main></html>
        """

    def raise_for_status(self) -> None:
        return None


def test_exact_medicine_lookup_uses_stable_nhs_pages_and_keeps_allergy_text(
    monkeypatch,
):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _MedicinePageResponse(url)

    monkeypatch.setattr("backend.official_guidance.requests.get", fake_get)

    sources = OfficialGuidanceEngine().search_medicine("flucloxacillin", limit=3)

    assert len(sources) == 3
    assert all(source["provider"] == "nhs" for source in sources)
    assert all("/medicines/flucloxacillin/" in source["url"] for source in sources)
    assert any("penicillin-type antibiotic" in source["detail_snippet"] for source in sources)
    assert len(calls) == 3


def test_exact_medicine_lookup_rejects_non_name_input_without_network(monkeypatch):
    monkeypatch.setattr(
        "backend.official_guidance.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )

    assert OfficialGuidanceEngine().search_medicine("https://example.com/?x=1") == []
