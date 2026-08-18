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


class _JsonResponse:
    status_code = 200

    def __init__(self, payload=None, text=""):
        self._payload = payload or {}
        self.text = text
        self.content = text.encode("utf-8")

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_cdc_search_returns_only_cdc_content_with_provenance(monkeypatch):
    payload = {
        "results": [
            {
                "name": "About seasonal influenza",
                "description": "CDC information about influenza symptoms and prevention.",
                "targetUrl": "https://www.cdc.gov/flu/about/index.html",
                "source": {"name": "Centers for Disease Control and Prevention", "acronym": "CDC"},
                "dateModified": "2026-01-15T00:00:00Z",
            },
            {
                "name": "Unrelated third-party item",
                "description": "Not a CDC source.",
                "targetUrl": "https://example.com/flu",
                "source": {"name": "Example", "acronym": "EX"},
            },
        ]
    }
    monkeypatch.setattr(
        "backend.official_guidance.requests.get",
        lambda *args, **kwargs: _JsonResponse(payload),
    )

    sources = OfficialGuidanceEngine()._search_cdc("seasonal influenza", 2)

    assert len(sources) == 1
    assert sources[0]["provider"] == "cdc"
    assert sources[0]["jurisdiction"] == "US"
    assert sources[0]["licence_status"] == "public_domain_us"
    assert sources[0]["updated_at"] == "2026-01-15T00:00:00Z"


def test_myhealthfinder_search_ranks_topics_and_preserves_attribution(monkeypatch):
    item_list = {
        "Result": {
            "Items": {
                "Item": [
                    {"Id": "10", "Title": "Eat Healthy"},
                    {"Id": "20", "Title": "Get Screened for Colorectal Cancer"},
                ]
            }
        }
    }
    topic = {
        "Result": {
            "Resources": {
                "Resource": [
                    {
                        "Id": "20",
                        "Title": "Get Screened for Colorectal Cancer",
                        "LastUpdate": "1745432126",
                        "AccessibleVersion": "https://odphp.health.gov/myhealthfinder/colorectal-screening",
                        "Sections": {
                            "section": [
                                {
                                    "Title": "The Basics",
                                    "Content": "<p>Screening can help find colorectal cancer early.</p>",
                                }
                            ]
                        },
                    }
                ]
            }
        }
    }

    def fake_get(url, **kwargs):
        if "itemlist" in url:
            return _JsonResponse(item_list)
        assert kwargs["params"]["TopicId"] == "20"
        return _JsonResponse(topic)

    monkeypatch.setattr("backend.official_guidance.requests.get", fake_get)

    sources = OfficialGuidanceEngine()._search_myhealthfinder(
        "colorectal cancer screening", 1
    )

    assert len(sources) == 1
    assert sources[0]["provider"] == "myhealthfinder"
    assert sources[0]["jurisdiction"] == "US"
    assert sources[0]["attribution"] == "MyHealthfinder, ODPHP"
    assert "find colorectal cancer early" in sources[0]["detail_snippet"]


def test_va_dod_search_uses_only_guideline_links(monkeypatch):
    html = """
    <a href="https://www.healthquality.va.gov/guidelines/CD/asthma/">The Primary Care Management of Asthma</a>
    <a href="/about/index.asp">About the VA</a>
    """
    engine = OfficialGuidanceEngine()
    monkeypatch.setattr(engine, "_fetch_va_index_html", lambda: html)

    sources = engine._search_va_dod("asthma management", 2)

    assert len(sources) == 1
    assert sources[0]["provider"] == "va/dod"
    assert sources[0]["jurisdiction"] == "US"
    assert sources[0]["url"].startswith("https://www.healthquality.va.gov/guidelines/")


def test_search_runs_all_official_providers_and_isolates_failure(monkeypatch):
    engine = OfficialGuidanceEngine()
    monkeypatch.setattr(engine, "_search_nice", lambda *args: [{"url": "https://nice/1"}])
    monkeypatch.setattr(engine, "_search_nhs", lambda *args: [{"url": "https://nhs/1"}])
    monkeypatch.setattr(
        engine,
        "_search_medlineplus",
        lambda *args: [{"url": "https://medlineplus/1"}],
    )
    monkeypatch.setattr(engine, "_search_cdc", lambda *args: [{"url": "https://cdc/1"}])
    monkeypatch.setattr(
        engine,
        "_search_myhealthfinder",
        lambda *args: (_ for _ in ()).throw(RuntimeError("temporary failure")),
    )
    monkeypatch.setattr(
        engine,
        "_search_va_dod",
        lambda *args: [{"url": "https://va/1"}],
    )
    monkeypatch.setattr(engine, "_enrich_with_page_content", lambda sources: sources)

    sources = engine.search("asthma", per_source_limit=1)

    assert {source["url"] for source in sources} == {
        "https://nice/1",
        "https://nhs/1",
        "https://medlineplus/1",
        "https://cdc/1",
        "https://va/1",
    }
    assert [source["source_id"] for source in sources] == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    ]
