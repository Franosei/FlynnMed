from backend.pubmed_search import PubMedCentralSearcher


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "resultList": {
                "result": [
                    {
                        "pmcid": "PMC123",
                        "title": "Dexamethasone adverse effects in older adults",
                        "journalTitle": "Example Journal",
                        "pubYear": "2025",
                        "authorString": "Example A",
                        "abstractText": "A structured abstract.",
                    }
                ]
            }
        }


def test_pubmed_search(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr("backend.pubmed_search.requests.get", fake_get)
    searcher = PubMedCentralSearcher()
    query = "dexamethasone elderly adverse effects"

    pmc_ids = searcher.search_articles(query, max_results=3)

    assert pmc_ids == ["PMC123"]
    assert searcher.search_cache[f"{query}::3"][0]["year"] == "2025"
    assert 'LICENSE:"CC BY"' in calls[0]["params"]["query"]
    assert 'LICENSE:"CC BY-ND"' not in calls[0]["params"]["query"]
    assert searcher.search_cache[f"{query}::3"][0]["licence_status"] == (
        "permissive_reuse_allowed"
    )


def test_search_article_records_survives_a_console_encoding_failure(monkeypatch):
    """
    Regression test: on a Windows console/log stream defaulting to cp1252,
    print()-ing a query containing a character outside that codec's range
    raises UnicodeEncodeError. That must not discard the records already
    parsed from a successful API response -- found via a real evaluation run
    where this silently zeroed out ~46% of PubMed searches, mislabeled in the
    log as a "JSON parse error" even though the API call and parsing had both
    already succeeded.
    """
    monkeypatch.setattr("backend.pubmed_search.requests.get", lambda *args, **kwargs: _Response())

    def _raise_unicode_error(*args, **kwargs):
        raise UnicodeEncodeError("charmap", "–", 0, 1, "character maps to <undefined>")

    monkeypatch.setattr("builtins.print", _raise_unicode_error)
    searcher = PubMedCentralSearcher()

    records = searcher.search_article_records("earache – duration guidance", max_results=3)

    assert len(records) == 1
    assert records[0]["pmcid"] == "PMC123"
