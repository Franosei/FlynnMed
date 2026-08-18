import html
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from xml.etree import ElementTree as ET

import requests

from backend.product_config import PRODUCT_NAME


class OfficialGuidanceEngine:
    """
    Live retriever for trusted public-health sources.
    Searches official sites in real time so the app can cite current guidance
    rather than relying on hard-coded case-specific responses.
    """

    NICE_SEARCH_URL = "https://www.nice.org.uk/search"
    NHS_SEARCH_URL = "https://www.nhs.uk/search/results"
    MEDLINEPLUS_SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"
    CDC_SEARCH_URL = "https://tools.cdc.gov/api/v2/resources/media"
    MYHEALTHFINDER_ITEM_URL = (
        "https://odphp.health.gov/myhealthfinder/api/v4/itemlist.json"
    )
    MYHEALTHFINDER_TOPIC_URL = (
        "https://odphp.health.gov/myhealthfinder/api/v4/topicsearch.json"
    )
    VA_DOD_GUIDELINES_URL = (
        "https://health.mil/About-MHS/MHS-Elements/DVPO/VADOD-CPGs?type=All"
    )
    USER_AGENT = (
        f"{PRODUCT_NAME.replace(' ', '')}/1.0 "
        "(+https://www.nhs.uk/; https://medlineplus.gov/)"
    )

    def __init__(self) -> None:
        self.search_cache: Dict[tuple, List[Dict]] = {}
        self.page_cache: Dict[str, str] = {}
        self.myhealthfinder_topics: List[Dict] | None = None

    def search(
        self,
        queries: str | List[str],
        per_source_limit: int = 1,
        preferred_sources: List[str] | None = None,
    ) -> List[Dict]:
        normalized_queries = self._normalize_queries(queries)
        if not normalized_queries:
            return []

        cache_key = (tuple(normalized_queries), per_source_limit, tuple(sorted(preferred_sources or [])))
        cached = self.search_cache.get(cache_key)
        if cached is not None:
            return [dict(source) for source in cached]

        search_methods = (
            self._search_nice,
            self._search_nhs,
            self._search_medlineplus,
            self._search_cdc,
            self._search_myhealthfinder,
            # _search_va_dod is disabled: every fetch to healthquality.va.gov
            # fails TLS verification (missing Federal PKI intermediate in the
            # public trust store), and the failure was silently swallowed
            # while the source stayed in the cited list -- producing
            # unfetchable, topically unrelated citations. Re-enable once the
            # CA trust chain is fixed and _search_va_dod's relevance
            # threshold is tightened.
        )
        futures = []
        with ThreadPoolExecutor(
            max_workers=min(12, max(2, len(normalized_queries) * len(search_methods)))
        ) as executor:
            for query in normalized_queries:
                for search_method in search_methods:
                    futures.append(
                        executor.submit(search_method, query, per_source_limit)
                    )

        collected = []
        for future in futures:
            try:
                collected.extend(future.result())
            except Exception as exc:
                print(f"OfficialGuidanceEngine search fallback: {exc}")

        preferred_ranked = self._apply_preferred_sources(collected, preferred_sources)
        deduped = self._dedupe_and_number(preferred_ranked)
        enriched = self._enrich_with_page_content(deduped)
        self.search_cache[cache_key] = [dict(source) for source in enriched]
        return enriched

    def search_medicine(self, medicine_name: str, limit: int = 4) -> List[Dict]:
        """Fetch exact NHS medicine pages without relying on the site search UI.

        NHS medicine pages use stable slugs.  This is especially useful when a
        user asks in conversational language or the general NHS search endpoint
        is temporarily unavailable.
        """
        cleaned_name = " ".join((medicine_name or "").split()).strip()
        if not cleaned_name or len(cleaned_name) > 80:
            return []
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '\-()]+", cleaned_name):
            return []

        slug = re.sub(r"[^a-z0-9]+", "-", cleaned_name.lower()).strip("-")
        if not slug:
            return []

        page_names = [
            f"about-{slug}",
            f"how-and-when-to-take-{slug}",
            f"common-questions-about-{slug}",
            f"who-can-and-cannot-take-{slug}",
            f"side-effects-of-{slug}",
        ][: max(1, min(limit, 4))]
        urls = [
            f"https://www.nhs.uk/medicines/{slug}/{page_name}/"
            for page_name in page_names
        ]

        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            futures = [
                executor.submit(
                    self._fetch_exact_nhs_medicine_page,
                    url,
                    cleaned_name,
                )
                for url in urls
            ]
            sources = []
            for future in futures:
                try:
                    source = future.result()
                except Exception as exc:
                    print(f"OfficialGuidanceEngine medicine-page fallback: {exc}")
                    source = None
                if source:
                    sources.append(source)

        return self._dedupe_and_number(sources)[:limit]

    def _fetch_exact_nhs_medicine_page(
        self, url: str, medicine_name: str
    ) -> Dict | None:
        response = requests.get(
            url,
            headers={"User-Agent": self.USER_AGENT},
            timeout=6,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()

        heading_match = re.search(
            r"<h1[^>]*>(?P<title>.*?)</h1>",
            response.text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        title = self._clean_html(
            heading_match.group("title") if heading_match else medicine_name
        )
        detail = self._extract_relevant_paragraphs(
            response.text,
            f"{medicine_name} used treat allergy contraindications side effects",
            max_paragraphs=5,
        )
        if not detail:
            return None
        return {
            "title": title,
            "journal": "NHS",
            "year": "",
            "section": "Medicine guidance",
            "url": url,
            "query": medicine_name,
            "snippet": detail[:500],
            "detail_snippet": detail,
            "provider": "nhs",
            "source_type": "official_guidance",
            "authority": "NHS",
            "jurisdiction": "UK",
            "licence_status": "source_terms",
            "licence_url": "https://www.nhs.uk/our-policies/terms-and-conditions/",
        }

    def _search_nice(self, query: str, limit: int) -> List[Dict]:
        response = requests.get(
            self.NICE_SEARCH_URL,
            params={"q": query},
            headers={"User-Agent": self.USER_AGENT},
            timeout=6,
        )
        response.raise_for_status()

        html_text = response.text
        pattern = re.compile(
            r'headinglink="(?P<href>/guidance/[^"]+)"[\s\S]*?'
            r'<a href="(?P=href)"><span>(?P<title>.*?)</span></a>[\s\S]*?'
            r'<p class="card__summary"><span>(?P<snippet>.*?)</span></p>',
            re.IGNORECASE,
        )

        matches = []
        for match in pattern.finditer(html_text):
            href = html.unescape(match.group("href"))
            title = self._clean_html(match.group("title"))
            snippet = self._clean_html(match.group("snippet"))
            if not href or not title:
                continue

            matches.append(
                {
                    "title": title,
                    "journal": "NICE",
                    "year": "",
                    "section": "Guidance summary",
                    "url": urljoin("https://www.nice.org.uk", href),
                    "query": query,
                    "snippet": snippet,
                    "provider": "nice",
                    "source_type": "official_guidance",
                    "authority": "NICE",
                    "jurisdiction": "UK",
                    "licence_status": "source_terms",
                    "licence_url": "https://www.nice.org.uk/reusing-our-content",
                }
            )
            if len(matches) >= limit:
                break

        return matches

    @staticmethod
    def _normalize_queries(queries: str | List[str]) -> List[str]:
        if isinstance(queries, str):
            candidates = [queries]
        else:
            candidates = list(queries)

        normalized = []
        seen = set()
        for query in candidates:
            cleaned = " ".join((query or "").split()).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized[:3]

    def _search_nhs(self, query: str, limit: int) -> List[Dict]:
        response = requests.get(
            self.NHS_SEARCH_URL,
            params={"q": query},
            headers={"User-Agent": self.USER_AGENT},
            timeout=6,
        )
        response.raise_for_status()

        html_text = response.text
        pattern = re.compile(
            r'<a class="app-search-results-item"[^>]*href="(?P<href>[^"]+)"[^>]*>\s*(?P<title>.*?)\s*</a>\s*'
            r'<p class="nhsuk-body-s nhsuk-u-margin-top-2">\s*(?P<snippet>.*?)\s*</p>',
            re.IGNORECASE | re.DOTALL,
        )

        matches = []
        for match in pattern.finditer(html_text):
            href = html.unescape(match.group("href"))
            title = self._clean_html(match.group("title"))
            snippet = self._clean_html(match.group("snippet"))
            target_url = self._resolve_nhs_result_url(href)
            if not target_url or not title:
                continue

            matches.append(
                {
                    "title": title,
                    "journal": "NHS",
                    "year": "",
                    "section": "Search result summary",
                    "url": target_url,
                    "query": query,
                    "snippet": snippet,
                    "provider": "nhs",
                    "source_type": "official_guidance",
                    "authority": "NHS",
                    "jurisdiction": "UK",
                    "licence_status": "source_terms",
                    "licence_url": "https://www.nhs.uk/our-policies/terms-and-conditions/",
                }
            )
            if len(matches) >= limit:
                break

        return matches

    def _search_medlineplus(self, query: str, limit: int) -> List[Dict]:
        response = requests.get(
            self.MEDLINEPLUS_SEARCH_URL,
            params={"db": "healthTopics", "term": query, "retmax": limit},
            headers={"User-Agent": self.USER_AGENT},
            timeout=6,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        documents = []
        for document in root.findall(".//document"):
            title = self._clean_html(self._xml_content(document, "title"))
            snippet = self._clean_html(self._xml_content(document, "snippet"))
            full_summary = self._clean_html(self._xml_content(document, "FullSummary"))
            url = document.attrib.get("url", "")
            if not title or not url:
                continue

            documents.append(
                {
                    "title": title,
                    "journal": "MedlinePlus",
                    "year": "",
                    "section": "Topic summary",
                    "url": url,
                    "query": query,
                    "snippet": snippet or full_summary[:500],
                    "provider": "medlineplus",
                    "source_type": "official_guidance",
                    "authority": "National Library of Medicine",
                    "jurisdiction": "US",
                    "licence_status": "mixed_content_check_notices",
                    "licence_url": "https://medlineplus.gov/about/using/usingcontent/",
                    "attribution": "MedlinePlus, National Library of Medicine",
                }
            )
        return documents[:limit]

    def _search_cdc(self, query: str, limit: int) -> List[Dict]:
        search_terms = [query]
        search_terms.extend(
            token
            for token in sorted(self._search_tokens(query), key=len, reverse=True)
            if token.lower() not in query.lower().split()[:1]
        )
        search_terms = list(dict.fromkeys(search_terms))[:3]

        def search_name(term: str) -> List[Dict]:
            response = requests.get(
                self.CDC_SEARCH_URL,
                params={
                    "name": term,
                    "max": max(1, min(limit * 3, 12)),
                    "mediatypes": "HTML",
                    "languagename": "English",
                },
                headers={"User-Agent": self.USER_AGENT},
                timeout=6,
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            if isinstance(results, dict):
                results = results.get("items") or results.get("results") or []
            return results if isinstance(results, list) else []

        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=len(search_terms)) as executor:
            for found in executor.map(search_name, search_terms):
                results.extend(found)

        sources: List[Dict] = []
        seen_urls: set[str] = set()
        for item in results:
            source = item.get("source") or {}
            acronym = str(source.get("acronym") or "").upper()
            provider_name = str(source.get("name") or "").lower()
            url = str(
                item.get("targetUrl")
                or item.get("sourceUrl")
                or item.get("persistentUrl")
                or ""
            )
            if (
                acronym != "CDC"
                and "centers for disease control" not in provider_name
                and "cdc.gov" not in url.lower()
            ):
                continue
            title = self._clean_html(
                str(item.get("name") or item.get("title") or "")
            )
            snippet = self._clean_html(
                str(item.get("description") or item.get("featuredText") or "")
            )
            if not title or not url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                {
                    "title": title,
                    "journal": "CDC",
                    "year": self._year_from_date(
                        str(item.get("dateModified") or item.get("datePublished") or "")
                    ),
                    "section": "CDC public health guidance",
                    "url": url,
                    "query": query,
                    "snippet": snippet,
                    "provider": "cdc",
                    "source_type": "official_guidance",
                    "authority": "Centers for Disease Control and Prevention",
                    "jurisdiction": "US",
                    "updated_at": str(
                        item.get("dateModified") or item.get("datePublished") or ""
                    ),
                    "licence_status": "public_domain_us",
                    "licence_url": "https://www.cdc.gov/other/agencymaterials.html",
                    "attribution": "Centers for Disease Control and Prevention",
                }
            )
            if len(sources) >= limit:
                break
        return sources

    def _search_myhealthfinder(self, query: str, limit: int) -> List[Dict]:
        topics = self._myhealthfinder_topic_index()
        query_tokens = self._search_tokens(query)
        ranked = []
        for topic in topics:
            title = str(topic.get("Title") or "")
            title_tokens = self._search_tokens(title)
            score = len(query_tokens & title_tokens)
            if score:
                ranked.append((score, title.lower(), topic))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        sources: List[Dict] = []
        for _score, _title, topic in ranked[: max(1, limit)]:
            topic_id = str(topic.get("Id") or "")
            if not topic_id:
                continue
            response = requests.get(
                self.MYHEALTHFINDER_TOPIC_URL,
                params={"TopicId": topic_id},
                headers={"User-Agent": self.USER_AGENT},
                timeout=6,
            )
            response.raise_for_status()
            resources = (
                response.json()
                .get("Result", {})
                .get("Resources", {})
                .get("Resource", [])
            )
            if isinstance(resources, dict):
                resources = [resources]
            for resource in resources:
                source = self._myhealthfinder_source(resource, query)
                if source:
                    sources.append(source)
                    break
        return sources[:limit]

    def _myhealthfinder_topic_index(self) -> List[Dict]:
        if self.myhealthfinder_topics is not None:
            return [dict(topic) for topic in self.myhealthfinder_topics]
        response = requests.get(
            self.MYHEALTHFINDER_ITEM_URL,
            params={"Type": "topic"},
            headers={"User-Agent": self.USER_AGENT},
            timeout=6,
        )
        response.raise_for_status()
        topics = (
            response.json().get("Result", {}).get("Items", {}).get("Item", [])
        )
        if isinstance(topics, dict):
            topics = [topics]
        self.myhealthfinder_topics = [dict(topic) for topic in topics]
        return [dict(topic) for topic in self.myhealthfinder_topics]

    def _myhealthfinder_source(self, resource: Dict, query: str) -> Dict | None:
        title = self._clean_html(str(resource.get("Title") or ""))
        url = str(resource.get("AccessibleVersion") or "")
        sections = (resource.get("Sections") or {}).get("section", [])
        if isinstance(sections, dict):
            sections = [sections]
        section_text = " ".join(
            self._clean_html(str(section.get("Content") or ""))
            for section in sections[:4]
        ).strip()
        if not title or not url or not section_text:
            return None
        updated_at = self._unix_timestamp_to_iso(str(resource.get("LastUpdate") or ""))
        return {
            "title": title,
            "journal": "MyHealthfinder",
            "year": updated_at[:4] if updated_at else "",
            "section": "Consumer prevention guidance",
            "url": url,
            "query": query,
            "snippet": section_text[:500],
            "detail_snippet": section_text[:1000],
            "provider": "myhealthfinder",
            "source_type": "official_guidance",
            "authority": "Office of Disease Prevention and Health Promotion",
            "jurisdiction": "US",
            "updated_at": updated_at,
            "licence_status": "api_terms",
            "licence_url": (
                "https://odphp.health.gov/our-work/national-health-initiatives/"
                "health-literacy/consumer-health-content/free-web-content/"
                "apis-developers/terms-use"
            ),
            "attribution": "MyHealthfinder, ODPHP",
        }

    def _search_va_dod(self, query: str, limit: int) -> List[Dict]:
        query_tokens = self._search_tokens(query)
        candidates = []
        for href, raw_title in re.findall(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            self._fetch_va_index_html(),
            flags=re.IGNORECASE | re.DOTALL,
        ):
            url = urljoin(self.VA_DOD_GUIDELINES_URL, html.unescape(href))
            parsed = urlparse(url)
            if parsed.netloc.lower() != "www.healthquality.va.gov":
                continue
            if "/guidelines/" not in parsed.path.lower() or parsed.path.lower().rstrip("/") == "/guidelines":
                continue
            title = self._clean_html(raw_title)
            title_tokens = self._search_tokens(title)
            score = len(query_tokens & title_tokens)
            if not title or not score:
                continue
            candidates.append((score, title.lower(), title, url))
        candidates.sort(key=lambda item: (-item[0], item[1]))

        return [
            {
                "title": title,
                "journal": "VA/DoD",
                "year": "",
                "section": "Clinical practice guideline",
                "url": url,
                "query": query,
                "snippet": f"VA/DoD clinical practice guidance: {title}.",
                "provider": "va/dod",
                "source_type": "official_guidance",
                "authority": "Department of Veterans Affairs and Department of Defense",
                "jurisdiction": "US",
                "licence_status": "federal_source_check_page_notices",
                "licence_url": "https://www.va.gov/web/standards/disclaimer.cfm",
                "attribution": "VA/DoD Clinical Practice Guidelines",
            }
            for _score, _sort_title, title, url in candidates[:limit]
        ]

    def _fetch_va_index_html(self) -> str:
        response = requests.get(
            self.VA_DOD_GUIDELINES_URL,
            headers={"User-Agent": self.USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        return response.text

    @staticmethod
    def _dedupe_and_number(sources: List[Dict]) -> List[Dict]:
        unique = []
        seen = set()
        for source in sources:
            key = source.get("url")
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(source)

        for index, source in enumerate(unique, start=1):
            source["source_id"] = f"S{index}"
        return unique

    @staticmethod
    def _apply_preferred_sources(sources: List[Dict], preferred_sources: List[str] | None) -> List[Dict]:
        if not preferred_sources:
            return sources

        preferred_tokens = {
            token.strip().lower()
            for item in preferred_sources
            for token in (item, item.replace("CKS", "").strip())
            if token.strip()
        }
        if not preferred_tokens:
            return sources

        preferred = []
        non_preferred = []
        for source in sources:
            haystack = " ".join(
                [
                    str(source.get("provider", "")),
                    str(source.get("journal", "")),
                    str(source.get("title", "")),
                ]
            ).lower()
            if any(token in haystack for token in preferred_tokens):
                preferred.append(source)
            else:
                non_preferred.append(source)
        return preferred + non_preferred

    @staticmethod
    def _resolve_nhs_result_url(href: str) -> str:
        parsed = urlparse(href)
        if parsed.path.startswith("/search/click"):
            query_params = parse_qs(parsed.query)
            raw_target = query_params.get("url", [""])[0]
            decoded = unquote(raw_target)
            return urljoin("https://www.nhs.uk", decoded)
        return urljoin("https://www.nhs.uk", href)

    @staticmethod
    def _xml_content(document: ET.Element, name: str) -> str:
        node = document.find(f".//content[@name='{name}']")
        return node.text if node is not None and node.text else ""

    @staticmethod
    def _clean_html(value: str) -> str:
        if not value:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", value)
        cleaned = html.unescape(cleaned)
        return " ".join(cleaned.split())

    @staticmethod
    def _search_tokens(value: str) -> set[str]:
        stopwords = {
            "about",
            "and",
            "for",
            "from",
            "guidance",
            "health",
            "management",
            "patient",
            "the",
            "with",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (value or "").lower())
            if len(token) >= 3 and token not in stopwords
        }

    @staticmethod
    def _year_from_date(value: str) -> str:
        match = re.search(r"\b(19|20)\d{2}\b", value or "")
        return match.group(0) if match else ""

    @staticmethod
    def _unix_timestamp_to_iso(value: str) -> str:
        try:
            timestamp = int(value)
        except (TypeError, ValueError):
            return ""
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            return ""

    def _enrich_with_page_content(self, sources: List[Dict]) -> List[Dict]:
        if not sources:
            return []

        with ThreadPoolExecutor(max_workers=min(6, len(sources))) as executor:
            futures = [executor.submit(self._fetch_page_excerpt, source) for source in sources]
            enriched = []
            for future in futures:
                try:
                    enriched.append(future.result())
                except Exception as exc:
                    print(f"OfficialGuidanceEngine page enrichment fallback: {exc}")
        return enriched

    def _fetch_page_excerpt(self, source: Dict) -> Dict:
        enriched = dict(source)
        if source.get("detail_snippet"):
            return enriched
        url = source.get("url", "")
        if not url:
            enriched["detail_snippet"] = source.get("snippet", "")
            return enriched

        cache_key = f"{url}::{source.get('query', '')}"
        cached_excerpt = self.page_cache.get(cache_key)
        if cached_excerpt is not None:
            enriched["detail_snippet"] = cached_excerpt
            return enriched

        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=6,
            )
            response.raise_for_status()
            paragraph_excerpt = self._extract_relevant_paragraphs(response.text, source.get("query", ""))
            detail_snippet = paragraph_excerpt or source.get("snippet", "")
            enriched["detail_snippet"] = detail_snippet
            self.page_cache[cache_key] = detail_snippet
        except Exception as exc:
            print(f"OfficialGuidanceEngine source fetch failed for {url}: {exc}")
            enriched["detail_snippet"] = source.get("snippet", "")

        return enriched

    def _extract_relevant_paragraphs(self, html_text: str, query: str, max_paragraphs: int = 3) -> str:
        cleaned_html = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.IGNORECASE)
        cleaned_html = re.sub(r"<style[\s\S]*?</style>", " ", cleaned_html, flags=re.IGNORECASE)
        paragraph_matches = re.findall(
            r"<(?:p|li)[^>]*>(.*?)</(?:p|li)>",
            cleaned_html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        paragraphs = []
        for paragraph in paragraph_matches:
            text = self._clean_html(paragraph)
            if len(text) < 80:
                continue
            paragraphs.append(text)

        if not paragraphs:
            return ""

        query_terms = {term for term in re.findall(r"[a-zA-Z]{4,}", query.lower())}
        scored = []
        for index, paragraph in enumerate(paragraphs):
            lower = paragraph.lower()
            score = sum(1 for term in query_terms if term in lower)
            scored.append((score, index, paragraph))

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        top = sorted(scored[:max_paragraphs], key=lambda item: item[1])
        return " ".join(item[2] for item in top)
