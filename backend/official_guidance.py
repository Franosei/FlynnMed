import html
import re
from concurrent.futures import ThreadPoolExecutor
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
    USER_AGENT = f"{PRODUCT_NAME.replace(' ', '')}/1.0 (+https://www.nhs.uk/ https://medlineplus.gov/)"

    def __init__(self) -> None:
        self.search_cache: Dict[tuple, List[Dict]] = {}
        self.page_cache: Dict[str, str] = {}

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

        futures = []
        with ThreadPoolExecutor(max_workers=max(2, len(normalized_queries) * 2)) as executor:
            for query in normalized_queries:
                futures.append(executor.submit(self._search_nice, query, per_source_limit))
                futures.append(executor.submit(self._search_nhs, query, per_source_limit))
                futures.append(executor.submit(self._search_medlineplus, query, per_source_limit))

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
                }
            )
        return documents[:limit]

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
