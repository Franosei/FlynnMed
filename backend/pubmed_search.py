import html
import re
import requests
from typing import List, Dict
from xml.etree import ElementTree as ET


class PubMedCentralSearcher:
    """
    Searches and retrieves open-access full-text biomedical articles from PubMed Central (PMC).
    Extracts structured sections such as Abstract, Introduction, Discussion, and Conclusion.
    """

    SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

    def __init__(self) -> None:
        self.search_cache: Dict[str, List[Dict[str, str]]] = {}
        self.section_cache: Dict[str, Dict[str, str]] = {}

    def search_article_records(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Searches Europe PMC for open-access article metadata.
        Returns normalized records that are easier to display in the UI and cite in answers.
        """
        cache_key = f"{query}::{max_results}"
        cached = self.search_cache.get(cache_key)
        if cached is not None:
            return [dict(record) for record in cached]

        params = {
            "query": query + " OPEN_ACCESS:Y",
            "format": "json",
            "pageSize": max_results,
            "resultType": "core",
        }

        try:
            response = requests.get(self.SEARCH_URL, params=params, timeout=6)
            response.raise_for_status()
            data = response.json()

            records = []
            for item in data.get("resultList", {}).get("result", []):
                pmcid = item.get("pmcid")
                if not pmcid:
                    continue

                journal = item.get("journalTitle") or item.get("journalInfo", {}).get("journal", {}).get("title", "")
                records.append(
                    {
                        "pmcid": pmcid,
                        "title": item.get("title", "Untitled article"),
                        "journal": journal,
                        "year": self._extract_year(item),
                        "authors": item.get("authorString", ""),
                        "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
                        "query": query,
                        "abstract": self._clean_abstract(item.get("abstractText", "")),
                    }
                )

            try:
                print("PubMed Query:", query)
                print("PMC IDs:", [record["pmcid"] for record in records])
            except UnicodeEncodeError:
                # A debug print failing to encode a query's Unicode characters
                # (e.g. Windows console defaulting to cp1252) must not discard
                # the records already parsed above -- caught here instead of
                # letting it fall into the broad except below, which was
                # silently returning [] and mislabeling this as a "JSON parse
                # error" even though the API call and parsing had both
                # already succeeded.
                pass
            self.search_cache[cache_key] = [dict(record) for record in records]
            return records

        except requests.exceptions.RequestException as e:
            print(f"PubMed API request failed: {e}")
            return []

        except Exception as e:
            print(f"PubMed JSON parse error: {e}")
            return []

    def search_articles(self, query: str, max_results: int = 3) -> List[str]:
        """
        Searches for PMC open-access article IDs using a query string.
        Returns an empty list on failure instead of raising errors.
        """
        return [record["pmcid"] for record in self.search_article_records(query, max_results=max_results)]

    def fetch_article_sections(self, pmcid: str) -> Dict[str, str]:
        """
        Retrieves full-text XML from PMC and extracts relevant sections.
        """
        cached = self.section_cache.get(pmcid)
        if cached is not None:
            return dict(cached)

        url = self.FULLTEXT_URL.format(pmcid=pmcid)
        sections = {
            "abstract": "", "introduction": "", "discussion": "", "conclusion": ""
        }

        section_keywords = {
            "abstract": ["abstract"],
            "introduction": ["introduction", "background"],
            "discussion": ["discussion"],
            "conclusion": ["conclusion", "summary", "concluding remarks"]
        }

        def extract_text(elem) -> str:
            return "".join(elem.itertext()).strip()

        try:
            response = requests.get(url, timeout=6)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            # Try section titles
            for sec in root.findall(".//sec"):
                title_elem = sec.find("title")
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.lower()
                    for key, keywords in section_keywords.items():
                        if any(kw in title for kw in keywords) and not sections[key]:
                            sections[key] = extract_text(sec)
                            print(f"Found section [{key}] in {pmcid}")

            # Backup: use <abstract>
            abstract_elem = root.find(".//abstract")
            if abstract_elem is not None and not sections["abstract"]:
                sections["abstract"] = extract_text(abstract_elem)
                print(f"Fallback abstract in {pmcid}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch full text for {pmcid}: {e}")

        except ET.ParseError as e:
            print(f"XML parsing error for {pmcid}: {e}")

        except Exception as e:
            print(f"Unexpected error processing {pmcid}: {e}")

        self.section_cache[pmcid] = dict(sections)
        return sections

    @staticmethod
    def _extract_year(item: Dict) -> str:
        for field in ("pubYear", "firstPublicationDate", "electronicPublicationDate", "firstIndexDate"):
            value = item.get(field)
            if not value:
                continue
            value = str(value)
            if len(value) >= 4:
                return value[:4]
        return ""

    @staticmethod
    def _clean_abstract(value: str) -> str:
        if not value:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", value)
        cleaned = html.unescape(cleaned)
        return " ".join(cleaned.split())
