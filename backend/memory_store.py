import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class MemoryStore:
    """
    Stores embedded biomedical content and supports similarity retrieval using OpenAI embeddings.
    """

    def __init__(self, embedding_model: Optional[str] = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        base_url = os.getenv("OPENAI_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.entries: List[Dict] = []
        self.entry_keys: set[str] = set()
        self.entry_key_to_index: Dict[str, int] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def add_entry(
        self,
        text: str,
        metadata: Dict[str, str],
        user: Optional[str] = None,
        entry_key: Optional[str] = None,
    ) -> None:
        stable_key = entry_key or metadata.get("entry_key")
        if stable_key and stable_key in self.entry_keys:
            return

        self.add_entries(
            [
                {
                    "text": text,
                    "metadata": metadata,
                    "user": user,
                    "entry_key": stable_key,
                }
            ]
        )

    def add_entries(self, items: List[Dict]) -> None:
        self._store_entries(items, replace=False)

    def upsert_entries(self, items: List[Dict]) -> None:
        self._store_entries(items, replace=True)

    def _store_entries(self, items: List[Dict], replace: bool) -> None:
        pending = []

        for item in items:
            stable_key = item.get("entry_key") or item.get("metadata", {}).get("entry_key")
            if stable_key and stable_key in self.entry_keys and not replace:
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            pending.append(
                {
                    "text": text,
                    "metadata": item.get("metadata", {}),
                    "user": item.get("user"),
                    "entry_key": stable_key,
                }
            )

        if not pending:
            return

        embeddings = self._embed_texts([item["text"] for item in pending])
        for item, embedding in zip(pending, embeddings):
            payload = {
                "text": item["text"],
                "embedding": embedding,
                "metadata": item["metadata"],
                "user": item["user"],
                "entry_key": item["entry_key"],
            }
            stable_key = item["entry_key"]
            if stable_key and stable_key in self.entry_key_to_index:
                index = self.entry_key_to_index[stable_key]
                self.entries[index] = payload
            else:
                self.entries.append(payload)
                if stable_key:
                    self.entry_key_to_index[stable_key] = len(self.entries) - 1
            if stable_key:
                self.entry_keys.add(stable_key)

    def search(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.45,
        user: Optional[str] = None,
        hypothetical_passage: str = "",
    ) -> List[Tuple[Dict, float]]:
        """
        hypothetical_passage: optional HyDE-style passage (a plausible excerpt from the
        kind of document that would answer the query). When supplied, it's embedded
        alongside the raw query and each entry is scored against whichever of the two
        matches best -- a hypothetical passage often lands closer to real stored
        entries in embedding space than a short question does. Embedding calls are
        cheap/fast, so this adds negligible latency next to the LLM call that produces
        the passage (which the caller generates once, in parallel with other retrieval
        work, and reuses here).
        """
        if not self.entries:
            return []

        filtered_entries = [entry for entry in self.entries if user is None or entry.get("user") == user]
        if not filtered_entries:
            return []

        all_vectors = np.array([entry["embedding"] for entry in filtered_entries], dtype=np.float32)

        query_vec = self._embed_text(query)
        similarities = np.dot(all_vectors, query_vec)

        if hypothetical_passage and hypothetical_passage.strip():
            hyde_vec = self._embed_text(hypothetical_passage)
            hyde_similarities = np.dot(all_vectors, hyde_vec)
            similarities = np.maximum(similarities, hyde_similarities)

        matches = [
            (filtered_entries[index], float(score))
            for index, score in enumerate(similarities)
            if score >= similarity_threshold
        ]
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:top_k]

    def remove_entry(self, entry_key: str) -> None:
        stable_key = (entry_key or "").strip()
        if not stable_key or stable_key not in self.entry_key_to_index:
            return

        index = self.entry_key_to_index.pop(stable_key)
        self.entries.pop(index)
        self.entry_keys.discard(stable_key)
        self.entry_key_to_index = {
            entry.get("entry_key"): idx
            for idx, entry in enumerate(self.entries)
            if entry.get("entry_key")
        }

    def remove_entries(self, entry_keys: List[str]) -> None:
        for entry_key in entry_keys:
            self.remove_entry(entry_key)

    def clear(self):
        self.entries.clear()
        self.entry_keys.clear()
        self.entry_key_to_index.clear()
        self.embedding_cache.clear()

    def _embed_text(self, text: str) -> np.ndarray:
        return self._embed_texts([text])[0]

    def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        cleaned_texts = [text.strip() for text in texts]
        results: List[Optional[np.ndarray]] = [None] * len(cleaned_texts)
        missing_texts = []
        missing_positions = []

        for index, text in enumerate(cleaned_texts):
            cached = self.embedding_cache.get(text)
            if cached is not None:
                results[index] = cached
            else:
                missing_texts.append(text)
                missing_positions.append(index)

        if missing_texts:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=missing_texts,
            )
            for index, data in enumerate(response.data):
                vector = np.array(data.embedding, dtype=np.float32)
                norm = np.linalg.norm(vector)
                normalized = vector if norm == 0 else vector / norm
                text = missing_texts[index]
                self.embedding_cache[text] = normalized
                results[missing_positions[index]] = normalized

        return [vector for vector in results if vector is not None]
