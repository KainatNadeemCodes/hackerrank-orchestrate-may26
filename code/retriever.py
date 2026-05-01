"""
retriever.py — TF-IDF retriever over the local support corpus.

Walks data/{hackerrank,claude,visa}/**/*.md, chunks each file into
overlapping windows, builds a TF-IDF index, and exposes a .search()
method that returns the top-k most relevant chunks for a query.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ── tuneable constants ───────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # words overlap between consecutive chunks
TOP_K_DEFAULT = 5     # chunks returned by default


class CorpusRetriever:
    """Index the corpus once; answer queries cheaply at runtime."""

    def __init__(self, data_dir: Path):
        self._chunks: List[dict] = []   # {text, source, company}
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

        self._index(data_dir)

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def doc_count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        company: str = "",
        top_k: int = TOP_K_DEFAULT,
    ) -> List[dict]:
        """
        Return up to *top_k* chunks most relevant to *query*.
        If company is given, we run two passes:
          1. corpus filtered to that company's files (weight ×2 for company match)
          2. global corpus for fallback coverage
        Results are deduplicated and re-ranked.
        """
        if not self._chunks:
            return []

        q_vec = self._vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self._matrix)[0]

        # boost chunks that belong to the requested company
        company_lower = company.lower()
        boosts = np.array([
            1.6 if (company_lower and company_lower in c["company"].lower()) else 1.0
            for c in self._chunks
        ])
        scores = sims * boosts

        top_indices = np.argsort(scores)[::-1][:top_k * 3]
        seen_texts: set = set()
        results: List[dict] = []
        for idx in top_indices:
            if scores[idx] < 0.01:
                break
            text = self._chunks[idx]["text"]
            key  = text[:120]
            if key not in seen_texts:
                seen_texts.add(key)
                results.append({**self._chunks[idx], "score": float(scores[idx])})
            if len(results) >= top_k:
                break

        return results

    # ── private ──────────────────────────────────────────────────────────────

    def _index(self, data_dir: Path) -> None:
        """Walk markdown files, chunk, fit TF-IDF."""
        md_files = list(data_dir.rglob("*.md"))

        for md_path in md_files:
            company = self._infer_company(md_path, data_dir)
            text    = md_path.read_text(encoding="utf-8", errors="ignore")
            text    = self._clean(text)
            for chunk in self._chunk(text, CHUNK_SIZE, CHUNK_OVERLAP):
                self._chunks.append({
                    "text":    chunk,
                    "source":  str(md_path.relative_to(data_dir)),
                    "company": company,
                })

        if not self._chunks:
            return

        corpus = [c["text"] for c in self._chunks]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    @staticmethod
    def _infer_company(path: Path, data_dir: Path) -> str:
        parts = path.relative_to(data_dir).parts
        top   = parts[0].lower() if parts else ""
        if "hackerrank" in top:
            return "HackerRank"
        if "claude" in top:
            return "Claude"
        if "visa" in top:
            return "Visa"
        return "Unknown"

    @staticmethod
    def _clean(text: str) -> str:
        # strip markdown links but keep link text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        words  = text.split()
        chunks = []
        start  = 0
        while start < len(words):
            end = start + size
            chunks.append(" ".join(words[start:end]))
            start += size - overlap
        return chunks
