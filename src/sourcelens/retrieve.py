"""Candidate retrieval: BM25, dense embeddings, and reciprocal rank fusion."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .models import Candidate, Requirements

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps c++, c#, node.js; drops 1-char tokens."""
    toks = (t.rstrip(".") for t in _TOKEN.findall(text.lower()))
    return [t for t in toks if len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.lens.mean()) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs))
        for t in query:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t, 0)
                if f:
                    norm = 1 - self.b + self.b * self.lens[i] / self.avgdl
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion. Ties break by id so output is deterministic."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda d: (-score[d], d))


def query_text(req: Requirements) -> str:
    return " ".join([req.title, *req.must_have, *req.nice_to_have, req.location])


_CITY_ALIASES = {"bangalore": "bengaluru", "bombay": "mumbai", "gurgaon": "gurugram",
                 "madras": "chennai", "calcutta": "kolkata", "new delhi": "delhi"}


def city(s: str) -> str:
    c = " ".join(s.lower().split())
    return _CITY_ALIASES.get(c, c)


class Retriever:
    """BM25, dense and hybrid search over candidate profiles."""

    def __init__(self, candidates: list[Candidate], llm):
        self.cands = candidates
        self.ids = [c.id for c in candidates]
        self.llm = llm
        self._bm25 = BM25([tokenize(c.text()) for c in candidates])
        self._doc_vecs: np.ndarray | None = None

    def _keep(self, location: str | None) -> set[str]:
        if not location:
            return set(self.ids)
        want = city(location)
        return {c.id for c in self.cands if city(c.location) == want or c.open_to_remote}

    def _ranked(self, scores: np.ndarray, k: int, location: str | None,
                positive_only: bool = False) -> list[str]:
        keep = self._keep(location)
        order = sorted(range(len(self.ids)), key=lambda i: (-scores[i], self.ids[i]))
        return [self.ids[i] for i in order
                if self.ids[i] in keep and (scores[i] > 0 or not positive_only)][:k]

    def bm25(self, query: str, k: int, location: str | None = None) -> list[str]:
        """Only profiles sharing at least one query term; zero-score order would be noise."""
        return self._ranked(self._bm25.scores(tokenize(query)), k, location, positive_only=True)

    def dense(self, query: str, k: int, location: str | None = None) -> list[str]:
        if self._doc_vecs is None:
            self._doc_vecs = self.llm.embed([c.text() for c in self.cands], task="RETRIEVAL_DOCUMENT")
        q = self.llm.embed([query], task="RETRIEVAL_QUERY")[0]
        return self._ranked(self._doc_vecs @ q, k, location)

    def hybrid(self, query: str, k: int = 15, location: str | None = None) -> list[str]:
        n = len(self.ids)
        return rrf([self.bm25(query, n, location), self.dense(query, n, location)])[:k]
