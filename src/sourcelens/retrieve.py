"""Candidate retrieval: BM25, dense embeddings, and reciprocal rank fusion."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .models import Requirements

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
