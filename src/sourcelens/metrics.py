"""Ranking metrics over graded labels (0 no, 1 partial, 2 strong)."""
from __future__ import annotations

import math


def _relevant(labels: dict[str, int]) -> set[str]:
    return {d for d, rel in labels.items() if rel >= 1}


def recall_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    rel = _relevant(labels)
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def precision_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    return sum(labels.get(d, 0) >= 1 for d in ranked[:k]) / k


def _dcg(rels: list[int]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    ideal = _dcg(sorted(labels.values(), reverse=True)[:k])
    return _dcg([labels.get(d, 0) for d in ranked[:k]]) / ideal if ideal else 0.0
