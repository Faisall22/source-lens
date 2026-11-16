import math

from sourcelens.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_ndcg_perfect_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a": 2, "b": 1, "c": 0}, 3) == 1.0


def test_ndcg_known_value():
    assert math.isclose(ndcg_at_k(["x", "y"], {"x": 0, "y": 2}, 2), 1 / math.log2(3), rel_tol=1e-9)


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["a"], {"a": 0}, 1) == 0.0


def test_recall_no_relevant_is_zero():
    assert recall_at_k(["a"], {"a": 0}, 5) == 0.0


def test_recall_and_precision():
    labels = {"a": 2, "b": 0, "c": 1, "d": 1}
    assert recall_at_k(["a", "b", "c"], labels, 3) == 2 / 3
    assert precision_at_k(["a", "b", "c"], labels, 2) == 0.5
