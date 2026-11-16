import numpy as np

from sourcelens.retrieve import BM25, rrf, tokenize


def test_tokenize_keeps_cpp_nodejs():
    assert tokenize("C++ and Node.js!") == ["c++", "and", "node.js"]


def test_tokenize_drops_single_chars():
    assert tokenize("a b go") == ["go"]


def test_bm25_ranks_matching_doc_first():
    docs = [tokenize("react frontend css"), tokenize("kafka postgres backend"), tokenize("spark airflow sql")]
    scores = BM25(docs).scores(tokenize("kafka backend"))
    assert int(np.argmax(scores)) == 1


def test_bm25_absent_token_scores_zero():
    docs = [tokenize("react css"), tokenize("kafka postgres")]
    assert np.all(BM25(docs).scores(["rust"]) == 0)


def test_rrf_merges():
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


def test_rrf_ties_break_by_id():
    assert rrf([["b"], ["a"]]) == ["a", "b"]
