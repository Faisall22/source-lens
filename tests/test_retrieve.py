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


from sourcelens.models import Candidate, Experience  # noqa: E402
from sourcelens.retrieve import Retriever, city  # noqa: E402


def cand(id, headline, location="Bengaluru", remote=False):
    return Candidate(id=id, name=f"N {id}", headline=headline, location=location, open_to_remote=remote,
                     years=5, skills=[], experience=[Experience(title="t", company="c", start="1", end="2", bullets=[])])


class FakeEmbedLLM:
    """Dense space where 'event streaming' is close to 'kafka' but shares no tokens."""

    def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        def vec(t):
            t = t.lower()
            return [1.0 if ("kafka" in t or "event streaming" in t) else 0.0, 1.0 if "react" in t else 0.0, 0.1]
        v = np.array([vec(t) for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_hybrid_combines():
    cands = [cand("lex", "kafka kafka backend"), cand("sem", "event streaming platform"), cand("fe", "react css")]
    r = Retriever(cands, FakeEmbedLLM())
    assert r.bm25("kafka", 3)[0] == "lex"
    assert set(r.hybrid("kafka", 2)) == {"lex", "sem"}


def test_location_filter():
    cands = [cand("blr", "kafka"), cand("pune", "kafka", "Pune"), cand("rem", "kafka", "Delhi", remote=True)]
    r = Retriever(cands, FakeEmbedLLM())
    assert set(r.hybrid("kafka", 5, location="Bangalore")) == {"blr", "rem"}


def test_city_aliases():
    assert city("Bangalore") == city(" bengaluru ") == "bengaluru"
    assert city("Gurgaon") == "gurugram"
