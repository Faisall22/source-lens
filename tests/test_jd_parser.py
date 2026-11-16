from sourcelens.jd_parser import norm, parse_jd, parse_scores
from sourcelens.models import JD, Requirements


def req(must, nice=()):
    return Requirements(title="t", must_have=list(must), nice_to_have=list(nice),
                        min_years=1, location="Pune", remote_ok=False)


def test_norm():
    assert norm("  Retrieval-Augmented   Generation. ") == "retrieval augmented generation"
    assert norm("C++") == "c++"


def test_parse_scores_alias_match():
    jd = JD(id="j", company="c", title="t", text="", gold=req(["kafka", "python", "sql"]),
            aliases={"kafka": ["event streaming"], "python": [], "sql": []}, labels={})
    s = parse_scores(req(["Event Streaming", "python"]), jd)
    assert s == {"must_precision": 1.0, "must_recall": 2 / 3}


def test_parse_scores_empty_parse():
    jd = JD(id="j", company="c", title="t", text="", gold=req(["sql"]), aliases={"sql": []}, labels={})
    assert parse_scores(req([]), jd) == {"must_precision": 0.0, "must_recall": 0.0}


class StubLLM:
    def __init__(self, out):
        self.out, self.prompts = out, []

    def generate(self, prompt, schema, system=""):
        self.prompts.append(prompt)
        return self.out


def test_parse_jd_uses_llm():
    fixed = req(["kafka"])
    stub = StubLLM(fixed)
    assert parse_jd("We need Kafka.", stub) == fixed
    assert "We need Kafka." in stub.prompts[0]
