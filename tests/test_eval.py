import hashlib
import json

import numpy as np

from sourcelens.data import load_candidates, load_jds
from sourcelens.eval import evaluate, write_report
from sourcelens.llm import Stats
from sourcelens.models import Requirements
from sourcelens.outreach import OutreachOut
from sourcelens.rerank import CheckOut, ChecksOut

JDS = load_jds()
CANDS = load_candidates()


class StubLLM:
    """Deterministic: parse returns gold, checks quote the first bullet when a must-have word appears in it."""

    def __init__(self):
        self.stats = Stats()

    def generate(self, prompt, schema, system=""):
        if schema is Requirements:
            return next(j.gold for j in JDS if j.text in prompt)
        if schema is ChecksOut:
            c = next(c for c in CANDS if f'id="{c.id}"' in prompt)
            jd = next(j for j in JDS if f'role "{j.gold.title}"' in prompt)
            out = []
            for r in jd.gold.must_have:
                for e in c.experience:
                    for b in e.bullets:
                        if r.split()[0] in b.lower():
                            out.append(CheckOut(requirement=r, met="yes", evidence_quote=b))
                            break
            return ChecksOut(checks=out)
        if schema is OutreachOut:
            name = prompt.split("message to ")[1].split(" ")[0]
            req = prompt.split("Verified facts")[1].split("- ")[1].split(":")[0]
            return OutreachOut(subject="Role", body=f"Hi {name}, your {req} work stood out. Talent team, Hirewise Labs")
        raise AssertionError(schema)

    def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        v = np.array([[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _numbers(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from _numbers(v)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield d


def test_evaluate_shape(tmp_path):
    res = evaluate(StubLLM(), CANDS, JDS)
    for key in ("parse", "retrieval", "rerank", "grounding", "traps", "outreach", "end_to_end", "cost"):
        assert key in res["summary"], key
    assert set(res["per_jd"]) == {j.id for j in JDS}
    for v in _numbers({k: res["summary"][k] for k in ("parse", "retrieval", "rerank", "end_to_end")}):
        assert 0.0 <= v <= 1.0
    assert res["summary"]["parse"]["must_recall"] == 1.0
    assert res["summary"]["grounding"]["rate"] == 1.0
    write_report(res, tmp_path)
    md = (tmp_path / "report.md").read_text()
    assert "| bm25 |" in md and "Injection" in md
    assert json.loads((tmp_path / "report.json").read_text())["summary"]


def test_injection_probe_scores_injected_profiles_directly(tmp_path):
    res = evaluate(StubLLM(), CANDS, JDS)
    probe = res["summary"]["injection_probe"]
    n_inj = sum(c.trap == "injection" for c in CANDS)
    assert probe["profiles_scored"] == n_inj * len(JDS)
    assert 0.0 <= probe["max_score"] <= 1.0
    assert probe["best_rank_if_inserted"] >= 1
    for jd_id, row in res["per_jd"].items():
        assert set(row["injection_probe"]) >= {"max_score", "best_rank_if_inserted", "claimed", "ungrounded"}
    write_report(res, tmp_path)
    assert "Injection probe" in (tmp_path / "report.md").read_text()
