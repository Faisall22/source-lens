import numpy as np

from sourcelens.agent import AgentStep, run_agent
from sourcelens.models import JD, Candidate, Experience, Requirements
from sourcelens.outreach import OutreachOut
from sourcelens.rerank import CheckOut, ChecksOut

REQ = Requirements(title="Backend", must_have=["kafka"], nice_to_have=[], min_years=1,
                   location="Bengaluru", remote_ok=False)
JDX = JD(id="j", company="Hirewise Labs", title="Backend", text="Need Kafka in Bengaluru.", gold=REQ,
         aliases={"kafka": []}, labels={})


def cand(id, bullet):
    return Candidate(id=id, name=f"Asha {id}", headline="Engineer", location="Bengaluru", open_to_remote=False,
                     years=5, skills=[], experience=[Experience(title="SE", company="Acme", start="1", end="2",
                                                                bullets=[bullet])])


CANDS = [cand("c1", "Built a Kafka pipeline for payments."), cand("c2", "Designed React pages for a shop.")]


class ScriptedLLM:
    def __init__(self, steps):
        self.steps = list(steps)
        self.agent_prompts = []

    def generate(self, prompt, schema, system=""):
        if schema is AgentStep:
            self.agent_prompts.append(prompt)
            return self.steps.pop(0)
        if schema is Requirements:
            return REQ
        if schema is ChecksOut:
            if 'id="c1"' in prompt:
                return ChecksOut(checks=[CheckOut(requirement="kafka", met="yes",
                                                  evidence_quote="Built a Kafka pipeline for payments")])
            return ChecksOut(checks=[])
        if schema is OutreachOut:
            return OutreachOut(subject="Role", body="Hi Asha, your Kafka pipeline stood out. Talent team, Hirewise Labs")
        raise AssertionError(schema)

    def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        v = np.array([[1.0 if "kafka" in t.lower() else 0.1, 0.5] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def S(action, **kw):
    return AgentStep(thought="t", action=action, **kw)


def test_agent_happy_path():
    llm = ScriptedLLM([S("parse_jd"), S("search_candidates", query="kafka", k=2),
                       S("score_candidates", candidate_ids=["c1", "c2"]),
                       S("draft_outreach", candidate_ids=["c1"]), S("finish", candidate_ids=["c1"])])
    r = run_agent("find kafka people", JDX, CANDS, llm)
    assert r.finished and r.shortlist == ["c1"] and r.steps == 5
    assert [d.candidate_id for d in r.drafts] == ["c1"]
    assert [t["action"] for t in r.trace] == ["parse_jd", "search_candidates", "score_candidates",
                                               "draft_outreach", "finish"]
    assert "c1" in llm.agent_prompts[-1]  # observations are fed back


def test_agent_bad_id_is_observation():
    llm = ScriptedLLM([S("parse_jd"), S("score_candidates", candidate_ids=["nope"]), S("finish", candidate_ids=[])])
    r = run_agent("x", JDX, CANDS, llm)
    assert "unknown candidate" in str(r.trace[1]["observation"])
    assert r.finished


def test_agent_score_before_parse_is_observation():
    llm = ScriptedLLM([S("score_candidates", candidate_ids=["c1"]), S("finish", candidate_ids=[])])
    r = run_agent("x", JDX, CANDS, llm)
    assert "parse_jd" in str(r.trace[0]["observation"])


def test_agent_max_steps():
    llm = ScriptedLLM([S("parse_jd"), S("score_candidates", candidate_ids=["c2", "c1"])] + [S("parse_jd")] * 10)
    r = run_agent("x", JDX, CANDS, llm, max_steps=4)
    assert not r.finished and r.steps == 4
    assert r.shortlist[0] == "c1"


def test_agent_finish_keeps_only_scored_ids_in_score_order():
    llm = ScriptedLLM([S("parse_jd"), S("score_candidates", candidate_ids=["c2", "c1"]),
                       S("finish", candidate_ids=["c2", "c1"])])
    cands = CANDS + [cand("c3", "Did nothing relevant.")]
    llm.steps.insert(2, S("search_candidates", query="kafka", k=3))
    r = run_agent("x", JDX, cands, llm)
    assert r.shortlist == ["c1", "c2"]
