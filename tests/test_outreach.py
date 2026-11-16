import pytest

from sourcelens.models import JD, Candidate, Experience, Requirements
from sourcelens.outreach import Outreach, OutreachOut, check_outreach, draft_outreach
from sourcelens.rerank import RequirementCheck, Scorecard

C = Candidate(id="c1", name="Asha Rao", headline="Backend Engineer", location="Bengaluru", open_to_remote=False,
              years=6, skills=["Kafka"], experience=[Experience(title="SE", company="Acme", start="2020", end="Present",
                                                                bullets=["Built a Kafka pipeline for payments."])])


def card(grounded=True):
    return Scorecard(candidate_id="c1", location_ok=True, years_ok=True, score=0.8, ungrounded=0, checks=[
        RequirementCheck(requirement="kafka", kind="must", met="yes",
                         evidence_quote="Built a Kafka pipeline for payments", grounded=grounded),
        RequirementCheck(requirement="postgresql", kind="must", met="no", evidence_quote="", grounded=False)])


GOOD = Outreach(candidate_id="c1", subject="Senior Backend role at Hirewise Labs",
                body="Hi Asha, your Kafka pipeline for payments caught my eye. We are hiring a Senior Backend "
                     "Engineer in Bengaluru. Open to a short call this week? Talent team, Hirewise Labs")


def test_check_passes_good_message():
    r = check_outreach(GOOD, C, card())
    assert r.passed and r.names_candidate and r.cites_evidence and r.no_placeholders


def test_check_flags_placeholder():
    bad = GOOD.model_copy(update={"body": "Hi [Name], your Kafka work is great. Talent team"})
    assert not check_outreach(bad, C, card()).no_placeholders


def test_check_flags_long():
    long = GOOD.model_copy(update={"body": "Hi Asha, Kafka " + "word " * 118})
    r = check_outreach(long, C, card())
    assert r.word_count == 121 and not r.passed


def test_check_requires_evidence():
    vague = GOOD.model_copy(update={"body": "Hi Asha, you look great for our role. Talent team, Hirewise Labs"})
    assert not check_outreach(vague, C, card()).cites_evidence


def test_ungrounded_evidence_does_not_count():
    assert not check_outreach(GOOD, C, card(grounded=False)).cites_evidence


class StubLLM:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, schema, system=""):
        self.prompt = prompt
        return OutreachOut(subject=GOOD.subject, body=GOOD.body)


JDX = JD(id="j", company="Hirewise Labs", title="Senior Backend Engineer", text="...",
         gold=Requirements(title="t", must_have=[], nice_to_have=[], min_years=0, location="x", remote_ok=False),
         aliases={}, labels={})


def test_draft_gets_only_grounded_evidence():
    stub = StubLLM()
    o = draft_outreach(C, card(), JDX, stub)
    assert o.candidate_id == "c1"
    assert "Built a Kafka pipeline for payments" in stub.prompt
    assert "postgresql" not in stub.prompt.lower()


def test_draft_without_evidence_raises():
    with pytest.raises(ValueError):
        draft_outreach(C, card(grounded=False), JDX, StubLLM())
