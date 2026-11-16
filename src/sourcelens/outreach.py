"""Write outreach that cites only verified evidence, then check it in code."""
from __future__ import annotations

from pydantic import BaseModel

from .jd_parser import norm
from .models import JD, Candidate
from .rerank import Scorecard
from .safety import DATA_RULE

MAX_WORDS = 120
PLACEHOLDER_CHARS = set("{}[]<>")
PLACEHOLDER_WORDS = ("todo", "your name", "lorem")


class OutreachOut(BaseModel):
    subject: str
    body: str


class Outreach(BaseModel):
    candidate_id: str
    subject: str
    body: str


class OutreachCheck(BaseModel):
    word_count: int
    names_candidate: bool
    cites_evidence: bool
    no_placeholders: bool
    passed: bool


PROMPT = """Write a short recruiter outreach message to {first_name} about the {title} role at {company}.

Verified facts about {first_name} (the only facts you may mention about them):
{facts}

Rules:
- At most {max_words} words in the body. Plain, warm, specific; no hype.
- Open with {first_name}'s first name. Mention one or two of the verified facts in your own words.
- Say what the role is and ask for a short call.
- Never claim anything about {first_name} that is not in the verified facts.
- No placeholders or brackets. Sign off as: Talent team, {company}"""


def _evidence(card: Scorecard):
    return [ch for ch in card.checks if ch.grounded and ch.met in ("yes", "partial")]


def draft_outreach(c: Candidate, card: Scorecard, jd: JD, llm) -> Outreach:
    ev = _evidence(card)
    if not ev:
        raise ValueError(f"{c.id}: no grounded evidence to cite")
    facts = "\n".join(f'- {ch.requirement}: "{ch.evidence_quote}"' for ch in ev)
    out = llm.generate(PROMPT.format(first_name=c.name.split()[0], title=jd.title, company=jd.company,
                                     facts=facts, max_words=MAX_WORDS), OutreachOut, system=DATA_RULE)
    return Outreach(candidate_id=c.id, subject=out.subject, body=out.body)


def _trigrams(s: str) -> set[str]:
    w = norm(s).split()
    return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}


def check_outreach(o: Outreach, c: Candidate, card: Scorecard) -> OutreachCheck:
    body = norm(o.body)
    words = len(o.body.split())
    named = c.name.split()[0].lower() in o.body.lower()
    ev = _evidence(card)
    cites = any(norm(ch.requirement) in body for ch in ev) or any(
        _trigrams(ch.evidence_quote) & _trigrams(o.body) for ch in ev)
    lowered = o.body.lower()
    clean = not (PLACEHOLDER_CHARS & set(o.body)) and not any(w in lowered for w in PLACEHOLDER_WORDS)
    return OutreachCheck(word_count=words, names_candidate=named, cites_evidence=cites, no_placeholders=clean,
                         passed=words <= MAX_WORDS and named and cites and clean)
