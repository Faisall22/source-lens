"""Re-rank retrieved candidates with per-requirement LLM checks.

The LLM only says, per requirement, met yes/partial/no plus a quote from the
profile. Code then (1) drops any quote not found in the profile and (2) computes
the score. The LLM never states the final score.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from .jd_parser import norm
from .models import Candidate, Requirements
from .retrieve import city
from .safety import DATA_RULE, profile_text, wrap_profile

Met = Literal["yes", "partial", "no"]
WEIGHT = {"must": 2.0, "nice": 1.0}
VALUE = {"yes": 1.0, "partial": 0.5, "no": 0.0}
LOCATION_WEIGHT = YEARS_WEIGHT = 2.0
MIN_QUOTE_CHARS = 8


class CheckOut(BaseModel):
    requirement: str
    met: Met
    evidence_quote: str


class ChecksOut(BaseModel):
    checks: list[CheckOut]


class RequirementCheck(BaseModel):
    requirement: str
    kind: Literal["must", "nice"]
    met: Met
    evidence_quote: str
    grounded: bool


class Scorecard(BaseModel):
    candidate_id: str
    checks: list[RequirementCheck]
    location_ok: bool
    years_ok: bool
    score: float
    ungrounded: int


PROMPT = """Check this candidate against each requirement of the role "{title}".

Requirements:
{reqs}

For every requirement return:
- met: "yes" if the profile shows real work using it, "partial" if only adjacent or shallow, "no" otherwise.
- evidence_quote: one sentence copied word for word from the profile that proves it, or "" when met is "no".
  A skill listed only under Skills is "partial" at most.

{profile}"""

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _clean(s: str) -> str:
    s = " ".join(s.translate(_QUOTES).lower().split())
    return re.sub(r"^[-\s]+", "", s).rstrip(".")


def is_grounded(quote: str, profile: str) -> bool:
    q = _clean(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in _clean(profile)


def only_in_skills(quote: str, c: Candidate) -> bool:
    """True when the quote appears in the Skills line but in no work bullet or headline."""
    q = _clean(quote)
    rest = profile_text(c).replace("Skills: " + ", ".join(c.skills), "")
    return q in _clean("Skills: " + ", ".join(c.skills)) and q not in _clean(rest)


def location_ok(c: Candidate, req: Requirements) -> bool:
    return city(c.location) == city(req.location) or (req.remote_ok and c.open_to_remote)


def years_ok(c: Candidate, req: Requirements) -> bool:
    return c.years >= req.min_years


def compute_score(checks: list[RequirementCheck], loc_ok: bool, yrs_ok: bool) -> float:
    points = sum(WEIGHT[c.kind] * (VALUE[c.met] if c.grounded else 0.0) for c in checks)
    points += LOCATION_WEIGHT * loc_ok + YEARS_WEIGHT * yrs_ok
    total = sum(WEIGHT[c.kind] for c in checks) + LOCATION_WEIGHT + YEARS_WEIGHT
    return points / total


def score_candidate(c: Candidate, req: Requirements, llm) -> Scorecard:
    kinds = [(r, "must") for r in req.must_have] + [(r, "nice") for r in req.nice_to_have]
    reqs = "\n".join(f"- {r} ({'required' if k == 'must' else 'nice to have'})" for r, k in kinds)
    out = llm.generate(PROMPT.format(title=req.title, reqs=reqs, profile=wrap_profile(c)), ChecksOut, system=DATA_RULE)
    by_name = {norm(o.requirement): o for o in out.checks}
    text = profile_text(c)
    checks = []
    for r, kind in kinds:
        o = by_name.get(norm(r))
        met, quote = (o.met, o.evidence_quote) if o else ("no", "")
        grounded = met != "no" and is_grounded(quote, text)
        if grounded and met == "yes" and only_in_skills(quote, c):
            met = "partial"  # a listed keyword is not shown work
        checks.append(RequirementCheck(requirement=r, kind=kind, met=met, evidence_quote=quote, grounded=grounded))
    loc, yrs = location_ok(c, req), years_ok(c, req)
    return Scorecard(candidate_id=c.id, checks=checks, location_ok=loc, years_ok=yrs,
                     score=compute_score(checks, loc, yrs),
                     ungrounded=sum(ch.met != "no" and not ch.grounded for ch in checks))


def rerank(cands: list[Candidate], req: Requirements, llm) -> list[Scorecard]:
    cards = [score_candidate(c, req, llm) for c in cands]
    return sorted(cards, key=lambda s: (-s.score, s.candidate_id))
