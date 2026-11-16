"""Core data types shared by every stage."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Trap = Literal["stuffer", "synonym", "location", "seniority", "injection"]


class Experience(BaseModel):
    title: str
    company: str
    start: str
    end: str
    bullets: list[str]


class Candidate(BaseModel):
    id: str
    name: str
    headline: str
    location: str
    open_to_remote: bool
    years: float
    skills: list[str]
    experience: list[Experience]
    trap: Trap | None = None  # eval-only tag; never shown to the LLM

    def text(self) -> str:
        """Full profile text. Grounding checks quotes against exactly this."""
        lines = [
            self.headline,
            f"Location: {self.location}",
            f"Experience: {self.years:g} years",
            "Skills: " + ", ".join(self.skills),
        ]
        for e in self.experience:
            lines.append(f"{e.title} at {e.company} ({e.start} - {e.end})")
            lines.extend(f"- {b}" for b in e.bullets)
        return "\n".join(lines)


class Requirements(BaseModel):
    title: str
    must_have: list[str]
    nice_to_have: list[str]
    min_years: float
    location: str
    remote_ok: bool


class JD(BaseModel):
    id: str
    company: str
    title: str
    text: str
    gold: Requirements
    aliases: dict[str, list[str]]
    labels: dict[str, int]  # candidate id -> 0 no, 1 partial, 2 strong


class AgentRequest(BaseModel):
    id: str
    jd_id: str
    request: str
    count: int
    location: str | None = None
    min_years: float | None = None
