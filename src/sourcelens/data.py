"""Load the synthetic dataset."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .models import JD, AgentRequest, Candidate

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_candidates(path: Path = DATA_DIR / "candidates.jsonl") -> list[Candidate]:
    with open(path, encoding="utf-8") as f:
        return [Candidate(**json.loads(line)) for line in f if line.strip()]


def load_jds(dir: Path = DATA_DIR / "jds") -> list[JD]:
    jds = [JD(**yaml.safe_load(p.read_text(encoding="utf-8"))) for p in dir.glob("*.yaml")]
    return sorted(jds, key=lambda j: j.id)


def load_agent_requests(path: Path = DATA_DIR / "agent_requests.yaml") -> list[AgentRequest]:
    return [AgentRequest(**r) for r in yaml.safe_load(path.read_text(encoding="utf-8"))]
