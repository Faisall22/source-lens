"""Run the pipeline on one JD and print the shortlist.

    python -m sourcelens.run data/jds/llm-engineer.yaml
    python -m sourcelens.run data/jds/llm-engineer.yaml --agent
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .agent import run_agent
from .data import load_agent_requests, load_candidates
from .jd_parser import parse_jd
from .llm import ROOT, LLM
from .models import JD
from .outreach import draft_outreach
from .rerank import rerank
from .retrieve import Retriever, query_text


def _print_card(card, c) -> None:
    print(f"\n{c.name} ({c.id}) - score {card.score:.2f} | {c.headline}, {c.location}, {c.years:g} yrs")
    for ch in card.checks:
        if ch.met != "no":
            mark = "ok " if ch.grounded else "DROPPED (quote not in profile)"
            print(f"  [{ch.met:7}] {ch.requirement}: \"{ch.evidence_quote}\" {mark}")
    print(f"  location_ok={card.location_ok} years_ok={card.years_ok}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sourcelens.run")
    ap.add_argument("jd", type=Path)
    ap.add_argument("--mode", choices=["live", "replay"], default="live")
    ap.add_argument("--agent", action="store_true", help="drive the pipeline with the agent")
    a = ap.parse_args(argv)

    jd = JD(**yaml.safe_load(a.jd.read_text(encoding="utf-8")))
    cands = load_candidates()
    by_id = {c.id: c for c in cands}
    llm = LLM.from_env(a.mode)
    record: dict = {"jd": jd.id}

    if a.agent:
        request = next(r.request for r in load_agent_requests() if r.jd_id == jd.id)
        res = run_agent(request, jd, cands, llm)
        print(f"Request: {request}\n")
        for t in res.trace:
            print(f"step {t['step']}: {t['action']} {json.dumps(t['args'])}  - {t['thought']}")
        print(f"\nShortlist: {[by_id[i].name for i in res.shortlist]} (finished={res.finished})")
        for d in res.drafts:
            print(f"\n--- to {by_id[d.candidate_id].name}: {d.subject}\n{d.body}")
        record["agent"] = res.model_dump()
    else:
        req = parse_jd(jd.text, llm)
        print("Parsed requirements:", json.dumps(req.model_dump()))
        ids = Retriever(cands, llm).hybrid(query_text(req), 15)
        cards = rerank([by_id[i] for i in ids], req, llm)
        for card in cards[:5]:
            _print_card(card, by_id[card.candidate_id])
        drafts = []
        for card in cards[:3]:
            try:
                o = draft_outreach(by_id[card.candidate_id], card, jd, llm)
                drafts.append(o.model_dump())
                print(f"\n--- to {by_id[o.candidate_id].name}: {o.subject}\n{o.body}")
            except ValueError as e:
                print(f"\n--- skipped: {e}")
        record.update(requirements=req.model_dump(), top=[c.model_dump() for c in cards[:5]], drafts=drafts)

    out = ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
