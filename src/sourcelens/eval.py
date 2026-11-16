"""Eval pipeline: score every stage against the labelled synthetic set.

    python -m sourcelens.eval --mode live     # call Gemini, record every response
    python -m sourcelens.eval --mode replay   # rerun offline from the recording, $0
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .data import load_candidates, load_jds
from .jd_parser import parse_jd, parse_scores
from .llm import ROOT, LLM
from .metrics import ndcg_at_k, precision_at_k, recall_at_k
from .models import JD, Candidate
from .outreach import check_outreach, draft_outreach
from .rerank import rerank, score_candidate
from .retrieve import Retriever, query_text

TOP_K = 15
TRAPS = ("stuffer", "synonym", "location", "seniority", "injection")
OUTREACH_TOP = 3
# Estimated USD per 1M tokens, gemini-2.5-flash paid-tier list price (Nov 2026). Estimate only.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50


def _eval_jd(jd: JD, cands: list[Candidate], llm) -> dict:
    by_id = {c.id: c for c in cands}
    trap_of = {c.id: c.trap for c in cands}
    r = Retriever(cands, llm)
    q = query_text(jd.gold)

    parsed = parse_jd(jd.text, llm)
    out: dict = {"parse": parse_scores(parsed, jd)}

    out["retrieval"] = {}
    for m in ("bm25", "dense", "hybrid"):
        ranked = getattr(r, m)(q, TOP_K)
        out["retrieval"][m] = {"recall@15": recall_at_k(ranked, jd.labels, TOP_K),
                               "ndcg@10": ndcg_at_k(ranked, jd.labels, 10)}

    hyb = r.hybrid(q, TOP_K)
    cards = rerank([by_id[i] for i in hyb], jd.gold, llm)
    reranked = [c.candidate_id for c in cards]
    out["rerank"] = {"hybrid": {"ndcg@5": ndcg_at_k(hyb, jd.labels, 5), "p@5": precision_at_k(hyb, jd.labels, 5)},
                     "reranked": {"ndcg@5": ndcg_at_k(reranked, jd.labels, 5),
                                  "p@5": precision_at_k(reranked, jd.labels, 5)}}

    claimed = [ch for c in cards for ch in c.checks if ch.met != "no" and ch.evidence_quote.strip()]
    grounded = sum(ch.grounded for ch in claimed)
    out["grounding"] = {"claimed": len(claimed), "grounded": grounded,
                        "rate": grounded / len(claimed) if claimed else 0.0}

    out["traps"] = {stage: {t: sum(trap_of[i] == t for i in ranking[:5]) for t in TRAPS}
                    for stage, ranking in (("hybrid", hyb), ("reranked", reranked))}

    # Injected profiles rarely survive retrieval, so score them directly: this tests the
    # re-rank defence itself, not the search that happened to filter them out.
    inj = [score_candidate(c, jd.gold, llm) for c in cands if c.trap == "injection"]
    top = max((c.score for c in inj), default=0.0)
    out["injection_probe"] = {
        "max_score": top,
        "best_rank_if_inserted": 1 + sum(c.score > top for c in cards),
        "claimed": sum(ch.met != "no" for c in inj for ch in c.checks),
        "ungrounded": sum(c.ungrounded for c in inj),
        "profiles_scored": len(inj)}

    drafts = []
    for card in cards[:OUTREACH_TOP]:
        c = by_id[card.candidate_id]
        try:
            o = draft_outreach(c, card, jd, llm)
            drafts.append({"outreach": o.model_dump(), "check": check_outreach(o, c, card).model_dump()})
        except ValueError as e:
            drafts.append({"outreach": None, "check": {"passed": False, "error": str(e)}})
    out["outreach"] = {"pass_rate": mean(d["check"]["passed"] for d in drafts), "n": len(drafts)}

    e2e_hyb = r.hybrid(query_text(parsed), TOP_K)
    e2e = [c.candidate_id for c in rerank([by_id[i] for i in e2e_hyb], parsed, llm)]
    out["end_to_end"] = {"ndcg@5": ndcg_at_k(e2e, jd.labels, 5)}

    out["sample"] = {"top": [c.model_dump() for c in cards[:OUTREACH_TOP]], "drafts": drafts,
                     "labels": {c.candidate_id: jd.labels[c.candidate_id] for c in cards[:5]},
                     "top5": reranked[:5]}
    return out


def _mean_tree(items: list[dict]) -> dict:
    out = {}
    for k, v in items[0].items():
        vals = [i[k] for i in items]
        out[k] = _mean_tree(vals) if isinstance(v, dict) else mean(vals)
    return out


def evaluate(llm, candidates: list[Candidate], jds: list[JD]) -> dict:
    per_jd = {jd.id: _eval_jd(jd, candidates, llm) for jd in jds}
    rows = list(per_jd.values())
    summary = {k: _mean_tree([r[k] for r in rows]) for k in ("parse", "retrieval", "rerank", "outreach", "end_to_end")}
    summary["grounding"] = {
        "claimed": sum(r["grounding"]["claimed"] for r in rows),
        "grounded": sum(r["grounding"]["grounded"] for r in rows)}
    summary["grounding"]["rate"] = (summary["grounding"]["grounded"] / summary["grounding"]["claimed"]
                                    if summary["grounding"]["claimed"] else 0.0)
    probes = [r["injection_probe"] for r in rows]
    summary["injection_probe"] = {
        "profiles_scored": sum(p["profiles_scored"] for p in probes),
        "max_score": max(p["max_score"] for p in probes),
        "best_rank_if_inserted": min(p["best_rank_if_inserted"] for p in probes),
        "claimed": sum(p["claimed"] for p in probes),
        "ungrounded": sum(p["ungrounded"] for p in probes)}
    summary["traps"] = {stage: {t: sum(r["traps"][stage][t] for r in rows) for t in TRAPS}
                        for stage in ("hybrid", "reranked")}
    s = llm.stats
    # reported cost when the backend gives one (Claude Code), else a Gemini list-price estimate
    usd = s.cost_usd or s.tokens_in / 1e6 * PRICE_IN_PER_M + s.tokens_out / 1e6 * PRICE_OUT_PER_M
    summary["cost"] = {**asdict(s), "est_usd_total": usd, "est_usd_per_jd": usd / len(jds),
                       "cost_source": "reported by backend" if s.cost_usd else "estimated at Gemini list price",
                       "model": getattr(llm, "model", "stub"), "embed_model": getattr(llm, "embed_model", "stub")}
    return {"summary": summary, "per_jd": per_jd}


def _f(x: float) -> str:
    return f"{x:.2f}"


def to_markdown(res: dict) -> str:
    s = res["summary"]
    L = ["# SourceLens eval report", "",
         f"Model `{s['cost']['model']}`, embeddings `{s['cost']['embed_model']}`. "
         f"{len(res['per_jd'])} JDs, means across JDs.", "",
         "## Retrieval (top 15, gold requirements)", "", "| method | recall@15 | nDCG@10 |", "|---|---|---|"]
    for m, v in s["retrieval"].items():
        L.append(f"| {m} | {_f(v['recall@15'])} | {_f(v['ndcg@10'])} |")
    L += ["", "## Re-rank (hybrid top 15 re-scored)", "", "| ranking | nDCG@5 | precision@5 |", "|---|---|---|"]
    for m, v in s["rerank"].items():
        L.append(f"| {m} | {_f(v['ndcg@5'])} | {_f(v['p@5'])} |")
    g = s["grounding"]
    L += ["", "## Grounding, outreach, JD parsing", "", "| metric | value |", "|---|---|",
          f"| evidence quotes found verbatim in profile | {g['grounded']}/{g['claimed']} ({_f(g['rate'])}) |",
          f"| outreach drafts passing all checks | {_f(s['outreach']['pass_rate'])} |",
          f"| JD parse must-have precision / recall | {_f(s['parse']['must_precision'])} / "
          f"{_f(s['parse']['must_recall'])} |",
          f"| end-to-end nDCG@5 (parsed requirements) | {_f(s['end_to_end']['ndcg@5'])} |"]
    L += ["", "## Trap candidates in top 5 (summed over JDs, lower is better)", "",
          "| trap | hybrid | reranked |", "|---|---|---|"]
    names = {"stuffer": "Keyword stuffer", "synonym": "Synonym (should rank: relevant)", "location": "Wrong location",
             "seniority": "Too junior", "injection": "Injection"}
    for t, n in names.items():
        L.append(f"| {n} | {s['traps']['hybrid'][t]} | {s['traps']['reranked'][t]} |")
    ip = s["injection_probe"]
    L += ["", "## Injection probe (injected profiles scored directly, bypassing retrieval)", "",
          "| metric | value |", "|---|---|",
          f"| injected profile x JD pairs scored | {ip['profiles_scored']} |",
          f"| highest score reached (0-1) | {_f(ip['max_score'])} |",
          f"| best rank it would reach among the re-ranked 15 | {ip['best_rank_if_inserted']} |",
          f"| requirements the model marked met | {ip['claimed']} |",
          f"| of those, quotes not found in the profile | {ip['ungrounded']} |"]
    c = s["cost"]
    L += ["", "## Cost", "", f"- live calls: {c['calls']}, cache hits: {c['cache_hits']}",
          f"- tokens in/out: {c['tokens_in']} / {c['tokens_out']}",
          f"- USD per JD: ${c['est_usd_per_jd']:.4f} ({c['cost_source']})",
          f"- model latency total: {c['latency_s']:.1f}s"]
    return "\n".join(L) + "\n"


def write_report(res: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(to_markdown(res), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sourcelens.eval")
    ap.add_argument("--mode", choices=["live", "replay"], default="replay")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    res = evaluate(LLM.from_env(a.mode), load_candidates(), load_jds())
    write_report(res, out)
    print(to_markdown(res))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
