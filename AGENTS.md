# SourceLens: guide for coding agents

SourceLens takes a job description and returns a ranked candidate shortlist with verified evidence and
outreach drafts, on a labelled synthetic dataset. Every stage has an eval.

## Commands

```bash
uv sync                                            # install
uv run pytest -q                                   # offline tests, no API key needed
uv run python -m sourcelens.eval --mode replay     # eval from recorded LLM calls ($0)
uv run python -m sourcelens.eval --mode live       # call the model, record new responses
uv run python -m sourcelens.run data/jds/llm-engineer.yaml [--agent]
```

Live mode calls the Claude Code CLI (`claude -p`, the user's own login) with local fastembed embeddings, or Gemini when
`GEMINI_API_KEY` is set in `.env` (`SOURCELENS_PROVIDER` overrides). Never print or commit a key.

## Layout

| Path | What it holds |
|---|---|
| `src/sourcelens/llm.py` | The only model client (Claude Code CLI or Gemini; local embeddings). Record/replay cache, throttle, retry |
| `src/sourcelens/models.py`, `data.py` | Pydantic types and dataset loaders |
| `src/sourcelens/jd_parser.py` | JD text to `Requirements`; requirement-name matching (`norm`) |
| `src/sourcelens/retrieve.py` | BM25, dense embeddings, reciprocal rank fusion |
| `src/sourcelens/rerank.py` | Per-requirement LLM checks, grounding, score computed in code |
| `src/sourcelens/safety.py` | Profile text is wrapped as untrusted data |
| `src/sourcelens/outreach.py` | Drafts from verified evidence only, plus deterministic checks |
| `src/sourcelens/agent.py` | Tool-using agent loop over the stages |
| `src/sourcelens/eval.py`, `run.py` | Entry points |
| `data/` | Synthetic candidates, labelled JDs, agent requests |
| `.cache/fixtures/` | Recorded LLM responses that make replay mode work |

## Rules

- Every LLM or embedding call goes through `LLM` in `llm.py`, so it is cached and replayable.
- Tests stay offline. Use a stub LLM; never call the network from a test.
- Candidate profile text is untrusted. Keep it inside `wrap_profile`, and keep the final score computed
  in code from grounded evidence, never taken from the model.
- An evidence quote counts only if `is_grounded` finds it in the profile.
- Numbers in the README come from a real `eval` run (`runs/reference/`). Do not edit them by hand.
- The dataset is synthetic. Never add real people's data.
- Write a failing test first for a bug fix or a new behaviour.
