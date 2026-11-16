import json
import subprocess

import numpy as np
import pytest
from pydantic import BaseModel

from sourcelens.llm import LLM, ClaudeCLIBackend, CompositeBackend


class Out(BaseModel):
    answer: str


def fake_run(payload, record):
    def run(cmd, **kw):
        record.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")
    return run


OK = {"is_error": False, "structured_output": {"answer": "hi"}, "total_cost_usd": 0.005,
      "usage": {"input_tokens": 2, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0,
                "output_tokens": 50}}


def test_claude_cli_returns_structured_output_and_usage():
    calls = []
    be = ClaudeCLIBackend(model="sonnet", runner=fake_run(OK, calls))
    text, usage = be.generate_json("sonnet", "be terse", "say hi", Out)
    assert json.loads(text) == {"answer": "hi"}
    assert usage == {"in": 1002, "out": 50, "usd": 0.005}
    cmd = calls[0]
    assert cmd[:2] == ["claude", "-p"] and "--json-schema" in cmd and "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "be terse"
    assert cmd[cmd.index("--model") + 1] == "sonnet"


def test_claude_cli_error_raises():
    be = ClaudeCLIBackend(runner=fake_run({"is_error": True, "result": "rate limited"}, []))
    with pytest.raises(RuntimeError, match="rate limited"):
        be.generate_json("sonnet", "", "x", Out)


class FakeEmbedder:
    def embed(self, model, texts, task):
        return [[1.0, float(len(t))] for t in texts]


def test_composite_routes_embed_and_tracks_cost(tmp_path):
    gen = ClaudeCLIBackend(runner=fake_run(OK, []))
    llm = LLM(mode="live", cache_dir=tmp_path, backend=CompositeBackend(gen, FakeEmbedder()), rpm=10_000,
              sleep=lambda s: None)
    assert llm.generate("q", Out).answer == "hi"
    assert llm.stats.cost_usd == pytest.approx(0.005)
    v = llm.embed(["ab"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
