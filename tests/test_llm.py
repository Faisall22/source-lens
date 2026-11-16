import json

import numpy as np
import pytest
from pydantic import BaseModel

from sourcelens.llm import LLM, CacheMiss


class Out(BaseModel):
    answer: str


class FakeBackend:
    def __init__(self, fail_times=0, bad_json_times=0):
        self.gen_calls = 0
        self.embed_calls: list[list[str]] = []
        self.fail_times = fail_times
        self.bad_json_times = bad_json_times

    def generate_json(self, model, system, prompt, schema):
        self.gen_calls += 1
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        if self.bad_json_times:
            self.bad_json_times -= 1
            return "not json", {"in": 1, "out": 1}
        return json.dumps({"answer": prompt.upper()}), {"in": 10, "out": 3}

    def embed(self, model, texts, task):
        self.embed_calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


def make(tmp_path, backend, mode="live"):
    return LLM(mode=mode, cache_dir=tmp_path, backend=backend, rpm=10_000, sleep=lambda s: None)


def test_live_records_then_replays(tmp_path):
    fb = FakeBackend()
    assert make(tmp_path, fb).generate("hi", Out).answer == "HI"
    replay = make(tmp_path, None, mode="replay")
    assert replay.generate("hi", Out).answer == "HI"
    assert fb.gen_calls == 1
    assert replay.stats.cache_hits == 1


def test_replay_miss_raises(tmp_path):
    with pytest.raises(CacheMiss, match="generate"):
        make(tmp_path, None, mode="replay").generate("never seen", Out)


def test_retry_then_cache(tmp_path):
    fb = FakeBackend(fail_times=1, bad_json_times=1)
    llm = make(tmp_path, fb)
    assert llm.generate("x", Out).answer == "X"
    assert fb.gen_calls == 3
    assert llm.stats.calls == 1
    assert llm.stats.tokens_in == 10
    assert list(tmp_path.rglob("*.json"))


def test_retry_gives_up_after_4(tmp_path):
    fb = FakeBackend(fail_times=10)
    with pytest.raises(RuntimeError):
        make(tmp_path, fb).generate("x", Out)
    assert fb.gen_calls == 4
    assert not list(tmp_path.rglob("*.json"))


def test_embed_normalized_and_per_text_cache(tmp_path):
    fb = FakeBackend()
    llm = make(tmp_path, fb)
    v = llm.embed(["abc", "de"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    llm.embed(["abc", "fghij"])
    assert fb.embed_calls == [["abc", "de"], ["fghij"]]


def test_embed_replay_miss_raises(tmp_path):
    with pytest.raises(CacheMiss, match="embed"):
        make(tmp_path, None, mode="replay").embed(["new"])
