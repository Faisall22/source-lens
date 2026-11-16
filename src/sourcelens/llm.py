"""Gemini client with a record/replay disk cache.

Every LLM and embedding call goes through `LLM`. In `live` mode a call that is
not cached goes to the backend and is written to disk as soon as it returns, so
an interrupted run resumes where it stopped. In `replay` mode a cache miss is an
error: evals rerun offline, for free, with identical numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / ".cache" / "fixtures"
MAX_ATTEMPTS = 4
EMBED_DIM = 768

T = TypeVar("T", bound=BaseModel)


class CacheMiss(RuntimeError):
    pass


class Backend(Protocol):
    def generate_json(self, model: str, system: str, prompt: str,
                      schema: type[BaseModel]) -> tuple[str, dict]: ...

    def embed(self, model: str, texts: list[str], task: str) -> list[list[float]]: ...


class GeminiBackend:
    def __init__(self, api_key: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def generate_json(self, model, system, prompt, schema):
        from google.genai import types

        resp = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
                # ponytail: no thinking budget; structured extraction does not need it and free tier is slow
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        u = resp.usage_metadata
        usage = {"in": u.prompt_token_count or 0,
                 "out": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)}
        return resp.text, usage

    def embed(self, model, texts, task):
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            resp = self.client.models.embed_content(
                model=model,
                contents=texts[i:i + 100],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM),
            )
            out.extend(e.values for e in resp.embeddings)
        return out


class ClaudeCLIBackend:
    """Structured output through a headless Claude Code call (`claude -p`), no API key.

    The session is stripped to a bare JSON API: own system prompt, no tools, no MCP
    servers, no skills, no settings, no saved session. That keeps each call to ~1K tokens.
    """

    def __init__(self, model: str = "sonnet", runner: Callable = subprocess.run, timeout: int = 300):
        self.model, self.runner, self.timeout = model, runner, timeout

    def generate_json(self, model, system, prompt, schema):
        cmd = ["claude", "-p", prompt, "--output-format", "json",
               "--json-schema", json.dumps(schema.model_json_schema()),
               "--model", self.model, "--system-prompt", system or "You are a JSON API. Follow the schema.",
               "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
               "--disable-slash-commands", "--setting-sources", "", "--no-session-persistence"]
        proc = self.runner(cmd, capture_output=True, text=True, timeout=self.timeout)
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude -p returned non-JSON: {proc.stdout[:200]!r} {proc.stderr[:200]!r}") from e
        if out.get("is_error") or out.get("structured_output") is None:
            raise RuntimeError(f"claude -p failed: {out.get('result') or out.get('terminal_reason')}")
        u = out.get("usage", {})
        usage = {"in": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                 + u.get("cache_read_input_tokens", 0),
                 "out": u.get("output_tokens", 0), "usd": out.get("total_cost_usd", 0.0)}
        return json.dumps(out["structured_output"]), usage

    def embed(self, model, texts, task):
        raise NotImplementedError("Claude Code has no embeddings; pair it with an embedder")


class FastEmbedBackend:
    """Local CPU embeddings (fastembed, ONNX). No key, no network after the first model download."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model)

    def embed(self, model, texts, task):
        return [list(map(float, v)) for v in self.model.embed(list(texts))]


class CompositeBackend:
    """One backend for generation, another for embeddings."""

    def __init__(self, generator, embedder):
        self.generator, self.embedder = generator, embedder

    def generate_json(self, model, system, prompt, schema):
        return self.generator.generate_json(model, system, prompt, schema)

    def embed(self, model, texts, task):
        return self.embedder.embed(model, texts, task)


@dataclass
class Stats:
    calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LLM:
    def __init__(self, mode: Literal["live", "replay"] = "live", cache_dir: Path = DEFAULT_CACHE,
                 backend: Backend | None = None, rpm: int | None = None,
                 model: str | None = None, embed_model: str | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.embed_model = embed_model or os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
        self.interval = 60.0 / (rpm or int(os.getenv("GEMINI_RPM", "8")))
        self.sleep = sleep
        self.stats = Stats()
        self._last = 0.0

    @classmethod
    def from_env(cls, mode: Literal["live", "replay"] = "live") -> "LLM":
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        key = os.getenv("GEMINI_API_KEY")
        provider = os.getenv("SOURCELENS_PROVIDER") or ("gemini" if key else "claude-cli")
        if provider == "claude-cli":
            # ponytail: model names live in the cache key, so Gemini and Claude runs never mix
            gen_model = os.getenv("CLAUDE_MODEL", "sonnet")
            emb_model = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
            backend = (CompositeBackend(ClaudeCLIBackend(gen_model), FastEmbedBackend(emb_model))
                       if mode == "live" else None)
            return cls(mode=mode, backend=backend, rpm=int(os.getenv("CLAUDE_RPM", "60")),
                       model=f"claude-cli:{gen_model}", embed_model=f"fastembed:{emb_model}")
        if mode == "live" and not key:
            raise RuntimeError("GEMINI_API_KEY not set (add it to .env, or use --mode replay)")
        return cls(mode=mode, backend=GeminiBackend(key) if mode == "live" else None)

    # -- internals -------------------------------------------------------
    def _path(self, kind: str, key: str) -> Path:
        return self.cache_dir / kind / f"{key}.json"

    def _live(self, kind: str, fn: Callable[[], T]) -> T:
        if self.mode == "replay" or self.backend is None:
            raise CacheMiss(f"{kind} cache miss in replay mode (run once with --mode live)")
        for attempt in range(MAX_ATTEMPTS):
            wait = self.interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            try:
                return fn()
            except Exception:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                self.sleep(min(60, 10 * 2 ** attempt))  # free-tier 429s ask for ~30-60s
        raise AssertionError("unreachable")

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    # -- public ----------------------------------------------------------
    def generate(self, prompt: str, schema: type[T], system: str = "") -> T:
        key = _key("generate", self.model, system, prompt, schema.model_json_schema())
        path = self._path("generate", key)
        if path.exists():
            self.stats.cache_hits += 1
            return schema.model_validate_json(json.loads(path.read_text(encoding="utf-8"))["response"])

        def call():
            t0 = time.monotonic()
            text, usage = self.backend.generate_json(self.model, system, prompt, schema)
            obj = schema.model_validate_json(text)  # invalid JSON raises -> retried
            return obj, text, usage, time.monotonic() - t0

        obj, text, usage, dt = self._live("generate", call)
        self._write(path, {"model": self.model, "response": text, "usage": usage, "latency_s": dt})
        self.stats.calls += 1
        self.stats.tokens_in += usage["in"]
        self.stats.tokens_out += usage["out"]
        self.stats.latency_s += dt
        self.stats.cost_usd += usage.get("usd", 0.0)
        return obj

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        keys = [_key("embed", self.embed_model, task, t) for t in texts]
        vecs: dict[str, list[float]] = {}
        missing = []
        for k, t in zip(keys, texts):
            p = self._path("embed", k)
            if p.exists():
                vecs[k] = json.loads(p.read_text(encoding="utf-8"))["vector"]
                self.stats.cache_hits += 1
            elif k not in vecs:
                missing.append((k, t))
                vecs[k] = []
        if missing:
            todo = [t for _, t in missing]
            out = self._live("embed", lambda: self.backend.embed(self.embed_model, todo, task))
            self.stats.calls += 1
            for (k, _), v in zip(missing, out):
                vecs[k] = list(v)
                self._write(self._path("embed", k), {"model": self.embed_model, "vector": vecs[k]})
        arr = np.array([vecs[k] for k in keys], dtype=float)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)
