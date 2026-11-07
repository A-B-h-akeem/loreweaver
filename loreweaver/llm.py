"""Thin, robust wrapper around a local Ollama server.

Structured output strategy (PLAN.md §8A): grammar-constrained generation through /api/generate
(which, unlike /api/chat, honours `format=` with think=False on qwen3.5), with the JSON template still
in the prompt for semantics; if that fails, free generation -> json-repair -> retry with the error.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Type, TypeVar

import numpy as np
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

from .config import settings

log = logging.getLogger("loreweaver.llm")
T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


class LLMError(RuntimeError):
    pass


@dataclass
class CallStats:
    calls: int = 0
    seconds: float = 0.0
    tokens_out: int = 0
    json_calls: int = 0
    json_failures: int = 0          # constrained attempts that still failed validation (fell back)
    ttft: list = field(default_factory=list)
    by_agent: dict = field(default_factory=dict)

    def add(self, agent: str, seconds: float, tokens: int = 0):
        self.calls += 1
        self.seconds += seconds
        self.tokens_out += tokens
        a = self.by_agent.setdefault(agent, {"calls": 0, "seconds": 0.0})
        a["calls"] += 1
        a["seconds"] += seconds


def clean_text(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_json_block(text: str) -> str:
    """Return the outermost {...} object in text (models sometimes add prose around it)."""
    text = _FENCE_RE.sub("", clean_text(text))
    start = text.find("{")
    if start < 0:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # truncated: let json-repair close it


def parse_model(text: str, model_cls: Type[T]) -> T:
    block = extract_json_block(text)
    try:
        return model_cls.model_validate_json(block)
    except ValidationError:
        repaired = repair_json(block, return_objects=True)
        if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
            repaired = repaired[0]
        if not isinstance(repaired, dict):
            raise
        return model_cls.model_validate(repaired)


class OllamaLLM:
    def __init__(self, model: str | None = None, embed_model: str | None = None, host: str | None = None):
        import ollama  # imported lazily so tests never need a server

        self.model = model or settings.llm_model
        self.embed_model = embed_model or settings.embed_model
        self.client = ollama.Client(host=host or settings.ollama_host)
        self.stats = CallStats()
        self._lock = threading.Lock()  # one generation at a time: keeps VRAM predictable

    # ------------------------------------------------------------------ helpers
    def _options(self, temperature: float, max_tokens: int | None) -> dict:
        opts = {"num_ctx": settings.num_ctx, "temperature": temperature,
                "top_p": 0.9, "top_k": 40, "presence_penalty": 0.4}
        if max_tokens:
            opts["num_predict"] = max_tokens
        return opts

    @staticmethod
    def _messages(system: str, prompt: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    # ------------------------------------------------------------------ generation
    def stream(self, system: str, prompt: str, temperature: float = 0.8,
               max_tokens: int | None = None, agent: str = "narrator") -> Iterator[str]:
        t0, n = time.time(), 0
        with self._lock:
            for chunk in self.client.chat(model=self.model, messages=self._messages(system, prompt),
                                          stream=True, think=False, keep_alive=settings.keep_alive,
                                          options=self._options(temperature, max_tokens)):
                piece = chunk.message.content or ""
                if piece:
                    if n == 0:
                        self.stats.ttft.append(time.time() - t0)      # time to first token (what players feel)
                    n += 1
                    yield piece
        self.stats.add(agent, time.time() - t0, n)

    def warm(self, system: str, prefix: str):
        """Prefill the server's KV cache with a prompt prefix (1 token generated). Ollama keeps one cached
        sequence per slot and reuses the longest common prefix (measured: 3.77 s -> 1.59 s for 4.5k tokens)."""
        t0 = time.time()
        with self._lock:
            self.client.chat(model=self.model, messages=self._messages(system, prefix), think=False,
                             keep_alive=settings.keep_alive, options=self._options(0.0, 1))
        self.stats.add("warmup", time.time() - t0)

    def complete(self, system: str, prompt: str, temperature: float = 0.7,
                 max_tokens: int | None = None, agent: str = "llm") -> str:
        t0 = time.time()
        with self._lock:
            r = self.client.chat(model=self.model, messages=self._messages(system, prompt), think=False,
                                 keep_alive=settings.keep_alive, options=self._options(temperature, max_tokens))
        self.stats.add(agent, time.time() - t0, r.eval_count or 0)
        return clean_text(r.message.content)

    def complete_constrained(self, system: str, prompt: str, schema: dict, temperature: float,
                             max_tokens: int | None, agent: str) -> str:
        """Grammar-constrained generation. /api/generate applies the JSON-schema mask with think=False;
        /api/chat does not on Ollama 0.17.6-0.30.7 (ollama#14645, verified on 0.23.2)."""
        t0 = time.time()
        with self._lock:
            r = self.client.generate(model=self.model, system=system, prompt=prompt, format=schema, think=False,
                                     keep_alive=settings.keep_alive, options=self._options(temperature, max_tokens))
        self.stats.add(agent, time.time() - t0, r.eval_count or 0)
        return clean_text(r.response)

    def json(self, system: str, prompt: str, model_cls: Type[T], temperature: float = 0.1,
             max_tokens: int | None = 1500, agent: str = "json", retries: int = 1) -> T:
        # Compact output: grammar-constrained JSON is otherwise pretty-printed, and restating empty/unchanged
        # fields wastes decode time (A/B on 5 real scenes: -38% tokens, -37% time, same validity; PLAN.md §11).
        sys_full = system + "\n\nRespond with ONLY one JSON object. No markdown, no commentary."
        if settings.compact_json:
            sys_full += (" Output MINIFIED JSON on ONE line with no indentation or line breaks. Omit any list that "
                         "would be empty and any field you would leave empty or unchanged.")
        self.stats.json_calls += 1
        last_err: Exception | None = None
        try:  # 1) grammar-constrained: the structure is guaranteed, pydantic checks the semantics
            raw = self.complete_constrained(sys_full, prompt, model_cls.model_json_schema(), temperature,
                                            max_tokens, agent)
            return parse_model(raw, model_cls)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_err = e
            self.stats.json_failures += 1
            log.warning("%s: constrained JSON invalid (often truncation): %s", agent, str(e)[:200])
        except Exception as e:  # noqa: BLE001 - e.g. a server that rejects the schema
            last_err = e
            log.warning("%s: constrained generation failed: %s", agent, e)
        attempt_prompt = prompt  # 2) fallback: free generation -> repair -> retry with the error fed back
        for attempt in range(retries + 1):
            raw = self.complete(sys_full, attempt_prompt, temperature, max_tokens, agent=agent)
            try:
                return parse_model(raw, model_cls)
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("%s: invalid JSON (fallback attempt %d): %s", agent, attempt + 1, str(e)[:200])
                attempt_prompt = (f"{prompt}\n\nYour previous reply was invalid:\n{str(e)[:600]}\n"
                                  "Reply again with ONLY the corrected JSON object.")
        raise LLMError(f"{agent}: could not obtain valid JSON: {last_err}")

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        t0, vecs = time.time(), []
        for i in range(0, len(texts), 32):
            r = self.client.embed(model=self.embed_model, input=[t[:6000] for t in texts[i:i + 32]],
                                  keep_alive=settings.keep_alive, options={"num_ctx": settings.embed_ctx})
            vecs.extend(r.embeddings)
        self.stats.add("embed", time.time() - t0)
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)

    # ------------------------------------------------------------------ health
    def health(self) -> tuple[bool, str]:
        try:
            names = {m.model for m in self.client.list().models}
        except Exception as e:  # noqa: BLE001
            return False, f"Cannot reach Ollama at {settings.ollama_host}: {e}"
        missing = [m for m in (self.model, self.embed_model)
                   if m not in names and f"{m}:latest" not in names]
        if missing:
            return False, "Missing models: " + ", ".join(missing) + " (run `ollama pull <model>`)"
        return True, f"Ollama OK · {self.model} + {self.embed_model}"
