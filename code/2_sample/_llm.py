"""Shared OpenAI helpers for Phase-3 clustering (Methods A and B).

- Loads the API key from $OPENAI_API_KEY only (via .env in the project root).
  The key is never logged, echoed, or written to any output file.
- Provides embedding + chat-JSON helpers and a token/cost tracker.

Pricing constants are USD per 1M tokens (approximate, for cost ESTIMATION only).
Token COUNTS come from the API response `usage` objects, never estimated.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# USD per 1,000,000 tokens (approximate list prices, May 2026).
PRICING = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def get_client():
    """Return an OpenAI client, loading the key from $OPENAI_API_KEY (.env supported).

    Stops with a clear error if the key is absent — never prompts, never logs the key.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to "
            f"{ROOT / '.env'} (OPENAI_API_KEY=...) or export it, then re-run."
        )
    from openai import OpenAI
    # per-request timeout so a stalled connection fails fast and retries
    # (instead of hanging on the SDK's long default), with bounded retries.
    return OpenAI(api_key=key, timeout=60.0, max_retries=4)


class Usage:
    """Accumulates token usage per model and computes estimated cost."""

    def __init__(self):
        import threading
        self.by_model: dict[str, dict[str, int]] = {}
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, model: str, usage_obj) -> None:
        with self._lock:  # thread-safe for concurrent classification calls
            self.calls += 1
            m = self.by_model.setdefault(model, {"input": 0, "output": 0})
            # embeddings usage has prompt_tokens only; chat has prompt+completion
            m["input"] += int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            m["output"] += int(getattr(usage_obj, "completion_tokens", 0) or 0)

    def cost(self) -> float:
        total = 0.0
        for model, t in self.by_model.items():
            p = PRICING.get(model, {"input": 0.0, "output": 0.0})
            total += t["input"] / 1e6 * p["input"] + t["output"] / 1e6 * p["output"]
        return total

    def to_dict(self) -> dict:
        return {"calls": self.calls, "by_model": self.by_model,
                "estimated_cost_usd": round(self.cost(), 4),
                "pricing_used_usd_per_1m": PRICING}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def summary(self) -> str:
        parts = [f"{m}: in={t['input']:,} out={t['output']:,}" for m, t in self.by_model.items()]
        return f"calls={self.calls} | " + " | ".join(parts) + f" | est ${self.cost():.4f}"


def embed_texts(client, texts: list[str], model: str, usage: Usage,
                batch_size: int = 256) -> "list[list[float]]":
    """Embed texts in batches; returns list of vectors in input order.

    Prints per-batch progress (flushed) so a stall is visible immediately.
    """
    import sys
    out: list[list[float]] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, len(texts), batch_size), start=1):
        chunk = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        usage.add(model, resp.usage)
        out.extend(d.embedding for d in resp.data)
        print(f"    embed batch {b}/{n_batches} ({len(out)}/{len(texts)})", flush=True)
    return out


def chat_json(client, model: str, system: str, user: str, usage: Usage,
              temperature: float = 0.0, max_retries: int = 5):
    """One chat call expecting a JSON object back (uses response_format json_object).

    Returns the parsed object. Retries on transient errors with backoff.
    """
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            usage.add(model, resp.usage)
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # transient (rate limit, 5xx, parse) -> backoff
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"chat_json failed after {max_retries} retries: {last}")