"""Shared OpenAI client wrapper for the coding phase.

Responsibilities
----------------
- Load credentials from the .env file at the Coding/ root (never echo them).
- Build a JSON-Schema response format derived from the v3.1 system prompt's
  column_definitions. The LLM emits every coded column EXCEPT the two derived
  scores (codifiability_score, evaluability_score), which are excluded from
  the schema and computed in Python after the response is parsed.
- Expose code_skill(skill_record, system_prompt) -> dict that codes one skill
  and returns the full row in required_column_order, plus a single non-column
  key "_audit" with the raw API response, usage and call metadata.

Model policy (see decisions.md, 2026-05-23)
-------------------------------------------
- Primary: gpt-5.2 with temperature=0 and reasoning_effort="none" (passed via
  extra_body, since openai 1.55.0 exposes neither a native reasoning_effort
  kwarg nor the Responses API).
- Fallback: if gpt-5.2 rejects the deterministic configuration (a 400 about
  temperature / reasoning_effort) OR is unavailable (model-not-found / 404),
  the call is retried ONCE with gpt-4.1 (temperature=0, no reasoning_effort).
  The fallback is always recorded in the returned metadata (never silent). If
  gpt-4.1 also fails, the error propagates and the run stops.

Formula policy
--------------
codifiability_score and evaluability_score are computed in Python with
round_half_up (standard "round half up", NOT Python's banker's round()).
The LLM only supplies the components.

Secrets rule: OPENAI_API_KEY is read from the environment and passed to the
client. It is never logged, printed, or written to any output file.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

# --------------------------------------------------------------------------
# Run parameters (see decisions.md)
# --------------------------------------------------------------------------
MODEL = "gpt-5.2"            # primary
FALLBACK_MODEL = "gpt-4.1"   # used if gpt-5.2 rejects the deterministic config
TEMPERATURE = 0
REASONING_EFFORT = "none"    # gpt-5.2 only; passed via extra_body
MAX_RETRIES = 3              # retries on transient errors (rate limit / 5xx / timeout)
BACKOFF_BASE = 2.0           # seconds; exponential: BACKOFF_BASE * 2**attempt

# The two aggregate scores are derived in Python, never emitted by the LLM.
DERIVED_FIELDS = ("codifiability_score", "evaluability_score")

# Approximate pricing (USD per 1M tokens) for the cost estimate only; never
# affects coded values. gpt-5.2 figures are placeholders pending confirmation.
PRICING_PER_1M = {
    "gpt-5.2": {"input": 1.25, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}
_DEFAULT_PRICING = {"input": 1.25, "output": 10.00}

BASE_DIR = Path(__file__).resolve().parent.parent          # Coding/
INPUTS_DIR = BASE_DIR / "inputs"
_ENV_PATH = BASE_DIR / ".env"                              # Coding/.env


# --------------------------------------------------------------------------
# Input loading (shared by the run + compile scripts)
# --------------------------------------------------------------------------
def find_system_prompt_path(inputs_dir: Path = INPUTS_DIR) -> Path:
    """Locate the system prompt JSON (System_Prompt.json)."""
    for name in ("System_Prompt.json", "System Prompt.json"):
        candidate = inputs_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(inputs_dir.glob("System*Prompt*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"System prompt JSON not found under {inputs_dir}")


def load_system_prompt(inputs_dir: Path = INPUTS_DIR) -> dict:
    return json.loads(find_system_prompt_path(inputs_dir).read_text())


def find_corpus_json(inputs_dir: Path = INPUTS_DIR) -> Path:
    matches = sorted(inputs_dir.glob("agent_skills_corpus*updated.json"))
    if not matches:
        raise FileNotFoundError(f"Corpus JSON not found under {inputs_dir}")
    return matches[-1]


def load_corpus(inputs_dir: Path = INPUTS_DIR) -> tuple[list[dict], dict, Path]:
    """Return (skills, metadata, corpus_path) from the corpus JSON."""
    path = find_corpus_json(inputs_dir)
    data = json.loads(path.read_text())
    return data.get("skills", []), data.get("metadata", {}), path


# --------------------------------------------------------------------------
# Errors — all inherit LLMError so callers can "stop and report" on any of
# them. The constraint is: never silently fix schema/param/auth problems.
# --------------------------------------------------------------------------
class LLMError(Exception):
    """Base class for fatal coding-call errors that should stop the run."""


class AuthError(LLMError):
    """401 / 403 — stop immediately, do not retry."""


class TransientError(LLMError):
    """Rate-limit / timeout / 5xx that persisted past MAX_RETRIES."""


class SchemaViolation(LLMError):
    """Structured-output schema violation or unparseable / refused response.

    Per the run constraints this is surfaced, not fixed by reformatting in
    Python — it signals a prompt/schema issue for the next session.
    """


class FatalAPIError(LLMError):
    """Any other non-retryable 400-class error."""


# --------------------------------------------------------------------------
# Environment / client
# --------------------------------------------------------------------------
def load_env(env_path: Path = _ENV_PATH) -> None:
    """Load KEY=VALUE pairs from the Coding/.env file into os.environ.

    Uses python-dotenv if available, otherwise parses the file manually.
    Existing environment variables are not overwritten. Never prints the
    file's contents.
    """
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=str(env_path), override=False)
        return
    except Exception:
        pass  # fall back to manual parsing

    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def get_api_key() -> str:
    """Return OPENAI_API_KEY, loading .env first. Raises a clear error if
    absent. Never logs or returns anything derived that exposes the key."""
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to Coding/.env "
            "(OPENAI_API_KEY=sk-...). The key is never logged."
        )
    return key


_CLIENT: OpenAI | None = None


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=get_api_key())
    return _CLIENT


# --------------------------------------------------------------------------
# Deriving column roles + JSON schema from the system prompt
# --------------------------------------------------------------------------
def required_column_order(system_prompt: dict) -> list[str]:
    return list(system_prompt["output_format"]["required_column_order"])


def echoed_columns(system_prompt: dict) -> list[str]:
    defs = system_prompt["column_definitions"]
    return [
        col for col in required_column_order(system_prompt)
        if defs.get(col, {}).get("source") == "echoed_from_input"
    ]


def coded_columns(system_prompt: dict) -> list[str]:
    """Columns whose source designates them as model-produced (coded /
    computed), in column order."""
    defs = system_prompt["column_definitions"]
    return [
        col for col in required_column_order(system_prompt)
        if defs.get(col, {}).get("source") in ("coded", "computed")
    ]


def llm_output_columns(system_prompt: dict) -> list[str]:
    """Columns the LLM must actually emit: coded columns minus the two
    Python-derived aggregate scores. These are the schema's properties.

    The exclusion is belt-and-braces: in the v3.1 prompt the score fields
    carry source "python_derived_after_api_call" (so they are already out of
    coded_columns), AND DERIVED_FIELDS removes them explicitly here. Either
    way the LLM is never asked to produce them."""
    return [c for c in coded_columns(system_prompt) if c not in DERIVED_FIELDS]


def _scale_bounds(scale: str) -> tuple[int, int]:
    lo, hi = scale.split("-")
    return int(lo), int(hi)


def _property_for(defn: dict) -> dict:
    """Build a JSON-schema property for one emitted column, embedding the full
    coding guidance (definition + scale anchors + notes) in the description so
    the model has the codebook detail at hand. Bounded scales are enforced
    with integer enums (strict structured outputs does not support numeric
    minimum/maximum)."""
    desc_parts: list[str] = []
    if defn.get("definition"):
        desc_parts.append(defn["definition"])
    if "scale_anchors" in defn:
        anchors = "; ".join(f"{k}={v}" for k, v in defn["scale_anchors"].items())
        desc_parts.append(f"Scale anchors: {anchors}")
    if "formula_treatment" in defn:
        desc_parts.append(defn["formula_treatment"])
    if "important_note" in defn:
        desc_parts.append(f"Note: {defn['important_note']}")
    description = " ".join(desc_parts)

    if "allowed_values" in defn:
        return {"type": "string", "enum": list(defn["allowed_values"]),
                "description": description}

    col_type = defn.get("type")
    scale = defn.get("scale")
    if col_type in ("integer", "binary_integer") and scale:
        lo, hi = _scale_bounds(scale)
        return {"type": "integer", "enum": list(range(lo, hi + 1)),
                "description": description}
    if col_type in ("integer", "binary_integer"):
        return {"type": "integer", "description": description}
    return {"type": "string", "description": description}


def build_schema(system_prompt: dict) -> dict:
    """Derive a strict JSON schema from the system prompt's column_definitions.
    Includes only the columns the LLM emits — the two derived scores are
    EXCLUDED (computed in Python after the call)."""
    defs = system_prompt["column_definitions"]
    cols = llm_output_columns(system_prompt)
    properties = {col: _property_for(defs[col]) for col in cols}
    return {
        "type": "object",
        "properties": properties,
        "required": cols,
        "additionalProperties": False,
    }


def build_system_message(system_prompt: dict) -> str:
    """Concatenate the system prompt's role + research_context + core_task +
    coding_rules + interpretive_guidance into a single system message. The
    per-column definitions and scale anchors ride along in the response
    schema's property descriptions (see build_schema)."""
    parts: list[str] = [system_prompt["role"]]
    parts.append("## Research context\n" + system_prompt["research_context"])
    parts.append("## Core task\n" + system_prompt["core_task"])

    rules = system_prompt["coding_rules"]
    parts.append("## Coding rules\n" + "\n".join(
        f"- {name}: {text}" for name, text in rules.items()))

    guidance = system_prompt["interpretive_guidance"]
    parts.append("## Interpretive guidance\n" + "\n".join(
        f"- {name}: {text}" for name, text in guidance.items()))

    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Input formatting
# --------------------------------------------------------------------------
# Fields the LLM needs in the user message (per the run spec).
_INPUT_FIELDS = [
    "skill_id", "skill_name", "repo_name", "repo_url", "skill_url",
    "category", "installs", "leaderboard_rank", "is_official",
    "github_stars_abbreviated", "snyk_status", "socket_status",
    "agent_trust_hub_status", "summary", "has_skill_md", "skill_md_content",
]


def format_skill_for_input(skill_record: dict) -> dict:
    """Select the fields the LLM needs from a raw corpus record.

    ``category`` is read directly from the corpus — the old domain->category
    shim was removed once the corpus column was renamed (decisions.md
    2026-05-23). ``has_skill_md`` is still derived ("Y" if skill_md_content is
    non-empty else "N"), since the corpus has no such column."""
    out: dict = {}
    for field in _INPUT_FIELDS:
        if field == "has_skill_md":
            md = skill_record.get("skill_md_content") or ""
            out["has_skill_md"] = "Y" if md.strip() else "N"
        else:
            out[field] = skill_record.get(field)
    return out


# --------------------------------------------------------------------------
# Formula computation (Python; standard round-half-up, not banker's)
# --------------------------------------------------------------------------
def round_half_up(x: float) -> int:
    """Standard round-half-up for non-negative x: 0.5->1, 1.5->2, 2.5->3,
    3.5->4. NOT Python's built-in round() (which uses banker's rounding,
    e.g. round(0.5)==0, round(2.5)==2)."""
    return int(math.floor(x + 0.5))


_CODIFIABILITY_COMPONENTS = ("explicitness", "documentation_quality",
                             "tacit_dependency", "context_sensitivity")
_EVALUABILITY_COMPONENTS = ("output_verifiability", "tests_quality",
                            "thresholds_quality", "error_handling")


def compute_derived_scores(record: dict) -> dict:
    """Compute codifiability_score and evaluability_score in Python from the
    component scores, using round_half_up. Mutates and returns record. If any
    component is missing or non-integer, that score is left as None."""
    def _ints(keys):
        vals = []
        for k in keys:
            try:
                vals.append(int(record[k]))
            except (KeyError, TypeError, ValueError):
                return None
        return vals

    cod = _ints(_CODIFIABILITY_COMPONENTS)
    record["codifiability_score"] = None if cod is None else round_half_up(
        (cod[0] + cod[1] + (4 - cod[2]) + (4 - cod[3])) / 4)

    ev = _ints(_EVALUABILITY_COMPONENTS)
    record["evaluability_score"] = None if ev is None else round_half_up(
        (ev[0] + ev[1] + ev[2] + ev[3]) / 4)

    return record


# --------------------------------------------------------------------------
# Response assembly (shared by live calls and resume-from-raw)
# --------------------------------------------------------------------------
def _extract_content(raw: dict) -> tuple[str, str | None]:
    """Return (content, refusal) from a raw chat-completion response dict."""
    choice = raw["choices"][0]
    message = choice["message"]
    return message.get("content"), message.get("refusal")


def assemble_record(raw_response: dict, formatted_input: dict,
                    system_prompt: dict) -> dict:
    """Parse a raw API response, validate the emitted columns, merge echoed
    fields, and compute the two derived scores in Python. Returns the row in
    required_column_order (no _audit key). Raises SchemaViolation on a
    refusal, unparseable content, or missing emitted columns."""
    content, refusal = _extract_content(raw_response)
    if refusal:
        raise SchemaViolation(f"Model refused to answer: {refusal}")
    if not content:
        raise SchemaViolation("Empty response content from model.")
    try:
        coded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"Could not parse structured output as JSON: {exc}")

    expected = llm_output_columns(system_prompt)
    missing = [c for c in expected if c not in coded]
    if missing:
        raise SchemaViolation(f"Structured output missing emitted columns: {missing}")

    echoed = set(echoed_columns(system_prompt))
    record: dict = {}
    for col in required_column_order(system_prompt):
        if col in echoed:
            record[col] = formatted_input.get(col)
        elif col in DERIVED_FIELDS:
            record[col] = None          # filled by compute_derived_scores
        else:
            record[col] = coded.get(col)

    compute_derived_scores(record)
    return record


# --------------------------------------------------------------------------
# The call itself — primary gpt-5.2, logged fallback to gpt-4.1
# --------------------------------------------------------------------------
def _request_kwargs(model: str, messages: list[dict], response_format: dict,
                    use_reasoning: bool) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": TEMPERATURE,
    }
    if use_reasoning:
        # openai 1.55.0 has no reasoning_effort kwarg -> pass via extra_body.
        kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}
    return kwargs


def _is_config_failure(exc: Exception) -> bool:
    """True if the error means the model cannot honor the deterministic config
    and we should fall back: a 400 mentioning temperature/reasoning_effort, or
    model-not-found (404, gpt-5.2 unavailable for this key)."""
    if isinstance(exc, NotFoundError):
        return True
    if isinstance(exc, BadRequestError):
        msg = str(exc).lower()
        return "temperature" in msg or "reasoning_effort" in msg or "reasoning" in msg
    return False


def _classify_fatal(exc: Exception) -> LLMError:
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return AuthError(f"Authentication/permission error (stop): {exc}")
    if isinstance(exc, BadRequestError):
        msg = str(exc)
        if "schema" in msg.lower() or "json" in msg.lower():
            return SchemaViolation(f"Structured-output / schema error: {msg}")
        return FatalAPIError(f"Non-retryable bad-request error: {msg}")
    return FatalAPIError(f"Non-retryable error: {type(exc).__name__}: {exc}")


def _create_one_model(client: OpenAI, model: str, messages: list[dict],
                      response_format: dict, use_reasoning: bool) -> dict:
    """Call one model, retrying transient errors. Returns the raw response
    dict. Auth errors -> AuthError immediately; persistent transient ->
    TransientError. BadRequestError / NotFoundError are NOT caught here so the
    caller can decide between a config-failure fallback and a fatal stop."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            kwargs = _request_kwargs(model, messages, response_format, use_reasoning)
            return client.chat.completions.create(**kwargs).model_dump()
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise _classify_fatal(exc) from exc
        except (RateLimitError, APITimeoutError, APIConnectionError,
                InternalServerError) as exc:
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE * (2 ** attempt))
                continue
            raise TransientError(
                f"Persistent transient error after {MAX_RETRIES} retries on "
                f"{model}: {exc}") from exc
    raise FatalAPIError("retry loop exited without a result")


def _call(client: OpenAI, messages: list[dict],
          response_format: dict) -> tuple[dict, dict]:
    """Run the primary model; on a config failure (or 404), fall back ONCE to
    gpt-4.1. Returns (raw_dict, meta). meta records model_used, fallback_used,
    fallback_reason, temperature, reasoning_effort."""
    try:
        raw = _create_one_model(client, MODEL, messages, response_format,
                                use_reasoning=True)
        return raw, {"model_used": MODEL, "fallback_used": False,
                     "fallback_reason": None, "temperature": TEMPERATURE,
                     "reasoning_effort": REASONING_EFFORT}
    except (BadRequestError, NotFoundError) as exc:
        if not _is_config_failure(exc):
            raise _classify_fatal(exc) from exc  # e.g. schema violation -> fatal
        reason = (f"{MODEL} rejected the deterministic config or is unavailable "
                  f"({type(exc).__name__}: {exc}); fell back to {FALLBACK_MODEL}.")

    # fallback: gpt-4.1, temperature=0, no reasoning_effort
    try:
        raw = _create_one_model(client, FALLBACK_MODEL, messages,
                                response_format, use_reasoning=False)
    except (BadRequestError, NotFoundError) as exc:
        raise _classify_fatal(exc) from exc
    return raw, {"model_used": FALLBACK_MODEL, "fallback_used": True,
                 "fallback_reason": reason, "temperature": TEMPERATURE,
                 "reasoning_effort": None}


def estimate_cost(usage: dict | None, model: str | None = None) -> float | None:
    """Approximate USD cost from a usage dict. Returns None if no usage info.
    completion_tokens already includes reasoning tokens."""
    if not usage:
        return None
    pricing = _DEFAULT_PRICING
    if model:
        if model in PRICING_PER_1M:
            pricing = PRICING_PER_1M[model]
        else:
            for prefix, rates in PRICING_PER_1M.items():
                if model.startswith(prefix):
                    pricing = rates
                    break
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return (prompt_tokens / 1_000_000 * pricing["input"]
            + completion_tokens / 1_000_000 * pricing["output"])


def code_skill(skill_record: dict, system_prompt: dict) -> dict:
    """Code one skill and return its full row.

    The returned dict has every column in required_column_order (with the two
    derived scores computed in Python), plus a single non-column key
    ``_audit`` carrying: raw_response, usage, cost_estimate_usd, latency_s,
    model_used, fallback_used, fallback_reason, temperature, reasoning_effort.

    Strip ``_audit`` before serializing the coded row. Raises an LLMError
    subclass on any fatal API/schema/auth condition (caller should stop)."""
    client = get_client()
    schema = build_schema(system_prompt)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_skill_coding",
            "strict": True,
            "schema": schema,
        },
    }
    formatted = format_skill_for_input(skill_record)
    messages = [
        {"role": "system", "content": build_system_message(system_prompt)},
        {"role": "user",
         "content": json.dumps(formatted, ensure_ascii=False, indent=2)},
    ]

    started = time.time()
    raw, meta = _call(client, messages, response_format)
    latency = time.time() - started

    record = assemble_record(raw, formatted, system_prompt)

    usage = raw.get("usage")
    record["_audit"] = {
        "raw_response": raw,
        "usage": usage,
        "cost_estimate_usd": estimate_cost(usage, meta.get("model_used")),
        "latency_s": round(latency, 2),
        "model_used": meta.get("model_used"),
        "fallback_used": meta.get("fallback_used"),
        "fallback_reason": meta.get("fallback_reason"),
        "temperature": meta.get("temperature"),
        "reasoning_effort": meta.get("reasoning_effort"),
    }
    return record


# --------------------------------------------------------------------------
# Rounding self-test (no API calls). Run:  python3 scripts/_llm.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("round_half_up boundary self-test (no API calls):")
    boundary_cases = [(0.25, 0), (0.5, 1), (0.75, 1), (1.5, 2), (2.5, 3), (3.5, 4)]
    all_ok = True
    for x, expected in boundary_cases:
        got = round_half_up(x)
        ok = got == expected
        all_ok &= ok
        builtin = round(x)  # banker's, for contrast
        print(f"  round_half_up({x:>4}) = {got}   expected {expected}   "
              f"[{'OK' if ok else 'FAIL'}]   (Python round() would give {builtin})")

    print("\ncompute_derived_scores examples (mirroring real pilot boundary cases):")
    examples = [
        # CC01-like: codif components avg 2.5 -> 3 ; eval avg 0.75 -> 1
        {"explicitness": 3, "documentation_quality": 3, "tacit_dependency": 3,
         "context_sensitivity": 1, "output_verifiability": 2, "tests_quality": 0,
         "thresholds_quality": 1, "error_handling": 0, "expect_cod": 3, "expect_ev": 1},
        # SD01-like: codif avg 1.75 -> 2 ; eval avg 0.5 -> 1
        {"explicitness": 2, "documentation_quality": 3, "tacit_dependency": 3,
         "context_sensitivity": 3, "output_verifiability": 2, "tests_quality": 0,
         "thresholds_quality": 0, "error_handling": 0, "expect_cod": 2, "expect_ev": 1},
    ]
    for ex in examples:
        expect_cod = ex.pop("expect_cod")
        expect_ev = ex.pop("expect_ev")
        compute_derived_scores(ex)
        ok = ex["codifiability_score"] == expect_cod and ex["evaluability_score"] == expect_ev
        all_ok &= ok
        print(f"  codif={ex['codifiability_score']} (exp {expect_cod}), "
              f"eval={ex['evaluability_score']} (exp {expect_ev})  "
              f"[{'OK' if ok else 'FAIL'}]")

    print("\nRESULT:", "ALL ROUNDING TESTS PASS" if all_ok else "ROUNDING TESTS FAILED")
    raise SystemExit(0 if all_ok else 1)