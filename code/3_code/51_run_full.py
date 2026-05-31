#!/usr/bin/env python3
"""Run the full corpus coding: all 120 skills -> coded/.

Identical pipeline to 50_run_pilot.py but over the whole corpus, with resume
logic: before coding a skill, if coded/raw/<skill_id>.json already exists, the
saved raw response is reused (no API call) and re-assembled. This lets an
interrupted run pick up where it left off.

Model gpt-5.2 (reasoning_effort=none, temperature=0) with logged fallback to
gpt-4.1; the two aggregate scores are computed in Python (round half up).

Isolated gpt-4.1 fallbacks are logged and the run continues. If more than 5
skills hit the fallback (a possible systemic gpt-5.2 issue) the run STOPS and
reports; re-running resumes from the cache.
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _llm  # noqa: E402

CODED_DIR = _llm.BASE_DIR / "coded"
RAW_DIR = CODED_DIR / "raw"
TODAY = _dt.date.today().isoformat()
MAX_FALLBACKS = 5  # stop if MORE than this many skills hit the gpt-4.1 fallback


def _estimate_from_pilot(n_skills: int):
    """Scale the latest v3.1 pilot's per-skill cost/runtime to n_skills.
    Returns (est_cost, est_runtime) or (None, None) if no usable pilot file."""
    pilots = sorted((_llm.BASE_DIR / "pilot").glob("pilot_coded_v3.1_*.json"))
    if not pilots:
        return None, None
    md = json.loads(pilots[-1].read_text()).get("metadata", {})
    n = md.get("n_skills_coded") or 0
    cost = md.get("approx_cost_usd")
    rt = md.get("runtime_seconds")
    if not n or cost is None or rt is None:
        return None, None
    return cost / n * n_skills, rt / n * n_skills


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- credentials (names + lengths only; NEVER the values) -------------
    key = _llm.get_api_key()
    gh = os.environ.get("GITHUB_TOKEN")
    if gh:
        print(f"GITHUB_TOKEN and OPENAI_API_KEY both loaded "
              f"(lengths {len(gh)}, {len(key)})")
    else:
        print(f"OPENAI_API_KEY loaded (length {len(key)}). GITHUB_TOKEN not "
              f"present in .env — not required for this coding run (OpenAI only).")

    system_prompt = _llm.load_system_prompt()
    skills, _meta, corpus_path = _llm.load_corpus()
    print(f"Loaded {len(skills)} skills from {corpus_path.name}")
    print(f"Codebook: {_llm.find_system_prompt_path().name} "
          f"(v{system_prompt.get('version')})")
    print(f"Primary model: {_llm.MODEL} (reasoning_effort={_llm.REASONING_EFFORT}, "
          f"temperature={_llm.TEMPERATURE}); fallback: {_llm.FALLBACK_MODEL}")

    est_cost, est_runtime = _estimate_from_pilot(len(skills))
    if est_cost is not None:
        print(f"Estimated cost (pilot rate):    ~${est_cost:.2f}")
        print(f"Estimated runtime (pilot rate): ~{est_runtime:.0f}s "
              f"(~{est_runtime / 60:.0f} min)")
    n_cached = len(list(RAW_DIR.glob("*.json")))
    if n_cached:
        print(f"Resume: {n_cached} skill(s) already cached in {RAW_DIR.name}/ "
              f"— loaded, not re-called.")
    print()

    columns = _llm.required_column_order(system_prompt)
    coded_records: list[dict] = []
    errored: list[dict] = []
    fallback_events: list[str] = []
    models_count: collections.Counter = collections.Counter()
    total_cost = 0.0
    cost_known = True
    n_resumed = 0

    run_start = time.time()
    stop_reason: str | None = None

    for idx, skill in enumerate(skills, 1):
        skill_id = skill.get("skill_id")
        category = skill.get("category")
        raw_path = RAW_DIR / f"{skill_id}.json"

        try:
            if raw_path.exists():
                # resume: reuse the saved raw response, no API call
                raw = json.loads(raw_path.read_text())
                formatted = _llm.format_skill_for_input(skill)
                record = _llm.assemble_record(raw, formatted, system_prompt)
                model_used = raw.get("model")
                cost = _llm.estimate_cost(raw.get("usage"), model_used)
                n_resumed += 1
                print(f"[{idx:3d}/{len(skills)}] {skill_id} ({category}) "
                      f"-> resumed from cache", flush=True)
            else:
                record = _llm.code_skill(skill, system_prompt)
                audit = record.pop("_audit")
                raw_path.write_text(json.dumps(audit["raw_response"],
                                               ensure_ascii=False, indent=2))
                model_used = audit.get("model_used")
                cost = audit.get("cost_estimate_usd")
                if audit.get("fallback_used"):
                    warn = f"{skill_id}: {audit.get('fallback_reason')}"
                    fallback_events.append(warn)
                    print(f"     ⚠ fallback: {warn}")
                print(f"[{idx:3d}/{len(skills)}] {skill_id} ({category}) -> "
                      f"[{model_used}] codif={record.get('codifiability_score')} "
                      f"eval={record.get('evaluability_score')} "
                      f"status={record.get('coding_status')} "
                      f"({audit.get('latency_s')}s)", flush=True)
        except _llm.LLMError as exc:
            errored.append({"skill_id": skill_id, "category": category,
                            "error": f"{type(exc).__name__}: {exc}"})
            stop_reason = (f"Fatal error on {skill_id} ({type(exc).__name__}). "
                           f"Stopping per constraints. Re-run to resume.")
            print(f"     !! {stop_reason}")
            break

        if model_used:
            models_count[model_used] += 1
        if cost is None:
            cost_known = False
        else:
            total_cost += cost
        coded_records.append(record)

        # systemic-fallback guard: stop if MORE than 5 skills fell back
        if len(fallback_events) > MAX_FALLBACKS:
            stop_reason = (f"gpt-4.1 fallback exceeded {MAX_FALLBACKS} skills "
                           f"({len(fallback_events)}) — possible systemic gpt-5.2 "
                           f"issue. Stopping per constraints; re-run to resume.")
            print(f"     !! {stop_reason}")
            break

    runtime = round(time.time() - run_start, 1)
    models_used = sorted(models_count)

    json_path = CODED_DIR / f"coded_{TODAY}.json"
    output = {
        "metadata": {
            "phase": "coding-full",
            "prompt_version": "3.1",
            "date": TODAY,
            "primary_model": _llm.MODEL,
            "fallback_model": _llm.FALLBACK_MODEL,
            "models_used": models_used,
            "model_breakdown": dict(models_count),
            "temperature": _llm.TEMPERATURE,
            "reasoning_effort": _llm.REASONING_EFFORT,
            "fallback_used": bool(fallback_events),
            "fallback_count": len(fallback_events),
            "scores_computed_in_python": True,
            "rounding": "round half up (math.floor(x + 0.5))",
            "system_prompt_version": system_prompt.get("version"),
            "codebook_file": _llm.find_system_prompt_path().name,
            "source_corpus": corpus_path.name,
            "n_skills_total": len(skills),
            "n_skills_coded": len(coded_records),
            "n_resumed_from_cache": n_resumed,
            "single_pass": True,
            "generated_by": "scripts/51_run_full.py",
            "input_field_mapping": {
                "has_skill_md": "derived: 'Y' if skill_md_content non-empty else 'N'",
            },
            "runtime_seconds": runtime,
            "approx_cost_usd": round(total_cost, 4) if cost_known else None,
            "stopped_early": stop_reason,
        },
        "skills": coded_records,
    }
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    csv_path = CODED_DIR / f"coded_{TODAY}.csv"
    pd.DataFrame(coded_records).reindex(columns=columns).to_csv(csv_path, index=False)

    print("\n" + "=" * 64)
    print("FULL RUN SUMMARY (v3.1)")
    print("=" * 64)
    print(f"Model breakdown:  "
          + (", ".join(f"{m}={c}" for m, c in models_count.most_common()) or "—"))
    print(f"Skills coded:     {len(coded_records)} / {len(skills)} "
          f"({n_resumed} resumed from cache)")
    print(f"gpt-4.1 fallback: {len(fallback_events)} skill(s)")
    print("Cost estimate:    "
          + (f"~${total_cost:.4f} (approx)" if cost_known else "unavailable"))
    print(f"Total runtime:    {runtime}s")
    print(f"API errors:       {len(errored)}")
    for e in errored:
        print(f"   - {e['skill_id']} ({e['category']}): {e['error']}")
    print(f"\nOutputs:\n  {json_path}\n  {csv_path}\n  {RAW_DIR}/")
    if stop_reason:
        print(f"\n** RUN STOPPED EARLY: {stop_reason}")

    _append_run_log(runtime, len(coded_records), len(skills), n_resumed,
                    total_cost if cost_known else None, errored,
                    fallback_events, models_count, stop_reason)
    return 1 if (errored or stop_reason) else 0


def _append_run_log(runtime, n_coded, n_total, n_resumed, cost, errored,
                    fallback_events, models_count, stop_reason) -> None:
    log_path = CODED_DIR / "coded_run_log.md"
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    breakdown = ", ".join(f"{m}={c}" for m, c in models_count.most_common()) or "—"
    lines = [
        f"\n## {TODAY} — full run, prompt v3.1 ({ts})",
        f"- Primary model: {_llm.MODEL} (reasoning_effort={_llm.REASONING_EFFORT}, "
        f"temperature {_llm.TEMPERATURE}); fallback {_llm.FALLBACK_MODEL}.",
        f"- Model used per skill (counts): {breakdown}.",
        f"- gpt-4.1 fallback count: {len(fallback_events)}.",
        "- Scores computed in Python with round-half-up; LLM does not emit them.",
        f"- Skills coded: {n_coded}/{n_total} ({n_resumed} resumed from cache).",
        f"- Runtime: {runtime}s; approx cost: "
        + (f"${cost:.4f}." if cost is not None else "unavailable."),
    ]
    anomalies = []
    if fallback_events:
        anomalies.append(f"gpt-4.1 fallback skills: " + "; ".join(fallback_events) + ".")
    if errored:
        anomalies.append("errors: " + "; ".join(
            f"{e['skill_id']} {e['error']}" for e in errored) + ".")
    if stop_reason:
        anomalies.append(stop_reason)
    lines.append("- Anomalies: " + (" ".join(anomalies) if anomalies else "none."))
    with log_path.open("a") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())