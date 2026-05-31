#!/usr/bin/env python3
"""Re-run the coding pilot under v3.1: 12 skills, one per category (highest
installs), model gpt-5.2 (reasoning_effort=none, temperature=0) with logged
fallback to gpt-4.1.

Outputs (v3.1, written alongside — never overwriting — the v3.0 pilot files):
  pilot/pilot_coded_v3.1_<today>.json
  pilot/pilot_coded_v3.1_<today>.csv
  pilot/raw_v3.1/<skill_id>.json        (per-skill raw API responses)
  pilot/v3_0_vs_v3_1_comparison.md      (score comparison vs the v3.0 pilot)

The two aggregate scores are computed in Python (round half up) inside
_llm.code_skill; this script does not touch coded values. On a fatal
API/schema/auth error the run stops, writes whatever was completed, and
reports.
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _llm  # noqa: E402

PILOT_DIR = _llm.BASE_DIR / "pilot"
RAW_DIR = PILOT_DIR / "raw_v3.1"
TODAY = _dt.date.today().isoformat()

CODIF_COMP = ("explicitness", "documentation_quality",
              "tacit_dependency", "context_sensitivity")
EVAL_COMP = ("output_verifiability", "tests_quality",
             "thresholds_quality", "error_handling")


def select_pilot_skills(skills: list[dict]) -> list[dict]:
    """One skill per category: the highest-installs skill in each category."""
    by_category: dict[str, list[dict]] = collections.defaultdict(list)
    for skill in skills:
        by_category[skill.get("category")].append(skill)
    return [max(by_category[c], key=lambda s: int(s.get("installs") or 0))
            for c in sorted(by_category)]


def load_v30_pilot() -> dict[str, dict]:
    """Return {skill_id: record} from the original (v3.0) pilot JSON, i.e. the
    most recent pilot_coded_*.json that is NOT a v3.1 file."""
    candidates = [p for p in PILOT_DIR.glob("pilot_coded_*.json")
                  if "v3.1" not in p.name]
    if not candidates:
        return {}
    src = sorted(candidates)[-1]
    data = json.loads(src.read_text())
    return {s["skill_id"]: s for s in data.get("skills", [])}


def classify_diff(old_rec, new_rec, comp_keys, score_key):
    """Classify a v3.0->v3.1 score difference for one dimension.
    Returns (label, old_score, new_score, components_changed)."""
    old_comp = tuple(old_rec.get(k) for k in comp_keys)
    new_comp = tuple(new_rec.get(k) for k in comp_keys)
    old_score = old_rec.get(score_key)
    new_score = new_rec.get(score_key)
    comp_changed = old_comp != new_comp
    if old_score == new_score:
        label = "unchanged"
    elif not comp_changed:
        # same components, different score => attributable to the rounding fix
        label = "rounding-fix"
    else:
        label = "genuine-disagreement"
    return label, old_score, new_score, comp_changed


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    key = _llm.get_api_key()
    print(f"OPENAI_API_KEY loaded (length {len(key)})")

    system_prompt = _llm.load_system_prompt()
    skills, _meta, corpus_path = _llm.load_corpus()
    print(f"Loaded {len(skills)} skills from {corpus_path.name}")
    print(f"Codebook: {_llm.find_system_prompt_path().name} "
          f"(v{system_prompt.get('version')})")
    print(f"Primary model: {_llm.MODEL} (reasoning_effort={_llm.REASONING_EFFORT}, "
          f"temperature={_llm.TEMPERATURE}); fallback: {_llm.FALLBACK_MODEL}")

    pilot_skills = select_pilot_skills(skills)
    v30 = load_v30_pilot()
    print(f"Selected {len(pilot_skills)} pilot skills; loaded "
          f"{len(v30)} v3.0 records for comparison.\n")

    columns = _llm.required_column_order(system_prompt)
    coded_records: list[dict] = []
    summary_rows: list[dict] = []
    errored: list[dict] = []
    fallback_events: list[str] = []
    models_used: set[str] = set()
    total_cost = 0.0
    cost_known = True

    run_start = time.time()
    stop_reason: str | None = None

    for idx, skill in enumerate(pilot_skills, 1):
        skill_id = skill.get("skill_id")
        category = skill.get("category")
        print(f"[{idx:2d}/{len(pilot_skills)}] coding {skill_id} ({category}) ...",
              flush=True)
        try:
            record = _llm.code_skill(skill, system_prompt)
        except _llm.LLMError as exc:
            errored.append({"skill_id": skill_id, "category": category,
                            "error": f"{type(exc).__name__}: {exc}"})
            stop_reason = (f"Fatal error on {skill_id} ({type(exc).__name__}). "
                           f"Stopping per constraints.")
            print(f"     !! {stop_reason}")
            break

        audit = record.pop("_audit")
        (RAW_DIR / f"{skill_id}.json").write_text(
            json.dumps(audit["raw_response"], ensure_ascii=False, indent=2))

        model_used = audit.get("model_used")
        models_used.add(model_used)
        if audit.get("fallback_used"):
            warn = f"{skill_id}: {audit.get('fallback_reason')}"
            fallback_events.append(warn)
            print(f"     ⚠ fallback: {warn}")

        cost = audit.get("cost_estimate_usd")
        if cost is None:
            cost_known = False
        else:
            total_cost += cost

        # comparison vs v3.0
        old = v30.get(skill_id, {})
        cl_cod = classify_diff(old, record, CODIF_COMP, "codifiability_score")
        cl_ev = classify_diff(old, record, EVAL_COMP, "evaluability_score")

        coded_records.append(record)
        summary_rows.append({
            "skill_id": skill_id, "category": category, "model": model_used,
            "codif_old": cl_cod[1], "codif_new": cl_cod[2], "codif_cls": cl_cod[0],
            "eval_old": cl_ev[1], "eval_new": cl_ev[2], "eval_cls": cl_ev[0],
            "coding_status": record.get("coding_status"),
            "latency_s": audit.get("latency_s"),
        })
        print(f"     ok [{model_used}] codif {cl_cod[1]}→{cl_cod[2]} "
              f"eval {cl_ev[1]}→{cl_ev[2]} status={record.get('coding_status')} "
              f"({audit.get('latency_s')}s)")

    runtime = round(time.time() - run_start, 1)

    # --- write aggregate JSON ---------------------------------------------
    json_path = PILOT_DIR / f"pilot_coded_v3.1_{TODAY}.json"
    output = {
        "metadata": {
            "phase": "coding-pilot",
            "prompt_version": "3.1",
            "date": TODAY,
            "primary_model": _llm.MODEL,
            "fallback_model": _llm.FALLBACK_MODEL,
            "models_used": sorted(m for m in models_used if m),
            "temperature": _llm.TEMPERATURE,
            "reasoning_effort": _llm.REASONING_EFFORT,
            "fallback_used": bool(fallback_events),
            "scores_computed_in_python": True,
            "rounding": "round half up (math.floor(x + 0.5))",
            "system_prompt_version": system_prompt.get("version"),
            "codebook_file": _llm.find_system_prompt_path().name,
            "source_corpus": corpus_path.name,
            "selection_rule": "one skill per category, highest installs (12)",
            "n_skills_selected": len(pilot_skills),
            "n_skills_coded": len(coded_records),
            "single_pass": True,
            "generated_by": "scripts/50_run_pilot.py",
            "input_field_mapping": {
                "has_skill_md": "derived: 'Y' if skill_md_content non-empty else 'N'",
            },
            "runtime_seconds": runtime,
            "approx_cost_usd": round(total_cost, 4) if cost_known else None,
        },
        "skills": coded_records,
    }
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    csv_path = PILOT_DIR / f"pilot_coded_v3.1_{TODAY}.csv"
    pd.DataFrame(coded_records).reindex(columns=columns).to_csv(csv_path, index=False)

    _write_comparison_report(summary_rows, v30)

    # --- run summary -------------------------------------------------------
    temp_effective = len(coded_records) > 0  # every successful call sent temperature=0
    print("\n" + "=" * 70)
    print("PILOT RUN SUMMARY (v3.1)")
    print("=" * 70)
    print(f"Model(s) actually used: {', '.join(sorted(m for m in models_used if m)) or '—'}")
    print(f"temperature=0 in effect: "
          f"{'yes (sent and accepted on every call)' if temp_effective else 'n/a (no successful calls)'}")
    if fallback_events:
        print(f"gpt-4.1 fallback used on {len(fallback_events)} skill(s):")
        for w in fallback_events:
            print(f"   ⚠ {w}")
    else:
        print(f"gpt-4.1 fallback used: no (gpt-5.2 honored the config)")
    print(f"Skills coded:           {len(coded_records)} / {len(pilot_skills)}")
    print("Cost estimate:          "
          + (f"~${total_cost:.4f} (approx, from token usage)" if cost_known
             else "unavailable"))
    print(f"Total runtime:          {runtime}s")
    print(f"API errors:             {len(errored)}")
    for e in errored:
        print(f"   - {e['skill_id']} ({e['category']}): {e['error']}")

    print("\nPer-skill (v3.0 → v3.1 scores; * = differs):")
    print(f"  {'skill':6s} {'model':8s} {'codif':>11s} {'eval':>11s}  status")
    for r in summary_rows:
        cflag = "*" if r["codif_cls"] != "unchanged" else " "
        eflag = "*" if r["eval_cls"] != "unchanged" else " "
        print(f"  {str(r['skill_id']):6s} {str(r['model'] or ''):8s} "
              f"{str(r['codif_old'])+'→'+str(r['codif_new'])+cflag:>11s} "
              f"{str(r['eval_old'])+'→'+str(r['eval_new'])+eflag:>11s}  "
              f"{r['coding_status']}")

    print(f"\nOutputs:\n  {json_path}\n  {csv_path}\n  {RAW_DIR}/ "
          f"({len(coded_records)} per-skill JSONs)\n  "
          f"{PILOT_DIR / 'v3_0_vs_v3_1_comparison.md'}")
    if stop_reason:
        print(f"\n** RUN STOPPED EARLY: {stop_reason}")

    _append_run_log(runtime, len(coded_records), len(pilot_skills),
                    total_cost if cost_known else None, errored,
                    fallback_events, sorted(m for m in models_used if m), stop_reason)
    return 1 if errored else 0


def _write_comparison_report(summary_rows, v30) -> None:
    path = PILOT_DIR / "v3_0_vs_v3_1_comparison.md"
    lines = [
        "# Pilot comparison — v3.0 vs v3.1",
        "",
        f"Generated {TODAY} by scripts/50_run_pilot.py.",
        "",
        "v3.0: gpt-5, temperature fell back to model default, scores emitted by "
        "the LLM. v3.1: gpt-5.2 (reasoning_effort=none, temperature=0) with "
        "gpt-4.1 fallback, scores computed in Python with round-half-up.",
        "",
        "## Scores side by side",
        "",
        "| skill | category | codif v3.0 | codif v3.1 | eval v3.0 | eval v3.1 |",
        "|-------|----------|:----------:|:----------:|:---------:|:---------:|",
    ]
    for r in summary_rows:
        cmark = "" if r["codif_cls"] == "unchanged" else " ←"
        emark = "" if r["eval_cls"] == "unchanged" else " ←"
        lines.append(
            f"| {r['skill_id']} | {r['category']} | {r['codif_old']} | "
            f"{r['codif_new']}{cmark} | {r['eval_old']} | {r['eval_new']}{emark} |")

    # attribution of differences
    diffs = []
    for r in summary_rows:
        if r["codif_cls"] != "unchanged":
            diffs.append((r["skill_id"], "codifiability",
                          r["codif_old"], r["codif_new"], r["codif_cls"]))
        if r["eval_cls"] != "unchanged":
            diffs.append((r["skill_id"], "evaluability",
                          r["eval_old"], r["eval_new"], r["eval_cls"]))

    n_round = sum(1 for d in diffs if d[4] == "rounding-fix")
    n_genuine = sum(1 for d in diffs if d[4] == "genuine-disagreement")
    lines += [
        "",
        "## Difference attribution",
        "",
        f"- Score changes total: {len(diffs)}",
        f"- Due to the rounding fix (same component scores, different aggregate): {n_round}",
        f"- Due to genuine LLM disagreement (component scores changed under the "
        f"new model/temperature): {n_genuine}",
        "",
    ]
    if diffs:
        lines.append("| skill | dimension | v3.0 | v3.1 | attribution |")
        lines.append("|-------|-----------|:----:|:----:|-------------|")
        for sid, dim, o, n, cls in diffs:
            lines.append(f"| {sid} | {dim} | {o} | {n} | {cls} |")
    else:
        lines.append("No score differences between v3.0 and v3.1.")
    lines += [
        "",
        "### Note on the rounding fix",
        "",
        "In the v3.0 pilot the LLM had already applied round-half-up to its own "
        "scores, so where v3.1 reproduces the same component scores the aggregate "
        "is unchanged. The Python computation guarantees this convention "
        "deterministically going forward; the three v3.0 'mismatches' (CC01 "
        "codifiability at raw 2.5; SD01/UI01 evaluability at raw 0.5) were a "
        "banker's-vs-half-up artefact in the old mismatch detector, not in the "
        "stored data.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _append_run_log(runtime, n_coded, n_selected, cost, errored,
                    fallback_events, models_used, stop_reason) -> None:
    log_path = PILOT_DIR / "pilot_run_log.md"
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n## {TODAY} — pilot re-run, prompt v3.1 ({ts})",
        f"- Primary model: {_llm.MODEL} (reasoning_effort={_llm.REASONING_EFFORT}, "
        f"temperature {_llm.TEMPERATURE}); fallback {_llm.FALLBACK_MODEL}. "
        f"Models that actually answered: {', '.join(models_used) or '—'}.",
        "- Scores (codifiability_score, evaluability_score) now computed in "
        "Python with round-half-up; LLM no longer emits them.",
        f"- Skills coded: {n_coded}/{n_selected}.",
        f"- Runtime: {runtime}s; approx cost: "
        + (f"${cost:.4f}." if cost is not None else "unavailable."),
    ]
    anomalies = []
    if fallback_events:
        anomalies.append(f"gpt-4.1 fallback used on {len(fallback_events)} skill(s): "
                         + "; ".join(fallback_events) + ".")
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