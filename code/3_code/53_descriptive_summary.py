#!/usr/bin/env python3
"""Produce coded/descriptive_summary.md — a descriptive run report over the
full coded corpus (Phase C). Reads the latest coded/coded_<date>.json and the
per-skill raw responses (for model attribution), and writes a markdown report
with: run summary, scaled-variable distributions, categorical/binary
distributions, category cross-tabs, a methodological watch-list, an outlier
list, and three representative-skill snapshots.

This report is the input to Phase D (double-coding subset selection). It does
no analysis beyond description and does not modify any coded value.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import statistics as stats
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _llm  # noqa: E402

CODED_DIR = _llm.BASE_DIR / "coded"
TODAY = _dt.date.today().isoformat()

SCALED_VARS = [
    "procedural", "analytical", "orchestration", "compliance", "documentation",
    "explicitness", "documentation_quality", "tacit_dependency",
    "context_sensitivity", "coordination_burden", "output_verifiability",
    "tests_quality", "thresholds_quality", "error_handling",
    "provenance_clarity", "maintenance_signals", "safety_notes", "scope_limits",
    "codifiability_score", "evaluability_score", "audit_confidence",
]
SCOPE_VALUES = ["micro_task", "single_bounded_task", "multi_step_workflow",
                "end_to_end_process", "unclear"]
EVAL_TYPES = ["objective", "hybrid", "holistic", "none_visible", "unclear"]
STATUS_VALUES = ["complete", "partial", "ambiguous",
                 "insufficiently_evidenced", "excluded"]
BINARY_VARS = ["requires_memory", "requires_tools", "requires_external_services"]
AUDIT_COLS = ["snyk_status", "socket_status", "agent_trust_hub_status"]
CODIF_COMP = ["explicitness", "documentation_quality",
              "tacit_dependency", "context_sensitivity"]
EVAL_COMP = ["output_verifiability", "tests_quality",
             "thresholds_quality", "error_handling"]


def _latest_coded() -> Path:
    cands = [p for p in CODED_DIR.glob("coded_*.json") if not p.stem.endswith("_final")]
    if not cands:
        raise FileNotFoundError("No coded/coded_*.json found.")
    return sorted(cands)[-1]


def _model_per_skill(skill_ids):
    """Map skill_id -> answering model (from coded/raw/<id>.json), classifying
    by the 'model' field prefix. Empty if raw files are absent."""
    out = {}
    raw_dir = CODED_DIR / "raw"
    for sid in skill_ids:
        p = raw_dir / f"{sid}.json"
        if p.exists():
            m = json.loads(p.read_text()).get("model", "")
            out[sid] = m
    return out


def _hist(values):
    return {k: sum(1 for v in values if v == k) for k in range(5)}


def _fmt_table(headers, rows):
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def main() -> int:
    src = _latest_coded()
    data = json.loads(src.read_text())
    md_meta = data.get("metadata", {})
    skills = data["skills"]
    df = pd.DataFrame(skills)
    n = len(skills)
    print(f"Loaded {n} coded records from {src.name}")

    models = _model_per_skill(df["skill_id"])
    df["_model"] = df["skill_id"].map(models).fillna("")
    fallback_skills = [s for s, m in models.items() if m and not m.startswith("gpt-5")]

    L: list[str] = []
    L.append(f"# Descriptive run summary — full corpus coding (v3.1)")
    L.append("")
    L.append(f"Generated {TODAY} by scripts/53_descriptive_summary.py from "
             f"`{src.name}`. Source of truth for Phase D subset selection. "
             f"Descriptive only — no values are altered here.")
    L.append("")

    # ---- (a) run summary -------------------------------------------------
    L.append("## (a) Run summary")
    L.append("")
    L.append(f"- Total skills coded: **{n}**")
    L.append(f"- Model breakdown: "
             + ", ".join(f"{k}={v}" for k, v in (md_meta.get('model_breakdown') or {}).items()))
    L.append(f"- gpt-4.1 fallback count: {md_meta.get('fallback_count', 0)}")
    L.append(f"- Total cost (approx): ${md_meta.get('approx_cost_usd')}")
    L.append(f"- Total runtime: {md_meta.get('runtime_seconds')}s")
    L.append(f"- Prompt version: {md_meta.get('system_prompt_version')}; "
             f"temperature={md_meta.get('temperature')}, "
             f"reasoning_effort={md_meta.get('reasoning_effort')}; single pass")
    L.append("")
    ac_hist = _hist(df["audit_confidence"])
    L.append("**audit_confidence distribution** (0–4):")
    L.append("")
    L.append(_fmt_table(["0", "1", "2", "3", "4", "mean"],
                        [[ac_hist[0], ac_hist[1], ac_hist[2], ac_hist[3], ac_hist[4],
                          round(df["audit_confidence"].mean(), 2)]]))
    L.append("")
    cs = df["coding_status"].value_counts().to_dict()
    L.append("**coding_status distribution:**")
    L.append("")
    L.append(_fmt_table(["status", "count"],
                        [[v, cs.get(v, 0)] for v in STATUS_VALUES if cs.get(v, 0)]))
    flagged = df[df["coding_status"].isin(
        ["partial", "ambiguous", "insufficiently_evidenced"])]
    L.append("")
    if len(flagged):
        L.append(f"**Skills flagged partial/ambiguous/insufficiently_evidenced "
                 f"({len(flagged)}):** "
                 + ", ".join(f"{r.skill_id} ({r.coding_status})"
                             for r in flagged.itertuples()))
    else:
        L.append("**Skills flagged partial/ambiguous/insufficiently_evidenced:** none")
    L.append("")

    # ---- (b) scaled variable distributions -------------------------------
    L.append("## (b) Scaled variable distributions (0–4)")
    L.append("")
    rows = []
    for v in SCALED_VARS:
        vals = list(df[v])
        h = _hist(vals)
        rows.append([v, round(stats.mean(vals), 2), stats.median(vals),
                     min(vals), max(vals), h[0], h[1], h[2], h[3], h[4]])
    L.append(_fmt_table(
        ["variable", "mean", "median", "min", "max", "n0", "n1", "n2", "n3", "n4"],
        rows))
    L.append("")

    # ---- (c) categorical distributions -----------------------------------
    L.append("## (c) Categorical variable distributions")
    L.append("")
    for var, vals in [("scope", SCOPE_VALUES), ("evaluation_type", EVAL_TYPES),
                      ("coding_status", STATUS_VALUES)]:
        vc = df[var].value_counts().to_dict()
        L.append(f"**{var}:**")
        L.append("")
        L.append(_fmt_table(["value", "count"],
                            [[x, vc.get(x, 0)] for x in vals if vc.get(x, 0)]))
        L.append("")

    # ---- (d) binary distributions ----------------------------------------
    L.append("## (d) Binary variable distributions")
    L.append("")
    brows = []
    for b in BINARY_VARS:
        ones = int((df[b] == 1).sum())
        brows.append([b, ones, n - ones, f"{ones / n:.0%}"])
    L.append(_fmt_table(["variable", "1 (yes)", "0 (no)", "% yes"], brows))
    L.append("")
    df["dependency_count"] = df[BINARY_VARS].sum(axis=1)
    dc = df["dependency_count"].value_counts().to_dict()
    L.append("**dependency_count** (sum of the three binaries, 0–3):")
    L.append("")
    L.append(_fmt_table(["0", "1", "2", "3", "mean"],
                        [[dc.get(0, 0), dc.get(1, 0), dc.get(2, 0), dc.get(3, 0),
                          round(df["dependency_count"].mean(), 2)]]))
    L.append("")

    # ---- (e) cross-tabs --------------------------------------------------
    L.append("## (e) Cross-tabs by category")
    L.append("")
    means = df.groupby("category")[["codifiability_score", "evaluability_score"]].mean().round(2)
    L.append("**Mean codifiability & evaluability per category:**")
    L.append("")
    L.append(_fmt_table(["category", "mean_codif", "mean_eval", "n"],
                        [[c, means.loc[c, "codifiability_score"],
                          means.loc[c, "evaluability_score"],
                          int((df["category"] == c).sum())]
                         for c in means.index]))
    L.append("")
    L.append("**category × scope (counts):**")
    L.append("")
    ct = pd.crosstab(df["category"], df["scope"])
    ct = ct.reindex(columns=[v for v in SCOPE_VALUES if v in ct.columns])
    L.append(_fmt_table(["category"] + list(ct.columns),
                        [[c] + list(ct.loc[c]) for c in ct.index]))
    L.append("")
    L.append("**category × evaluation_type (counts):**")
    L.append("")
    ct2 = pd.crosstab(df["category"], df["evaluation_type"])
    ct2 = ct2.reindex(columns=[v for v in EVAL_TYPES if v in ct2.columns])
    L.append(_fmt_table(["category"] + list(ct2.columns),
                        [[c] + list(ct2.loc[c]) for c in ct2.index]))
    L.append("")
    off = df.groupby("is_official")[["codifiability_score", "evaluability_score"]].mean().round(2)
    L.append("**is_official × mean scores:**")
    L.append("")
    L.append(_fmt_table(["is_official", "mean_codif", "mean_eval", "n"],
                        [[i, off.loc[i, "codifiability_score"],
                          off.loc[i, "evaluability_score"],
                          int((df["is_official"] == i).sum())] for i in off.index]))
    L.append("")
    df["_any_audit_fail"] = df[AUDIT_COLS].apply(lambda r: (r == "fail").any(), axis=1)
    af = df.groupby("_any_audit_fail")["safety_notes"].agg(["mean", "count"]).round(2)
    L.append("**audit status (any of snyk/socket/trust-hub == 'fail') × mean safety_notes:**")
    L.append("")
    L.append(_fmt_table(["any_audit_fail", "mean_safety_notes", "n"],
                        [[bool(i), af.loc[i, "mean"], int(af.loc[i, "count"])]
                         for i in af.index]))
    L.append("")

    # ---- (f) watch list --------------------------------------------------
    L.append("## (f) Watch list — variables that may need methodological attention")
    L.append("")
    unimodal = []
    for v in SCALED_VARS:
        h = _hist(list(df[v]))
        top_val = max(h, key=h.get)
        share = h[top_val] / n
        if share >= 0.80:
            unimodal.append((v, top_val, f"{share:.0%}"))
    if unimodal:
        L.append("**Heavily unimodal scaled variables (≥80% at one value — weak "
                 "discrimination):**")
        L.append("")
        L.append(_fmt_table(["variable", "dominant value", "share"],
                            [[v, tv, sh] for v, tv, sh in unimodal]))
    else:
        L.append("No scaled variable is ≥80% concentrated at a single value.")
    L.append("")
    tq0 = int((df["tests_quality"] == 0).sum())
    L.append(f"- **tests_quality == 0:** {tq0} of {n} skills ({tq0 / n:.0%}).")
    eq = int((df["explicitness"] == df["documentation_quality"]).sum())
    L.append(f"- **explicitness == documentation_quality:** {eq} of {n} "
             f"({eq / n:.0%}) — checks whether the two are being collapsed.")
    L.append("")

    # ---- (g) outlier list ------------------------------------------------
    L.append("## (g) Outlier list")
    L.append("")
    low_conf = df[df["audit_confidence"].isin([0, 1, 2])]
    L.append(f"**Low audit_confidence (0–2) — flag for human review "
             f"({len(low_conf)}):** "
             + (", ".join(f"{r.skill_id}({r.audit_confidence})"
                          for r in low_conf.itertuples()) or "none"))
    L.append("")
    extreme = df[df["codifiability_score"].isin([0, 4]) |
                 df["evaluability_score"].isin([0, 4])]
    L.append(f"**Extreme scores (codif or eval == 0 or 4) ({len(extreme)}):** "
             + (", ".join(f"{r.skill_id}(c{r.codifiability_score}/e{r.evaluability_score})"
                          for r in extreme.itertuples()) or "none"))
    L.append("")
    L.append(f"**gpt-4.1 fallback skills ({len(fallback_skills)}):** "
             + (", ".join(fallback_skills) or "none"))
    L.append("")

    # ---- (h) representative skills ---------------------------------------
    L.append("## (h) Representative skill snapshots")
    L.append("")
    df["_prod"] = df["codifiability_score"] * df["evaluability_score"]
    df["_sum"] = df["codifiability_score"] + df["evaluability_score"]
    med_c, med_e = df["codifiability_score"].median(), df["evaluability_score"].median()
    df["_dist"] = ((df["codifiability_score"] - med_c) ** 2 +
                   (df["evaluability_score"] - med_e) ** 2) ** 0.5

    most = df.sort_values(["_prod", "_sum", "codifiability_score"],
                          ascending=False).iloc[0]
    least = df.sort_values(["_prod", "_sum", "codifiability_score"],
                           ascending=True).iloc[0]
    median = df.sort_values(["_dist", "skill_id"]).iloc[0]

    coded_vars = [c for c in _llm.required_column_order(_llm.load_system_prompt())
                  if c not in set(_llm.echoed_columns(_llm.load_system_prompt()))]

    def snapshot(row, label):
        out = [f"### {label}: {row['skill_id']} — {row['category']}",
               f"(codifiability={row['codifiability_score']}, "
               f"evaluability={row['evaluability_score']}, "
               f"status={row['coding_status']}, "
               f"audit_confidence={row['audit_confidence']})", ""]
        kv = [[v, row[v]] for v in coded_vars
              if v not in ("codifiability_rationale", "evaluability_rationale",
                           "coding_notes")]
        out.append(_fmt_table(["variable", "value"], kv))
        out.append("")
        out.append(f"- **codifiability_rationale:** {row['codifiability_rationale']}")
        out.append(f"- **evaluability_rationale:** {row['evaluability_rationale']}")
        out.append(f"- **coding_notes:** {row['coding_notes'] or '(none)'}")
        out.append("")
        return out

    L += snapshot(most, "Most codified × evaluable")
    L += snapshot(least, "Least codified × evaluable")
    L += snapshot(median, "Closest to corpus median")

    # ---- (i) data integrity notes ---------------------------------------
    # Cross-check the corpus skill_name against the SKILL.md frontmatter name.
    # These are pre-existing corpus labels (inputs/ is read-only); the coding
    # itself used skill_md_content per the prompt's evidence rule, so codings
    # reflect the actual artefact regardless of the label. Flagged for Phase D.
    L.append("## (i) Data integrity notes (corpus labels vs SKILL.md)")
    L.append("")
    corpus_skills, _m, _p = _llm.load_corpus()
    md_by_id = {s["skill_id"]: (s.get("skill_md_content") or "") for s in corpus_skills}

    def _fm_name(md):
        m = re.match(r"^\s*---\s*\n(.*?)\n---", md, re.S)
        block = m.group(1) if m else md[:600]
        nm = re.search(r"^\s*name:\s*[\"']?([A-Za-z0-9._\-/ ]+)", block, re.M)
        return nm.group(1).strip() if nm else None

    name_mismatch = []
    for r in df.itertuples():
        fn = _fm_name(md_by_id.get(r.skill_id, ""))
        if fn and fn.lower().strip() != str(r.skill_name).lower().strip():
            name_mismatch.append([r.skill_id, r.category, r.skill_name, fn])
    if name_mismatch:
        L.append(f"{len(name_mismatch)} skill(s) have a corpus `skill_name` that "
                 f"differs from the `name:` in their SKILL.md frontmatter. Coding "
                 f"used the SKILL.md content (the artefact), per the evidence rule; "
                 f"some are benign abbreviations, others (e.g. BS02, SM01, UI03–05) "
                 f"look like genuine label/content mismatches in the corpus. Inputs "
                 f"are read-only — surfaced here for Phase D, not fixed.")
        L.append("")
        L.append(_fmt_table(["skill_id", "category", "corpus skill_name",
                             "SKILL.md name"], name_mismatch))
    else:
        L.append("No skill_name / SKILL.md frontmatter-name mismatches detected.")
    L.append("")

    report = CODED_DIR / "descriptive_summary.md"
    report.write_text("\n".join(L) + "\n")
    print(f"Wrote {report} ({len(L)} lines)")
    print(f"Representative skills: most={most['skill_id']} "
          f"least={least['skill_id']} median={median['skill_id']}")
    print(f"Flagged (partial/ambiguous/insuff): {len(flagged)} | "
          f"low-confidence: {len(low_conf)} | extreme scores: {len(extreme)} | "
          f"fallback: {len(fallback_skills)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())