#!/usr/bin/env python3
"""
_build_datasets.py — one-time consolidation step for the findings-chapter analyses.

Run once (this is the Phase-E plumbing step). It reads the canonical source
data scattered across the three project phases and writes five working files
into Coding/analysis/data/. Every subsequent analysis script reads ONLY from
Coding/analysis/data/; nothing downstream reaches back into the source phases.

Outputs (in analysis/data/):
  - population.parquet        ~5,235 rows, one per detail-complete population skill
  - corpus.parquet            120 rows, the coded corpus (authoritative v2 coding)
  - double_coded.parquet      24 rows, H1 / H2 / LLM codings of the reliability subset
  - category_mapping.csv      12-row lookup: category -> display label + id-prefix
  - codebook.csv              one row per coded variable: definition + anchors

Field provenance is documented inline at each build step.

Conventions:
  - Working directory name has a TRAILING SPACE; all paths are absolute string
    literals that include it.
  - Audit statuses are lowercased everywhere (Pass -> pass).
  - codifiability_score / evaluability_score use round-half-up (math.floor(x+0.5)),
    matching the LLM pipeline (scripts/_llm.py) and methodology 3.3.4.
  - No secrets are read or written.
"""

import csv
import json
import math
import sys

import pandas as pd

# --------------------------------------------------------------------------
# Paths (absolute; trailing space in the project root is intentional)
# --------------------------------------------------------------------------
ROOT = "/Users/Niklas/Master Thesis Claude Code "
POP_SRC      = ROOT + "/Analysis and Sampling/processed/skills_with_text_2026-05-20.json"
METHODB_SRC  = ROOT + "/Analysis and Sampling/analysis/clustering/method_b_results.json"
CODED_SRC    = ROOT + "/Coding/coded/coded_2026-05-23_v2.json"
CORPUS_V3    = ROOT + "/Coding/inputs/agent_skills_corpus_v3_2026-05-23.json"
PROMPT_SRC   = ROOT + "/Coding/inputs/System_Prompt.json"
HUMAN1_SRC   = ROOT + "/Coding/inputs/coded_human_01.csv.xlsx"
HUMAN2_SRC   = ROOT + "/Coding/inputs/coded_human_02.csv.xlsx"

DATA_DIR     = ROOT + "/Coding/analysis/data"

# The 24-skill double-coded reliability subset (methodology 3.3.8).
DOUBLE_CODED_IDS = [
    "BS02", "BS09", "CC03", "CC09", "CT02", "CT10", "DM04", "DM06",
    "ET05", "ET09", "FA03", "FA08", "PM07", "PM10", "RD01", "RD06",
    "SC05", "SC07", "SD02", "SD08", "SM04", "SM05", "UI01", "UI08",
]

# Short display labels for figures (keyed by the 12 functional categories).
CATEGORY_DISPLAY = {
    "business-strategy":       "Business Strategy",
    "communication-tools":     "Comms Tools",
    "content-creation":        "Content",
    "document-management":     "Doc Management",
    "education-and-training":  "Education",
    "financial-analysis":      "Finance",
    "project-management":      "Project Mgmt",
    "research-and-analysis":   "Research",
    "security-and-compliance": "Security",
    "social-media-management": "Social Media",
    "software-development":    "Software Dev",
    "ui-ux-design":            "UI/UX",
}

# Canonical per-category plot colour (single source of truth: written to
# category_mapping.csv's display_color column, then read back by
# scripts/_style.py — colours are NEVER hardcoded in section scripts).
# Tableau-derived qualitative palette; perceptually distinguishable and
# reasonably colour-blind-aware. Adjacencies (alphabetical category order)
# deliberately avoid placing a saturated red next to a saturated green.
CATEGORY_DISPLAY_COLOR = {
    "business-strategy":       "#4E79A7",  # blue
    "communication-tools":     "#F28E2B",  # orange
    "content-creation":        "#59A14F",  # green
    "document-management":     "#B07AA1",  # purple
    "education-and-training":  "#EDC948",  # yellow
    "financial-analysis":      "#76B7B2",  # teal
    "project-management":      "#FF9DA7",  # pink
    "research-and-analysis":   "#9C755F",  # brown
    "security-and-compliance": "#E15759",  # red
    "social-media-management": "#76549A",  # violet
    "software-development":    "#B6992D",  # olive/gold
    "ui-ux-design":            "#59636E",  # slate
}

# Components of the two derived scores (methodology 3.3.4 / scripts/_llm.py).
CODIFIABILITY_COMPONENTS = ("explicitness", "documentation_quality",
                            "tacit_dependency", "context_sensitivity")
EVALUABILITY_COMPONENTS  = ("output_verifiability", "tests_quality",
                            "thresholds_quality", "error_handling")

# Coded variables triple-coded in the double-coded dataset (H1/H2/LLM).
# Order mirrors the codebook; the two derived scores and the free-text
# rationales/notes are appended.
DOUBLE_CODED_VARS = [
    "scope", "procedural", "analytical", "orchestration", "compliance",
    "documentation", "explicitness", "documentation_quality",
    "tacit_dependency", "context_sensitivity",
    "codifiability_score", "codifiability_rationale",
    "coordination_burden", "evaluation_type", "output_verifiability",
    "tests_quality", "thresholds_quality", "error_handling",
    "evaluability_score", "evaluability_rationale",
    "requires_memory", "requires_tools", "requires_external_services",
    "provenance_clarity", "maintenance_signals", "safety_notes",
    "scope_limits", "coding_status", "audit_confidence", "coding_notes",
]


def round_half_up(x):
    """Standard round-half-up for non-negative x (0.5->1, 2.5->3). Matches
    scripts/_llm.py; NOT Python's banker's round()."""
    return int(math.floor(x + 0.5))


def codifiability_from(rec):
    try:
        e, d, t, c = (int(rec[k]) for k in CODIFIABILITY_COMPONENTS)
    except (KeyError, TypeError, ValueError):
        return None
    return round_half_up((e + d + (4 - t) + (4 - c)) / 4)


def evaluability_from(rec):
    try:
        ov, tq, th, eh = (int(rec[k]) for k in EVALUABILITY_COMPONENTS)
    except (KeyError, TypeError, ValueError):
        return None
    return round_half_up((ov + tq + th + eh) / 4)


# --------------------------------------------------------------------------
# (a) population.parquet
# --------------------------------------------------------------------------
def build_population():
    """Source: skills_with_text_2026-05-20.json (5,235 detail-complete skills,
    from the Analysis-and-Sampling phase — the superset that carries
    summary_clean; skills_combined lacks it). Method B primary/secondary
    left-joined on skill_url from method_b_results.json."""
    pop = json.load(open(POP_SRC))["skills"]
    mb = json.load(open(METHODB_SRC))["assignments"]
    mb_by_url = {a["skill_url"]: a for a in mb}

    rows = []
    for s in pop:
        audits = {a["slug"]: a.get("status") for a in (s.get("audits") or [])}

        def lc(v):  # lowercase audit status, keep None as None
            return v.lower() if isinstance(v, str) else None

        skill_md = s.get("skill_md") or {}
        has_md = "Y" if (skill_md.get("raw") or "").strip() else "N"
        mbrec = mb_by_url.get(s["skill_url"]) or {}

        rows.append({
            # --- identity / from skills_with_text_2026-05-20.json ---
            "skill_id":                 s.get("skill_id"),
            "skill_name":               s.get("skill_name"),
            "repo_name":                s.get("repo"),          # 'repo' -> repo_name
            "repo_url":                 s.get("repo_url"),
            "skill_url":                s.get("skill_url"),
            "summary":                  s.get("summary"),
            "summary_clean":            s.get("summary_clean"),
            # --- ecosystem signals ---
            "installs":                 s.get("installs"),
            "leaderboard_rank":         s.get("leaderboard_rank"),
            "is_official":              s.get("is_official"),
            "github_stars_abbreviated": s.get("github_stars_abbreviated"),
            # --- audit statuses (flattened from the audits list, lowercased) ---
            "snyk_status":              lc(audits.get("snyk")),
            "socket_status":            lc(audits.get("socket")),
            "agent_trust_hub_status":   lc(audits.get("agent-trust-hub")),
            # --- skill.md presence (derived: 'Y' if raw non-empty else 'N') ---
            "has_skill_md":             has_md,
            # --- Method B classification (left-join on skill_url) ---
            "method_b_category":        mbrec.get("primary"),
            "method_b_secondary":       mbrec.get("secondary"),
        })
    df = pd.DataFrame(rows)
    out = DATA_DIR + "/population.parquet"
    df.to_parquet(out, index=False)
    n_no_mb = df["method_b_category"].isna().sum()
    print(f"[population] {len(df)} rows, {len(df.columns)} cols -> population.parquet "
          f"(no Method B label: {n_no_mb})")
    return df


# --------------------------------------------------------------------------
# (b) corpus.parquet
# --------------------------------------------------------------------------
def build_corpus():
    """Source: coded_2026-05-23_v2.json (authoritative full-corpus coding after
    the BS02/SM01/UI04/UI05 fix). skill_md_content joined by skill_id from
    agent_skills_corpus_v3_2026-05-23.json. in_double_coded_subset flag added."""
    coded = json.load(open(CODED_SRC))["skills"]
    v3 = json.load(open(CORPUS_V3))["skills"]
    md_by_id = {s["skill_id"]: s.get("skill_md_content") for s in v3}

    # The coded JSON echoes audit statuses in mixed case (~9 skills carry
    # 'Pass'/'Warn'). Lowercase them to match the population convention
    # (Part 4a) so the two datasets group/join consistently. Casing only;
    # no substantive value change.
    AUDIT_COLS = ("snyk_status", "socket_status", "agent_trust_hub_status")

    subset = set(DOUBLE_CODED_IDS)
    rows = []
    for s in coded:
        row = dict(s)  # all coded fields, flattened
        for col in AUDIT_COLS:
            if isinstance(row.get(col), str):
                row[col] = row[col].lower()
        row["skill_md_content"] = md_by_id.get(s["skill_id"])
        row["in_double_coded_subset"] = s["skill_id"] in subset
        rows.append(row)
    df = pd.DataFrame(rows)
    out = DATA_DIR + "/corpus.parquet"
    df.to_parquet(out, index=False)
    print(f"[corpus] {len(df)} rows, {len(df.columns)} cols -> corpus.parquet "
          f"(in_double_coded_subset=True: {int(df['in_double_coded_subset'].sum())})")
    return df


# --------------------------------------------------------------------------
# (c) double_coded.parquet
# --------------------------------------------------------------------------
def _read_human(path):
    """Read the 'Human Coding' sheet (NOT the first reference sheet) and return
    a dict keyed by skill_id. Verifies exactly 24 rows."""
    df = pd.read_excel(path, sheet_name="Human Coding")
    assert len(df) == 24, f"{path}: expected 24 rows in 'Human Coding', got {len(df)}"
    return {r["skill_id"]: r.to_dict() for _, r in df.iterrows()}


def build_double_coded():
    """Sources: coded_human_01/02 ('Human Coding' sheet, 24 rows each) and the
    LLM coding from coded_2026-05-23_v2.json filtered to the 24 subset skills.
    Output: one row per skill, columns <var>_h1 / <var>_h2 / <var>_llm.

    Score handling (methodology 3.3.4; convention decided 2026-05-26):
      - The two derived scores are formula values with round-half-up for ALL
        three coders, so derived-score agreement reflects component agreement
        rather than rounding-convention differences.
      - H1's scores are null in the xlsx -> computed from H1's components.
      - H2's scores are present in the xlsx but H2 hand-rounded DOWN at the .5
        boundary in 6 cases -> recomputed from H2's components with round-half-up
        (canonical *_h2 columns), and the original hand-entered values are
        preserved in *_h2_provided columns for transparency.
      - LLM scores are already derived in coded_2026-05-23_v2.json.
    """
    h1 = _read_human(HUMAN1_SRC)
    h2 = _read_human(HUMAN2_SRC)
    coded = {s["skill_id"]: s for s in json.load(open(CODED_SRC))["skills"]
             if s["skill_id"] in set(DOUBLE_CODED_IDS)}

    # Recompute / fill the derived scores per coder; preserve H2's hand entries.
    n_h1_computed = 0
    h2_provided = {}
    h2_score_divergences = []
    for sid in DOUBLE_CODED_IDS:
        # H1: both scores are null in the source -> compute from components.
        for fld, fn in (("codifiability_score", codifiability_from),
                        ("evaluability_score", evaluability_from)):
            if pd.isna(h1[sid].get(fld)):
                h1[sid][fld] = fn(h1[sid])
                n_h1_computed += 1
        # H2: stash the hand entry, then overwrite with the round-half-up formula.
        h2_provided[sid] = {}
        for fld, fn in (("codifiability_score", codifiability_from),
                        ("evaluability_score", evaluability_from)):
            provided = h2[sid].get(fld)
            h2_provided[sid][fld] = None if pd.isna(provided) else int(provided)
            recomputed = fn(h2[sid])
            if (h2_provided[sid][fld] is not None and recomputed is not None
                    and h2_provided[sid][fld] != recomputed):
                h2_score_divergences.append((sid, fld, h2_provided[sid][fld], recomputed))
            h2[sid][fld] = recomputed  # canonical = round-half-up formula

    rows = []
    for sid in DOUBLE_CODED_IDS:
        llm = coded[sid]
        row = {
            "skill_id":   sid,
            "skill_name": llm.get("skill_name"),
            "category":   llm.get("category"),
        }
        for var in DOUBLE_CODED_VARS:
            row[f"{var}_h1"]  = h1[sid].get(var)
            row[f"{var}_h2"]  = h2[sid].get(var)
            row[f"{var}_llm"] = llm.get(var)
        # Provenance: H2's original hand-entered derived scores (pre-recompute).
        row["codifiability_score_h2_provided"] = h2_provided[sid]["codifiability_score"]
        row["evaluability_score_h2_provided"]  = h2_provided[sid]["evaluability_score"]
        rows.append(row)
    df = pd.DataFrame(rows)
    out = DATA_DIR + "/double_coded.parquet"
    df.to_parquet(out, index=False)
    print(f"[double_coded] {len(df)} rows, {len(df.columns)} cols -> double_coded.parquet")
    print(f"               H1 derived scores computed (were null): {n_h1_computed} (expected 48 = 24x2)")
    print(f"               H2 derived scores recomputed with round-half-up; "
          f"{len(h2_score_divergences)} differed from H2's hand entry (kept in *_h2_provided):")
    for sid, fld, prov, recomp in h2_score_divergences:
        print(f"                 {sid} {fld}: hand-entered={prov} -> recomputed={recomp}")
    return df


# --------------------------------------------------------------------------
# (d) category_mapping.csv
# --------------------------------------------------------------------------
def build_category_mapping(corpus_df):
    """12 functional categories: display label, id-prefix, n in corpus.
    Prefix is derived from the corpus skill_ids (e.g. BS02 -> BS)."""
    import re
    pref = {}
    counts = corpus_df["category"].value_counts().to_dict()
    for _, r in corpus_df.iterrows():
        m = re.match(r"^([A-Za-z]+)\d+$", str(r["skill_id"]))
        if m:
            pref.setdefault(r["category"], m.group(1))
    rows = []
    for cat in sorted(CATEGORY_DISPLAY):
        rows.append({
            "category":       cat,
            "display_label":  CATEGORY_DISPLAY[cat],
            "skill_id_prefix": pref.get(cat, ""),
            "n_in_corpus":    int(counts.get(cat, 0)),
            # Canonical plot colour — single source of truth for _style.py.
            "display_color":  CATEGORY_DISPLAY_COLOR[cat],
        })
    df = pd.DataFrame(rows)
    out = DATA_DIR + "/category_mapping.csv"
    df.to_csv(out, index=False)
    print(f"[category_mapping] {len(df)} rows -> category_mapping.csv")
    return df


# --------------------------------------------------------------------------
# (e) codebook.csv
# --------------------------------------------------------------------------
def build_codebook():
    """Source: System_Prompt.json column_definitions. One row per analytic
    variable (source 'coded' or python-derived score): definition, scale,
    and flattened anchors / allowed values. For human reference and for any
    script that needs to cite anchors."""
    sp = json.load(open(PROMPT_SRC))
    defs = sp["column_definitions"]
    rows = []
    for var, d in defs.items():
        src = d.get("source", "")
        if src not in ("coded", "python_derived_after_api_call"):
            continue  # skip echoed-from-input identity/ecosystem fields
        anchors = ""
        if "scale_anchors" in d and isinstance(d["scale_anchors"], dict):
            anchors = " | ".join(f"{k}: {v}" for k, v in d["scale_anchors"].items())
        elif "allowed_values" in d:
            anchors = "allowed: " + ", ".join(str(v) for v in d["allowed_values"])
        note = d.get("important_note") or d.get("note") or d.get("formula_treatment") or ""
        rows.append({
            "variable":   var,
            "source":     src,
            "type":       d.get("type", ""),
            "scale":      d.get("scale", ""),
            "definition": d.get("definition", ""),
            "anchors_or_allowed_values": anchors,
            "note":       note,
        })
    df = pd.DataFrame(rows)
    out = DATA_DIR + "/codebook.csv"
    df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[codebook] {len(df)} variables -> codebook.csv")
    return df


def main():
    print("Building consolidated working datasets in analysis/data/ ...\n")
    pop = build_population()
    corpus = build_corpus()
    build_double_coded()
    build_category_mapping(corpus)
    build_codebook()
    print("\nDone. Run scripts/00_validate_data.py next.")


if __name__ == "__main__":
    sys.exit(main())
