#!/usr/bin/env python3
"""
09_reliability_analysis.py — Section 9 of the findings chapter: coding
reliability and human-LLM convergence on the 24-skill double-coded subset.

The reliability subset (methodology 3.3.8) was independently coded by two human
coders (H1, H2) WITHOUT seeing the LLM output; the LLM also coded all 120
skills, the same 24 among them. That gives three parallel codings of the same
24 skills and three pairwise comparisons per variable: H1-H2 (the human-human
reference), H1-LLM, and H2-LLM. Per methodology 3.3.9 the PATTERN of
disagreement is treated as an analytical output, not as noise to reconcile;
this script operationalises that framing.

Variable typing drives the statistic (see the brief and methodology 3.3):
  - 18 ordinal (0-4)   : mean absolute deviation (MAD) + exact-match rate
  - 2 derived ordinal  : same; canonical round-half-up scores for every coder
                         (H2 uses *_h2, NOT *_h2_provided)
  - 3 binary           : % exact agreement
  - 2 categorical      : Cohen's kappa + % exact agreement

Self-contained: reads ONLY from analysis/data/double_coded.parquet, imports the
locked visual style from _style.py, writes figures to analysis/figures/
(.png + .svg), tables to analysis/tables/ (.csv). Descriptive only;
interpretation is deferred to the chapter draft.

Run from anywhere:
    python3 scripts/09_reliability_analysis.py
"""

import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as st  # noqa: E402

DATA_DIR = st.DATA_DIR
TBL_DIR = st.TBL_DIR

N = 24  # the double-coded subset size

# --------------------------------------------------------------------------
# Variable groups (fixed canonical order used in every table/figure).
# --------------------------------------------------------------------------
# 18 ordinal 0-4 (knowledge types -> codifiability comps -> horizon ->
# evaluability comps -> governance), grouped so related variables sit together.
ORDINAL = [
    "procedural", "analytical", "orchestration", "compliance", "documentation",
    "explicitness", "documentation_quality", "tacit_dependency", "context_sensitivity",
    "coordination_burden",
    "output_verifiability", "tests_quality", "thresholds_quality", "error_handling",
    "provenance_clarity", "maintenance_signals", "safety_notes", "scope_limits",
]
# 2 derived ordinal 0-4 (canonical round-half-up scores for all coders).
DERIVED = ["codifiability_score", "evaluability_score"]
# 3 binary 0/1.
BINARY = ["requires_memory", "requires_tools", "requires_external_services"]
# 2 categorical (Cohen's kappa).
CATEGORICAL = ["scope", "evaluation_type"]

# The 20 numeric variables that carry MAD / per-skill divergence / bias.
NUMERIC = ORDINAL + DERIVED

# The two inverted-direction components (higher raw code = LESS codifiable);
# flagged so the bias section can note that a positive LLM deviation there means
# the LLM read MORE tacit dependence / context-boundness, i.e. less codifiable.
INVERTED = {"tacit_dependency", "context_sensitivity"}

PAIRS = [("h1", "h2"), ("h1", "llm"), ("h2", "llm")]
PAIR_KEYS = {("h1", "h2"): "h1_h2", ("h1", "llm"): "h1_llm", ("h2", "llm"): "h2_llm"}

# Display labels for variables (figures/tables).
VAR_LABELS = {
    "procedural": "procedural", "analytical": "analytical",
    "orchestration": "orchestration", "compliance": "compliance",
    "documentation": "documentation", "explicitness": "explicitness",
    "documentation_quality": "documentation_quality",
    "tacit_dependency": "tacit_dependency *(inv.)*",
    "context_sensitivity": "context_sensitivity *(inv.)*",
    "coordination_burden": "coordination_burden",
    "output_verifiability": "output_verifiability", "tests_quality": "tests_quality",
    "thresholds_quality": "thresholds_quality", "error_handling": "error_handling",
    "provenance_clarity": "provenance_clarity",
    "maintenance_signals": "maintenance_signals", "safety_notes": "safety_notes",
    "scope_limits": "scope_limits",
    "codifiability_score": "codifiability_score", "evaluability_score": "evaluability_score",
    "requires_memory": "requires_memory", "requires_tools": "requires_tools",
    "requires_external_services": "requires_external_services",
    "scope": "scope", "evaluation_type": "evaluation_type",
}
# Plain figure labels (no markdown), with the inverted marker spelled out.
FIG_LABELS = {v: v for v in VAR_LABELS}
FIG_LABELS["tacit_dependency"] = "tacit_dependency (inv.)"
FIG_LABELS["context_sensitivity"] = "context_sensitivity (inv.)"


# ==========================================================================
# Helpers
# ==========================================================================
def load_double_coded():
    df = pd.read_parquet(DATA_DIR / "double_coded.parquet")
    assert len(df) == N, f"expected {N} double-coded skills, got {len(df)}"
    return df


def round_half_up(x):
    """Standard round-half-up (0.5 -> 1), the project-wide convention."""
    return int(math.floor(x + 0.5))


def col(df, var, coder):
    return df[f"{var}_{coder}"]


def formula_guard(df):
    """Re-derive each coder's codifiability_score and evaluability_score from
    that coder's OWN components (round-half-up) and assert it matches the stored
    canonical score. By construction this holds for all three coders (H1 and the
    LLM were computed round-half-up; H2's canonical *_h2 was recomputed the same
    way). A failure would mean a build/rounding problem worth stopping for."""
    for coder in ["h1", "h2", "llm"]:
        codif = [
            round_half_up((col(df, "explicitness", coder)[i]
                           + col(df, "documentation_quality", coder)[i]
                           + (4 - col(df, "tacit_dependency", coder)[i])
                           + (4 - col(df, "context_sensitivity", coder)[i])) / 4)
            for i in df.index
        ]
        ev = [
            round_half_up((col(df, "output_verifiability", coder)[i]
                           + col(df, "tests_quality", coder)[i]
                           + col(df, "thresholds_quality", coder)[i]
                           + col(df, "error_handling", coder)[i]) / 4)
            for i in df.index
        ]
        assert (pd.Series(codif, index=df.index)
                == col(df, "codifiability_score", coder)).all(), \
            f"codifiability_score_{coder} does not match round-half-up of its components"
        assert (pd.Series(ev, index=df.index)
                == col(df, "evaluability_score", coder)).all(), \
            f"evaluability_score_{coder} does not match round-half-up of its components"


def single_value_guard(df):
    """Per the brief: if any variable has all-identical H1 values across the 24
    skills, that is a coding-frame issue, not a normal finding — STOP."""
    flat = []
    for v in ORDINAL + DERIVED + BINARY + CATEGORICAL:
        if col(df, v, "h1").nunique(dropna=False) == 1:
            flat.append(v)
    if flat:
        raise SystemExit(
            "STOP — these variables have a single unique H1 value across all 24 "
            f"skills (coding-frame issue, investigate): {flat}"
        )


# ==========================================================================
# (a) Per-variable agreement summary
# ==========================================================================
def agreement_summary(df):
    rows = []

    def base_row(var, vtype):
        r = {"variable": var, "variable_type": vtype, "n": N}
        for pa, pb in PAIRS:
            k = PAIR_KEYS[(pa, pb)]
            r[f"mad_{k}"] = np.nan
            r[f"exact_match_{k}"] = np.nan
            r[f"pct_agree_{k}"] = np.nan
            r[f"kappa_{k}"] = np.nan
        return r

    # ordinal + derived: MAD + exact-match rate
    for var, vtype in [(v, "ordinal") for v in ORDINAL] + \
                      [(v, "derived") for v in DERIVED]:
        r = base_row(var, vtype)
        for pa, pb in PAIRS:
            k = PAIR_KEYS[(pa, pb)]
            a, b = col(df, var, pa), col(df, var, pb)
            r[f"mad_{k}"] = round(float((a - b).abs().mean()), 4)
            r[f"exact_match_{k}"] = round(float((a == b).mean()), 4)
        rows.append(r)

    # binary: % exact agreement (mad/exact_match left NaN)
    for var in BINARY:
        r = base_row(var, "binary")
        for pa, pb in PAIRS:
            k = PAIR_KEYS[(pa, pb)]
            a, b = col(df, var, pa), col(df, var, pb)
            r[f"pct_agree_{k}"] = round(float((a == b).mean()), 4)
        rows.append(r)

    # categorical: Cohen's kappa + % exact agreement
    for var in CATEGORICAL:
        r = base_row(var, "categorical")
        for pa, pb in PAIRS:
            k = PAIR_KEYS[(pa, pb)]
            a, b = col(df, var, pa), col(df, var, pb)
            r[f"pct_agree_{k}"] = round(float((a == b).mean()), 4)
            r[f"kappa_{k}"] = cohen_kappa_value(a, b)
        rows.append(r)

    cols = ["variable", "variable_type", "n"]
    for pa, pb in PAIRS:
        cols.append(f"mad_{PAIR_KEYS[(pa, pb)]}")
    for pa, pb in PAIRS:
        cols.append(f"exact_match_{PAIR_KEYS[(pa, pb)]}")
    for pa, pb in PAIRS:
        cols.append(f"pct_agree_{PAIR_KEYS[(pa, pb)]}")
    for pa, pb in PAIRS:
        cols.append(f"kappa_{PAIR_KEYS[(pa, pb)]}")

    out = pd.DataFrame(rows)[cols]
    out.to_csv(TBL_DIR / "09_agreement_summary.csv", index=False)
    return out


def cohen_kappa_value(a, b):
    """Cohen's kappa, reported as-is. NaN if either column is single-valued
    (kappa undefined) or if sklearn returns a non-finite value."""
    if a.nunique(dropna=False) == 1 or b.nunique(dropna=False) == 1:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = cohen_kappa_score(a.astype(str), b.astype(str))
    return round(float(kappa), 4) if np.isfinite(kappa) else np.nan


# ==========================================================================
# (b) Agreement profile figure (ordinal/derived, ranked by H1-H2 exact match)
# ==========================================================================
def fig_agreement_profile(summary):
    sub = summary[summary["variable_type"].isin(["ordinal", "derived"])].copy()
    sub = sub.sort_values("exact_match_h1_h2", ascending=True)  # top of axis = highest
    yvars = sub["variable"].tolist()
    y = np.arange(len(yvars))

    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    # thin connector per row, then the three coloured dots
    for i, var in enumerate(yvars):
        xs = [sub.loc[sub.variable == var, f"exact_match_{k}"].iloc[0]
              for k in ["h1_h2", "h1_llm", "h2_llm"]]
        ax.plot([min(xs), max(xs)], [i, i], color="#CCCCCC", lw=1.0, zorder=1)
    for k in ["h1_h2", "h1_llm", "h2_llm"]:
        ax.scatter(sub[f"exact_match_{k}"], y, s=58,
                   color=st.CODER_PAIR_COLORS[k], label=st.CODER_PAIR_LABELS[k],
                   zorder=3, edgecolor="white", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([FIG_LABELS[v] for v in yvars])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(yvars) - 0.4)
    ax.set_xlabel("Exact-match rate (proportion of 24 skills)")
    ax.set_title("Per-variable agreement profile\n(ordinal/derived, ranked by H1–H2 exact match)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="coder pair", frameon=True, framealpha=0.9)
    fig.tight_layout()
    st.save_fig(fig, "09_agreement_profile")
    plt.close(fig)


# ==========================================================================
# (c) LLM systematic-bias check
# ==========================================================================
def llm_bias(df):
    rows = []
    for var in NUMERIC:
        human_mean = (col(df, var, "h1") + col(df, var, "h2")) / 2.0
        dev = col(df, var, "llm") - human_mean
        rows.append({
            "variable": var,
            "variable_type": "derived" if var in DERIVED else "ordinal",
            "inverted": var in INVERTED,
            "mean_llm_minus_human": round(float(dev.mean()), 4),
            "sd_llm_minus_human": round(float(dev.std(ddof=1)), 4),
            "n": N,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "09_llm_bias.csv", index=False)
    return out


def fig_llm_bias(bias):
    sub = bias.copy()
    sub["abs"] = sub["mean_llm_minus_human"].abs()
    sub = sub.sort_values("abs", ascending=True)
    y = np.arange(len(sub))

    rdbu = colormaps["RdBu"]
    pos_color = rdbu(0.85)   # blue  — LLM scores HIGHER than human consensus
    neg_color = rdbu(0.15)   # red   — LLM scores LOWER than human consensus
    colors = [pos_color if v >= 0 else neg_color for v in sub["mean_llm_minus_human"]]

    fig, ax = plt.subplots(figsize=(7.4, 8.0))
    ax.barh(y, sub["mean_llm_minus_human"], color=colors, edgecolor="#333333", linewidth=0.4)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([FIG_LABELS[v] for v in sub["variable"]])
    ax.set_xlabel("Mean signed deviation:  LLM − mean(H1, H2)")
    ax.set_title("LLM systematic-bias check\n(mean signed deviation from the human consensus, per variable)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    # value annotations
    for i, v in zip(y, sub["mean_llm_minus_human"]):
        ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=pos_color),
               plt.Rectangle((0, 0), 1, 1, color=neg_color)]
    ax.legend(handles, ["LLM scores higher (+)", "LLM scores lower (−)"],
              loc="lower right", frameon=True, framealpha=0.9)
    pad = max(0.18, sub["abs"].max() * 1.25)
    ax.set_xlim(-pad, pad)
    fig.tight_layout()
    st.save_fig(fig, "09_llm_bias")
    plt.close(fig)


# ==========================================================================
# (d) Per-skill divergence
# ==========================================================================
def per_skill_divergence(df):
    """For each skill: total_disagreement = sum over the 20 numeric variables of
    the max absolute pairwise deviation among the three coders on that variable
    (= max(values) - min(values))."""
    rows = []
    for idx in df.index:
        per_var = {}
        for var in NUMERIC:
            vals = [int(col(df, var, c)[idx]) for c in ["h1", "h2", "llm"]]
            per_var[var] = max(vals) - min(vals)
        total = sum(per_var.values())
        top3 = sorted(per_var.items(), key=lambda kv: (-kv[1], NUMERIC.index(kv[0])))[:3]
        row = {
            "skill_id": df.loc[idx, "skill_id"],
            "category": df.loc[idx, "category"],
            "total_disagreement": total,
        }
        for n, (var, spread) in enumerate(top3, start=1):
            row[f"top{n}_variable"] = var
            row[f"top{n}_spread"] = spread
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("total_disagreement", ascending=False).reset_index(drop=True)
    out.to_csv(TBL_DIR / "09_per_skill_divergence.csv", index=False)
    return out


def fig_per_skill_divergence(per_skill):
    sub = per_skill.sort_values("total_disagreement", ascending=True).reset_index(drop=True)
    y = np.arange(len(sub))
    top3_ids = set(per_skill.nlargest(3, "total_disagreement")["skill_id"])
    colors = [st.BINARY_COLORS["positive"] if sid in top3_ids
              else st.BINARY_COLORS["negative"] for sid in sub["skill_id"]]

    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    ax.barh(y, sub["total_disagreement"], color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["skill_id"])
    ax.set_xlabel("Total disagreement (sum of max pairwise spread over 20 numeric variables)")
    ax.set_title("Per-skill divergence across the three coders\n(higher = the three coders disagree more on that skill)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    for i, v in zip(y, sub["total_disagreement"]):
        ax.text(v + 0.2, i, str(int(v)), va="center", ha="left", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=st.BINARY_COLORS["positive"]),
               plt.Rectangle((0, 0), 1, 1, color=st.BINARY_COLORS["negative"])]
    ax.legend(handles, ["top-3 most-divergent skills", "other skills"],
              loc="lower right", frameon=True, framealpha=0.9)
    ax.set_xlim(0, sub["total_disagreement"].max() * 1.12)
    fig.tight_layout()
    st.save_fig(fig, "09_per_skill_divergence")
    plt.close(fig)


# ==========================================================================
# (e) By-category divergence (within the 24-skill subset; n=2 per category)
# ==========================================================================
def by_category_divergence(per_skill):
    g = (per_skill.groupby("category")["total_disagreement"]
         .agg(n="count", mean_total_disagreement="mean").reset_index())
    g["mean_total_disagreement"] = g["mean_total_disagreement"].round(2)
    g = g.sort_values("mean_total_disagreement", ascending=False).reset_index(drop=True)
    g.to_csv(TBL_DIR / "09_by_category_divergence.csv", index=False)
    return g


# ==========================================================================
# (f) Well-anchored vs contested classification (on H1-H2 agreement)
# ==========================================================================
def anchored_vs_contested(summary):
    rows = []
    for _, r in summary.iterrows():
        var, vtype = r["variable"], r["variable_type"]
        if vtype in ("ordinal", "derived"):
            metric = r["exact_match_h1_h2"]
            count = int(round(metric * N))
            if count >= 17:
                bucket = "well-anchored"
                crit = f"H1–H2 exact match {count}/24 ≥ 17/24 (≥70%)"
            elif count < 10:
                bucket = "contested"
                crit = f"H1–H2 exact match {count}/24 < 10/24 (<42%)"
            else:
                bucket = "middle"
                crit = f"H1–H2 exact match {count}/24 (10–16/24)"
            metric_str = f"exact_match={metric:.3f} ({count}/24)"
        elif vtype == "binary":
            metric = r["pct_agree_h1_h2"]
            if metric >= 0.80:
                bucket = "well-anchored"
                crit = f"H1–H2 agreement {metric:.1%} ≥ 80%"
            elif metric < 0.60:
                bucket = "contested"
                crit = f"H1–H2 agreement {metric:.1%} < 60%"
            else:
                bucket = "middle"
                crit = f"H1–H2 agreement {metric:.1%} (60–80%)"
            metric_str = f"pct_agree={metric:.3f}"
        else:  # categorical
            metric = r["kappa_h1_h2"]
            if pd.isna(metric):
                bucket = "undefined"
                crit = "H1–H2 kappa undefined (single-value column)"
            elif metric >= 0.60:
                bucket = "well-anchored"
                crit = f"H1–H2 kappa {metric:.3f} ≥ 0.60"
            elif metric < 0.40:
                bucket = "contested"
                crit = f"H1–H2 kappa {metric:.3f} < 0.40"
            else:
                bucket = "middle"
                crit = f"H1–H2 kappa {metric:.3f} (0.40–0.60)"
            metric_str = f"kappa={metric:.3f}" if pd.notna(metric) else "kappa=NaN"
        rows.append({
            "variable": var, "variable_type": vtype, "bucket": bucket,
            "h1_h2_metric": metric_str, "criterion": crit,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "09_anchored_vs_contested.csv", index=False)
    return out


# ==========================================================================
# (g) LLM disagreement profile vs human-human (MAD), ordinal/derived
# ==========================================================================
def fig_llm_vs_human_human(summary):
    sub = summary[summary["variable_type"].isin(["ordinal", "derived"])].copy()
    sub["human_llm_mad"] = (sub["mad_h1_llm"] + sub["mad_h2_llm"]) / 2.0
    sub = sub.sort_values("mad_h1_h2", ascending=False).reset_index(drop=True)  # top = best agreement
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    for i in y:
        x0 = sub.loc[i, "mad_h1_h2"]
        x1 = sub.loc[i, "human_llm_mad"]
        ax.plot([x0, x1], [i, i], color="#CCCCCC", lw=1.0, zorder=1)
    ax.scatter(sub["mad_h1_h2"], y, s=58, color=st.CODER_PAIR_COLORS["h1_h2"],
               label="H1–H2 MAD (human–human reference)", zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.scatter(sub["human_llm_mad"], y, s=58, color=st.CODER_PAIR_COLORS["h1_llm"],
               label="mean human–LLM MAD  (½·[H1–LLM + H2–LLM])", zorder=3,
               edgecolor="white", linewidth=0.6, marker="D")

    ax.set_yticks(y)
    ax.set_yticklabels([FIG_LABELS[v] for v in sub["variable"]])
    ax.set_xlabel("Mean absolute deviation (scale points, 0–4)")
    ax.set_title("LLM vs human–human disagreement\n(low MAD = high agreement; ordinal/derived)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(sub["mad_h1_h2"].max(), sub["human_llm_mad"].max()) * 1.15)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    # Top rows are low-MAD (left side), so the upper-right corner is empty —
    # put the legend there to avoid covering the high-MAD points at the bottom.
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    st.save_fig(fig, "09_llm_vs_human_human")
    plt.close(fig)
    return sub


# ==========================================================================
# main
# ==========================================================================
def main():
    st.apply_style()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_double_coded()
    single_value_guard(df)
    formula_guard(df)

    print("=" * 72)
    print("SECTION 9 — CODING RELIABILITY & HUMAN-LLM CONVERGENCE")
    print("=" * 72)
    print(f"  n = {N} double-coded skills; 3 coders (H1, H2, LLM); 3 pairwise comparisons")
    print("  single-value-H1 guard: PASS (no variable has identical H1 across 24)")
    print("  formula guard: PASS (every coder's derived scores = round-half-up of its components)")

    # (a)
    summary = agreement_summary(df)
    print("\n[a] tables/09_agreement_summary.csv  (25 rows: 18 ordinal + 2 derived + 3 binary + 2 categorical)")
    show = summary.copy()
    print(show.to_string(index=False))

    # (b)
    fig_agreement_profile(summary)
    print("\n[b] figures/09_agreement_profile.{png,svg} (ordinal/derived, ranked by H1-H2 exact match)")

    # (c)
    bias = llm_bias(df)
    fig_llm_bias(bias)
    print("\n[c] tables/09_llm_bias.csv + figures/09_llm_bias.{png,svg}")
    print(bias.to_string(index=False))
    n_pos = int((bias["mean_llm_minus_human"] > 0).sum())
    n_neg = int((bias["mean_llm_minus_human"] < 0).sum())
    n_zero = int((bias["mean_llm_minus_human"] == 0).sum())
    print(f"    direction: {n_pos} variables LLM-higher, {n_neg} LLM-lower, {n_zero} exactly 0")
    print(f"    mean of per-variable signed deviations = {bias['mean_llm_minus_human'].mean():+.4f}")
    print(f"    mean of per-variable |signed deviation| = {bias['mean_llm_minus_human'].abs().mean():.4f}")

    # (d)
    per_skill = per_skill_divergence(df)
    fig_per_skill_divergence(per_skill)
    print("\n[d] tables/09_per_skill_divergence.csv + figures/09_per_skill_divergence.{png,svg}")
    print(per_skill.to_string(index=False))
    print(f"    total_disagreement: min={per_skill['total_disagreement'].min()}, "
          f"max={per_skill['total_disagreement'].max()}, "
          f"mean={per_skill['total_disagreement'].mean():.2f}, "
          f"median={per_skill['total_disagreement'].median():.1f}")

    # (e)
    by_cat = by_category_divergence(per_skill)
    print("\n[e] tables/09_by_category_divergence.csv (n=2 per category — coarse, directional only)")
    print(by_cat.to_string(index=False))

    # (f)
    avc = anchored_vs_contested(summary)
    print("\n[f] tables/09_anchored_vs_contested.csv")
    print(avc.to_string(index=False))
    for b in ["well-anchored", "middle", "contested", "undefined"]:
        vs = avc.loc[avc.bucket == b, "variable"].tolist()
        if vs:
            print(f"    {b:<13}: {vs}")

    # (g)
    g = fig_llm_vs_human_human(summary)
    print("\n[g] figures/09_llm_vs_human_human.{png,svg} (H1-H2 MAD vs mean human-LLM MAD)")
    n_higher = int((g["human_llm_mad"] > g["mad_h1_h2"]).sum())
    n_lower = int((g["human_llm_mad"] < g["mad_h1_h2"]).sum())
    n_eq = int(np.isclose(g["human_llm_mad"], g["mad_h1_h2"]).sum())
    print(f"    human-LLM MAD vs H1-H2 MAD: higher in {n_higher} vars, lower in {n_lower}, ~equal in {n_eq}")
    print(f"    mean H1-H2 MAD = {g['mad_h1_h2'].mean():.4f}; "
          f"mean human-LLM MAD = {g['human_llm_mad'].mean():.4f}; "
          f"mean gap (human-LLM − H1-H2) = {(g['human_llm_mad'] - g['mad_h1_h2']).mean():+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
