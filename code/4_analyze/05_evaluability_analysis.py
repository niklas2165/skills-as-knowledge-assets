#!/usr/bin/env python3
"""
05_evaluability_analysis.py — Section 5 of the findings chapter:
evaluability across the 120-skill coded corpus.

The coding frame measures evaluability via four components, each scored 0-4:
    output_verifiability — in-principle checkability of the skill's outputs,
                           REGARDLESS of whether tests are actually present
    tests_quality        — presence and quality of tests / test-like validation
                           in the skill artefact
    thresholds_quality   — presence and quality of pass/fail thresholds or
                           acceptance bounds
    error_handling       — clarity on failure modes, recovery, escalation

Plus one DESCRIPTIVE categorical that is NOT in the formula:
    evaluation_type      — objective / hybrid / holistic / none_visible / unclear

The derived evaluability_score is computed in Python (round half up):
    (output_verifiability + tests_quality
     + thresholds_quality + error_handling) / 4

All four components push the score in the SAME direction (higher = more
evaluable), so — unlike Section 4 — no inversion handling is needed anywhere.

The central empirical finding of this section is that tests_quality is at or
near zero for the overwhelming majority of the corpus, which structurally
compresses the score. Methodology section 3.3.10 documents this as a property
of the public ecosystem rather than a flaw in the design; this analysis reports
it as the section's headline observation, not a problem to engineer around.

Self-contained: reads ONLY from analysis/data/, imports the locked visual style
from _style.py, writes figures to analysis/figures/ (.png + .svg) and tables to
analysis/tables/ (.csv). Descriptive only.

Run from anywhere:
    python3 scripts/05_evaluability_analysis.py
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as st  # noqa: E402

DATA_DIR = st.DATA_DIR
TBL_DIR = st.TBL_DIR

# The four evaluability components, fixed canonical order used everywhere.
COMPONENTS = [
    "output_verifiability",
    "tests_quality",
    "thresholds_quality",
    "error_handling",
]
# Score + components, the five numeric variables this section reports on.
ALL_VARS = ["evaluability_score"] + COMPONENTS
INTENSITIES = [0, 1, 2, 3, 4]

# Display labels.
VAR_LABELS = {
    "evaluability_score":  "Evaluability score",
    "output_verifiability": "Output verifiability",
    "tests_quality":        "Tests quality",
    "thresholds_quality":   "Thresholds quality",
    "error_handling":       "Error handling",
}
# Compact labels for tight figures (heatmap columns etc.).
SHORT_LABELS = {
    "output_verifiability": "output\nverifiability",
    "tests_quality":        "tests quality",
    "thresholds_quality":   "thresholds\nquality",
    "error_handling":       "error handling",
}

# The five knowledge types (for the by-dominant-type breakdown, Section 3).
KT = ["procedural", "analytical", "orchestration", "compliance", "documentation"]

SEQ_CMAP = "viridis"        # same sequential map as Sections 3-4 heatmaps
DIV_CMAP = "RdBu"           # same diverging map as Section 4 fig (g)


# ==========================================================================
# Helpers
# ==========================================================================
def round_half_up(x):
    """Standard round-half-up (math.floor(x + 0.5)), matching the pipeline."""
    return int(math.floor(x + 0.5))


def load_corpus():
    df = pd.read_parquet(DATA_DIR / "corpus.parquet")
    for c in ALL_VARS:
        if c not in df.columns:
            raise RuntimeError(f"missing column: {c}")
        if df[c].isna().any():
            raise RuntimeError(f"null values in {c}")
        if not df[c].between(0, 4).all():
            raise RuntimeError(f"{c} has values outside 0-4")
    if "evaluation_type" not in df.columns:
        raise RuntimeError("missing column: evaluation_type")
    # evaluation_type must be one of the five allowed values (or empty).
    bad = set(df["evaluation_type"].dropna().unique()) - set(st.EVALUATION_TYPE_ORDER)
    if bad:
        raise RuntimeError(f"ANOMALY: unexpected evaluation_type value(s): {bad}")

    # ANOMALY GUARD: every evaluability_score must equal the round-half-up
    # formula applied to the four components. Stop loudly if not.
    calc = (
        (df["output_verifiability"] + df["tests_quality"]
         + df["thresholds_quality"] + df["error_handling"]) / 4.0
    ).apply(round_half_up)
    mism = df.loc[calc != df["evaluability_score"]]
    if len(mism):
        ids = ", ".join(mism["skill_id"].tolist())
        raise RuntimeError(
            f"ANOMALY: {len(mism)} skill(s) have evaluability_score that does "
            f"not match the round-half-up formula applied to components: {ids}"
        )
    return df


# ==========================================================================
# (a) Evaluability summary table
# ==========================================================================
def summary_table(df):
    n = len(df)
    rows = []
    for v in ALL_VARS:
        s = df[v]
        vc = s.value_counts().reindex(INTENSITIES, fill_value=0)
        rows.append({
            "variable": v,
            "mean": round(float(s.mean()), 3),
            "median": float(s.median()),
            "std": round(float(s.std()), 3),
            "n_at_0": int(vc[0]),
            "n_at_1": int(vc[1]),
            "n_at_2": int(vc[2]),
            "n_at_3": int(vc[3]),
            "n_at_4": int(vc[4]),
            "n_at_3_or_higher": int((s >= 3).sum()),
            "share_at_zero": round(float((s == 0).mean()), 4),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TBL_DIR / "05_evaluability_summary.csv", index=False)
    return tbl


# ==========================================================================
# (b) Evaluation_type distribution + category cross-tab + stacked-bar figure
# ==========================================================================
def evaluation_type_distribution(df):
    order = st.EVALUATION_TYPE_ORDER
    n = len(df)

    # ---- corpus-level distribution (all five allowed values, unclear may = 0) ----
    counts = df["evaluation_type"].value_counts().reindex(order, fill_value=0)
    dist = pd.DataFrame({
        "evaluation_type": order,
        "count": [int(counts[t]) for t in order],
        "share": [round(float(counts[t] / n), 4) for t in order],
    })
    dist.to_csv(TBL_DIR / "05_evaluation_type_distribution.csv", index=False)

    # ---- 12 x 5 category x evaluation_type cross-tab (counts) ----
    cm = st.CATEGORY_MAPPING
    cats = cm["category"].tolist()
    ctab = (pd.crosstab(df["category"], df["evaluation_type"])
            .reindex(index=cats, columns=order, fill_value=0))
    ctab.index.name = "category"
    ctab.to_csv(TBL_DIR / "05_evaluation_type_by_category.csv")

    # ---- stacked horizontal-bar figure (shares per category) ----
    # Order categories by the "formal-evaluation" share (objective + hybrid),
    # descending, so the gradient from most-formal to least reads top-to-bottom.
    share_mat = ctab.div(ctab.sum(axis=1), axis=0)            # rows sum to 1
    formal = share_mat["objective"] + share_mat["hybrid"]
    cat_order = formal.sort_values(ascending=False).index.tolist()
    share_mat = share_mat.reindex(cat_order)
    labels = [st.CATEGORY_LABELS[c] for c in cat_order]

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ypos = np.arange(len(cat_order))[::-1]                    # first row on top
    left = np.zeros(len(cat_order))
    for t in order:
        widths = share_mat[t].values
        ax.barh(ypos, widths, left=left, height=0.74,
                color=st.EVALUATION_TYPE_COLORS[t], edgecolor="white",
                linewidth=0.6, label=st.EVALUATION_TYPE_LABELS[t])
        # Annotate the underlying count inside segments wide enough to hold text.
        for y, w, l, c in zip(ypos, widths, left, cat_order):
            if w >= 0.07:
                cnt = int(ctab.loc[c, t])
                ax.text(l + w / 2, y, str(cnt), ha="center", va="center",
                        fontsize=8,
                        color=st.text_color_for_bg(
                            mcolors.to_rgba(st.EVALUATION_TYPE_COLORS[t])))
        left = left + widths

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of category's skills (n = 10 per category; segment label = count)")
    ax.set_title("Evaluation type by category\n"
                 "(ordered by objective + hybrid share; descending)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=5, frameon=False, fontsize=9)
    fig.tight_layout()
    st.save_fig(fig, "05_evaluation_type_by_category")
    plt.close(fig)
    return dist, ctab


# ==========================================================================
# (c) Evaluability score distribution
# ==========================================================================
def fig_score_distribution(df):
    s = df["evaluability_score"]
    vc = s.value_counts().reindex(INTENSITIES, fill_value=0)
    mean, median = s.mean(), s.median()

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ytop = int(np.ceil((vc.max() + 12) / 10.0) * 10)
    ax.bar(INTENSITIES, vc.values, color=st.NEUTRAL_ACCENT,
           edgecolor="white", linewidth=0.6, width=0.78)
    for x, val in zip(INTENSITIES, vc.values):
        if val > 0:
            ax.text(x, val + ytop * 0.012, str(int(val)), ha="center",
                    va="bottom", fontsize=9, color="#333333")

    # Mean (amber dashed) / median (red solid), matching Section 4 fig (b).
    ax.axvline(mean, color="#E8A33D", linestyle="--", linewidth=1.8, zorder=5)
    ax.axvline(median, color="#D1495B", linestyle="-", linewidth=1.8, zorder=5)
    ax.text(0.97, 0.95, f"mean = {mean:.2f}", transform=ax.transAxes,
            color="#B5781F", ha="right", va="top", fontsize=10, fontweight="bold")
    ax.text(0.97, 0.87, f"median = {median:.1f}", transform=ax.transAxes,
            color="#D1495B", ha="right", va="top", fontsize=10, fontweight="bold")

    ax.set_xticks(INTENSITIES)
    ax.set_xlabel("Evaluability score (0–4, round-half-up of the formula)")
    ax.set_ylabel("Number of skills")
    ax.set_ylim(0, ytop)
    ax.set_title("Evaluability score distribution across the 120-skill corpus")
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    st.save_fig(fig, "05_evaluability_distribution")
    plt.close(fig)


# ==========================================================================
# (d) Component distributions — small multiples (shared y-axis)
# ==========================================================================
def fig_component_distributions(df):
    n = len(df)
    counts = {c: df[c].value_counts().reindex(INTENSITIES, fill_value=0)
              for c in COMPONENTS}
    ymax = max(int(v.max()) for v in counts.values())   # driven by tests_quality=0
    # IMPORTANT: do NOT truncate the axis — the tall tests_quality=0 bar (105
    # skills) is the central finding and must dwarf everything, by design.
    ytop = int(np.ceil((ymax + 12) / 10.0) * 10)

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.9), sharey=True)
    for ax, c in zip(axes, COMPONENTS):
        vc = counts[c]
        ax.bar(INTENSITIES, vc.values, color=st.NEUTRAL_ACCENT,
               edgecolor="white", linewidth=0.6, width=0.78)
        for x, val in zip(INTENSITIES, vc.values):
            if val > 0:
                ax.text(x, val + ytop * 0.012, str(int(val)), ha="center",
                        va="bottom", fontsize=8, color="#333333")
        share0 = (df[c] == 0).mean()
        ax.set_title(f"{VAR_LABELS[c]}\n(0 in {share0*100:.1f}% of skills)",
                     fontsize=10)
        ax.set_xticks(INTENSITIES)
        ax.set_xlabel("Intensity (0–4)")
        ax.set_ylim(0, ytop)
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Number of skills")
    fig.suptitle("Evaluability component distributions across the 120-skill corpus\n"
                 "(all four push the score the same way: higher = more evaluable)",
                 y=1.07)
    fig.tight_layout()
    st.save_fig(fig, "05_evaluability_components")
    plt.close(fig)


# ==========================================================================
# (e) Evaluability by category — two panels (ranked means + component heatmap)
# ==========================================================================
def fig_by_category(df):
    cm = st.CATEGORY_MAPPING
    rows = []
    for cat in cm["category"]:
        sub = df.loc[df["category"] == cat]
        rec = {"category": cat,
               "display_label": st.CATEGORY_LABELS[cat],
               "n": len(sub),
               "evaluability_score_mean": round(float(sub["evaluability_score"].mean()), 4),
               "evaluability_score_std": round(float(sub["evaluability_score"].std()), 4)}
        for c in COMPONENTS:
            rec[f"{c}_mean"] = round(float(sub[c].mean()), 4)
        rows.append(rec)
    bycat = pd.DataFrame(rows).sort_values(
        "evaluability_score_mean", ascending=False).reset_index(drop=True)
    bycat.to_csv(TBL_DIR / "05_evaluability_by_category.csv", index=False)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.5, 6.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- Left: ranked mean evaluability_score with std error bars ----
    order = bycat["category"].tolist()
    labels = bycat["display_label"].tolist()
    means = bycat["evaluability_score_mean"].values
    stds = bycat["evaluability_score_std"].values
    ypos = np.arange(len(order))[::-1]              # highest mean on top
    colors = [st.CATEGORY_COLORS[c] for c in order]
    axL.barh(ypos, means, xerr=stds, color=colors, edgecolor="white",
             linewidth=0.6, height=0.72,
             error_kw=dict(ecolor="#555555", elinewidth=1.1, capsize=3))
    for y, m in zip(ypos, means):
        axL.text(m + 0.04, y, f"{m:.1f}", va="center", ha="left",
                 fontsize=8.5, color="#333333")
    axL.set_yticks(ypos)
    axL.set_yticklabels(labels)
    axL.set_xlabel("Mean evaluability score (0–4); bars = ±1 SD")
    axL.set_xlim(0, 4.0)
    axL.set_title("Mean evaluability score by category\n(ranked; n = 10 per category)")
    axL.grid(axis="x", visible=True)
    axL.grid(axis="y", visible=False)

    # ---- Right: 12 x 4 component heatmap (raw means; no inversion needed) ----
    disp = np.zeros((len(order), len(COMPONENTS)))
    col_labels = []
    for j, c in enumerate(COMPONENTS):
        disp[:, j] = bycat[f"{c}_mean"].values
        col_labels.append(SHORT_LABELS[c])

    cmap = colormaps[SEQ_CMAP]
    norm = mcolors.Normalize(vmin=0, vmax=4)
    im = axR.imshow(disp, cmap=cmap, norm=norm, aspect="auto")
    axR.set_xticks(range(len(COMPONENTS)))
    axR.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=8.5)
    axR.set_yticks(range(len(order)))
    axR.set_yticklabels(labels)
    axR.set_title("Mean component intensity by category\n"
                  "(viridis 0–4; brighter = more evaluable, all four columns)")
    axR.grid(False)
    axR.set_xticks(np.arange(-0.5, len(COMPONENTS), 1), minor=True)
    axR.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    axR.grid(which="minor", color="white", linewidth=1.2)
    axR.tick_params(which="minor", length=0)
    for i in range(len(order)):
        for j in range(len(COMPONENTS)):
            val = disp[i, j]
            axR.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8.5,
                     color=st.text_color_for_bg(cmap(norm(val))))
    cbar = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.03)
    cbar.set_label("Mean component intensity (0–4)")
    cbar.set_ticks([0, 1, 2, 3, 4])

    st.save_fig(fig, "05_evaluability_by_category")
    plt.close(fig)
    return bycat


# ==========================================================================
# (f) The tests_quality central finding
# ==========================================================================
def tests_quality_finding(df):
    n = len(df)
    tq = df["tests_quality"]
    share0 = float((tq == 0).mean())

    # ---- by-category count of skills at each tests_quality level (0-4) ----
    cm = st.CATEGORY_MAPPING
    cats = cm["category"].tolist()
    by_cat = (pd.crosstab(df["category"], df["tests_quality"])
              .reindex(index=cats, columns=INTENSITIES, fill_value=0))
    by_cat.columns = [f"tests_quality_{i}" for i in INTENSITIES]
    by_cat.insert(0, "n_with_tests_gt0",
                  [int((df.loc[df["category"] == c, "tests_quality"] > 0).sum())
                   for c in cats])
    by_cat.index.name = "category"
    by_cat.to_csv(TBL_DIR / "05_tests_quality_summary.csv")

    # ---- the handful of skills with tests_quality > 0 ----
    pos = df.loc[df["tests_quality"] > 0].copy()
    pos = pos.sort_values(["category", "skill_id"])

    def _excerpt(row):
        # Prefer the evaluability_rationale (references specific evidence);
        # fall back to the skill summary. Collapse whitespace, truncate.
        txt = str(row.get("evaluability_rationale", "") or "").strip()
        if not txt:
            txt = str(row.get("summary", "") or "").strip()
        txt = " ".join(txt.split())
        return (txt[:237] + "…") if len(txt) > 238 else txt

    pos_tbl = pd.DataFrame({
        "skill_id": pos["skill_id"].values,
        "category": pos["category"].values,
        "tests_quality": pos["tests_quality"].astype(int).values,
        "evaluation_type": pos["evaluation_type"].values,
        "evaluability_score": pos["evaluability_score"].astype(int).values,
        "summary": [_excerpt(r) for _, r in pos.iterrows()],
    })
    pos_tbl.to_csv(TBL_DIR / "05_skills_with_tests.csv", index=False)

    cats_with_tests = (pos.groupby("category")["tests_quality"].count()
                       .sort_values(ascending=False))

    # ---- small figure: cumulative share at tests_quality >= k ----
    thresholds = [(0, "= 0"), (1, "≥ 1"), (2, "≥ 2"), (3, "≥ 3")]
    bar_counts = []
    for k, _ in thresholds:
        if k == 0:
            bar_counts.append(int((tq == 0).sum()))
        else:
            bar_counts.append(int((tq >= k).sum()))
    bar_labels = [lab for _, lab in thresholds]
    # colours: the "= 0" bar in a muted red (the finding), the rest in the accent.
    bar_colors = ["#D1495B"] + [st.NEUTRAL_ACCENT] * (len(thresholds) - 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    xpos = np.arange(len(thresholds))
    ax.bar(xpos, bar_counts, color=bar_colors, edgecolor="white",
           linewidth=0.6, width=0.7)
    ytop = int(np.ceil((max(bar_counts) + 14) / 10.0) * 10)
    for x, v in zip(xpos, bar_counts):
        ax.text(x, v + ytop * 0.012, f"{v}\n({v/n*100:.1f}%)", ha="center",
                va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"tests_quality {lab}" for lab in bar_labels])
    ax.set_ylim(0, ytop)
    ax.set_ylabel("Number of skills (of 120)")
    ax.set_title("Tests quality across the corpus\n"
                 f"(no tests of any kind in {bar_counts[0]} of {n} skills "
                 f"= {share0*100:.1f}%)")
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    st.save_fig(fig, "05_tests_quality_finding")
    plt.close(fig)

    return {
        "share0": share0,
        "n_zero": int((tq == 0).sum()),
        "n_pos": int((tq > 0).sum()),
        "by_cat": by_cat,
        "pos_tbl": pos_tbl,
        "cats_with_tests": cats_with_tests,
        "bar_counts": bar_counts,
        "bar_labels": bar_labels,
    }


# ==========================================================================
# (g) Evaluability by dominant knowledge type
# ==========================================================================
def by_dominant_type(df):
    """Partition the 120 skills by dominant (==4) knowledge type, with the same
    single/none/two split as Sections 3-4 so the n's reconcile to 120."""
    ndom = sum((df[k] == 4).astype(int) for k in KT)
    rows = []
    for k in KT:
        mask = (df[k] == 4) & (ndom == 1)
        if mask.sum() > 0:
            rows.append({
                "dominant_group": f"{k} (sole dominant)",
                "n": int(mask.sum()),
                "evaluability_score_mean": round(float(df.loc[mask, "evaluability_score"].mean()), 3),
                "evaluability_score_std": round(float(df.loc[mask, "evaluability_score"].std(ddof=0)), 3),
            })
    for label, mask in [("no dominant type", ndom == 0),
                        ("two dominant types", ndom == 2)]:
        rows.append({
            "dominant_group": label,
            "n": int(mask.sum()),
            "evaluability_score_mean": round(float(df.loc[mask, "evaluability_score"].mean()), 3),
            "evaluability_score_std": round(float(df.loc[mask, "evaluability_score"].std(ddof=0)), 3),
        })
    tbl = pd.DataFrame(rows).sort_values(
        "evaluability_score_mean", ascending=False).reset_index(drop=True)
    tbl.to_csv(TBL_DIR / "05_evaluability_by_dominant_type.csv", index=False)

    spread = tbl["evaluability_score_mean"].max() - tbl["evaluability_score_mean"].min()
    made_fig = spread > 0.2
    if made_fig:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        ypos = np.arange(len(tbl))[::-1]
        ax.barh(ypos, tbl["evaluability_score_mean"].values,
                color=st.NEUTRAL_ACCENT, edgecolor="white", linewidth=0.6, height=0.7)
        for y, (_, r) in zip(ypos, tbl.iterrows()):
            ax.text(r["evaluability_score_mean"] + 0.03, y,
                    f"{r['evaluability_score_mean']:.2f}  (n={r['n']})",
                    va="center", ha="left", fontsize=8.5, color="#333333")
        ax.set_yticks(ypos)
        ax.set_yticklabels(tbl["dominant_group"].tolist())
        ax.set_xlabel("Mean evaluability score (0–4)")
        ax.set_xlim(0, 4.0)
        ax.set_title("Mean evaluability score by dominant knowledge type\n"
                     "(sole-dominant per type, plus no-/two-dominant; groups partition the 120)")
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        st.save_fig(fig, "05_evaluability_by_dominant_type")
        plt.close(fig)
    return tbl, spread, made_fig


# ==========================================================================
# (h) Component correlation matrix (5x5: score + 4 components)
# ==========================================================================
def fig_correlations(df):
    corr = df[ALL_VARS].corr(method="pearson")
    corr.round(4).to_csv(TBL_DIR / "05_evaluability_component_correlations.csv")

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    cmap = colormaps[DIV_CMAP]
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(ALL_VARS)))
    ax.set_xticklabels([VAR_LABELS[v] for v in ALL_VARS], rotation=30, ha="right",
                       fontsize=8.5)
    ax.set_yticks(range(len(ALL_VARS)))
    ax.set_yticklabels([VAR_LABELS[v] for v in ALL_VARS], fontsize=8.5)
    ax.set_title("Evaluability score & component correlations\n"
                 "(Pearson r across the 120 skills)")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(ALL_VARS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ALL_VARS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(ALL_VARS)):
        for j in range(len(ALL_VARS)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color=st.text_color_for_bg(cmap(norm(val))))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    fig.tight_layout()
    st.save_fig(fig, "05_evaluability_component_correlations")
    plt.close(fig)
    return corr


# ==========================================================================
# (i) Codifiability <-> evaluability preview (single statistic + 5x5 shares)
# ==========================================================================
def codif_eval_preview(df):
    r = float(df["codifiability_score"].corr(df["evaluability_score"]))
    # 5x5 cross-tab of SHARES (fraction of the 120), rows = codifiability,
    # cols = evaluability. Preview only — Section 8 builds the full map.
    counts = (pd.crosstab(df["codifiability_score"], df["evaluability_score"])
              .reindex(index=INTENSITIES, columns=INTENSITIES, fill_value=0))
    shares = (counts / len(df)).round(4)
    shares.index.name = "codifiability_score"
    shares.columns.name = "evaluability_score"
    shares.to_csv(TBL_DIR / "05_codif_x_eval_preview.csv")
    return r, counts, shares


# ==========================================================================
# main
# ==========================================================================
def main():
    st.apply_style()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus()

    print("=" * 72)
    print("SECTION 5 — EVALUABILITY")
    print("=" * 72)
    print("  formula-consistency guard: PASS (all 120 scores match round-half-up)")

    summ = summary_table(df)
    print("\n[a] Evaluability summary -> tables/05_evaluability_summary.csv")
    print("    (share_at_zero is the headline column for tests_quality)")
    print(summ.to_string(index=False))

    dist, ctab = evaluation_type_distribution(df)
    print("\n[b] figures/05_evaluation_type_by_category.{png,svg}")
    print("    + tables/05_evaluation_type_distribution.csv")
    print("    + tables/05_evaluation_type_by_category.csv")
    print(dist.to_string(index=False))
    print("    category x evaluation_type (counts):")
    print(ctab.to_string())

    fig_score_distribution(df)
    print("\n[c] figures/05_evaluability_distribution.{png,svg}")
    sc = df["evaluability_score"]
    print(f"    score mean={sc.mean():.3f}, median={sc.median():.1f}, "
          f"std={sc.std():.3f}; "
          + ", ".join(f"{i}:{int((sc==i).sum())}" for i in INTENSITIES))

    fig_component_distributions(df)
    print("\n[d] figures/05_evaluability_components.{png,svg}")

    bycat = fig_by_category(df)
    print("\n[e] figures/05_evaluability_by_category.{png,svg} "
          "+ tables/05_evaluability_by_category.csv")
    print(bycat[["display_label", "evaluability_score_mean",
                 "evaluability_score_std"] +
                [f"{c}_mean" for c in COMPONENTS]].to_string(index=False))

    tq = tests_quality_finding(df)
    print("\n[f] TESTS_QUALITY CENTRAL FINDING")
    print("    figures/05_tests_quality_finding.{png,svg}")
    print("    + tables/05_tests_quality_summary.csv "
          "+ tables/05_skills_with_tests.csv")
    print(f"    SHARE AT ZERO: {tq['n_zero']}/120 = {tq['share0']*100:.2f}%  "
          f"(methodology cites ~88%)")
    print(f"    tests_quality > 0: {tq['n_pos']} skills across "
          f"{tq['cats_with_tests'].shape[0]} categories")
    print("    categories with any tests_quality>0 (count of such skills):")
    for c, k in tq["cats_with_tests"].items():
        print(f"      {c:<26} {int(k)}")
    print("    skills with tests_quality > 0:")
    print(tq["pos_tbl"][["skill_id", "category", "tests_quality",
                         "evaluation_type", "evaluability_score"]].to_string(index=False))
    print("    by-category tests_quality counts (0-4):")
    print(tq["by_cat"].to_string())

    domtbl, spread, made_fig = by_dominant_type(df)
    print("\n[g] tables/05_evaluability_by_dominant_type.csv")
    print(f"    spread of group means = {spread:.3f} "
          f"({'>0.2 -> figure produced' if made_fig else '<=0.2 -> figure skipped'})")
    print(domtbl.to_string(index=False))

    corr = fig_correlations(df)
    print("\n[h] figures/05_evaluability_component_correlations.{png,svg} "
          "+ tables/05_evaluability_component_correlations.csv")
    print(corr.round(3).to_string())

    r, counts, shares = codif_eval_preview(df)
    print("\n[i] CODIFIABILITY <-> EVALUABILITY PREVIEW (Section 8 builds the map)")
    print("    + tables/05_codif_x_eval_preview.csv (5x5 shares)")
    print(f"    Pearson r(codifiability_score, evaluability_score) = {r:.4f}")
    print("    5x5 counts (rows=codifiability, cols=evaluability):")
    print(counts.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
