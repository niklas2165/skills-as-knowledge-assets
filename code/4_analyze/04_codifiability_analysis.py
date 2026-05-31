#!/usr/bin/env python3
"""
04_codifiability_analysis.py — Section 4 of the findings chapter:
codifiability across the 120-skill coded corpus.

The coding frame measures codifiability via four components, each scored 0-4:
    explicitness          — clarity of the CORE SKILL.md instructions
    documentation_quality — quality of supporting material BEYOND the core
                            instructions (examples, templates, helper scripts)
    tacit_dependency      — reliance on unstated judgement / expertise
                            (HIGHER = LOWER codifiability)
    context_sensitivity   — dependence on a specific org / repo / toolchain /
                            legal setting (HIGHER = LOWER codifiability)

The derived codifiability_score is computed in Python (round half up):
    (explicitness + documentation_quality
     + (4 - tacit_dependency) + (4 - context_sensitivity)) / 4

This section characterises the corpus along that dimension and surfaces a
methodological finding: how often explicitness and documentation_quality
actually distinguish from each other in practice.

Self-contained: reads ONLY from analysis/data/, imports the locked visual style
from _style.py, writes figures to analysis/figures/ (.png + .svg) and tables to
analysis/tables/ (.csv). Descriptive only.

Run from anywhere:
    python3 scripts/04_codifiability_analysis.py
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

# The four codifiability components, fixed canonical order used everywhere.
COMPONENTS = [
    "explicitness",
    "documentation_quality",
    "tacit_dependency",
    "context_sensitivity",
]
# The two components that INVERT in the formula (higher = lower codifiability).
INVERTED = {"tacit_dependency", "context_sensitivity"}
# Score + components, the five variables this section reports on.
ALL_VARS = ["codifiability_score"] + COMPONENTS
INTENSITIES = [0, 1, 2, 3, 4]

# Display labels.
VAR_LABELS = {
    "codifiability_score":   "Codifiability score",
    "explicitness":          "Explicitness",
    "documentation_quality": "Documentation quality",
    "tacit_dependency":      "Tacit dependency",
    "context_sensitivity":   "Context sensitivity",
}
# Compact labels for tight figures (heatmap columns etc.).
SHORT_LABELS = {
    "explicitness":          "explicitness",
    "documentation_quality": "documentation\nquality",
    "tacit_dependency":      "tacit dependency",
    "context_sensitivity":   "context sensitivity",
}

# The five knowledge types (for the by-dominant-type breakdown, Section 3).
KT = ["procedural", "analytical", "orchestration", "compliance", "documentation"]

SEQ_CMAP = "viridis"        # same sequential map as Section 3's heatmap
DIV_CMAP = "RdBu"           # same diverging map as Section 3 fig (e)


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

    # ANOMALY GUARD: every codifiability_score must equal the round-half-up
    # formula applied to the four components. Stop loudly if not.
    calc = (
        (df["explicitness"] + df["documentation_quality"]
         + (4 - df["tacit_dependency"]) + (4 - df["context_sensitivity"])) / 4.0
    ).apply(round_half_up)
    mism = df.loc[calc != df["codifiability_score"]]
    if len(mism):
        ids = ", ".join(mism["skill_id"].tolist())
        raise RuntimeError(
            f"ANOMALY: {len(mism)} skill(s) have codifiability_score that does "
            f"not match the round-half-up formula applied to components: {ids}"
        )
    return df


# ==========================================================================
# (a) Codifiability summary table
# ==========================================================================
def summary_table(df):
    rows = []
    for v in ALL_VARS:
        s = df[v]
        vc = s.value_counts().reindex(INTENSITIES, fill_value=0)
        rows.append({
            "variable": v,
            "inverted_in_formula": v in INVERTED,
            "mean": round(float(s.mean()), 3),
            "median": float(s.median()),
            "std": round(float(s.std()), 3),
            "n_at_0": int(vc[0]),
            "n_at_1": int(vc[1]),
            "n_at_2": int(vc[2]),
            "n_at_3": int(vc[3]),
            "n_at_4": int(vc[4]),
            "n_at_3_or_higher": int((s >= 3).sum()),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TBL_DIR / "04_codifiability_summary.csv", index=False)
    return tbl


# ==========================================================================
# (b) Codifiability score distribution
# ==========================================================================
def fig_score_distribution(df):
    s = df["codifiability_score"]
    vc = s.value_counts().reindex(INTENSITIES, fill_value=0)
    mean, median = s.mean(), s.median()

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ytop = int(np.ceil((vc.max() + 8) / 10.0) * 10)
    ax.bar(INTENSITIES, vc.values, color=st.NEUTRAL_ACCENT,
           edgecolor="white", linewidth=0.6, width=0.78)
    for x, val in zip(INTENSITIES, vc.values):
        if val > 0:
            ax.text(x, val + ytop * 0.012, str(int(val)), ha="center",
                    va="bottom", fontsize=9, color="#333333")

    # Mean / median vertical lines (mean amber, median red — matching Sec 2).
    # The two lines sit close together (2.79 vs 3.0) on the tall x=3 bar, so the
    # numeric labels go in a clear corner box rather than on the lines.
    ax.axvline(mean, color="#E8A33D", linestyle="--", linewidth=1.8, zorder=5)
    ax.axvline(median, color="#D1495B", linestyle="-", linewidth=1.8, zorder=5)
    ax.text(0.03, 0.95, f"mean = {mean:.2f}", transform=ax.transAxes,
            color="#B5781F", ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.03, 0.87, f"median = {median:.1f}", transform=ax.transAxes,
            color="#D1495B", ha="left", va="top", fontsize=10, fontweight="bold")

    ax.set_xticks(INTENSITIES)
    ax.set_xlabel("Codifiability score (0–4, round-half-up of the formula)")
    ax.set_ylabel("Number of skills")
    ax.set_ylim(0, ytop)
    ax.set_title("Codifiability score distribution across the 120-skill corpus")
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    st.save_fig(fig, "04_codifiability_distribution")
    plt.close(fig)


# ==========================================================================
# (c) Component distributions — small multiples (shared y-axis)
# ==========================================================================
def fig_component_distributions(df):
    counts = {c: df[c].value_counts().reindex(INTENSITIES, fill_value=0)
              for c in COMPONENTS}
    ymax = max(int(v.max()) for v in counts.values())
    # Leave a clear band above the tallest bar for the direction arrow.
    ytop = int(np.ceil((ymax + 16) / 10.0) * 10)
    band_lo = ymax + ytop * 0.02            # just above the tallest bar
    arrow_y = band_lo + (ytop - band_lo) * 0.30
    label_y = band_lo + (ytop - band_lo) * 0.62

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.9), sharey=True)
    for ax, c in zip(axes, COMPONENTS):
        vc = counts[c]
        ax.bar(INTENSITIES, vc.values, color=st.NEUTRAL_ACCENT,
               edgecolor="white", linewidth=0.6, width=0.78)
        for x, val in zip(INTENSITIES, vc.values):
            if val > 0:
                ax.text(x, val + ytop * 0.012, str(int(val)), ha="center",
                        va="bottom", fontsize=8, color="#333333")
        inv = c in INVERTED
        title = VAR_LABELS[c] + ("\n(higher = LOWER codifiability)" if inv
                                 else "\n(higher = higher codifiability)")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(INTENSITIES)
        ax.set_xlabel("Intensity (0–4)")
        ax.set_ylim(0, ytop)
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)

        # Direction arrow in the clear top band: "more codifiable" points
        # toward 0 for inverted components, toward 4 for the direct ones.
        if inv:
            ax.annotate("", xy=(0.3, arrow_y), xytext=(3.5, arrow_y),
                        arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.6))
        else:
            ax.annotate("", xy=(3.7, arrow_y), xytext=(0.5, arrow_y),
                        arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.6))
        ax.text(2.0, label_y, "more codifiable", ha="center", va="center",
                fontsize=8.5, color="#1B5E20", fontweight="bold")

    axes[0].set_ylabel("Number of skills")
    fig.suptitle("Codifiability component distributions across the 120-skill corpus",
                 y=1.05)
    fig.tight_layout()
    st.save_fig(fig, "04_codifiability_components")
    plt.close(fig)


# ==========================================================================
# (d) Codifiability by category — two panels (ranked means + component heatmap)
# ==========================================================================
def fig_by_category(df):
    cm = st.CATEGORY_MAPPING
    # Per-category means (raw component means + score mean/std).
    rows = []
    for cat in cm["category"]:
        sub = df.loc[df["category"] == cat]
        rec = {"category": cat,
               "display_label": st.CATEGORY_LABELS[cat],
               "n": len(sub),
               "codifiability_score_mean": round(float(sub["codifiability_score"].mean()), 4),
               "codifiability_score_std": round(float(sub["codifiability_score"].std()), 4)}
        for c in COMPONENTS:
            rec[f"{c}_mean"] = round(float(sub[c].mean()), 4)
        rows.append(rec)
    bycat = pd.DataFrame(rows).sort_values(
        "codifiability_score_mean", ascending=False).reset_index(drop=True)
    bycat.to_csv(TBL_DIR / "04_codifiability_by_category.csv", index=False)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.5, 6.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- Left: ranked mean codifiability_score with std error bars ----
    order = bycat["category"].tolist()
    labels = bycat["display_label"].tolist()
    means = bycat["codifiability_score_mean"].values
    stds = bycat["codifiability_score_std"].values
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
    axL.set_xlabel("Mean codifiability score (0–4); bars = ±1 SD")
    axL.set_xlim(0, 4.0)
    axL.set_title("Mean codifiability score by category\n(ranked; n = 10 per category)")
    axL.grid(axis="x", visible=True)
    axL.grid(axis="y", visible=False)

    # ---- Right: 12 x 4 component heatmap ----
    # INVERSION DISPLAY CHOICE: the two inverted components are shown as their
    # (4 - x) value so a single viridis scale means "brighter = more codifiable"
    # uniformly across ALL four columns (otherwise a bright tacit/context cell
    # would mean the opposite of a bright explicitness cell). Direct components
    # are shown as-is. Column headers state the (4 - x) transform explicitly.
    disp = np.zeros((len(order), len(COMPONENTS)))
    col_labels = []
    for j, c in enumerate(COMPONENTS):
        raw = bycat[f"{c}_mean"].values
        if c in INVERTED:
            disp[:, j] = 4.0 - raw
            col_labels.append(f"4 − {SHORT_LABELS[c]}")
        else:
            disp[:, j] = raw
            col_labels.append(SHORT_LABELS[c])

    cmap = colormaps[SEQ_CMAP]
    norm = mcolors.Normalize(vmin=0, vmax=4)
    im = axR.imshow(disp, cmap=cmap, norm=norm, aspect="auto")
    axR.set_xticks(range(len(COMPONENTS)))
    axR.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=8.5)
    axR.set_yticks(range(len(order)))
    axR.set_yticklabels(labels)
    axR.set_title("Mean component intensity by category\n"
                  "(all columns oriented so brighter = more codifiable)")
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
    cbar.set_label("Mean intensity, codifiability-oriented (0–4)")
    cbar.set_ticks([0, 1, 2, 3, 4])

    st.save_fig(fig, "04_codifiability_by_category")
    plt.close(fig)
    return bycat


# ==========================================================================
# (e) Codifiability by dominant knowledge type
# ==========================================================================
def by_dominant_type(df):
    """Partition the 120 skills by dominant (==4) knowledge type, using the
    same single/none/two split as Section 3 so the n's reconcile to 120:
    a per-type row counts skills where that type is the SOLE dominant type;
    plus 'no dominant type' and 'two dominant types' rows."""
    ndom = sum((df[k] == 4).astype(int) for k in KT)
    rows = []
    for k in KT:
        mask = (df[k] == 4) & (ndom == 1)
        if mask.sum() > 0:
            rows.append({
                "dominant_group": f"{k} (sole dominant)",
                "n": int(mask.sum()),
                "codifiability_score_mean": round(float(df.loc[mask, "codifiability_score"].mean()), 3),
                "codifiability_score_std": round(float(df.loc[mask, "codifiability_score"].std(ddof=0)), 3),
            })
    for label, mask in [("no dominant type", ndom == 0),
                        ("two dominant types", ndom == 2)]:
        rows.append({
            "dominant_group": label,
            "n": int(mask.sum()),
            "codifiability_score_mean": round(float(df.loc[mask, "codifiability_score"].mean()), 3),
            "codifiability_score_std": round(float(df.loc[mask, "codifiability_score"].std(ddof=0)), 3),
        })
    tbl = pd.DataFrame(rows).sort_values(
        "codifiability_score_mean", ascending=False).reset_index(drop=True)
    tbl.to_csv(TBL_DIR / "04_codifiability_by_dominant_type.csv", index=False)

    # Figure: the spread among the well-populated groups exceeds ±0.2, so a
    # small horizontal-bar figure is informative. n is annotated on each bar so
    # the tiny (n=2) groups are not over-read.
    spread = tbl["codifiability_score_mean"].max() - tbl["codifiability_score_mean"].min()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ypos = np.arange(len(tbl))[::-1]
    ax.barh(ypos, tbl["codifiability_score_mean"].values,
            color=st.NEUTRAL_ACCENT, edgecolor="white", linewidth=0.6, height=0.7)
    for y, (_, r) in zip(ypos, tbl.iterrows()):
        ax.text(r["codifiability_score_mean"] + 0.03, y,
                f"{r['codifiability_score_mean']:.2f}  (n={r['n']})",
                va="center", ha="left", fontsize=8.5, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels(tbl["dominant_group"].tolist())
    ax.set_xlabel("Mean codifiability score (0–4)")
    ax.set_xlim(0, 4.0)
    ax.set_title("Mean codifiability score by dominant knowledge type\n"
                 "(sole-dominant per type, plus no-/two-dominant; groups partition the 120)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    st.save_fig(fig, "04_codifiability_by_dominant_type")
    plt.close(fig)
    return tbl, spread


# ==========================================================================
# (f) Explicitness vs documentation_quality collapse analysis
# ==========================================================================
def collapse_analysis(df):
    e, d = df["explicitness"], df["documentation_quality"]
    diff = e - d
    n = len(df)

    exact_match = int((diff == 0).sum())
    exact_rate = exact_match / n
    expl_higher = int((diff > 0).sum())
    doc_higher = int((diff < 0).sum())

    # Distribution of (explicitness - documentation_quality).
    diff_dist = diff.value_counts().reindex(range(-4, 5), fill_value=0).astype(int)
    # When they collapse, at which intensity.
    collapse_at = e[diff == 0].value_counts().reindex(INTENSITIES, fill_value=0).astype(int)

    # Joint 5x5 distribution (rows = explicitness, cols = documentation_quality).
    joint = pd.crosstab(e, d).reindex(index=INTENSITIES, columns=INTENSITIES,
                                      fill_value=0)
    joint.index.name = "explicitness"
    joint.columns.name = "documentation_quality"
    joint.to_csv(TBL_DIR / "04_explicitness_vs_doc_quality.csv")

    # ---- Figure 1: the 5x5 joint heatmap with counts overlaid ----
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cmap = colormaps[SEQ_CMAP]
    vmax = int(joint.values.max())
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(joint.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(INTENSITIES)
    ax.set_yticks(range(5)); ax.set_yticklabels(INTENSITIES)
    ax.set_xlabel("documentation_quality (0–4)")
    ax.set_ylabel("explicitness (0–4)")
    ax.set_title("Joint distribution: explicitness × documentation quality\n"
                 "(counts overlaid; diagonal = the two collapse to one score)")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(5):
        for j in range(5):
            v = int(joint.values[i, j])
            on_diag = (i == j)
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=10 if on_diag else 9,
                    fontweight="bold" if on_diag else "normal",
                    color=st.text_color_for_bg(cmap(norm(v))))
    # Outline the diagonal (collapse) cells.
    for k in range(5):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                   edgecolor="#D1495B", linewidth=2.0, zorder=6))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Number of skills")
    fig.tight_layout()
    st.save_fig(fig, "04_explicitness_vs_doc_quality")
    plt.close(fig)

    # ---- Figure 2: collapse-at-intensity bar + diff distribution bar ----
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    # left: collapse-at-intensity
    ax1.bar(INTENSITIES, collapse_at.values, color=st.NEUTRAL_ACCENT,
            edgecolor="white", linewidth=0.6, width=0.78)
    top1 = int(np.ceil((collapse_at.max() + 5) / 5.0) * 5)
    for x, v in zip(INTENSITIES, collapse_at.values):
        if v > 0:
            ax1.text(x, v + top1 * 0.015, str(int(v)), ha="center", va="bottom",
                     fontsize=9, color="#333333")
    ax1.set_xticks(INTENSITIES)
    ax1.set_ylim(0, top1)
    ax1.set_xlabel("Shared intensity")
    ax1.set_ylabel("Number of skills")
    ax1.set_title(f"Where the two collapse (diff = 0)\n"
                  f"{exact_match} of {n} skills")
    ax1.grid(axis="y", visible=True); ax1.grid(axis="x", visible=False)
    # right: signed difference distribution
    xs = list(range(-2, 3))
    vals = [int(diff_dist[k]) for k in xs]
    bar_colors = ["#D1495B" if k < 0 else ("#9E9E9E" if k == 0 else "#2C5F8A")
                  for k in xs]
    ax2.bar(xs, vals, color=bar_colors, edgecolor="white", linewidth=0.6, width=0.78)
    top2 = int(np.ceil((max(vals) + 6) / 10.0) * 10)
    for x, v in zip(xs, vals):
        if v > 0:
            ax2.text(x, v + top2 * 0.012, str(int(v)), ha="center", va="bottom",
                     fontsize=9, color="#333333")
    ax2.set_xticks(xs)
    ax2.set_ylim(0, top2)
    ax2.set_xlabel("explicitness − documentation_quality")
    ax2.set_ylabel("Number of skills")
    ax2.set_title(f"Signed difference\n(expl > doc: {expl_higher}; "
                  f"expl < doc: {doc_higher})")
    ax2.grid(axis="y", visible=True); ax2.grid(axis="x", visible=False)
    fig2.suptitle("Explicitness vs documentation quality: collapse and direction",
                  y=1.04)
    fig2.tight_layout()
    st.save_fig(fig2, "04_explicitness_vs_doc_quality_collapse")
    plt.close(fig2)

    return {
        "exact_match": exact_match, "exact_rate": exact_rate,
        "expl_higher": expl_higher, "doc_higher": doc_higher,
        "diff_dist": diff_dist, "collapse_at": collapse_at, "joint": joint,
    }


# ==========================================================================
# (g) Component correlation matrix (5x5: score + 4 components)
# ==========================================================================
def fig_correlations(df):
    corr = df[ALL_VARS].corr(method="pearson")
    corr.round(4).to_csv(TBL_DIR / "04_codifiability_component_correlations.csv")

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    cmap = colormaps[DIV_CMAP]
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect="auto")

    # Inverted variables are kept RAW here (so the sign of each correlation is
    # faithful to the actual codes); their labels are marked with ↑=less
    # codifiable so the direction stays unambiguous.
    def axlabel(v):
        base = VAR_LABELS[v]
        return base + " (↑ = less codif.)" if v in INVERTED else base

    ax.set_xticks(range(len(ALL_VARS)))
    ax.set_xticklabels([axlabel(v) for v in ALL_VARS], rotation=30, ha="right",
                       fontsize=8.5)
    ax.set_yticks(range(len(ALL_VARS)))
    ax.set_yticklabels([axlabel(v) for v in ALL_VARS], fontsize=8.5)
    ax.set_title("Codifiability score & component correlations\n"
                 "(Pearson r across the 120 skills; raw components)")
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
    st.save_fig(fig, "04_codifiability_component_correlations")
    plt.close(fig)
    return corr


# ==========================================================================
# main
# ==========================================================================
def main():
    st.apply_style()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus()

    print("=" * 72)
    print("SECTION 4 — CODIFIABILITY")
    print("=" * 72)
    print(f"  formula-consistency guard: PASS (all 120 scores match round-half-up)")

    summ = summary_table(df)
    print("\n[a] Codifiability summary -> tables/04_codifiability_summary.csv")
    print("    (tacit_dependency & context_sensitivity: HIGHER = LOWER codifiability)")
    print(summ.to_string(index=False))

    fig_score_distribution(df)
    print("\n[b] figures/04_codifiability_distribution.{png,svg}")
    sc = df["codifiability_score"]
    print(f"    score mean={sc.mean():.3f}, median={sc.median():.1f}, "
          f"std={sc.std():.3f}; "
          + ", ".join(f"{i}:{int((sc==i).sum())}" for i in INTENSITIES))

    fig_component_distributions(df)
    print("\n[c] figures/04_codifiability_components.{png,svg}")

    bycat = fig_by_category(df)
    print("\n[d] figures/04_codifiability_by_category.{png,svg} "
          "+ tables/04_codifiability_by_category.csv")
    print("    (heatmap shows inverted (4 - x) for tacit_dependency & "
          "context_sensitivity)")
    print(bycat[["display_label", "codifiability_score_mean",
                 "codifiability_score_std"] +
                [f"{c}_mean" for c in COMPONENTS]].to_string(index=False))

    domtbl, spread = by_dominant_type(df)
    print("\n[e] figures/04_codifiability_by_dominant_type.{png,svg} "
          "+ tables/04_codifiability_by_dominant_type.csv")
    print(f"    spread of group means = {spread:.3f} (>0.2 -> figure produced)")
    print(domtbl.to_string(index=False))

    cres = collapse_analysis(df)
    print("\n[f] figures/04_explicitness_vs_doc_quality.{png,svg} "
          "(+ _collapse.{png,svg})")
    print("    + tables/04_explicitness_vs_doc_quality.csv")
    print(f"    EXACT-MATCH RATE: {cres['exact_match']}/120 = "
          f"{cres['exact_rate']*100:.1f}%")
    print(f"    BIAS DIRECTION: explicitness > documentation_quality in "
          f"{cres['expl_higher']} skills; explicitness < documentation_quality "
          f"in {cres['doc_higher']} skills "
          + ("(skews toward explicitness)" if cres['expl_higher'] > cres['doc_higher']
             else "(skews toward documentation_quality)"))
    print("    signed diff (expl - doc) distribution:")
    for k in range(-2, 3):
        print(f"      {k:+d}: {int(cres['diff_dist'][k])}")
    print("    collapse-at-intensity (diff = 0):")
    for i in INTENSITIES:
        print(f"      intensity {i}: {int(cres['collapse_at'][i])}")
    print("    joint 5x5 (rows=explicitness, cols=documentation_quality):")
    print(cres["joint"].to_string())

    corr = fig_correlations(df)
    print("\n[g] figures/04_codifiability_component_correlations.{png,svg} "
          "+ tables/04_codifiability_component_correlations.csv")
    print(corr.round(3).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
