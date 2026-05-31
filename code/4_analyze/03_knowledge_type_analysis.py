#!/usr/bin/env python3
"""
03_knowledge_type_analysis.py — Section 3 of the findings chapter:
knowledge-type distributions across the 120-skill coded corpus.

The coding frame measures five knowledge types as intensity scores 0-4,
coded independently (a skill can score 4 on procedural AND 3 on documentation):
    procedural, analytical, orchestration, compliance, documentation
This section characterises the corpus along that dimension.

Self-contained: reads ONLY from analysis/data/, imports visual style and the
knowledge-type palette from _style.py, writes figures to analysis/figures/
(.png + .svg) and tables to analysis/tables/ (.csv). Descriptive only.

Run from anywhere:
    python3 scripts/03_knowledge_type_analysis.py
"""

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

# The five knowledge types, in a fixed canonical order used everywhere.
KT = ["procedural", "analytical", "orchestration", "compliance", "documentation"]
KT_LABELS = st.KNOWLEDGE_TYPE_LABELS
INTENSITIES = [0, 1, 2, 3, 4]

# Sequential colormap for the category x knowledge-type heatmap (low = dark,
# high = light). One colormap used throughout this section's sequential views.
SEQ_CMAP = "viridis"


# ==========================================================================
# Helpers
# ==========================================================================
def load_corpus():
    df = pd.read_parquet(DATA_DIR / "corpus.parquet")
    # Guard: knowledge-type columns must be present, integer-valued, in 0-4,
    # non-null, and not identically zero. Stop loudly if not.
    for c in KT:
        if c not in df.columns:
            raise RuntimeError(f"missing knowledge-type column: {c}")
        if df[c].isna().any():
            raise RuntimeError(f"null values in {c}")
        if not df[c].between(0, 4).all():
            raise RuntimeError(f"{c} has values outside 0-4")
        if (df[c] == 0).all():
            raise RuntimeError(f"{c} is identically zero across all 120 skills")
    return df


def text_color_for(rgba):
    """Black on light cells, white on dark cells, by WCAG relative luminance."""
    r, g, b = rgba[0], rgba[1], rgba[2]

    def lin(ch):
        return ch / 12.92 if ch <= 0.03928 else ((ch + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "black" if lum > 0.4 else "white"


# ==========================================================================
# (a) Knowledge-type summary table
# ==========================================================================
def summary_table(df):
    rows = []
    for c in KT:
        s = df[c]
        vc = s.value_counts().reindex(INTENSITIES, fill_value=0)
        rows.append({
            "knowledge_type": c,
            "mean": round(float(s.mean()), 3),
            "median": float(s.median()),
            "std": round(float(s.std()), 3),
            "n_at_0": int(vc[0]),
            "n_at_1": int(vc[1]),
            "n_at_2": int(vc[2]),
            "n_at_3": int(vc[3]),
            "n_at_4": int(vc[4]),
            "n_at_3_or_higher": int((s >= 3).sum()),
            "n_dominant_at_4": int((s == 4).sum()),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TBL_DIR / "03_knowledge_type_summary.csv", index=False)
    return tbl


# ==========================================================================
# (b) Distribution histogram per knowledge type — small multiples
# ==========================================================================
def fig_distributions(df, summ):
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4), sharey=True)
    ymax = 0
    counts_by_kt = {}
    for c in KT:
        vc = df[c].value_counts().reindex(INTENSITIES, fill_value=0)
        counts_by_kt[c] = vc
        ymax = max(ymax, int(vc.max()))
    ytop = int(np.ceil((ymax + 4) / 10.0) * 10)  # headroom for bar labels

    for ax, c in zip(axes, KT):
        vc = counts_by_kt[c]
        ax.bar(INTENSITIES, vc.values, color=st.NEUTRAL_ACCENT,
               edgecolor="white", linewidth=0.6, width=0.78)
        for x, v in zip(INTENSITIES, vc.values):
            if v > 0:
                ax.text(x, v + ytop * 0.012, str(int(v)), ha="center",
                        va="bottom", fontsize=8, color="#333333")
        row = summ.loc[summ["knowledge_type"] == c].iloc[0]
        ax.set_title(f"{KT_LABELS[c]}\n(mean={row['mean']:.1f}, "
                     f"n≥3={int(row['n_at_3_or_higher'])})", fontsize=10)
        ax.set_xticks(INTENSITIES)
        ax.set_xlabel("Intensity (0–4)")
        ax.set_ylim(0, ytop)
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Number of skills")
    fig.suptitle("Knowledge-type intensity distributions across the 120-skill corpus",
                 y=1.04)
    fig.tight_layout()
    st.save_fig(fig, "03_knowledge_type_distributions")
    plt.close(fig)


# ==========================================================================
# (c) Category x knowledge-type heatmap — THE figure for this section
# ==========================================================================
def fig_category_heatmap(df):
    cm_map = st.CATEGORY_MAPPING                       # CSV order = display order
    cats = list(cm_map["category"])
    labels = list(cm_map["display_label"])

    # Mean intensity matrix: rows = categories, cols = knowledge types.
    mat = np.zeros((len(cats), len(KT)))
    for i, cat in enumerate(cats):
        sub = df.loc[df["category"] == cat]
        for j, c in enumerate(KT):
            mat[i, j] = sub[c].mean()

    # Persist the underlying matrix so the figure can be regenerated.
    mat_df = pd.DataFrame(mat, index=cats, columns=KT)
    mat_df.index.name = "category"
    mat_df.round(4).to_csv(TBL_DIR / "03_knowledge_type_by_category.csv")

    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    cmap = colormaps[SEQ_CMAP]
    norm = mcolors.Normalize(vmin=0, vmax=4)           # full 0-4 scale
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(KT)))
    ax.set_xticklabels([KT_LABELS[c] for c in KT], rotation=30, ha="right")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(labels)
    ax.set_title("Mean knowledge-type intensity by category\n"
                 "(0–4 scale, mean over the 10 skills per category)")
    ax.grid(False)
    # Light cell separators.
    ax.set_xticks(np.arange(-0.5, len(KT), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(cats)):
        for j in range(len(KT)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    color=text_color_for(cmap(norm(val))))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean intensity (0–4)")
    cbar.set_ticks([0, 1, 2, 3, 4])

    fig.tight_layout()
    st.save_fig(fig, "03_knowledge_type_by_category")
    plt.close(fig)
    return mat_df


# ==========================================================================
# (d) Per-category knowledge profile — horizontal stacked bars
# ==========================================================================
def fig_profile_by_category(df):
    cm_map = st.CATEGORY_MAPPING.set_index("category")
    cats = list(cm_map.index)

    # Mean intensity of each knowledge type per category.
    means = {cat: {c: df.loc[df["category"] == cat, c].mean() for c in KT}
             for cat in cats}
    totals = {cat: sum(means[cat].values()) for cat in cats}

    # Order categories by total bar length (sum of the five means) descending.
    ordered = sorted(cats, key=lambda c: totals[c], reverse=True)
    labels = [cm_map.loc[c, "display_label"] for c in ordered]
    ypos = np.arange(len(ordered))[::-1]               # first (longest) on top

    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    for cat, y in zip(ordered, ypos):
        left = 0.0
        for c in KT:
            w = means[cat][c]
            ax.barh(y, w, left=left, color=st.KNOWLEDGE_TYPE_COLORS[c],
                    edgecolor="white", linewidth=0.6, height=0.72)
            left += w
        ax.text(left + 0.05, y, f"{totals[cat]:.1f}", va="center", ha="left",
                fontsize=8, color="#333333")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Sum of mean knowledge-type intensities (0–20 possible)")
    ax.set_title("Knowledge profile by category\n"
                 "(stacked mean intensity of the five knowledge types)")
    ax.set_xlim(0, max(totals.values()) * 1.10)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=st.KNOWLEDGE_TYPE_COLORS[c])
               for c in KT]
    ax.legend(handles, [KT_LABELS[c] for c in KT], loc="lower center",
              bbox_to_anchor=(0.5, -0.16), ncol=5)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.save_fig(fig, "03_knowledge_profile_by_category")
    plt.close(fig)
    return ordered, totals


# ==========================================================================
# (e) Knowledge-type correlation matrix
# ==========================================================================
def fig_correlations(df):
    corr = df[KT].corr(method="pearson")
    corr.round(4).to_csv(TBL_DIR / "03_knowledge_type_correlations.csv")

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    cmap = colormaps["RdBu"]
    norm = mcolors.Normalize(vmin=-1, vmax=1)          # diverging, centred on 0
    im = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(KT)))
    ax.set_xticklabels([KT_LABELS[c] for c in KT], rotation=30, ha="right")
    ax.set_yticks(range(len(KT)))
    ax.set_yticklabels([KT_LABELS[c] for c in KT])
    ax.set_title("Knowledge-type correlations\n"
                 "(Pearson r across the 120 skills)")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(KT), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(KT), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(KT)):
        for j in range(len(KT)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color=text_color_for(cmap(norm(val))))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    fig.tight_layout()
    st.save_fig(fig, "03_knowledge_type_correlations")
    plt.close(fig)
    return corr


# ==========================================================================
# (f) Co-presence at high intensity (>= 3)
# ==========================================================================
def copresence_table(df):
    high = {c: (df[c] >= 3) for c in KT}
    mat = np.zeros((len(KT), len(KT)), dtype=int)
    for i, ci in enumerate(KT):
        for j, cj in enumerate(KT):
            mat[i, j] = int((high[ci] & high[cj]).sum())
    cp = pd.DataFrame(mat, index=KT, columns=KT)
    cp.index.name = "knowledge_type"
    cp.to_csv(TBL_DIR / "03_knowledge_type_copresence.csv")
    return cp


# ==========================================================================
# (g) Dominant knowledge type per skill (intensity == 4)
# ==========================================================================
def dominant_table(df):
    out = df[["skill_id", "category"]].copy()
    for c in KT:
        out[f"dominant_{c}"] = (df[c] == 4).astype(int)
    out["n_dominant"] = out[[f"dominant_{c}" for c in KT]].sum(axis=1)
    out.to_csv(TBL_DIR / "03_dominant_knowledge_types.csv", index=False)

    summary = out["n_dominant"].value_counts().reindex(
        range(0, 6), fill_value=0).astype(int)
    return out, summary


# ==========================================================================
# main
# ==========================================================================
def main():
    st.apply_style()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus()

    print("=" * 70)
    print("SECTION 3 — KNOWLEDGE-TYPE DISTRIBUTIONS")
    print("=" * 70)

    summ = summary_table(df)
    print("\n[a] Knowledge-type summary -> tables/03_knowledge_type_summary.csv")
    print(summ.to_string(index=False))

    fig_distributions(df, summ)
    print("\n[b] figures/03_knowledge_type_distributions.{png,svg}")

    mat_df = fig_category_heatmap(df)
    print("[c] figures/03_knowledge_type_by_category.{png,svg} "
          "+ tables/03_knowledge_type_by_category.csv")
    print(mat_df.round(2).to_string())

    ordered, totals = fig_profile_by_category(df)
    print("\n[d] figures/03_knowledge_profile_by_category.{png,svg}")
    print("    category profile totals (sum of 5 means), descending:")
    for c in ordered:
        print(f"      {st.CATEGORY_LABELS[c]:<16s} {totals[c]:.2f}")

    corr = fig_correlations(df)
    print("\n[e] figures/03_knowledge_type_correlations.{png,svg} "
          "+ tables/03_knowledge_type_correlations.csv")
    print(corr.round(2).to_string())

    cp = copresence_table(df)
    print("\n[f] tables/03_knowledge_type_copresence.csv "
          "(count with BOTH types >= 3; diagonal = that type alone at 3+)")
    print(cp.to_string())

    dom, dom_summary = dominant_table(df)
    print("\n[g] tables/03_dominant_knowledge_types.csv")
    print("    skills by number of dominant (==4) knowledge types:")
    for k, v in dom_summary.items():
        print(f"      {k} dominant type(s): {v} skills")
    # Most common single-dominant pattern.
    single = dom.loc[dom["n_dominant"] == 1]
    if len(single):
        which = single[[f"dominant_{c}" for c in KT]].idxmax(axis=1).str.replace(
            "dominant_", "", regex=False)
        print("    among single-dominant skills, dominant type counts:")
        for c in KT:
            print(f"      {KT_LABELS[c]:<14s} {int((which == c).sum())}")

    print("\nDone.")


if __name__ == "__main__":
    main()
