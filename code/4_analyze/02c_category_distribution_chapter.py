#!/usr/bin/env python3
"""
02c_category_distribution_chapter.py — chapter-specific variant of the Section 2
category-distribution figure.

This is a single-figure touch-up for chapter inclusion; it does NOT replace the
analysis-pipeline figure. The pipeline figure (02_category_distribution.* from
02_population_analysis.py) is a two-panel chart — skill count + install volume —
over all 25 LLM-induced categories plus the "none" and "uncategorized" buckets,
and is left intact.

The chapter version differs in four ways:
  (1) single panel (the install-volume panel is dropped; that view is text-only);
  (2) percentages instead of absolute counts, with the base being the SUBSET of
      skills carrying an LLM-induced category label, n = 3,949 (NOT the full
      5,235-skill population);
  (3) the two trailing buckets ("none (no cluster)" and "uncategorized (no
      label)") are excluded — only the 25 LLM-induced categories are shown
      (they cover 24.6% of the full population and are described in chapter text);
  (4) no "Method A"/"Method B" wording — generic "LLM-induced" language only.

Unchanged from the pipeline figure: 25 categories ordered by share descending,
the 12 retained categories in their CATEGORY_COLORS hues, the 13 excluded
categories in neutral grey, the retained/excluded legend, a percentage label on
every bar, and the horizontal-bar orientation.

Reads ONLY from data/population.parquet and data/category_mapping.csv (via the
CATEGORY_COLORS / CATEGORY_MAPPING loaded in _style.py). Uses the locked style.

Run from anywhere:
    python3 scripts/02c_category_distribution_chapter.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as st  # noqa: E402

DATA_DIR = st.DATA_DIR

# Representative swatch for the "retained" legend entry (each retained category
# carries its own CATEGORY_COLORS hue; this single swatch is illustrative only —
# same convention as the pipeline figure's legend).
RETAINED_SWATCH = "#4E79A7"


def fig_category_distribution_chapter():
    pop = pd.read_parquet(DATA_DIR / "population.parquet")
    retained = set(st.CATEGORY_COLORS)
    cm = st.CATEGORY_MAPPING.set_index("category")

    raw = pop["method_b_category"]
    # The percentage base: skills with an LLM-induced category label only
    # (drops the "none" no-cluster bucket and the null no-label bucket).
    real = raw[raw.notna() & (raw != "none")]
    base = len(real)  # 3,949

    counts = real.value_counts()              # 25 categories, count descending
    ordered = list(counts.index)              # share-descending == count-descending

    def disp(cat):
        return cm.loc[cat, "display_label"] if cat in retained else cat

    def color_for(cat):
        return st.CATEGORY_COLORS[cat] if cat in retained else st.EXCLUDED_COLOR

    rows = []
    for cat in ordered:
        n = int(counts[cat])
        rows.append({
            "category": cat,
            "label": disp(cat),
            "count": n,
            "pct": n / base * 100.0,
            "color": color_for(cat),
            "kind": "retained" if cat in retained else "excluded",
        })
    rdf = pd.DataFrame(rows)
    ypos = np.arange(len(rdf))[::-1]          # first (largest) row on top

    fig, ax = plt.subplots(figsize=(8.6, 9.4))
    ax.barh(ypos, rdf["pct"], color=rdf["color"], edgecolor="white",
            linewidth=0.4, height=0.74)
    for y, p in zip(ypos, rdf["pct"]):
        ax.text(p + rdf["pct"].max() * 0.012, y, f"{p:.1f}%", va="center",
                ha="left", fontsize=7.5, color="#333333")

    ax.set_yticks(ypos)
    ax.set_yticklabels(rdf["label"], fontsize=8)
    ax.set_xlabel("Share of LLM-induced–categorised skills (%)")
    ax.set_xlim(0, rdf["pct"].max() * 1.12)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)

    # Subtitle (de-bolded) carries the denominator so the base is unambiguous.
    ax.set_title(
        f"Percentages of the {base:,} skills with an LLM-induced category label "
        f"(not the full 5,235-skill population)",
        fontsize=9.5, fontweight="normal", color="#444444",
    )

    # Legend: two roles only (no "none"/"no-label" entries — those are excluded).
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=RETAINED_SWATCH),
        plt.Rectangle((0, 0), 1, 1, color=st.EXCLUDED_COLOR),
    ]
    ax.legend(handles, ["retained (12, sampled)", "excluded (13)"],
              loc="lower right", frameon=True, framealpha=0.95)

    fig.suptitle("LLM-induced category distribution across the population", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    st.save_fig(fig, "02_category_distribution_chapter")
    plt.close(fig)
    return rdf, base


def main():
    st.apply_style()
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SECTION 2 — CHAPTER FIGURE: LLM-induced category distribution (%)")
    print("=" * 70)

    rdf, base = fig_category_distribution_chapter()
    print(f"\nfigures/02_category_distribution_chapter.{{png,svg}}")
    print(f"  base (denominator) = {base:,} skills with an LLM-induced category label")
    print(f"  categories shown   = {len(rdf)} (retained {int((rdf['kind']=='retained').sum())}, "
          f"excluded {int((rdf['kind']=='excluded').sum())})")
    print(f"  total of shown shares = {rdf['pct'].sum():.1f}%  (should be 100.0%)")
    print("\n  per-category share (descending):")
    for _, r in rdf.iterrows():
        print(f"    {r['label']:<34} {r['count']:>5,}  {r['pct']:5.1f}%  [{r['kind']}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
