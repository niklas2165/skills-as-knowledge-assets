#!/usr/bin/env python3
"""
02_population_analysis.py — Section 2 of the findings chapter:
population-level patterns across the 5,235-skill skills.sh population.

Self-contained: reads ONLY from analysis/data/, imports visual style from
_style.py, writes figures to analysis/figures/ (.png + .svg) and tables to
analysis/tables/ (.csv). Descriptive only — no interpretation.

Run from anywhere:
    python3 scripts/02_population_analysis.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as st  # noqa: E402

DATA_DIR = st.DATA_DIR
TBL_DIR = st.TBL_DIR
AUDIT_COLS = ["snyk_status", "socket_status", "agent_trust_hub_status"]
AUDIT_LABELS = {
    "snyk_status": "Snyk",
    "socket_status": "Socket",
    "agent_trust_hub_status": "Agent Trust Hub",
}


# ==========================================================================
# Helpers
# ==========================================================================
def parse_stars(s):
    """Parse the abbreviated GitHub-stars string ('1.2K', '102.7K', '1') into a
    float. Returns NaN for non-strings (null inputs)."""
    if not isinstance(s, str):
        return np.nan
    s = s.strip().replace(",", "")
    mult = 1.0
    if s[-1:].upper() == "K":
        mult, s = 1e3, s[:-1]
    elif s[-1:].upper() == "M":
        mult, s = 1e6, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def thousands(x, _pos=None):
    """Thousand-separator tick formatter."""
    return f"{int(round(x)):,}"


def fmt_int(x):
    return f"{int(round(x)):,}"


def audit_proportions(series):
    """Map an audit-status series (pass/warn/fail/None) to counts in the fixed
    AUDIT_ORDER, treating null as 'missing'."""
    s = series.fillna("missing").str.lower()
    counts = {k: int((s == k).sum()) for k in st.AUDIT_ORDER}
    return counts


def load_population():
    pop = pd.read_parquet(DATA_DIR / "population.parquet")
    # Derived columns used throughout.
    pop["publisher"] = pop["repo_name"].str.split("/").str[0]
    pop["stars_num"] = pop["github_stars_abbreviated"].apply(parse_stars)
    pop["has_md_bool"] = pop["has_skill_md"].eq("Y")
    pop["any_audit"] = pop[AUDIT_COLS].notna().any(axis=1)
    pop["all_audit"] = pop[AUDIT_COLS].notna().all(axis=1)
    return pop


# ==========================================================================
# (a) Population summary table
# ==========================================================================
def population_summary(pop):
    inst = pop["installs"]
    stars = pop["stars_num"]
    n = len(pop)

    rows = [
        ("total_population_skills", fmt_int(n)),
        ("unique_repos", fmt_int(pop["repo_name"].nunique())),
        ("unique_publishers", fmt_int(pop["publisher"].nunique())),
        ("pct_official", f"{pop['is_official'].mean() * 100:.1f}%"),
        ("pct_with_any_audit", f"{pop['any_audit'].mean() * 100:.1f}%"),
        ("pct_with_all_three_audits", f"{pop['all_audit'].mean() * 100:.1f}%"),
        ("pct_with_skill_md", f"{pop['has_md_bool'].mean() * 100:.1f}%"),
        ("installs_min", fmt_int(inst.min())),
        ("installs_p10", fmt_int(inst.quantile(0.10))),
        ("installs_p25", fmt_int(inst.quantile(0.25))),
        ("installs_median_p50", fmt_int(inst.median())),
        ("installs_mean", fmt_int(inst.mean())),
        ("installs_p75", fmt_int(inst.quantile(0.75))),
        ("installs_p90", fmt_int(inst.quantile(0.90))),
        ("installs_p99", fmt_int(inst.quantile(0.99))),
        ("installs_max", fmt_int(inst.max())),
        ("github_stars_median", fmt_int(stars.median())),
        ("github_stars_mean", fmt_int(stars.mean())),
        ("github_stars_parsed_n", fmt_int(stars.notna().sum())),
    ]
    df = pd.DataFrame(rows, columns=["statistic", "value"])
    df.to_csv(TBL_DIR / "02_population_summary.csv", index=False)
    return df


# ==========================================================================
# (b) Install distribution (overall)
# ==========================================================================
def fig_install_distribution(pop):
    inst = pop["installs"].values
    med, mean = np.median(inst), np.mean(inst)
    p25, p75, p99 = (np.percentile(inst, q) for q in (25, 75, 99))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.logspace(np.log10(inst.min()), np.log10(inst.max()), 45)
    ax.hist(inst, bins=bins, color=st.BINARY_COLORS["positive"],
            edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")

    ax.axvline(med, color="#D1495B", linestyle="-", linewidth=1.6,
               label=f"median = {fmt_int(med)}")
    ax.axvline(mean, color="#E8A33D", linestyle="--", linewidth=1.6,
               label=f"mean = {fmt_int(mean)}")
    ax.legend(loc="upper right")

    ax.set_xlabel("Installs (log scale)")
    ax.set_ylabel("Number of skills")
    ax.set_title("Install distribution across the 5,235-skill population")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands))

    # Long-tail annotation pointing at the high end.
    ax.annotate(
        "Long tail: a handful of skills\nexceed 100,000 installs",
        xy=(0.97, 0.55), xycoords="axes fraction", ha="right", va="center",
        fontsize=8.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#CCCCCC", lw=0.6),
    )
    # Percentile text note.
    note = (f"p25 = {fmt_int(p25)}\np50 = {fmt_int(med)}\n"
            f"p75 = {fmt_int(p75)}\np99 = {fmt_int(p99)}")
    ax.text(0.025, 0.97, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, family="monospace", color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.6))

    fig.tight_layout()
    st.save_fig(fig, "02_install_distribution")
    plt.close(fig)


# ==========================================================================
# (c) Install distribution: official vs unofficial
# ==========================================================================
def fig_install_by_official(pop):
    off = pop.loc[pop["is_official"], "installs"].values
    unoff = pop.loc[~pop["is_official"], "installs"].values

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.logspace(np.log10(pop["installs"].min()),
                       np.log10(pop["installs"].max()), 40)
    ax.hist(unoff, bins=bins, color=st.BINARY_COLORS["unofficial"],
            edgecolor="white", linewidth=0.3, alpha=0.95,
            label=f"unofficial  (n = {fmt_int(len(unoff))}, "
                  f"median {fmt_int(np.median(unoff))})")
    ax.hist(off, bins=bins, color=st.BINARY_COLORS["official"],
            edgecolor="white", linewidth=0.3, alpha=0.75,
            label=f"official  (n = {fmt_int(len(off))}, "
                  f"median {fmt_int(np.median(off))})")
    ax.set_xscale("log")

    ax.axvline(np.median(unoff), color=st.BINARY_COLORS["unofficial"],
               linestyle="--", linewidth=1.4)
    ax.axvline(np.median(off), color=st.BINARY_COLORS["official"],
               linestyle="--", linewidth=1.6)

    ax.set_xlabel("Installs (log scale)")
    ax.set_ylabel("Number of skills")
    ax.set_title("Install distribution: official vs. community-published")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands))
    ax.legend(loc="upper right")

    fig.tight_layout()
    st.save_fig(fig, "02_install_by_official")
    plt.close(fig)


# ==========================================================================
# (d) Audit coverage (overall) — stacked horizontal bars, one per auditor
# ==========================================================================
def fig_audit_coverage_overall(pop):
    n = len(pop)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y_positions = list(range(len(AUDIT_COLS)))[::-1]  # first auditor on top

    for col, y in zip(AUDIT_COLS, y_positions):
        counts = audit_proportions(pop[col])
        left = 0.0
        for status in st.AUDIT_ORDER:
            c = counts[status]
            pct = c / n * 100
            ax.barh(y, pct, left=left, color=st.AUDIT_COLORS[status],
                    edgecolor="white", linewidth=0.6, height=0.62)
            if pct >= 6:  # label inside if it fits
                txt_color = "white" if status in ("pass", "fail") else "#333333"
                ax.text(left + pct / 2, y, f"{status}\n{fmt_int(c)}",
                        ha="center", va="center", fontsize=8,
                        color=txt_color, linespacing=0.95)
            left += pct

    ax.set_yticks(y_positions)
    ax.set_yticklabels([AUDIT_LABELS[c] for c in AUDIT_COLS])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of population (%)")
    ax.set_title("External audit coverage across the population")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)

    # Legend with the four statuses.
    handles = [plt.Rectangle((0, 0), 1, 1, color=st.AUDIT_COLORS[s])
               for s in st.AUDIT_ORDER]
    ax.legend(handles, st.AUDIT_ORDER, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), ncol=4)

    fig.tight_layout()
    st.save_fig(fig, "02_audit_coverage_overall")
    plt.close(fig)


# ==========================================================================
# (e) Audit coverage by category — 3-panel small multiples
# ==========================================================================
def fig_audit_coverage_by_category(pop):
    cm = st.CATEGORY_MAPPING
    cats = list(cm["category"])              # alphabetical (retained order)
    labels = [cm.set_index("category").loc[c, "display_label"] for c in cats]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), sharey=True)
    y_positions = list(range(len(cats)))[::-1]  # first category on top

    for ax, col in zip(axes, AUDIT_COLS):
        for cat, y in zip(cats, y_positions):
            sub = pop.loc[pop["method_b_category"] == cat]
            n = len(sub)
            counts = audit_proportions(sub[col])
            left = 0.0
            for status in st.AUDIT_ORDER:
                pct = (counts[status] / n * 100) if n else 0
                ax.barh(y, pct, left=left, color=st.AUDIT_COLORS[status],
                        edgecolor="white", linewidth=0.4, height=0.7)
                left += pct
            ax.text(101, y, f"n={n}", va="center", ha="left",
                    fontsize=7, color="#777777")
        ax.set_title(AUDIT_LABELS[col])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share (%)")
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels)

    handles = [plt.Rectangle((0, 0), 1, 1, color=st.AUDIT_COLORS[s])
               for s in st.AUDIT_ORDER]
    fig.legend(handles, st.AUDIT_ORDER, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Audit outcomes by Method B category (population)", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    st.save_fig(fig, "02_audit_coverage_by_category")
    plt.close(fig)


# ==========================================================================
# (f) Publisher concentration — top-20 bar + Lorenz curve
# ==========================================================================
def fig_publisher_concentration(pop):
    pc = pop["publisher"].value_counts()
    n_pub = len(pc)
    n_skills = len(pop)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 6.0))

    # --- left: top 20 publishers by skill count ---
    top20 = pc.head(20)
    ypos = np.arange(len(top20))[::-1]
    ax1.barh(ypos, top20.values, color=st.BINARY_COLORS["positive"],
             edgecolor="white", linewidth=0.4, height=0.72)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(top20.index, fontsize=8)
    for y, v in zip(ypos, top20.values):
        ax1.text(v + n_skills * 0.004, y, fmt_int(v), va="center",
                 ha="left", fontsize=8, color="#333333")
    ax1.set_xlabel("Number of skills")
    ax1.set_title(f"Top 20 publishers by skill count (of {fmt_int(n_pub)})")
    ax1.set_xlim(0, top20.max() * 1.12)
    ax1.grid(axis="x", visible=True)
    ax1.grid(axis="y", visible=False)

    # --- right: Lorenz / cumulative-share curve ---
    sorted_counts = np.sort(pc.values)[::-1]          # largest first
    cum_skills = np.cumsum(sorted_counts) / n_skills * 100
    cum_pub = np.arange(1, n_pub + 1) / n_pub * 100
    x = np.concatenate([[0], cum_pub])
    y = np.concatenate([[0], cum_skills])
    ax2.plot(x, y, color=st.BINARY_COLORS["positive"], linewidth=2.0)
    ax2.plot([0, 100], [0, 100], color="#999999", linestyle=":", linewidth=1.2,
             label="equal share (1:1)")

    # Mark the top-10 publishers' cumulative share.
    top10_pub_pct = 10 / n_pub * 100
    top10_skill_pct = sorted_counts[:10].sum() / n_skills * 100
    ax2.scatter([top10_pub_pct], [top10_skill_pct], color="#D1495B", zorder=5, s=40)
    ax2.annotate(
        f"Top 10 publishers\n({top10_pub_pct:.1f}% of publishers)\n"
        f"= {top10_skill_pct:.1f}% of skills",
        xy=(top10_pub_pct, top10_skill_pct),
        xytext=(top10_pub_pct + 22, top10_skill_pct - 18),
        fontsize=8.5, color="#333333",
        arrowprops=dict(arrowstyle="->", color="#D1495B", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#CCCCCC", lw=0.6),
    )
    ax2.set_xlabel("Cumulative % of publishers (ranked, largest first)")
    ax2.set_ylabel("Cumulative % of skills")
    ax2.set_title("Publisher concentration (cumulative-share curve)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="both", visible=True)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    st.save_fig(fig, "02_publisher_concentration")
    plt.close(fig)

    # Companion table: top 50 publishers by skill count, with total installs.
    inst_by_pub = pop.groupby("publisher")["installs"].sum()
    off_by_pub = pop.groupby("publisher")["is_official"].mean()
    top50 = pc.head(50)
    tbl = pd.DataFrame({
        "publisher": top50.index,
        "n_skills": top50.values,
        "total_installs": [int(inst_by_pub[p]) for p in top50.index],
        "pct_official": [round(off_by_pub[p] * 100, 1) for p in top50.index],
    })
    tbl.to_csv(TBL_DIR / "02_top_publishers.csv", index=False)
    return top10_skill_pct, top10_pub_pct, n_pub


# ==========================================================================
# (g) SKILL.md retrievability
# ==========================================================================
def fig_skill_md_retrievability(pop):
    overall = pop["has_md_bool"].mean() * 100
    by_off = pop.groupby("is_official")["has_md_bool"].mean() * 100
    off_rate = by_off.get(True, np.nan)
    unoff_rate = by_off.get(False, np.nan)

    groups = ["Overall", "Official", "Community\n(unofficial)"]
    rates = [overall, off_rate, unoff_rate]
    ns = [len(pop), int(pop["is_official"].sum()), int((~pop["is_official"]).sum())]
    colors = [st.BINARY_COLORS["positive"], st.BINARY_COLORS["official"],
              st.BINARY_COLORS["unofficial"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xpos = np.arange(len(groups))
    bars = ax.bar(xpos, rates, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.62)
    for x, r, nn in zip(xpos, rates, ns):
        ax.text(x, r + 1.2, f"{r:.1f}%\n(n={fmt_int(nn)})", ha="center",
                va="bottom", fontsize=8.5, color="#333333", linespacing=1.0)

    ax.axhline(overall, color="#999999", linestyle=":", linewidth=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(groups)
    ax.set_ylabel("% with retrievable SKILL.md")
    ax.set_ylim(0, 100)
    ax.set_title("SKILL.md retrievability across the population")
    ax.grid(axis="y", visible=True)

    fig.tight_layout()
    st.save_fig(fig, "02_skill_md_retrievability")
    plt.close(fig)
    return overall, off_rate, unoff_rate


# ==========================================================================
# (h) Category distribution: retained vs excluded (count + install volume)
# ==========================================================================
def fig_category_distribution(pop):
    retained = set(st.CATEGORY_COLORS)
    cm = st.CATEGORY_MAPPING.set_index("category")

    # Build ordered list of real categories (sorted by skill count desc),
    # then the two non-category buckets at the bottom.
    raw = pop["method_b_category"]
    real = raw[raw.notna() & (raw != "none")]
    counts = real.value_counts()                      # real categories only
    ordered = list(counts.index)                      # desc by count

    n_none = int((raw == "none").sum())
    n_null = int(raw.isna().sum())

    # Assemble rows: (display_label, count, total_installs, color, kind)
    def disp(cat):
        return cm.loc[cat, "display_label"] if cat in retained else cat

    def color_for(cat):
        return st.CATEGORY_COLORS[cat] if cat in retained else st.EXCLUDED_COLOR

    inst_by_cat = real.to_frame().join(pop["installs"]).groupby("method_b_category")["installs"].sum()

    rows = []
    for cat in ordered:
        rows.append((disp(cat), int(counts[cat]), int(inst_by_cat[cat]),
                     color_for(cat), "retained" if cat in retained else "excluded"))
    # Two trailing buckets.
    none_inst = int(pop.loc[raw == "none", "installs"].sum())
    null_inst = int(pop.loc[raw.isna(), "installs"].sum())
    rows.append(("none (Method B: no cluster)", n_none, none_inst,
                 st.NONE_COLOR, "none"))
    rows.append(("uncategorized (no Method B label)", n_null, null_inst,
                 st.UNCATEGORIZED_COLOR, "uncategorized"))

    rdf = pd.DataFrame(rows, columns=["label", "count", "installs", "color", "kind"])
    ypos = np.arange(len(rdf))[::-1]  # first row on top

    fig, (axc, axi) = plt.subplots(1, 2, figsize=(13.5, 8.0), sharey=True)

    # --- left: counts ---
    axc.barh(ypos, rdf["count"], color=rdf["color"], edgecolor="white",
             linewidth=0.4, height=0.74)
    for y, v in zip(ypos, rdf["count"]):
        axc.text(v + rdf["count"].max() * 0.01, y, fmt_int(v), va="center",
                 ha="left", fontsize=7.5, color="#333333")
    axc.set_yticks(ypos)
    axc.set_yticklabels(rdf["label"], fontsize=8)
    axc.set_xlabel("Number of skills")
    axc.set_title("Category distribution — by skill count")
    axc.set_xlim(0, rdf["count"].max() * 1.12)
    axc.grid(axis="x", visible=True)
    axc.grid(axis="y", visible=False)

    # --- right: install volume (same ordering) ---
    axi.barh(ypos, rdf["installs"], color=rdf["color"], edgecolor="white",
             linewidth=0.4, height=0.74)
    for y, v in zip(ypos, rdf["installs"]):
        axi.text(v + rdf["installs"].max() * 0.01, y, fmt_int(v), va="center",
                 ha="left", fontsize=7.5, color="#333333")
    axi.set_xlabel("Total installs")
    axi.set_title("Category distribution — by install volume")
    axi.set_xlim(0, rdf["installs"].max() * 1.16)
    axi.xaxis.set_major_formatter(mticker.FuncFormatter(thousands))
    axi.grid(axis="x", visible=True)
    axi.grid(axis="y", visible=False)

    # Legend for the three colour roles.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4E79A7"),
        plt.Rectangle((0, 0), 1, 1, color=st.EXCLUDED_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=st.NONE_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=st.UNCATEGORIZED_COLOR),
    ]
    fig.legend(handles,
               ["retained (12, sampled — own colour)", "excluded (13, grey)",
                "Method B 'none'", "no Method B label"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("LLM-induced (Method B) category distribution across the population",
                 y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    st.save_fig(fig, "02_category_distribution")
    plt.close(fig)


# ==========================================================================
# (i) Per-category context table (retained categories)
# ==========================================================================
def category_context_table(pop):
    cm = st.CATEGORY_MAPPING.set_index("category")
    rows = []
    for cat in cm.index:  # alphabetical retained order
        sub = pop.loc[pop["method_b_category"] == cat]
        n_pop = len(sub)
        rows.append({
            "category": cat,
            "display_label": cm.loc[cat, "display_label"],
            "n_in_population": n_pop,
            "n_in_corpus": int(cm.loc[cat, "n_in_corpus"]),
            "pct_official_in_pop": round(sub["is_official"].mean() * 100, 1) if n_pop else None,
            "pct_with_any_audit_in_pop": round(sub["any_audit"].mean() * 100, 1) if n_pop else None,
            "median_installs_in_pop": int(sub["installs"].median()) if n_pop else None,
            "retrievability_rate_in_pop": round(sub["has_md_bool"].mean() * 100, 1) if n_pop else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "02_category_context.csv", index=False)
    return df


# ==========================================================================
# main
# ==========================================================================
def main():
    st.apply_style()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    pop = load_population()

    print("=" * 70)
    print("SECTION 2 — POPULATION-LEVEL PATTERNS")
    print("=" * 70)

    summ = population_summary(pop)
    print("\n[a] Population summary -> tables/02_population_summary.csv")
    print(summ.to_string(index=False))

    fig_install_distribution(pop)
    print("\n[b] figures/02_install_distribution.{png,svg}")

    fig_install_by_official(pop)
    print("[c] figures/02_install_by_official.{png,svg}")

    fig_audit_coverage_overall(pop)
    print("[d] figures/02_audit_coverage_overall.{png,svg}")

    fig_audit_coverage_by_category(pop)
    print("[e] figures/02_audit_coverage_by_category.{png,svg}")

    t10_skill, t10_pub, n_pub = fig_publisher_concentration(pop)
    print("[f] figures/02_publisher_concentration.{png,svg} "
          "+ tables/02_top_publishers.csv")
    print(f"    top-10 publishers = {t10_skill:.1f}% of skills "
          f"({t10_pub:.1f}% of {n_pub:,} publishers)")

    overall_md, off_md, unoff_md = fig_skill_md_retrievability(pop)
    print("[g] figures/02_skill_md_retrievability.{png,svg}")
    print(f"    has_skill_md: overall {overall_md:.1f}%, official {off_md:.1f}%, "
          f"unofficial {unoff_md:.1f}%")

    fig_category_distribution(pop)
    print("[h] figures/02_category_distribution.{png,svg}")

    ctx = category_context_table(pop)
    print("[i] tables/02_category_context.csv")
    print(ctx.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()