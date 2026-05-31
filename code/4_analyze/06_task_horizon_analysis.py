#!/usr/bin/env python3
"""
06_task_horizon_analysis.py — Section 6 of the findings chapter: task horizon
across the 120-skill coded corpus.

The coding frame operationalises TASK HORIZON through two deliberately
complementary variables plus three binary architectural dependencies:

    scope                — categorical breadth of work:
                           micro_task / single_bounded_task /
                           multi_step_workflow / end_to_end_process / unclear
    coordination_burden  — 0-4 intensity of cross-step / cross-tool / branching
                           coordination (higher = longer effective horizon)

    requires_memory             — binary: persistent state across invocations
    requires_tools              — binary: specific tools beyond the base agent
    requires_external_services  — binary: external APIs / services / credentials

scope (breadth) and coordination_burden (coordination intensity) are kept
SEPARATE on purpose (methodology 3.2): a multi-step workflow can be purely
sequential (low coordination_burden) or heavily branching (high
coordination_burden). Whether the two move together or vary independently is
itself the empirical question this section answers. Statefulness and
error-surface expansion are NOT separately coded — methodology 3.2 folds them
into coordination_burden and the architectural binaries rather than adding more
horizon variables.

Self-contained: reads ONLY from analysis/data/, imports the locked visual style
from _style.py, writes figures to analysis/figures/ (.png + .svg) and tables to
analysis/tables/ (.csv). Descriptive only — no structural inference.

Run from anywhere:
    python3 scripts/06_task_horizon_analysis.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as st  # noqa: E402

DATA_DIR = st.DATA_DIR
TBL_DIR = st.TBL_DIR

INTENSITIES = [0, 1, 2, 3, 4]

# The three architectural-dependency binaries, fixed canonical order.
BINARIES = ["requires_memory", "requires_tools", "requires_external_services"]
BINARY_LABELS = {
    "requires_memory":            "Requires memory",
    "requires_tools":             "Requires tools",
    "requires_external_services": "Requires external services",
}
BINARY_SHORT = {
    "requires_memory":            "memory",
    "requires_tools":             "tools",
    "requires_external_services": "external",
}

# Scope ordering / colours / ordinal encoding all come from _style.py.
SCOPE_ORDER = st.SCOPE_ORDER          # micro -> ... -> end_to_end -> unclear
SCOPE_LABELS = st.SCOPE_LABELS
SCOPE_COLORS = st.SCOPE_COLORS
SCOPE_ORDINAL = st.SCOPE_ORDINAL      # the 4 substantive values -> 1..4

# The five knowledge types (for the by-dominant-type breakdown, Sections 3-5).
KT = ["procedural", "analytical", "orchestration", "compliance", "documentation"]

SEQ_CMAP = "viridis"        # same sequential map as Sections 3-5 heatmaps


# ==========================================================================
# Load + integrity / anomaly guards
# ==========================================================================
def load_corpus():
    df = pd.read_parquet(DATA_DIR / "corpus.parquet")

    # coordination_burden must be a clean 0-4 integer intensity.
    if df["coordination_burden"].isna().any():
        raise RuntimeError("null values in coordination_burden")
    if not df["coordination_burden"].between(0, 4).all():
        raise RuntimeError("coordination_burden has values outside 0-4")

    # The three binaries must be 0/1 with no nulls.
    for b in BINARIES:
        if df[b].isna().any():
            raise RuntimeError(f"null values in {b}")
        if not df[b].isin([0, 1]).all():
            raise RuntimeError(f"{b} has values outside {{0,1}}")

    # scope must be one of the five allowed values (no nulls).
    if df["scope"].isna().any():
        raise RuntimeError("null values in scope")
    bad = set(df["scope"].unique()) - set(SCOPE_ORDER)
    if bad:
        raise RuntimeError(f"ANOMALY: unexpected scope value(s): {bad}")

    # codifiability/evaluability scores are needed for part (h).
    for c in ["codifiability_score", "evaluability_score"]:
        if df[c].isna().any():
            raise RuntimeError(f"null values in {c}")

    # ----------------------------------------------------------------------
    # ANOMALY GUARD: a contradiction between breadth and coordination would be
    # an end-to-end process with NO coordination (cb 0), or a micro task with
    # FULL orchestration (cb 4). Stop loudly if any such pair exists — we want
    # edge cases surfaced, not smoothed.
    # ----------------------------------------------------------------------
    contra = df.loc[
        ((df["scope"] == "end_to_end_process") & (df["coordination_burden"] == 0))
        | ((df["scope"] == "micro_task") & (df["coordination_burden"] == 4))
    ]
    if len(contra):
        ids = ", ".join(contra["skill_id"].tolist())
        raise RuntimeError(
            f"ANOMALY: {len(contra)} skill(s) pair an extreme scope with a "
            f"contradictory coordination_burden: {ids}"
        )
    return df


# ==========================================================================
# (a) Task-horizon summary table  (+ separate scope distribution table)
# ==========================================================================
def summary_tables(df):
    n = len(df)
    cols = ["variable", "mean", "median", "std",
            "n_at_0", "n_at_1", "n_at_2", "n_at_3", "n_at_4",
            "n_at_3_or_higher", "share_true"]
    rows = []

    # --- coordination_burden: full 0-4 intensity stats ---
    s = df["coordination_burden"]
    vc = s.value_counts().reindex(INTENSITIES, fill_value=0)
    rows.append({
        "variable": "coordination_burden",
        "mean": round(float(s.mean()), 3),
        "median": float(s.median()),
        "std": round(float(s.std()), 3),
        "n_at_0": int(vc[0]), "n_at_1": int(vc[1]), "n_at_2": int(vc[2]),
        "n_at_3": int(vc[3]), "n_at_4": int(vc[4]),
        "n_at_3_or_higher": int((s >= 3).sum()),
        "share_true": "",
    })

    # --- the three binaries: share true (= mean of a 0/1 var) ---
    for b in BINARIES:
        sb = df[b]
        rows.append({
            "variable": b,
            "mean": round(float(sb.mean()), 3),
            "median": float(sb.median()),
            "std": round(float(sb.std()), 3),
            "n_at_0": int((sb == 0).sum()), "n_at_1": int((sb == 1).sum()),
            "n_at_2": "", "n_at_3": "", "n_at_4": "",
            "n_at_3_or_higher": "",
            "share_true": round(float(sb.mean()), 4),
        })

    # --- n_requirements: how many of the three binaries each skill requires ---
    nreq = df[BINARIES].sum(axis=1)
    nvc = nreq.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    rows.append({
        "variable": "n_requirements",
        "mean": round(float(nreq.mean()), 3),
        "median": float(nreq.median()),
        "std": round(float(nreq.std()), 3),
        "n_at_0": int(nvc[0]), "n_at_1": int(nvc[1]), "n_at_2": int(nvc[2]),
        "n_at_3": int(nvc[3]), "n_at_4": "",
        "n_at_3_or_higher": int((nreq >= 3).sum()),
        "share_true": "",
    })

    summ = pd.DataFrame(rows)[cols]
    summ.to_csv(TBL_DIR / "06_task_horizon_summary.csv", index=False)

    # --- separate small table: scope counts + shares (categorical) ---
    counts = df["scope"].value_counts().reindex(SCOPE_ORDER, fill_value=0)
    scope_tbl = pd.DataFrame({
        "scope": SCOPE_ORDER,
        "count": [int(counts[s]) for s in SCOPE_ORDER],
        "share": [round(float(counts[s] / n), 4) for s in SCOPE_ORDER],
    })
    scope_tbl.to_csv(TBL_DIR / "06_scope_distribution.csv", index=False)

    return summ, scope_tbl, nreq


# ==========================================================================
# (b) Scope distribution — horizontal bar, shortest -> longest, unclear last
# ==========================================================================
def fig_scope_distribution(df, scope_tbl):
    n = len(df)
    counts = {r["scope"]: r["count"] for _, r in scope_tbl.iterrows()}

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ypos = np.arange(len(SCOPE_ORDER))[::-1]      # micro_task at top
    vals = [counts[s] for s in SCOPE_ORDER]
    ax.barh(ypos, vals, color=st.NEUTRAL_ACCENT, edgecolor="white",
            linewidth=0.6, height=0.72)
    xmax = max(vals)
    for y, v in zip(ypos, vals):
        ax.text(v + xmax * 0.012, y, f"{v}  ({v / n * 100:.1f}%)",
                va="center", ha="left", fontsize=9, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels([SCOPE_LABELS[s] for s in SCOPE_ORDER])
    ax.set_xlim(0, xmax * 1.16)
    ax.set_xlabel("Number of skills (of 120)")
    ax.set_title("Scope distribution across the 120-skill corpus\n"
                 "(ordered shortest → longest horizon; 'unclear' last)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    st.save_fig(fig, "06_scope_distribution")
    plt.close(fig)


# ==========================================================================
# (c) Coordination_burden distribution — bar 0-4, mean/median marked
# ==========================================================================
def fig_coordination_burden_distribution(df):
    s = df["coordination_burden"]
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

    ax.axvline(mean, color="#E8A33D", linestyle="--", linewidth=1.8, zorder=5)
    ax.axvline(median, color="#D1495B", linestyle="-", linewidth=1.8, zorder=5)
    ax.text(0.97, 0.95, f"mean = {mean:.2f}", transform=ax.transAxes,
            color="#B5781F", ha="right", va="top", fontsize=10, fontweight="bold")
    ax.text(0.97, 0.87, f"median = {median:.1f}", transform=ax.transAxes,
            color="#D1495B", ha="right", va="top", fontsize=10, fontweight="bold")

    ax.set_xticks(INTENSITIES)
    ax.set_xlabel("Coordination burden (0–4; higher = more coordination)")
    ax.set_ylabel("Number of skills")
    ax.set_ylim(0, ytop)
    ax.set_title("Coordination-burden distribution across the 120-skill corpus")
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    st.save_fig(fig, "06_coordination_burden_distribution")
    plt.close(fig)


# ==========================================================================
# (d) scope x coordination_burden — cross-tab + heatmap + Spearman
# ==========================================================================
def scope_x_coordination(df):
    # ---- count matrix: rows = scope (canonical order), cols = cb 0-4 ----
    ct = (pd.crosstab(df["scope"], df["coordination_burden"])
          .reindex(index=SCOPE_ORDER, columns=INTENSITIES, fill_value=0))
    ct.index.name = "scope"
    ct.to_csv(TBL_DIR / "06_scope_x_coordination_burden.csv")

    # ---- Spearman: ordinal scope (unclear -> NaN, excluded) vs cb ----
    ord_scope = df["scope"].map(SCOPE_ORDINAL)        # unclear -> NaN
    mask = ord_scope.notna()
    n_excluded = int((~mask).sum())
    rho, pval = stats.spearmanr(ord_scope[mask], df.loc[mask, "coordination_burden"])

    # ---- heatmap of the joint counts ----
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    mat = ct.values.astype(float)
    cmap = colormaps[SEQ_CMAP]
    norm = mcolors.Normalize(vmin=0, vmax=mat.max())
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(INTENSITIES)))
    ax.set_xticklabels(INTENSITIES)
    ax.set_yticks(range(len(SCOPE_ORDER)))
    ax.set_yticklabels([SCOPE_LABELS[s] for s in SCOPE_ORDER])
    ax.set_xlabel("Coordination burden (0–4)")
    ax.set_ylabel("Scope (shortest → longest)")
    ax.set_title("Scope × coordination burden joint counts\n"
                 f"(Spearman ρ = {rho:.2f}, p = {pval:.1e}, n = {int(mask.sum())}; "
                 "'unclear' excluded)")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(INTENSITIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SCOPE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(SCOPE_ORDER)):
        for j in range(len(INTENSITIES)):
            v = int(mat[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color=st.text_color_for_bg(cmap(norm(v))))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Number of skills")
    fig.tight_layout()
    st.save_fig(fig, "06_scope_x_coordination_burden")
    plt.close(fig)

    return ct, rho, pval, int(mask.sum()), n_excluded


# ==========================================================================
# (e) Task horizon by category — stacked scope shares + ranked mean cb
# ==========================================================================
def fig_by_category(df):
    cm = st.CATEGORY_MAPPING
    cats = cm["category"].tolist()

    # ---- per-category table: scope counts + mean/SD coordination_burden ----
    rows = []
    sc_ct = (pd.crosstab(df["category"], df["scope"])
             .reindex(index=cats, columns=SCOPE_ORDER, fill_value=0))
    for cat in cats:
        sub = df.loc[df["category"] == cat]
        rec = {"category": cat, "display_label": st.CATEGORY_LABELS[cat],
               "n": len(sub),
               "coordination_burden_mean": round(float(sub["coordination_burden"].mean()), 4),
               "coordination_burden_std": round(float(sub["coordination_burden"].std()), 4)}
        for s in SCOPE_ORDER:
            rec[f"scope_{s}"] = int(sc_ct.loc[cat, s])
        rows.append(rec)
    bycat = pd.DataFrame(rows)
    bycat.to_csv(TBL_DIR / "06_task_horizon_by_category.csv", index=False)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14.0, 6.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- Left: stacked horizontal scope-share bars per category ----
    # Order categories by their "longer-horizon" share
    # (multi_step_workflow + end_to_end_process), descending.
    share_mat = sc_ct.div(sc_ct.sum(axis=1), axis=0)
    longer = share_mat["multi_step_workflow"] + share_mat["end_to_end_process"]
    cat_order = longer.sort_values(ascending=False).index.tolist()
    share_mat_o = share_mat.reindex(cat_order)
    labels = [st.CATEGORY_LABELS[c] for c in cat_order]

    ypos = np.arange(len(cat_order))[::-1]
    left = np.zeros(len(cat_order))
    for s in SCOPE_ORDER:
        widths = share_mat_o[s].values
        axL.barh(ypos, widths, left=left, height=0.74,
                 color=SCOPE_COLORS[s], edgecolor="white", linewidth=0.6,
                 label=SCOPE_LABELS[s])
        for y, w, l, c in zip(ypos, widths, left, cat_order):
            if w >= 0.08:
                cnt = int(sc_ct.loc[c, s])
                axL.text(l + w / 2, y, str(cnt), ha="center", va="center",
                         fontsize=8,
                         color=st.text_color_for_bg(mcolors.to_rgba(SCOPE_COLORS[s])))
        left = left + widths
    axL.set_yticks(ypos)
    axL.set_yticklabels(labels)
    axL.set_xlim(0, 1.0)
    axL.set_xlabel("Share of category's skills (n = 10 each; segment label = count)")
    axL.set_title("Scope mix by category\n(ordered by multi-step + end-to-end share)")
    axL.grid(axis="x", visible=True)
    axL.grid(axis="y", visible=False)
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
               frameon=False, fontsize=8.5)

    # ---- Right: ranked mean coordination_burden, ±1 SD ----
    bycat_r = bycat.sort_values("coordination_burden_mean",
                                ascending=False).reset_index(drop=True)
    order = bycat_r["category"].tolist()
    rlabels = bycat_r["display_label"].tolist()
    means = bycat_r["coordination_burden_mean"].values
    stds = bycat_r["coordination_burden_std"].values
    ypos2 = np.arange(len(order))[::-1]
    colors = [st.CATEGORY_COLORS[c] for c in order]
    axR.barh(ypos2, means, xerr=stds, color=colors, edgecolor="white",
             linewidth=0.6, height=0.72,
             error_kw=dict(ecolor="#555555", elinewidth=1.1, capsize=3))
    for y, m in zip(ypos2, means):
        axR.text(m + 0.05, y, f"{m:.1f}", va="center", ha="left",
                 fontsize=8.5, color="#333333")
    axR.set_yticks(ypos2)
    axR.set_yticklabels(rlabels)
    axR.set_xlabel("Mean coordination burden (0–4); bars = ±1 SD")
    axR.set_xlim(0, 4.0)
    axR.set_title("Mean coordination burden by category\n(ranked; n = 10 per category)")
    axR.grid(axis="x", visible=True)
    axR.grid(axis="y", visible=False)

    st.save_fig(fig, "06_task_horizon_by_category")
    plt.close(fig)
    return bycat, sc_ct


# ==========================================================================
# (f) Architectural dependencies — n_requirements, marginals, joint
# ==========================================================================
def architectural_dependencies(df, nreq):
    n = len(df)

    # ---- joint distribution over the 8 (memory, tools, external) combos ----
    combo_rows = []
    for m in (0, 1):
        for t in (0, 1):
            for e in (0, 1):
                mask = ((df["requires_memory"] == m)
                        & (df["requires_tools"] == t)
                        & (df["requires_external_services"] == e))
                cnt = int(mask.sum())
                present = [BINARY_SHORT[b] for b, v in
                           zip(BINARIES, (m, t, e)) if v == 1]
                combo_rows.append({
                    "requires_memory": m, "requires_tools": t,
                    "requires_external_services": e,
                    "n_requirements": m + t + e,
                    "combo": "+".join(present) if present else "none",
                    "count": cnt,
                    "share": round(cnt / n, 4),
                })
    combo = (pd.DataFrame(combo_rows)
             .sort_values(["n_requirements", "count"], ascending=[True, False])
             .reset_index(drop=True))
    combo.to_csv(TBL_DIR / "06_architectural_dependencies.csv", index=False)

    # counts of skills requiring 0/1/2/3 of the three
    nvc = nreq.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    # marginals (how many require each individually)
    marg = {b: int(df[b].sum()) for b in BINARIES}

    # ---- per-category breakdown ----
    cm = st.CATEGORY_MAPPING
    cats = cm["category"].tolist()
    bc_rows = []
    for cat in cats:
        sub = df.loc[df["category"] == cat]
        rec = {"category": cat, "display_label": st.CATEGORY_LABELS[cat],
               "n": len(sub)}
        for b in BINARIES:
            rec[f"n_{BINARY_SHORT[b]}"] = int(sub[b].sum())
        rec["mean_n_requirements"] = round(float(sub[BINARIES].sum(axis=1).mean()), 3)
        bc_rows.append(rec)
    bycat = pd.DataFrame(bc_rows)
    bycat.to_csv(TBL_DIR / "06_architectural_dependencies_by_category.csv", index=False)

    # ---- figure: two panels ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Left: n_requirements 0-3
    xs = [0, 1, 2, 3]
    vals = [int(nvc[k]) for k in xs]
    ytop = int(np.ceil((max(vals) + 10) / 10.0) * 10)
    axL.bar(xs, vals, color=st.NEUTRAL_ACCENT, edgecolor="white",
            linewidth=0.6, width=0.7)
    for x, v in zip(xs, vals):
        axL.text(x, v + ytop * 0.012, f"{v}\n({v / n * 100:.1f}%)",
                 ha="center", va="bottom", fontsize=9, color="#333333")
    axL.set_xticks(xs)
    axL.set_xlabel("Number of architectural dependencies required (of 3)")
    axL.set_ylabel("Number of skills (of 120)")
    axL.set_ylim(0, ytop)
    axL.set_title("How many of the three dependencies\neach skill requires")
    axL.grid(axis="y", visible=True)
    axL.grid(axis="x", visible=False)

    # Right: share with each individual requirement
    bx = np.arange(len(BINARIES))
    shares = [marg[b] / n for b in BINARIES]
    axR.bar(bx, shares, color=st.NEUTRAL_ACCENT, edgecolor="white",
            linewidth=0.6, width=0.62)
    for x, b in zip(bx, BINARIES):
        axR.text(x, marg[b] / n + 0.012, f"{marg[b]}\n({marg[b] / n * 100:.1f}%)",
                 ha="center", va="bottom", fontsize=9, color="#333333")
    axR.set_xticks(bx)
    axR.set_xticklabels([BINARY_LABELS[b].replace("Requires ", "") for b in BINARIES])
    axR.set_ylim(0, max(shares) * 1.25)
    axR.set_ylabel("Share of skills requiring it")
    axR.set_title("Share requiring each dependency\n(marginals)")
    axR.grid(axis="y", visible=True)
    axR.grid(axis="x", visible=False)

    fig.suptitle("Architectural dependencies across the 120-skill corpus", y=1.04)
    fig.tight_layout()
    st.save_fig(fig, "06_architectural_dependencies")
    plt.close(fig)

    return combo, nvc, marg, bycat


# ==========================================================================
# (g) Task horizon by dominant knowledge type
# ==========================================================================
def by_dominant_type(df):
    """Partition the 120 skills by dominant (==4) knowledge type, with the same
    single/none/two split as Sections 3-5 so the n's reconcile to 120. For each
    group: mean coordination_burden and the MODAL scope value."""
    ndom = sum((df[k] == 4).astype(int) for k in KT)

    def _modal_scope(mask):
        sub = df.loc[mask, "scope"]
        if len(sub) == 0:
            return ""
        # value_counts is descending; ties broken by canonical scope order.
        vc = sub.value_counts()
        top = vc.max()
        tied = [s for s in SCOPE_ORDER if s in vc.index and vc[s] == top]
        return tied[0]

    rows = []
    for k in KT:
        mask = (df[k] == 4) & (ndom == 1)
        if mask.sum() > 0:
            rows.append({
                "dominant_group": f"{k} (sole dominant)",
                "n": int(mask.sum()),
                "coordination_burden_mean": round(float(df.loc[mask, "coordination_burden"].mean()), 3),
                "coordination_burden_std": round(float(df.loc[mask, "coordination_burden"].std(ddof=0)), 3),
                "modal_scope": _modal_scope(mask),
            })
    for label, mask in [("no dominant type", ndom == 0),
                        ("two dominant types", ndom == 2)]:
        rows.append({
            "dominant_group": label,
            "n": int(mask.sum()),
            "coordination_burden_mean": round(float(df.loc[mask, "coordination_burden"].mean()), 3),
            "coordination_burden_std": round(float(df.loc[mask, "coordination_burden"].std(ddof=0)), 3),
            "modal_scope": _modal_scope(mask),
        })
    tbl = pd.DataFrame(rows).sort_values(
        "coordination_burden_mean", ascending=False).reset_index(drop=True)
    tbl.to_csv(TBL_DIR / "06_task_horizon_by_dominant_type.csv", index=False)

    spread = tbl["coordination_burden_mean"].max() - tbl["coordination_burden_mean"].min()
    made_fig = spread > 0.2
    if made_fig:
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        ypos = np.arange(len(tbl))[::-1]
        ax.barh(ypos, tbl["coordination_burden_mean"].values,
                color=st.NEUTRAL_ACCENT, edgecolor="white", linewidth=0.6, height=0.7)
        for y, (_, r) in zip(ypos, tbl.iterrows()):
            ax.text(r["coordination_burden_mean"] + 0.04, y,
                    f"{r['coordination_burden_mean']:.2f}  (n={r['n']}; "
                    f"{SCOPE_LABELS.get(r['modal_scope'], r['modal_scope'])})",
                    va="center", ha="left", fontsize=8.3, color="#333333")
        ax.set_yticks(ypos)
        ax.set_yticklabels(tbl["dominant_group"].tolist())
        ax.set_xlabel("Mean coordination burden (0–4)")
        ax.set_xlim(0, 4.0)
        ax.set_title("Mean coordination burden by dominant knowledge type\n"
                     "(sole-dominant per type, plus no-/two-dominant; groups partition the 120; "
                     "label shows modal scope)")
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        st.save_fig(fig, "06_task_horizon_by_dominant_type")
        plt.close(fig)
    return tbl, spread, made_fig


# ==========================================================================
# (h) Horizon x codifiability x evaluability
# ==========================================================================
def horizon_x_codif_eval(df):
    # ---- Spearman: coordination_burden with each derived score ----
    rho_c, p_c = stats.spearmanr(df["coordination_burden"], df["codifiability_score"])
    rho_e, p_e = stats.spearmanr(df["coordination_burden"], df["evaluability_score"])

    # ---- mean codifiability / evaluability by scope (one row per scope) ----
    rows = []
    for s in SCOPE_ORDER:
        sub = df.loc[df["scope"] == s]
        if len(sub) == 0:
            rows.append({"scope": s, "n": 0,
                         "mean_codifiability_score": "",
                         "mean_evaluability_score": "",
                         "mean_coordination_burden": ""})
            continue
        rows.append({
            "scope": s, "n": len(sub),
            "mean_codifiability_score": round(float(sub["codifiability_score"].mean()), 3),
            "mean_evaluability_score": round(float(sub["evaluability_score"].mean()), 3),
            "mean_coordination_burden": round(float(sub["coordination_burden"].mean()), 3),
        })
    byscope = pd.DataFrame(rows)
    byscope.to_csv(TBL_DIR / "06_horizon_x_codif_eval.csv", index=False)

    # ---- optional figure: mean codif & eval as function of scope ----
    present = [s for s in SCOPE_ORDER if (df["scope"] == s).any()]
    codif_means = [float(df.loc[df["scope"] == s, "codifiability_score"].mean())
                   for s in present]
    eval_means = [float(df.loc[df["scope"] == s, "evaluability_score"].mean())
                  for s in present]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(present))
    w = 0.38
    ax.bar(x - w / 2, codif_means, width=w, color=st.NEUTRAL_ACCENT,
           edgecolor="white", linewidth=0.6, label="Codifiability score")
    ax.bar(x + w / 2, eval_means, width=w, color="#E8A33D",
           edgecolor="white", linewidth=0.6, label="Evaluability score")
    for xi, cv, ev in zip(x, codif_means, eval_means):
        ax.text(xi - w / 2, cv + 0.05, f"{cv:.2f}", ha="center", va="bottom",
                fontsize=8.5, color="#333333")
        ax.text(xi + w / 2, ev + 0.05, f"{ev:.2f}", ha="center", va="bottom",
                fontsize=8.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([SCOPE_LABELS[s] for s in present], rotation=12, ha="right")
    ax.set_ylim(0, 4.0)
    ax.set_ylabel("Mean score (0–4)")
    ax.set_title("Mean codifiability & evaluability by scope\n"
                 "(descriptive; Section 8 builds the full map)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    st.save_fig(fig, "06_horizon_vs_codif_eval")
    plt.close(fig)

    return (rho_c, p_c), (rho_e, p_e), byscope


# ==========================================================================
# main
# ==========================================================================
def main():
    st.apply_style()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus()

    print("=" * 72)
    print("SECTION 6 — TASK HORIZON")
    print("=" * 72)
    print("  anomaly guard: PASS (no end_to_end@cb0 / micro_task@cb4 contradiction)")

    summ, scope_tbl, nreq = summary_tables(df)
    print("\n[a] tables/06_task_horizon_summary.csv")
    print(summ.to_string(index=False))
    print("\n    tables/06_scope_distribution.csv")
    print(scope_tbl.to_string(index=False))

    fig_scope_distribution(df, scope_tbl)
    print("\n[b] figures/06_scope_distribution.{png,svg}")

    fig_coordination_burden_distribution(df)
    cb = df["coordination_burden"]
    print("\n[c] figures/06_coordination_burden_distribution.{png,svg}")
    print(f"    cb mean={cb.mean():.3f}, median={cb.median():.1f}, std={cb.std():.3f}; "
          + ", ".join(f"{i}:{int((cb == i).sum())}" for i in INTENSITIES))

    ct, rho, pval, n_used, n_excl = scope_x_coordination(df)
    print("\n[d] figures/06_scope_x_coordination_burden.{png,svg} "
          "+ tables/06_scope_x_coordination_burden.csv")
    print(ct.to_string())
    print(f"    Spearman ρ(scope_ordinal, coordination_burden) = {rho:.4f}  "
          f"(p = {pval:.3e}, n = {n_used}; 'unclear' excluded: {n_excl} skill)")

    bycat, sc_ct = fig_by_category(df)
    print("\n[e] figures/06_task_horizon_by_category.{png,svg} "
          "+ tables/06_task_horizon_by_category.csv")
    print(bycat[["display_label", "coordination_burden_mean", "coordination_burden_std"]
                + [f"scope_{s}" for s in SCOPE_ORDER]].to_string(index=False))

    combo, nvc, marg, depbycat = architectural_dependencies(df, nreq)
    print("\n[f] figures/06_architectural_dependencies.{png,svg}")
    print("    + tables/06_architectural_dependencies.csv (8-combo joint)")
    print("    + tables/06_architectural_dependencies_by_category.csv")
    print("    n_requirements counts (0/1/2/3): "
          + ", ".join(f"{k}:{int(nvc[k])}" for k in [0, 1, 2, 3]))
    print("    marginals: " + ", ".join(f"{BINARY_SHORT[b]}={marg[b]}" for b in BINARIES))
    print("    joint combinations:")
    print(combo[["combo", "n_requirements", "count", "share"]].to_string(index=False))
    print("    per-category dependency counts:")
    print(depbycat.to_string(index=False))

    domtbl, spread, made_fig = by_dominant_type(df)
    print("\n[g] tables/06_task_horizon_by_dominant_type.csv")
    print(f"    spread of group mean coordination_burden = {spread:.3f} "
          f"({'>0.2 -> figure produced' if made_fig else '<=0.2 -> figure skipped'})")
    print(domtbl.to_string(index=False))

    (rho_c, p_c), (rho_e, p_e), byscope = horizon_x_codif_eval(df)
    print("\n[h] figures/06_horizon_vs_codif_eval.{png,svg} "
          "+ tables/06_horizon_x_codif_eval.csv")
    print(f"    Spearman ρ(coordination_burden, codifiability_score) = {rho_c:.4f} (p={p_c:.3e})")
    print(f"    Spearman ρ(coordination_burden, evaluability_score)  = {rho_e:.4f} (p={p_e:.3e})")
    print("    mean codif/eval/cb by scope:")
    print(byscope.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
