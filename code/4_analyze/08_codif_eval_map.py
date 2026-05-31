#!/usr/bin/env python3
"""
08_codif_eval_map.py — Section 8 of the findings chapter: the codifiability–
evaluability map. This is the chapter's CENTERPIECE section. It locates the
120-skill corpus in the 2-D space of codifiability_score (Section 4) ×
evaluability_score (Section 5) and overlays the analytical lenses developed in
Sections 3, 6 and 7.

The map's conceptual basis is methodology 3.1's automation-threshold argument:
reliable agentic delegation needs HIGH codifiability AND HIGH evaluability
together. The "high" threshold is score >= 3 on each axis (the methodology's
"substantial presence" cut, NOT a median/mean). That convention defines four
quadrants:

    Q1  codif>=3 & eval>=3  — "Above threshold"            (top-right)
    Q2  codif>=3 & eval<3   — "Codified but unverifiable"  (bottom-right)
    Q3  codif<3  & eval>=3  — "Unanchored evaluability"    (top-left)
    Q4  codif<3  & eval<3   — "Below threshold"            (bottom-left)

Visual-choice split (documented in the report):
  * the PRIMARY map (part a) is a 5x5 COUNT HEATMAP (option A) — unambiguous for
    cell counts and for showing exactly where the empty cells are, which is the
    centerpiece message (the quadrant distribution);
  * every OVERLAY map (parts b-f) is a JITTERED SCATTER (option B) with one point
    per skill, because each overlay encodes a per-skill attribute (cohort
    membership, dominant knowledge type, category, scope/coordination_burden,
    governance) that 25 aggregated heatmap cells cannot carry. The jitter is
    seeded ONCE and shared, so a given skill sits in the same spot in every
    overlay figure; jitter radius (0.3) is < half a cell (0.5) so no point ever
    crosses a quadrant divider.

Self-contained: reads ONLY from analysis/data/corpus.parquet, imports the locked
visual style from _style.py, writes figures to analysis/figures/ (.png + .svg)
and tables to analysis/tables/ (.csv). Descriptive only — no structural
inference (that is the chapter draft's job).

Run from anywhere:
    python3 scripts/08_codif_eval_map.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as st  # noqa: E402

DATA_DIR = st.DATA_DIR
TBL_DIR = st.TBL_DIR

INTENSITIES = [0, 1, 2, 3, 4]

# Quadrant boundary: score >= 3 is "high" on each axis (methodology 3.1).
HIGH = 3
# Divider lines sit between code 2 and 3 on each axis.
DIVIDER = 2.5

# The five knowledge types (dominant-type partition, Sections 3-6).
KT = ["procedural", "analytical", "orchestration", "compliance", "documentation"]

# The four governance variables (Section 7), split into its two independent pairs.
IDENTITY_LIFECYCLE = ["provenance_clarity", "maintenance_signals"]
BOUNDARY_SAFETY = ["safety_notes", "scope_limits"]
GOV_VARS = IDENTITY_LIFECYCLE + BOUNDARY_SAFETY

# Cohorts referenced by the chapter (from Sections 5, 6, 7).
HIGH_EVAL = ["DM09", "ET09", "SC07", "SD09", "SM08"]              # eval >= 3 (Section 5)
AUDIT_FAIL = ["CT10", "DM01", "ET02", "FA06", "RD09", "SC01", "SC07", "SM07"]  # Section 7
ARCH_ALL3 = ["BS03", "CC07", "ET07", "FA03", "RD01", "RD06", "SM03", "SM06"]   # memory+tools+external
ARCH_MEMTOOLS = ["BS01", "CC08", "ET04", "SC08", "UI03"]          # memory+tools (no external)
ARCH_HEAVY = ARCH_ALL3 + ARCH_MEMTOOLS                           # 13 architecturally-heaviest

# Scope marker shapes for the horizon overlay (corpus has NO micro_task).
SCOPE_MARKERS = {
    "single_bounded_task": "o",   # circle
    "multi_step_workflow": "s",   # square
    "end_to_end_process":  "^",   # triangle
    "unclear":             "X",   # cross
}

JITTER_SEED = 8
JITTER_R = 0.30                    # < 0.5, so points never cross a quadrant divider

SEQ_CMAP = "viridis"               # governance overlays (same as Sections 3-7)


# ==========================================================================
# Load + anomaly guard + derived layout columns
# ==========================================================================
def load_and_prepare():
    df = pd.read_parquet(DATA_DIR / "corpus.parquet")

    # --- score sanity ---
    for c in ["codifiability_score", "evaluability_score"]:
        if df[c].isna().any():
            raise RuntimeError(f"null values in {c}")
        if not df[c].between(0, 4).all():
            raise RuntimeError(f"{c} has values outside 0-4")

    # ----------------------------------------------------------------------
    # ANOMALY GUARD. The brief flags the unexpected boundary case explicitly:
    # a skill at evaluability_score == 4 (Section 5 found NONE; an eval-4 skill,
    # especially the codif==3 & eval==4 case, would be a surprise to surface
    # before any output). Stop loudly if one exists.
    # ----------------------------------------------------------------------
    if (df["evaluability_score"] == 4).any():
        ids = ", ".join(df.loc[df["evaluability_score"] == 4, "skill_id"].tolist())
        raise RuntimeError(
            f"ANOMALY: {(df['evaluability_score'] == 4).sum()} skill(s) at "
            f"evaluability_score==4 ({ids}) — Section 5 reported none. Inspect "
            "before producing the map."
        )

    # --- derived governance pairs (Section 7's two-pair structure) ---
    df["identity_lifecycle"] = df[IDENTITY_LIFECYCLE].mean(axis=1)
    df["boundary_safety"] = df[BOUNDARY_SAFETY].mean(axis=1)
    df["governance_total"] = df[GOV_VARS].mean(axis=1)

    # --- scope ordinal (Section 6 style; unclear -> NaN) ---
    df["scope_ord"] = df["scope"].map(st.SCOPE_ORDINAL)

    # --- quadrant assignment (>= 3 high on each axis) ---
    hi_c = df["codifiability_score"] >= HIGH
    hi_e = df["evaluability_score"] >= HIGH
    df["quadrant"] = np.select(
        [hi_c & hi_e, hi_c & ~hi_e, ~hi_c & hi_e, ~hi_c & ~hi_e],
        ["Q1", "Q2", "Q3", "Q4"], default="?")

    # --- dominant-type partition group (sole / none / two) ---
    ndom = sum((df[k] == 4).astype(int) for k in KT)
    df["n_dominant"] = ndom
    def _domgroup(row):
        if row["n_dominant"] == 0:
            return "no dominant"
        if row["n_dominant"] >= 2:
            return "two dominant"
        for k in KT:
            if row[k] == 4:
                return k
        return "?"
    df["dom_group"] = df.apply(_domgroup, axis=1)

    # --- shared seeded jitter (one position per skill, reused by all overlays) ---
    rng = np.random.default_rng(JITTER_SEED)
    df["jx"] = df["codifiability_score"] + rng.uniform(-JITTER_R, JITTER_R, len(df))
    df["jy"] = df["evaluability_score"] + rng.uniform(-JITTER_R, JITTER_R, len(df))

    return df


# Quadrant metadata (label, short label, condition strings), fixed order.
QUADRANTS = [
    ("Q1", "Above threshold",          "codif>=3 & eval>=3"),
    ("Q2", "Codified but unverifiable", "codif>=3 & eval<3"),
    ("Q3", "Unanchored evaluability",  "codif<3 & eval>=3"),
    ("Q4", "Below threshold",          "codif<3 & eval<3"),
]
QUAD_LABEL = {q: lbl for q, lbl, _ in QUADRANTS}
QUAD_COND = {q: cond for q, _, cond in QUADRANTS}


# ==========================================================================
# Shared scatter frame: quadrant dividers, axes, ticks
# ==========================================================================
def draw_frame(ax, xlabel=True, ylabel=True):
    ax.axvline(DIVIDER, color="#555555", ls="--", lw=1.0, zorder=1)
    ax.axhline(DIVIDER, color="#555555", ls="--", lw=1.0, zorder=1)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.6, 4.6)
    ax.set_xticks(INTENSITIES)
    ax.set_yticks(INTENSITIES)
    if xlabel:
        ax.set_xlabel("Codifiability score (0–4)")
    if ylabel:
        ax.set_ylabel("Evaluability score (0–4)")
    ax.set_aspect("equal")
    ax.grid(False)


# ==========================================================================
# (a) Main map — 5x5 count heatmap with quadrant lines, labels, counts
# ==========================================================================
def fig_main_map(df):
    # M[eval, codif] counts (rows = evaluability so y increases upward).
    M = np.zeros((5, 5), dtype=int)
    for _, r in df.iterrows():
        M[int(r["evaluability_score"]), int(r["codifiability_score"])] += 1

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    cmap = colormaps["Blues"]
    vmax = int(M.max())
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(M, origin="lower", cmap=cmap, norm=norm, aspect="equal")

    # cell counts: bold for >0, faint "0" so empty cells are explicit
    for e in INTENSITIES:
        for c in INTENSITIES:
            v = M[e, c]
            if v > 0:
                ax.text(c, e, str(v), ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=st.text_color_for_bg(cmap(norm(v))))
            else:
                ax.text(c, e, "0", ha="center", va="center",
                        fontsize=8, color="#C0C0C0")

    # quadrant dividers (between code 2 and 3 on each axis)
    ax.axvline(DIVIDER, color="#444444", ls="--", lw=1.4)
    ax.axhline(DIVIDER, color="#444444", ls="--", lw=1.4)

    ax.set_xticks(INTENSITIES)
    ax.set_yticks(INTENSITIES)
    ax.set_xlabel("Codifiability score (0–4)")
    ax.set_ylabel("Evaluability score (0–4)")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.grid(False)
    ax.set_title("Codifiability × evaluability map of the 120-skill corpus\n"
                 "(cell counts; dashed lines mark the score≥3 'high' threshold)")

    # quadrant corner labels + counts
    counts = df["quadrant"].value_counts().reindex(["Q1", "Q2", "Q3", "Q4"], fill_value=0)
    corner = {  # (x, y, ha, va)
        "Q1": (4.42, 4.42, "right", "top"),
        "Q2": (4.42, -0.42, "right", "bottom"),
        "Q3": (-0.42, 4.42, "left", "top"),
        "Q4": (-0.42, -0.42, "left", "bottom"),
    }
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        x, y, ha, va = corner[q]
        n = int(counts[q])
        ax.text(x, y, f"{q} · {QUAD_LABEL[q]}\nn = {n}",
                ha=ha, va=va, fontsize=8.5, color="#222222", linespacing=1.1,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888",
                          alpha=0.88, lw=0.7))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Number of skills in cell")
    fig.tight_layout()
    st.save_fig(fig, "08_codif_eval_map_main")
    plt.close(fig)

    # quadrant counts table
    rows = []
    n_total = len(df)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        n = int(counts[q])
        rows.append({"quadrant": q, "quadrant_label": QUAD_LABEL[q],
                     "condition": QUAD_COND[q], "n_skills": n,
                     "share": round(n / n_total, 4)})
    qc = pd.DataFrame(rows)
    qc.to_csv(TBL_DIR / "08_quadrant_counts.csv", index=False)
    return M, qc


# ==========================================================================
# (b) Map with cohort callouts — 3 cohorts on the jittered base
# ==========================================================================
def fig_cohorts(df):
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    draw_frame(ax)

    cohort_ids = set(HIGH_EVAL) | set(AUDIT_FAIL) | set(ARCH_HEAVY)
    base = df[~df["skill_id"].isin(cohort_ids)]
    ax.scatter(base["jx"], base["jy"], s=26, color="#D5D5D5",
               edgecolor="white", linewidth=0.3, zorder=2, label="_nolegend_")

    # architecturally-heaviest (13): hollow indigo squares
    arch = df[df["skill_id"].isin(ARCH_HEAVY)]
    ax.scatter(arch["jx"], arch["jy"], s=70, facecolors="none",
               edgecolors="#7570B3", linewidths=1.6, marker="s", zorder=3)

    # audit-failing (8): red X
    fail = df[df["skill_id"].isin(AUDIT_FAIL)]
    ax.scatter(fail["jx"], fail["jy"], s=95, color="#D1495B",
               marker="X", linewidths=0.6, edgecolor="white", zorder=5)

    # high-evaluability (5 = Q1): gold filled circles, labelled by skill_id
    heval = df[df["skill_id"].isin(HIGH_EVAL)]
    ax.scatter(heval["jx"], heval["jy"], s=120, color="#E6AB02",
               edgecolor="#333333", linewidth=0.8, marker="o", zorder=4)
    # the 5 Q1 skills sit in a tight cluster (4 at codif=3, eval=3; DM09 at
    # codif=4) — fan their labels out radially so they stay readable
    label_off = {"SD09": (-30, 12), "ET09": (-4, 16), "DM09": (12, 10),
                 "SC07": (16, -16), "SM08": (-32, -16)}
    for _, r in heval.iterrows():
        ax.annotate(r["skill_id"], (r["jx"], r["jy"]),
                    textcoords="offset points",
                    xytext=label_off.get(r["skill_id"], (8, 6)),
                    fontsize=8.5, fontweight="bold", color="#5A4500", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#9A7A00", lw=0.6,
                                    shrinkA=0, shrinkB=3))

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor="#E6AB02", markeredgecolor="#333333",
               label="High-evaluability (eval≥3), n=5"),
        Line2D([0], [0], marker="X", linestyle="none", markersize=10,
               markerfacecolor="#D1495B", markeredgecolor="white",
               label="Audit-failing cohort, n=8"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=9,
               markerfacecolor="none", markeredgecolor="#7570B3",
               markeredgewidth=1.6, label="Architecturally-heaviest, n=13"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=7,
               markerfacecolor="#D5D5D5", markeredgecolor="white",
               label="Other skills"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("Cohort callouts on the codifiability × evaluability map\n"
                 "(points jittered within their score cell; SC07 is in two cohorts)")
    fig.tight_layout()
    st.save_fig(fig, "08_codif_eval_map_cohorts")
    plt.close(fig)

    # record cohort placements + intersections for the report
    def _place(ids):
        sub = df[df["skill_id"].isin(ids)]
        return sub["quadrant"].value_counts().reindex(["Q1", "Q2", "Q3", "Q4"],
                                                      fill_value=0).to_dict()
    placements = {"high_eval": _place(HIGH_EVAL), "audit_fail": _place(AUDIT_FAIL),
                  "arch_heavy": _place(ARCH_HEAVY)}
    inter = {
        "high_eval & audit_fail": sorted(set(HIGH_EVAL) & set(AUDIT_FAIL)),
        "high_eval & arch_heavy": sorted(set(HIGH_EVAL) & set(ARCH_HEAVY)),
        "audit_fail & arch_heavy": sorted(set(AUDIT_FAIL) & set(ARCH_HEAVY)),
        "all three": sorted(set(HIGH_EVAL) & set(AUDIT_FAIL) & set(ARCH_HEAVY)),
    }
    return placements, inter


# ==========================================================================
# (c) Map by dominant knowledge type — 7 small-multiple panels
# ==========================================================================
def fig_by_dominant_type(df):
    # panel order: 5 sole-dominant types, then no-dominant, then two-dominant
    panels = []
    for k in KT:
        panels.append((k, st.KNOWLEDGE_TYPE_LABELS[k] + " (sole)",
                       st.KNOWLEDGE_TYPE_COLORS[k]))
    panels.append(("no dominant", "No dominant type", st.NONE_COLOR))
    panels.append(("two dominant", "Two dominant types", "#444444"))

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.8))
    axes = axes.ravel()
    # bottom-most used panel per column gets the x-label (col 3 ends at panel 3);
    # left column gets the y-label — avoids titles colliding with x-labels above
    xlabel_idx = {3, 4, 5, 6}
    ylabel_idx = {0, 4}
    for idx, (ax, (grp, title, color)) in enumerate(zip(axes, panels)):
        draw_frame(ax, xlabel=False, ylabel=False)
        sub = df[df["dom_group"] == grp]
        ax.scatter(sub["jx"], sub["jy"], s=34, color=color,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_title(f"{title}\n(n = {len(sub)})", fontsize=10)
        if idx in xlabel_idx:
            ax.set_xlabel("Codifiability", fontsize=8.5)
        if idx in ylabel_idx:
            ax.set_ylabel("Evaluability", fontsize=8.5)
        ax.tick_params(labelsize=8)
    # hide the unused 8th panel
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Codifiability × evaluability by dominant knowledge type\n"
                 "(groups partition the 120; dashed lines mark the score≥3 threshold)",
                 y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.42)
    st.save_fig(fig, "08_codif_eval_map_by_dominant_type")
    plt.close(fig)

    # quadrant breakdown per group, for the report
    tbl = (df.groupby(["dom_group", "quadrant"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["Q1", "Q2", "Q3", "Q4"], fill_value=0))
    return tbl


# ==========================================================================
# (d) Map by category — 12 small-multiple panels
# ==========================================================================
def fig_by_category(df):
    cats = list(st.CATEGORY_MAPPING["category"])
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 10.2))
    axes = axes.ravel()
    for ax, cat in zip(axes, cats):
        draw_frame(ax)
        sub = df[df["category"] == cat]
        ax.scatter(sub["jx"], sub["jy"], s=46, color=st.CATEGORY_COLORS[cat],
                   edgecolor="white", linewidth=0.5, zorder=3)
        mc = sub["codifiability_score"].mean()
        me = sub["evaluability_score"].mean()
        ax.set_title(f"{st.CATEGORY_LABELS[cat]}  (n={len(sub)})\n"
                     f"mean codif {mc:.1f} · mean eval {me:.1f}", fontsize=9.5)
        ax.set_xlabel("Codifiability", fontsize=8.5)
        ax.set_ylabel("Evaluability", fontsize=8.5)
        ax.tick_params(labelsize=8)

    fig.suptitle("Codifiability × evaluability by category (10 skills each)\n"
                 "(dashed lines mark the score≥3 threshold)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    st.save_fig(fig, "08_codif_eval_map_by_category")
    plt.close(fig)


# ==========================================================================
# (e) Task-horizon overlay — scope = marker shape, coord_burden = marker size
# ==========================================================================
def fig_with_horizon(df):
    fig, ax = plt.subplots(figsize=(7.4, 7.8))
    draw_frame(ax)

    def _size(cb):
        return 30 + cb * 55   # cb 1->85, 2->140, 3->195, 4->250

    for scope, marker in SCOPE_MARKERS.items():
        sub = df[df["scope"] == scope]
        if not len(sub):
            continue
        ax.scatter(sub["jx"], sub["jy"],
                   s=[_size(cb) for cb in sub["coordination_burden"]],
                   marker=marker, color=st.SCOPE_COLORS[scope],
                   edgecolor="#333333", linewidth=0.5, alpha=0.85, zorder=3)

    # two legends BELOW the plot (scope left, burden right) — keeps the full
    # "(n=…)" text on-canvas, which a right-side anchor was clipping
    scope_handles = [
        Line2D([0], [0], marker=SCOPE_MARKERS[s], linestyle="none", markersize=9,
               markerfacecolor=st.SCOPE_COLORS[s], markeredgecolor="#333333",
               label=f"{st.SCOPE_LABELS[s]} (n={int((df['scope'] == s).sum())})")
        for s in SCOPE_MARKERS if (df["scope"] == s).any()
    ]
    leg1 = ax.legend(handles=scope_handles, title="Scope (marker shape)",
                     loc="upper left", bbox_to_anchor=(-0.02, -0.10),
                     frameon=False, fontsize=8.5, title_fontsize=9)
    ax.add_artist(leg1)
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markersize=np.sqrt(_size(cb)) * 0.55,
               markerfacecolor="#BBBBBB", markeredgecolor="#333333",
               label=f"burden {cb} (n={int((df['coordination_burden'] == cb).sum())})")
        for cb in [1, 2, 3, 4]
    ]
    ax.legend(handles=size_handles, title="Coordination burden (size)",
              loc="upper right", bbox_to_anchor=(1.02, -0.10),
              frameon=False, fontsize=8.5, title_fontsize=9)

    ax.set_title("Task-horizon overlay on the codifiability × evaluability map\n"
                 "(shape = scope, size = coordination_burden)")
    fig.tight_layout()
    st.save_fig(fig, "08_codif_eval_map_with_horizon")
    plt.close(fig)


# ==========================================================================
# (f) Governance overlay — two panels (identity/lifecycle, boundary/safety)
# ==========================================================================
def fig_with_governance(df):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    cmap = colormaps[SEQ_CMAP]
    norm = mcolors.Normalize(vmin=0, vmax=4)

    panel_specs = [
        ("identity_lifecycle", "Identity / lifecycle governance\n"
         "(mean of provenance_clarity & maintenance_signals)"),
        ("boundary_safety", "Boundary / safety governance\n"
         "(mean of safety_notes & scope_limits)"),
    ]
    sc = None
    for ax, (col, title) in zip(axes, panel_specs):
        draw_frame(ax)
        sc = ax.scatter(df["jx"], df["jy"], c=df[col], cmap=cmap, norm=norm,
                        s=46, edgecolor="#333333", linewidth=0.4, zorder=3)
        ax.set_title(title, fontsize=10.5)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Governance-pair mean intensity (0–4)")
    fig.suptitle("Governance overlay on the codifiability × evaluability map\n"
                 "(Section 7's two independent governance pairs; viridis 0–4)", y=1.0)
    st.save_fig(fig, "08_codif_eval_map_with_governance")
    plt.close(fig)


# ==========================================================================
# (g) Quadrant interpretation table
# ==========================================================================
def quadrant_analysis_table(df):
    n_total = len(df)
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[df["quadrant"] == q]
        n = len(sub)
        rec = {
            "quadrant": q,
            "quadrant_label": QUAD_LABEL[q],
            "n_skills": n,
            "share": round(n / n_total, 4),
        }
        if n == 0:
            # empty quadrant: leave numeric fields blank, note explicitly
            for f in ["mean_codifiability", "mean_evaluability",
                      "top3_categories", "mean_coordination_burden",
                      "mean_scope_ordinal", "mean_governance_total",
                      "mean_identity_lifecycle", "mean_boundary_safety"]:
                rec[f] = ""
            for k in KT:
                rec[f"n_{k}_dom"] = 0
            rec["n_no_dominant"] = 0
            rec["n_two_dominant"] = 0
            rows.append(rec)
            continue

        rec["mean_codifiability"] = round(float(sub["codifiability_score"].mean()), 3)
        rec["mean_evaluability"] = round(float(sub["evaluability_score"].mean()), 3)
        top3 = sub["category"].value_counts().head(3)
        rec["top3_categories"] = "; ".join(
            f"{st.CATEGORY_LABELS[c]} ({n_})" for c, n_ in top3.items())
        # dominant-type breakdown
        for k in KT:
            rec[f"n_{k}_dom"] = int((sub["dom_group"] == k).sum())
        rec["n_no_dominant"] = int((sub["dom_group"] == "no dominant").sum())
        rec["n_two_dominant"] = int((sub["dom_group"] == "two dominant").sum())
        rec["mean_coordination_burden"] = round(float(sub["coordination_burden"].mean()), 3)
        rec["mean_scope_ordinal"] = round(float(sub["scope_ord"].mean(skipna=True)), 3)
        rec["mean_governance_total"] = round(float(sub["governance_total"].mean()), 3)
        rec["mean_identity_lifecycle"] = round(float(sub["identity_lifecycle"].mean()), 3)
        rec["mean_boundary_safety"] = round(float(sub["boundary_safety"].mean()), 3)
        rows.append(rec)

    col_order = (["quadrant", "quadrant_label", "n_skills", "share",
                  "mean_codifiability", "mean_evaluability", "top3_categories"]
                 + [f"n_{k}_dom" for k in KT] + ["n_no_dominant", "n_two_dominant"]
                 + ["mean_coordination_burden", "mean_scope_ordinal",
                    "mean_governance_total", "mean_identity_lifecycle",
                    "mean_boundary_safety"])
    qa = pd.DataFrame(rows)[col_order]
    qa.to_csv(TBL_DIR / "08_quadrant_analysis.csv", index=False)
    return qa


# ==========================================================================
# (h) The high/high corner (Q1) named
# ==========================================================================
def q1_skills_table(df):
    q1 = df[df["quadrant"] == "Q1"].copy()
    cols = ["skill_id", "skill_name", "category", "codifiability_score",
            "evaluability_score", "evaluation_type", "tests_quality",
            "thresholds_quality", "output_verifiability", "error_handling",
            "scope", "coordination_burden", "dom_group",
            "provenance_clarity", "maintenance_signals", "safety_notes",
            "scope_limits", "snyk_status", "socket_status",
            "agent_trust_hub_status"]
    out = q1[cols].sort_values(["codifiability_score", "skill_id"],
                               ascending=[False, True]).reset_index(drop=True)
    out.to_csv(TBL_DIR / "08_q1_skills_named.csv", index=False)
    return out


# ==========================================================================
# main
# ==========================================================================
def main():
    st.apply_style()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare()

    print("=" * 72)
    print("SECTION 8 — CODIFIABILITY × EVALUABILITY MAP (centerpiece)")
    print("=" * 72)
    print("  anomaly guard: PASS (no skill at evaluability_score==4)")
    print(f"  threshold: score >= {HIGH} = 'high' on each axis (methodology 3.1)")

    M, qc = fig_main_map(df)
    print("\n[a] figures/08_codif_eval_map_main.{png,svg} "
          "+ tables/08_quadrant_counts.csv")
    print("    5x5 joint counts (rows=eval 0..4 top-down shown low..high; cols=codif 0..4):")
    print(pd.DataFrame(M, index=[f"eval{e}" for e in INTENSITIES],
                       columns=[f"codif{c}" for c in INTENSITIES]).to_string())
    print(qc.to_string(index=False))

    placements, inter = fig_cohorts(df)
    print("\n[b] figures/08_codif_eval_map_cohorts.{png,svg}")
    print("    cohort quadrant placements:")
    for name, p in placements.items():
        print(f"      {name:<12} {p}")
    print("    cohort intersections:")
    for name, members in inter.items():
        print(f"      {name:<26} {members}")

    domtbl = fig_by_dominant_type(df)
    print("\n[c] figures/08_codif_eval_map_by_dominant_type.{png,svg}")
    print(domtbl.to_string())

    fig_by_category(df)
    print("\n[d] figures/08_codif_eval_map_by_category.{png,svg}")
    cat_means = (df.groupby("category")[["codifiability_score", "evaluability_score"]]
                   .mean().round(2))
    print(cat_means.to_string())

    fig_with_horizon(df)
    print("\n[e] figures/08_codif_eval_map_with_horizon.{png,svg}")
    print("    mean codif/eval by scope:")
    print(df.groupby("scope")[["codifiability_score", "evaluability_score",
                               "coordination_burden"]].mean().round(2).to_string())

    fig_with_governance(df)
    print("\n[f] figures/08_codif_eval_map_with_governance.{png,svg}")
    print("    mean governance-pair by quadrant:")
    print(df.groupby("quadrant")[["identity_lifecycle", "boundary_safety"]]
            .mean().round(2).to_string())

    qa = quadrant_analysis_table(df)
    print("\n[g] tables/08_quadrant_analysis.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(qa.to_string(index=False))

    q1 = q1_skills_table(df)
    print("\n[h] tables/08_q1_skills_named.csv")
    print(q1[["skill_id", "skill_name", "category", "codifiability_score",
              "evaluability_score", "evaluation_type", "tests_quality",
              "scope", "coordination_burden"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
