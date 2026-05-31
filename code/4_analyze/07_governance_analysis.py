#!/usr/bin/env python3
"""
07_governance_analysis.py — Section 7 of the findings chapter: governance signals
across the 120-skill coded corpus.

The coding frame measures GOVERNANCE through four artefactual variables scored
0-4 (all push the SAME direction: higher = better governance, so — like
evaluability in Section 5 — no inversion handling is needed anywhere here):

    provenance_clarity   — clarity of authorship, versioning, publisher identity
    maintenance_signals  — visible evidence of ongoing maintenance (changelogs,
                           dates, version markers)
    safety_notes         — presence of safety / limitation / trust-boundary notes
    scope_limits         — clarity of OUT-OF-SCOPE boundaries and escalation rules
                           (a governance property, NOT codification: methodology
                           3.3.5; in-scope clarity is explicitness, Section 4)

Plus three externally-sourced AUDIT statuses (categorical pass/warn/fail/missing,
lowercased per the population-phase convention) — coded by external auditors
OUTSIDE the skill artefact, so they are independent of the four artefactual
variables above:

    snyk_status / socket_status / agent_trust_hub_status

Self-contained: reads ONLY from analysis/data/ (corpus.parquet + population.parquet
for the audit comparison), imports the locked visual style from _style.py, writes
figures to analysis/figures/ (.png + .svg) and tables to analysis/tables/ (.csv).
Descriptive only — no structural inference.

Run from anywhere:
    python3 scripts/07_governance_analysis.py
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

INTENSITIES = [0, 1, 2, 3, 4]

# The four artefactual governance variables, fixed canonical order. All push the
# same direction (higher = better governance) — no inversion handling needed.
GOV_VARS = ["provenance_clarity", "maintenance_signals", "safety_notes", "scope_limits"]
GOV_LABELS = {
    "provenance_clarity":  "Provenance clarity",
    "maintenance_signals": "Maintenance signals",
    "safety_notes":        "Safety notes",
    "scope_limits":        "Scope limits",
}
# Derived: mean of the four governance variables (kept on the same 0-4 scale).
GOV_TOTAL = "governance_total"

# The three external audit statuses (categorical), fixed canonical order.
AUDIT_COLS = ["snyk_status", "socket_status", "agent_trust_hub_status"]
AUDIT_LABELS = {
    "snyk_status":            "Snyk",
    "socket_status":          "Socket",
    "agent_trust_hub_status": "Agent Trust Hub",
}

# Variables on the 7x7 correlation matrix (4 gov + governance_total + the two
# derived scores). Audit statuses are categorical and are EXCLUDED here.
CORR_VARS = GOV_VARS + [GOV_TOTAL, "codifiability_score", "evaluability_score"]
CORR_LABELS = {**GOV_LABELS,
               GOV_TOTAL:             "Governance total",
               "codifiability_score": "Codifiability score",
               "evaluability_score":  "Evaluability score"}

# Scope encoding for part (i). NOTE: this is the Section-7-specific 1/2/3 encoding
# the brief specifies (single_bounded_task=1, multi_step_workflow=2,
# end_to_end_process=3; unclear excluded). The corpus has NO micro_task skill, so
# the three present substantive values map to 1/2/3.
SCOPE_ENC_07 = {
    "single_bounded_task": 1,
    "multi_step_workflow": 2,
    "end_to_end_process":  3,
}

# The five knowledge types (for the by-dominant-type breakdown, Sections 3-6).
KT = ["procedural", "analytical", "orchestration", "compliance", "documentation"]

SEQ_CMAP = "viridis"        # same sequential map as Sections 3-6 heatmaps
DIV_CMAP = "RdBu"           # same diverging map as Sections 4-5 correlation figs


# ==========================================================================
# Helpers
# ==========================================================================
def fmt_int(x):
    return f"{int(round(x)):,}"


def audit_proportions(series):
    """Map an audit-status series to counts in the fixed AUDIT_ORDER, treating
    null as 'missing' (corpus already stores 'missing' as a literal string; the
    population stores nulls — fillna handles both)."""
    s = series.fillna("missing").astype(str).str.lower()
    return {k: int((s == k).sum()) for k in st.AUDIT_ORDER}


# ==========================================================================
# Load + integrity / anomaly guards
# ==========================================================================
def load_corpus():
    df = pd.read_parquet(DATA_DIR / "corpus.parquet")

    # The four governance variables must be clean 0-4 integer intensities.
    for v in GOV_VARS:
        if df[v].isna().any():
            raise RuntimeError(f"null values in {v}")
        if not df[v].between(0, 4).all():
            raise RuntimeError(f"{v} has values outside 0-4")

    # Audit statuses must be one of the four allowed values (no surprise labels).
    allowed = set(st.AUDIT_ORDER)
    for a in AUDIT_COLS:
        s = df[a].fillna("missing").astype(str).str.lower()
        bad = set(s.unique()) - allowed
        if bad:
            raise RuntimeError(f"ANOMALY: unexpected {a} value(s): {bad}")

    # Derived scores needed for parts (f) and (g).
    for c in ["codifiability_score", "evaluability_score"]:
        if df[c].isna().any():
            raise RuntimeError(f"null values in {c}")

    # Derived governance_total = mean of the four governance variables (0-4).
    df[GOV_TOTAL] = df[GOV_VARS].mean(axis=1)

    # ----------------------------------------------------------------------
    # ANOMALY GUARD. The brief flags the contradictory case: a skill that FAILS
    # all three external auditors yet carries high artefactual governance (or
    # vice versa). The only logically extreme audit case is failing all three —
    # stop loudly if any exists so it is surfaced, not smoothed.
    # ----------------------------------------------------------------------
    fail_all = ((df["snyk_status"] == "fail")
                & (df["socket_status"] == "fail")
                & (df["agent_trust_hub_status"] == "fail"))
    if fail_all.any():
        ids = ", ".join(df.loc[fail_all, "skill_id"].tolist())
        raise RuntimeError(
            f"ANOMALY: {int(fail_all.sum())} skill(s) FAIL all three auditors: {ids} "
            "— inspect their governance scores before proceeding."
        )
    return df


# ==========================================================================
# (a) Governance summary table (4 governance ordinals + 3 audit categoricals)
# ==========================================================================
def summary_table(df):
    cols = ["variable", "kind", "mean", "median", "std",
            "n_at_0", "n_at_1", "n_at_2", "n_at_3", "n_at_4", "n_at_3_or_higher",
            "n_pass", "n_warn", "n_fail", "n_missing", "n_not_pass"]
    rows = []

    # --- the four governance ordinals: full 0-4 intensity stats ---
    for v in GOV_VARS:
        s = df[v]
        vc = s.value_counts().reindex(INTENSITIES, fill_value=0)
        rows.append({
            "variable": v, "kind": "governance_ordinal",
            "mean": round(float(s.mean()), 3),
            "median": float(s.median()),
            "std": round(float(s.std()), 3),
            "n_at_0": int(vc[0]), "n_at_1": int(vc[1]), "n_at_2": int(vc[2]),
            "n_at_3": int(vc[3]), "n_at_4": int(vc[4]),
            "n_at_3_or_higher": int((s >= 3).sum()),
            "n_pass": "", "n_warn": "", "n_fail": "", "n_missing": "",
            "n_not_pass": "",
        })

    # --- the three audit statuses: categorical pass/warn/fail/missing counts ---
    for a in AUDIT_COLS:
        c = audit_proportions(df[a])
        rows.append({
            "variable": a, "kind": "audit_status",
            "mean": "", "median": "", "std": "",
            "n_at_0": "", "n_at_1": "", "n_at_2": "", "n_at_3": "", "n_at_4": "",
            "n_at_3_or_higher": "",
            "n_pass": c["pass"], "n_warn": c["warn"], "n_fail": c["fail"],
            "n_missing": c["missing"],
            "n_not_pass": int(c["warn"] + c["fail"] + c["missing"]),
        })

    summ = pd.DataFrame(rows)[cols]
    summ.to_csv(TBL_DIR / "07_governance_summary.csv", index=False)
    return summ


# ==========================================================================
# (b) Governance variable distributions — 4 panels, shared y, single colour
# ==========================================================================
def fig_governance_distributions(df):
    n = len(df)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharey=True)
    axes = axes.ravel()
    ymax = max(int(df[v].value_counts().reindex(INTENSITIES, fill_value=0).max())
               for v in GOV_VARS)
    ytop = int(np.ceil((ymax + 12) / 10.0) * 10)

    for ax, v in zip(axes, GOV_VARS):
        vc = df[v].value_counts().reindex(INTENSITIES, fill_value=0)
        mean = df[v].mean()
        share0 = (df[v] == 0).mean()
        ax.bar(INTENSITIES, vc.values, color=st.NEUTRAL_ACCENT,
               edgecolor="white", linewidth=0.6, width=0.78)
        for x, val in zip(INTENSITIES, vc.values):
            if val > 0:
                ax.text(x, val + ytop * 0.012, str(int(val)), ha="center",
                        va="bottom", fontsize=8.5, color="#333333")
        ax.set_xticks(INTENSITIES)
        ax.set_ylim(0, ytop)
        ax.set_xlabel("Intensity (0–4; higher = better governance)")
        ax.set_title(f"{GOV_LABELS[v]}\n(mean = {mean:.2f}; share@0 = {share0 * 100:.1f}%)",
                     fontsize=11)
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Number of skills")
    axes[2].set_ylabel("Number of skills")

    fig.suptitle("Governance variable distributions across the 120-skill corpus", y=1.0)
    fig.tight_layout()
    st.save_fig(fig, "07_governance_distributions")
    plt.close(fig)


# ==========================================================================
# (c) Audit status within the corpus — stacked horizontal bar per auditor
# ==========================================================================
def fig_audit_status_corpus(df):
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y_positions = list(range(len(AUDIT_COLS)))[::-1]   # first auditor on top

    for col, y in zip(AUDIT_COLS, y_positions):
        counts = audit_proportions(df[col])
        left = 0.0
        for status in st.AUDIT_ORDER:
            c = counts[status]
            pct = c / n * 100
            ax.barh(y, pct, left=left, color=st.AUDIT_COLORS[status],
                    edgecolor="white", linewidth=0.6, height=0.62)
            if pct >= 6:
                txt_color = "white" if status in ("pass", "fail") else "#333333"
                ax.text(left + pct / 2, y, f"{status}\n{c}",
                        ha="center", va="center", fontsize=8,
                        color=txt_color, linespacing=0.95)
            left += pct

    ax.set_yticks(y_positions)
    ax.set_yticklabels([AUDIT_LABELS[c] for c in AUDIT_COLS])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of corpus (%)")
    ax.set_title("External audit status within the 120-skill corpus")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=st.AUDIT_COLORS[s])
               for s in st.AUDIT_ORDER]
    ax.legend(handles, st.AUDIT_ORDER, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), ncol=4)
    fig.tight_layout()
    st.save_fig(fig, "07_audit_status_corpus")
    plt.close(fig)


def audit_comparison(df):
    """Corpus vs population audit rates, for the report. Returns a tidy frame plus
    the two 'priority' rates tied to the sampling rule (Snyk+Socket Pass)."""
    pop = pd.read_parquet(DATA_DIR / "population.parquet")

    def _rates(frame):
        n = len(frame)
        out = {}
        for a in AUDIT_COLS:
            c = audit_proportions(frame[a])
            out[a] = {k: c[k] / n for k in st.AUDIT_ORDER}
        # pass on >=1 of the three auditors ("pass-anywhere")
        passmat = pd.DataFrame(
            {a: frame[a].fillna("missing").astype(str).str.lower() == "pass"
             for a in AUDIT_COLS})
        out["pass_anywhere"] = float(passmat.any(axis=1).mean())
        # Snyk Pass AND Socket Pass (the methodology's selection priority)
        snyk_p = frame["snyk_status"].fillna("missing").astype(str).str.lower() == "pass"
        sock_p = frame["socket_status"].fillna("missing").astype(str).str.lower() == "pass"
        out["snyk_and_socket_pass"] = float((snyk_p & sock_p).mean())
        return out, n

    corp_rates, n_corp = _rates(df)
    pop_rates, n_pop = _rates(pop)

    rows = []
    for a in AUDIT_COLS:
        for status in st.AUDIT_ORDER:
            rows.append({
                "metric": f"{a}:{status}",
                "corpus_share": round(corp_rates[a][status], 4),
                "population_share": round(pop_rates[a][status], 4),
            })
    for key in ["pass_anywhere", "snyk_and_socket_pass"]:
        rows.append({
            "metric": key,
            "corpus_share": round(corp_rates[key], 4),
            "population_share": round(pop_rates[key], 4),
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(TBL_DIR / "07_audit_corpus_vs_population.csv", index=False)
    return comp, corp_rates, pop_rates, n_pop


# ==========================================================================
# (d) Audit status by category (within corpus) — 3-panel small multiples
# ==========================================================================
def fig_audit_status_by_category(df):
    cm = st.CATEGORY_MAPPING
    cats = list(cm["category"])                          # alphabetical (as fig 02e)
    labels = [st.CATEGORY_LABELS[c] for c in cats]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), sharey=True)
    y_positions = list(range(len(cats)))[::-1]

    for ax, col in zip(axes, AUDIT_COLS):
        for cat, y in zip(cats, y_positions):
            sub = df.loc[df["category"] == cat]
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
    fig.suptitle("External audit status by category (120-skill corpus)", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    st.save_fig(fig, "07_audit_status_corpus_by_category")
    plt.close(fig)


# ==========================================================================
# (e) Governance by category — 12 x 4 heatmap of mean governance var
# ==========================================================================
def governance_by_category(df):
    cm = st.CATEGORY_MAPPING
    cats = cm["category"].tolist()

    rows = []
    for cat in cats:
        sub = df.loc[df["category"] == cat]
        rec = {"category": cat, "display_label": st.CATEGORY_LABELS[cat], "n": len(sub)}
        for v in GOV_VARS:
            rec[f"{v}_mean"] = round(float(sub[v].mean()), 4)
        rec[f"{GOV_TOTAL}_mean"] = round(float(sub[GOV_TOTAL].mean()), 4)
        rows.append(rec)
    bycat = pd.DataFrame(rows)
    bycat.to_csv(TBL_DIR / "07_governance_by_category.csv", index=False)

    # ---- heatmap: rows = categories (cm order), cols = the 4 gov vars ----
    mat = np.array([[bycat.loc[bycat["category"] == c, f"{v}_mean"].iloc[0]
                     for v in GOV_VARS] for c in cats], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    cmap = colormaps[SEQ_CMAP]
    norm = mcolors.Normalize(vmin=0, vmax=4)   # all four oriented brighter = better
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(GOV_VARS)))
    ax.set_xticklabels([GOV_LABELS[v] for v in GOV_VARS], rotation=20, ha="right",
                       fontsize=9)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([st.CATEGORY_LABELS[c] for c in cats], fontsize=9)
    ax.set_title("Mean governance variable by category\n"
                 "(viridis 0–4; brighter = better governance on all four)")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(GOV_VARS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(cats)):
        for j in range(len(GOV_VARS)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                    color=st.text_color_for_bg(cmap(norm(v))))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean intensity (0–4)")
    fig.tight_layout()
    st.save_fig(fig, "07_governance_by_category")
    plt.close(fig)
    return bycat


# ==========================================================================
# (f) Governance correlation matrix — 7 x 7 Pearson
# ==========================================================================
def fig_correlations(df):
    corr = df[CORR_VARS].corr(method="pearson")
    corr.round(4).to_csv(TBL_DIR / "07_governance_correlations.csv")

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    cmap = colormaps[DIV_CMAP]
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(CORR_VARS)))
    ax.set_xticklabels([CORR_LABELS[v] for v in CORR_VARS], rotation=30, ha="right",
                       fontsize=8.5)
    ax.set_yticks(range(len(CORR_VARS)))
    ax.set_yticklabels([CORR_LABELS[v] for v in CORR_VARS], fontsize=8.5)
    ax.set_title("Governance, codifiability & evaluability correlations\n"
                 "(Pearson r across the 120 skills; governance_total = mean of the four)")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(CORR_VARS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CORR_VARS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(CORR_VARS)):
        for j in range(len(CORR_VARS)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                    color=st.text_color_for_bg(cmap(norm(val))))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    fig.tight_layout()
    st.save_fig(fig, "07_governance_correlations")
    plt.close(fig)
    return corr


# ==========================================================================
# (g) Audit-failing cohort — any 'fail' across the three auditors
# ==========================================================================
def audit_failing_cohort(df):
    fail = ((df["snyk_status"] == "fail")
            | (df["socket_status"] == "fail")
            | (df["agent_trust_hub_status"] == "fail"))
    cohort = df.loc[fail].copy()

    cols = (["skill_id", "category"] + AUDIT_COLS + GOV_VARS
            + ["codifiability_score", "evaluability_score"])
    cohort_out = cohort[cols].sort_values("skill_id").reset_index(drop=True)
    cohort_out.to_csv(TBL_DIR / "07_audit_failing_cohort.csv", index=False)

    rest = df.loc[~fail]
    means = pd.DataFrame({
        "variable": GOV_VARS,
        "cohort_mean": [round(float(cohort[v].mean()), 3) for v in GOV_VARS],
        "rest_mean": [round(float(rest[v].mean()), 3) for v in GOV_VARS],
    })
    cat_dist = cohort["category"].value_counts()
    return cohort_out, means, cat_dist, int(fail.sum())


# ==========================================================================
# (h) Governance by dominant knowledge type
# ==========================================================================
def by_dominant_type(df):
    """Partition the 120 skills by dominant (==4) knowledge type, with the same
    single/none/two split as Sections 3-6 so the n's reconcile to 120. For each
    group: the mean of each of the four governance variables."""
    ndom = sum((df[k] == 4).astype(int) for k in KT)

    def _row(label, mask):
        sub = df.loc[mask]
        rec = {"dominant_group": label, "n": int(mask.sum())}
        for v in GOV_VARS:
            rec[f"{v}_mean"] = round(float(sub[v].mean()), 3)
        rec[f"{GOV_TOTAL}_mean"] = round(float(sub[GOV_TOTAL].mean()), 3)
        return rec

    rows = []
    for k in KT:
        mask = (df[k] == 4) & (ndom == 1)
        if mask.sum() > 0:
            rows.append(_row(f"{k} (sole dominant)", mask))
    rows.append(_row("no dominant type", ndom == 0))
    rows.append(_row("two dominant types", ndom == 2))
    tbl = pd.DataFrame(rows).sort_values(
        f"{GOV_TOTAL}_mean", ascending=False).reset_index(drop=True)
    tbl.to_csv(TBL_DIR / "07_governance_by_dominant_type.csv", index=False)

    # Figure only if the spread of group means on ANY single governance var > 0.3.
    spreads = {v: float(tbl[f"{v}_mean"].max() - tbl[f"{v}_mean"].min()) for v in GOV_VARS}
    max_spread = max(spreads.values())
    made_fig = max_spread > 0.3
    if made_fig:
        fig, ax = plt.subplots(figsize=(10.0, 5.4))
        groups = tbl["dominant_group"].tolist()
        ypos = np.arange(len(groups))[::-1]
        h = 0.8 / len(GOV_VARS)
        # one grouped bar per governance variable, coloured by a viridis sample
        bar_cols = [colormaps[SEQ_CMAP](x) for x in np.linspace(0.12, 0.85, len(GOV_VARS))]
        for gi, v in enumerate(GOV_VARS):
            offs = (gi - (len(GOV_VARS) - 1) / 2) * h
            ax.barh(ypos + offs, tbl[f"{v}_mean"].values, height=h,
                    color=bar_cols[gi], edgecolor="white", linewidth=0.4,
                    label=GOV_LABELS[v])
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{g}\n(n={n})" for g, n in zip(groups, tbl['n'])],
                           fontsize=8.5)
        ax.set_xlabel("Mean intensity (0–4; higher = better governance)")
        ax.set_xlim(0, 4.0)
        ax.set_title("Mean governance variables by dominant knowledge type\n"
                     "(groups partition the 120; sole-dominant per type, plus no-/two-dominant)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4,
                  frameon=False, fontsize=8.5)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        st.save_fig(fig, "07_governance_by_dominant_type")
        plt.close(fig)
    return tbl, spreads, max_spread, made_fig


# ==========================================================================
# (i) Governance by scope — Pearson corr with ordinal scope + total by scope
# ==========================================================================
def governance_by_scope(df):
    enc = df["scope"].map(SCOPE_ENC_07)         # unclear / micro -> NaN (excluded)
    mask = enc.notna()
    n_used = int(mask.sum())
    n_excl = int((~mask).sum())

    corr_rows = []
    for v in GOV_VARS + [GOV_TOTAL]:
        r = float(np.corrcoef(enc[mask], df.loc[mask, v])[0, 1])
        corr_rows.append({"variable": v, "pearson_r_with_scope": round(r, 4)})
    corr_tbl = pd.DataFrame(corr_rows)

    # mean governance_total by scope (one row per present substantive scope value)
    tot_rows = []
    for s, code in sorted(SCOPE_ENC_07.items(), key=lambda kv: kv[1]):
        sub = df.loc[df["scope"] == s]
        tot_rows.append({"scope": s, "scope_code": code, "n": len(sub),
                         "mean_governance_total": round(float(sub[GOV_TOTAL].mean()), 3)
                         if len(sub) else ""})
    tot_tbl = pd.DataFrame(tot_rows)
    return corr_tbl, tot_tbl, n_used, n_excl


# ==========================================================================
# main
# ==========================================================================
def main():
    st.apply_style()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    st.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus()

    print("=" * 72)
    print("SECTION 7 — GOVERNANCE SIGNALS")
    print("=" * 72)
    print("  anomaly guard: PASS (no skill fails all three external auditors)")

    summ = summary_table(df)
    print("\n[a] tables/07_governance_summary.csv")
    print(summ.to_string(index=False))

    fig_governance_distributions(df)
    print("\n[b] figures/07_governance_distributions.{png,svg}")
    for v in GOV_VARS:
        vc = df[v].value_counts().reindex(INTENSITIES, fill_value=0)
        print(f"    {v:<20} mean={df[v].mean():.3f} share@0={(df[v]==0).mean()*100:.1f}%  "
              + ", ".join(f"{i}:{int(vc[i])}" for i in INTENSITIES))

    fig_audit_status_corpus(df)
    comp, corp_rates, pop_rates, n_pop = audit_comparison(df)
    print("\n[c] figures/07_audit_status_corpus.{png,svg} "
          "+ tables/07_audit_corpus_vs_population.csv")
    print(f"    population n = {n_pop}")
    print("    per-auditor PASS share (corpus vs population):")
    for a in AUDIT_COLS:
        print(f"      {AUDIT_LABELS[a]:<16} corpus={corp_rates[a]['pass']*100:5.1f}%  "
              f"population={pop_rates[a]['pass']*100:5.1f}%")
    print(f"    pass-anywhere (>=1 auditor): corpus={corp_rates['pass_anywhere']*100:.1f}%  "
          f"population={pop_rates['pass_anywhere']*100:.1f}%")
    print(f"    Snyk&Socket both pass (selection rule): "
          f"corpus={corp_rates['snyk_and_socket_pass']*100:.1f}%  "
          f"population={pop_rates['snyk_and_socket_pass']*100:.1f}%")

    fig_audit_status_by_category(df)
    print("\n[d] figures/07_audit_status_corpus_by_category.{png,svg}")

    bycat = governance_by_category(df)
    print("\n[e] figures/07_governance_by_category.{png,svg} "
          "+ tables/07_governance_by_category.csv")
    print(bycat[["display_label"] + [f"{v}_mean" for v in GOV_VARS]
                + [f"{GOV_TOTAL}_mean"]].to_string(index=False))

    corr = fig_correlations(df)
    print("\n[f] figures/07_governance_correlations.{png,svg} "
          "+ tables/07_governance_correlations.csv")
    print(corr.round(3).to_string())

    cohort_out, means, cat_dist, n_fail = audit_failing_cohort(df)
    print("\n[g] tables/07_audit_failing_cohort.csv")
    print(f"    audit-failing cohort (ANY fail across 3 auditors): n = {n_fail}")
    print(cohort_out.to_string(index=False))
    print("    category distribution:", dict(cat_dist))
    print("    governance means — cohort vs rest:")
    print(means.to_string(index=False))

    domtbl, spreads, max_spread, made_fig = by_dominant_type(df)
    print("\n[h] tables/07_governance_by_dominant_type.csv")
    print("    per-variable spread of group means: "
          + ", ".join(f"{v}={spreads[v]:.2f}" for v in GOV_VARS))
    print(f"    max single-variable spread = {max_spread:.3f} "
          f"({'>0.3 -> figure produced' if made_fig else '<=0.3 -> figure skipped'})")
    print(domtbl.to_string(index=False))

    corr_tbl, tot_tbl, n_used, n_excl = governance_by_scope(df)
    print("\n[i] governance by scope (inline; no figure)")
    print(f"    Pearson r with ordinal scope (single=1/multi=2/end2end=3; "
          f"n={n_used}, excluded={n_excl}):")
    print(corr_tbl.to_string(index=False))
    print("    mean governance_total by scope:")
    print(tot_tbl.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
