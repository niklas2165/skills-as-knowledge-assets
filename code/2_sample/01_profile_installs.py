"""01 - Install-count distribution.

Uses the exact integer `installs` from the leaderboard API (NOT the abbreviated
K/M detail-page string) for all numeric analysis.

Outputs:
- plots/01_installs_hist_log.png  (log-scaled histogram)
- plots/01_installs_lorenz.png    (Lorenz / cumulative curve + Gini)
- sections/01_installs.md / .json

Run:  python3 scripts/01_profile_installs.py
"""
from __future__ import annotations

import numpy as np

import _lib as L

NUM, TOPIC = "01", "installs"


def main() -> None:
    _, skills = L.load()
    installs = np.array([r["installs"] for r in skills], dtype=float)
    n = len(installs)

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    pvals = {p: float(np.percentile(installs, p)) for p in pcts}
    mean = float(installs.mean())
    median = float(np.median(installs))
    g = L.gini(installs.tolist())

    # Concentration: share of all installs held by the top X% of skills.
    desc = np.sort(installs)[::-1]
    total = desc.sum()
    top_share = {}
    for frac in (0.01, 0.05, 0.10, 0.25):
        k = max(1, int(round(frac * n)))
        top_share[frac] = float(desc[:k].sum() / total)

    # Tier counts
    tier_counts = {lbl: 0 for lbl, _, _ in L.INSTALL_TIERS}
    for v in installs:
        tier_counts[L.tier_of(int(v))] += 1

    # Ties: install counts are not independent across skills (same-repo skills
    # tend to share similar counts). Quantify with distinct values / tied share.
    from collections import Counter
    vc = Counter(int(v) for v in installs)
    distinct_vals = len(vc)
    tied_skills = sum(c for c in vc.values() if c > 1)

    # ------------------------------------------------------------------ plots
    # Histogram, log-spaced bins on a log x-axis.
    fig, ax = L.plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(installs.min()), np.log10(installs.max()), 40)
    ax.hist(installs, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Install count (log scale, exact integer)")
    ax.set_ylabel("Number of skills")
    ax.set_title(f"Install-count distribution (n={n:,}; 40 log-spaced bins)")
    ax.axvline(median, color="#C44E52", ls="--", lw=1.2, label=f"median {median:,.0f}")
    ax.axvline(mean, color="#55A868", ls=":", lw=1.4, label=f"mean {mean:,.0f}")
    ax.legend()
    hist_path = L.save_fig(fig, f"{NUM}_installs_hist_log")

    # Lorenz curve
    pop, val = L.lorenz_points(installs.tolist())
    fig, ax = L.plt.subplots(figsize=(6.5, 6.5))
    ax.plot(pop, val, color="#4C72B0", lw=2, label="Lorenz curve")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="equality")
    ax.fill_between(pop, val, pop, color="#4C72B0", alpha=0.12)
    ax.set_xlabel("Cumulative share of skills (ascending by installs)")
    ax.set_ylabel("Cumulative share of installs")
    ax.set_title(f"Lorenz curve of installs — Gini = {g:.3f}")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    lorenz_path = L.save_fig(fig, f"{NUM}_installs_lorenz")

    # ------------------------------------------------------------------ md
    md = [f"## {NUM} — Install-count distribution", ""]
    md.append("Based on the exact integer `installs` (leaderboard API). The abbreviated "
              "`installs_abbreviated` string is not used for numeric analysis.")
    md += ["", "### Summary statistics", ""]
    md.append(L.md_table(
        ["Statistic", "Installs"],
        [["count", L.fmt_int(n)],
         ["min", L.fmt_int(installs.min())],
         ["mean", L.fmt_int(round(mean))],
         ["median", L.fmt_int(round(median))],
         ["max", L.fmt_int(installs.max())],
         ["std", L.fmt_int(round(installs.std()))],
         ["Gini", f"{g:.3f}"]]))
    md += ["", "### Percentiles", ""]
    md.append(L.md_table(["Percentile", "Installs"],
                         [[f"p{p:g}", L.fmt_int(round(pvals[p]))] for p in pcts]))
    md += ["", "### Concentration (share of all installs held by the top X% of skills)", ""]
    md.append(L.md_table(["Top X% of skills", "Share of all installs"],
                         [[L.fmt_pct(f, 0), L.fmt_pct(top_share[f])] for f in (0.01, 0.05, 0.10, 0.25)]))
    md += ["", "### Skills per install tier (decade bins)", ""]
    md.append(L.md_table(["Tier", "Skills", "Share"],
                         [[lbl, L.fmt_int(tier_counts[lbl]), L.fmt_pct(tier_counts[lbl] / n)]
                          for lbl, _, _ in L.INSTALL_TIERS]))
    md += ["", f"![Install histogram]({hist_path})", "",
           f"![Lorenz curve]({lorenz_path})", ""]
    md.append(f"The mean ({mean:,.0f}) sits above the p{_pct_rank(installs, mean):.0f} "
              "percentile, i.e. the distribution is heavily right-skewed: most skills "
              f"cluster near the floor (median {median:,.0f}) while a few dominate the totals.")
    md.append("")
    md.append(f"Install counts are also heavily **tied**: only {L.fmt_int(distinct_vals)} "
              f"distinct values across {L.fmt_int(n)} skills, and {L.fmt_int(tied_skills)} "
              f"skills ({L.fmt_pct(tied_skills / n)}) share their exact count with at least "
              "one other skill. The secondary bump visible near ~10^4 in the histogram is "
              "dominated by a few large repos whose skills cluster at similar install levels "
              "(see §02). Install count is therefore **not independent across skills** and is "
              "partly collinear with repo — a flag for any install-tier stratification, "
              "stated here without resolving it.")

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "n": n, "min": int(installs.min()), "max": int(installs.max()),
        "mean": mean, "median": median, "std": float(installs.std()),
        "gini": g, "percentiles": pvals, "top_share": top_share,
        "tier_counts": tier_counts,
        "distinct_install_values": distinct_vals, "tied_skills": tied_skills,
        "tied_share": tied_skills / n,
    })
    print(f"[01] median={median:,.0f} mean={mean:,.0f} gini={g:.3f} "
          f"top1%={top_share[0.01]:.1%} tiers={tier_counts}")


def _pct_rank(arr, value) -> float:
    return float((arr < value).mean() * 100)


if __name__ == "__main__":
    main()
