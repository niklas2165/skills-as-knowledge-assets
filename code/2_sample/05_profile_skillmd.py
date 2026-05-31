"""05 - SKILL.md coverage.

Where is SKILL.md present vs missing? Broken down by official status, install
tier, and repo/owner, to show whether missingness is concentrated.

`has_md` := skill_md.fetch_status == 'found' (equivalently skill_md.raw present).

Outputs:
- plots/05_skillmd_coverage.png
- sections/05_skillmd.md / .json

Run:  python3 scripts/05_profile_skillmd.py
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

import _lib as L

NUM, TOPIC = "05", "skillmd"
MIN_REPO_SIZE = 10  # only rank coverage for repos with >= this many skills


def has_md(r) -> bool:
    return (r.get("skill_md") or {}).get("fetch_status") == "found"


def main() -> None:
    _, skills = L.load()
    n = len(skills)
    n_md = sum(1 for r in skills if has_md(r))

    # by official
    off = [r for r in skills if r["is_official"]]
    unoff = [r for r in skills if not r["is_official"]]
    cov_off = sum(has_md(r) for r in off) / len(off)
    cov_unoff = sum(has_md(r) for r in unoff) / len(unoff)

    # by install tier
    tiers = L.INSTALL_TIER_LABELS
    tier_tot = Counter()
    tier_md = Counter()
    for r in skills:
        t = L.tier_of(r["installs"])
        tier_tot[t] += 1
        if has_md(r):
            tier_md[t] += 1

    # by repo
    repo_tot = Counter()
    repo_md = Counter()
    for r in skills:
        repo_tot[r["repo"]] += 1
        if has_md(r):
            repo_md[r["repo"]] += 1
    # repos where missingness concentrates: most missing in absolute terms
    repo_missing = Counter({rp: repo_tot[rp] - repo_md[rp] for rp in repo_tot})
    # large repos with 0% and 100% coverage
    big = [rp for rp in repo_tot if repo_tot[rp] >= MIN_REPO_SIZE]
    big_zero = [(rp, repo_tot[rp]) for rp in big if repo_md[rp] == 0]
    big_full = [(rp, repo_tot[rp]) for rp in big if repo_md[rp] == repo_tot[rp]]

    # share of all missing held by top-k repos
    total_missing = n - n_md
    miss_top10 = sum(c for _, c in repo_missing.most_common(10))

    # ------------------------------------------------------------------ plot
    fig, axes = L.plt.subplots(1, 2, figsize=(11, 4.6))
    # left: coverage by tier
    cov_t = [tier_md[t] / tier_tot[t] if tier_tot[t] else 0 for t in tiers]
    axes[0].bar(tiers, [c * 100 for c in cov_t], color="#4C72B0")
    for i, t in enumerate(tiers):
        axes[0].text(i, cov_t[i] * 100, f"{cov_t[i]:.0%}\n(n={tier_tot[t]})",
                     ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("SKILL.md coverage (%)")
    axes[0].set_title("Coverage by install tier")
    axes[0].set_xlabel("Install tier")
    # right: coverage by official
    axes[1].bar(["official", "unofficial"], [cov_off * 100, cov_unoff * 100],
                color=["#4C72B0", "#C4C4C4"])
    for i, (lbl, c, k) in enumerate([("official", cov_off, len(off)), ("unofficial", cov_unoff, len(unoff))]):
        axes[1].text(i, c * 100, f"{c:.0%}\n(n={k})", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Coverage by official status")
    fig.suptitle(f"SKILL.md coverage — {n_md:,}/{n:,} = {n_md/n:.0%} overall")
    plot_path = L.save_fig(fig, f"{NUM}_skillmd_coverage")

    # ------------------------------------------------------------------ md
    md = [f"## {NUM} — SKILL.md coverage", ""]
    md.append(f"**{L.fmt_int(n_md)}** of {L.fmt_int(n)} skills "
              f"({L.fmt_pct(n_md / n)}) have a fetched SKILL.md "
              f"(`fetch_status == 'found'`). The remaining {L.fmt_int(n - n_md)} "
              "are `not_found_via_patterns` (or 1 cache-only); no GitHub Tree-API "
              "fallback was used in the scrape, so missingness reflects scrape method "
              "as well as true absence.")
    md += ["", "### Coverage by official status", ""]
    md.append(L.md_table(["Group", "Skills", "With SKILL.md", "Coverage"],
                         [["official", L.fmt_int(len(off)), L.fmt_int(sum(has_md(r) for r in off)), L.fmt_pct(cov_off)],
                          ["unofficial", L.fmt_int(len(unoff)), L.fmt_int(sum(has_md(r) for r in unoff)), L.fmt_pct(cov_unoff)]]))
    md += ["", "### Coverage by install tier", ""]
    md.append(L.md_table(["Tier", "Skills", "With SKILL.md", "Coverage"],
                         [[t, L.fmt_int(tier_tot[t]), L.fmt_int(tier_md[t]),
                           L.fmt_pct(tier_md[t] / tier_tot[t]) if tier_tot[t] else "n/a"] for t in tiers]))
    md += ["", "### Where missingness concentrates", ""]
    md.append(f"- Total missing: **{L.fmt_int(total_missing)}**. The 10 repos with the "
              f"most missing SKILL.md account for **{L.fmt_int(miss_top10)}** of them "
              f"({L.fmt_pct(miss_top10 / total_missing)} of all missing).")
    md.append(f"- Repos with >= {MIN_REPO_SIZE} skills and **0% coverage**: "
              f"{len(big_zero)}; with **100% coverage**: {len(big_full)}.")
    md.append("")
    md.append("Top 10 repos by count of missing SKILL.md:")
    md.append("")
    md.append(L.md_table(["Repo", "Skills", "Missing", "Repo coverage"],
                         [[rp, L.fmt_int(repo_tot[rp]), L.fmt_int(miss),
                           L.fmt_pct(repo_md[rp] / repo_tot[rp])]
                          for rp, miss in repo_missing.most_common(10)]))
    if big_zero:
        md += ["", f"Large repos (>= {MIN_REPO_SIZE} skills) with **no** SKILL.md at all:", ""]
        md.append(L.md_table(["Repo", "Skills"],
                             [[rp, L.fmt_int(k)] for rp, k in sorted(big_zero, key=lambda x: -x[1])[:15]]))
    md += ["", f"![SKILL.md coverage]({plot_path})", ""]

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "n_md": n_md, "coverage_overall": n_md / n,
        "coverage_official": cov_off, "coverage_unofficial": cov_unoff,
        "coverage_by_tier": {t: (tier_md[t] / tier_tot[t] if tier_tot[t] else None) for t in tiers},
        "total_missing": total_missing, "missing_top10_repos": repo_missing.most_common(10),
        "missing_top10_share": miss_top10 / total_missing if total_missing else None,
        "big_repos_zero_coverage": big_zero, "big_repos_full_coverage_count": len(big_full),
    })
    print(f"[05] md={n_md} ({n_md/n:.1%}) cov_off={cov_off:.1%} cov_unoff={cov_unoff:.1%} "
          f"cov_by_tier={ {t: round(tier_md[t]/tier_tot[t],3) if tier_tot[t] else None for t in tiers} } "
          f"big_zero={len(big_zero)} big_full={len(big_full)}")


if __name__ == "__main__":
    main()
