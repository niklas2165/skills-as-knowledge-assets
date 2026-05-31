"""03 - Official vs unofficial.

`is_official` is a publisher-claimed flag (not externally verified). Counts,
overall share, and distribution across install tiers.

Outputs:
- plots/03_official_by_tier.png
- sections/03_official.md / .json

Run:  python3 scripts/03_profile_official.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import _lib as L

NUM, TOPIC = "03", "official"


def main() -> None:
    _, skills = L.load()
    n = len(skills)
    n_off = sum(1 for r in skills if r["is_official"] is True)
    n_unoff = n - n_off

    # cross-tab tier x official
    tiers = L.INSTALL_TIER_LABELS
    off_by_tier = Counter()
    unoff_by_tier = Counter()
    for r in skills:
        t = L.tier_of(r["installs"])
        (off_by_tier if r["is_official"] else unoff_by_tier)[t] += 1
    tot_by_tier = {t: off_by_tier[t] + unoff_by_tier[t] for t in tiers}

    off_installs = [r["installs"] for r in skills if r["is_official"]]
    unoff_installs = [r["installs"] for r in skills if not r["is_official"]]

    # ------------------------------------------------------------------ plot
    fig, ax = L.plt.subplots(figsize=(8, 5))
    x = np.arange(len(tiers))
    off_vals = [off_by_tier[t] for t in tiers]
    unoff_vals = [unoff_by_tier[t] for t in tiers]
    ax.bar(x, unoff_vals, color="#C4C4C4", label="unofficial")
    ax.bar(x, off_vals, bottom=unoff_vals, color="#4C72B0", label="official")
    for i, t in enumerate(tiers):
        if tot_by_tier[t]:
            ax.text(i, tot_by_tier[t], f"{off_by_tier[t]/tot_by_tier[t]:.0%} off.",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_xlabel("Install tier")
    ax.set_ylabel("Number of skills")
    ax.set_title("Official vs unofficial by install tier (label = official share of tier)")
    ax.legend()
    plot_path = L.save_fig(fig, f"{NUM}_official_by_tier")

    # ------------------------------------------------------------------ md
    md = [f"## {NUM} — Official vs unofficial", ""]
    md.append("`is_official` is a **publisher-claimed** flag, not externally verified.")
    md += ["", "### Overall", ""]
    md.append(L.md_table(["Group", "Skills", "Share"],
                         [["official", L.fmt_int(n_off), L.fmt_pct(n_off / n)],
                          ["unofficial", L.fmt_int(n_unoff), L.fmt_pct(n_unoff / n)]]))
    md += ["", "### Install distribution by group (exact installs)", ""]
    md.append(L.md_table(
        ["Group", "Median", "Mean", "p90", "Max"],
        [["official", L.fmt_int(round(np.median(off_installs))), L.fmt_int(round(np.mean(off_installs))),
          L.fmt_int(round(np.percentile(off_installs, 90))), L.fmt_int(max(off_installs))],
         ["unofficial", L.fmt_int(round(np.median(unoff_installs))), L.fmt_int(round(np.mean(unoff_installs))),
          L.fmt_int(round(np.percentile(unoff_installs, 90))), L.fmt_int(max(unoff_installs))]]))
    md += ["", "### Cross-tab: install tier x official status", ""]
    md.append(L.md_table(
        ["Tier", "Official", "Unofficial", "Total", "Official share of tier"],
        [[t, L.fmt_int(off_by_tier[t]), L.fmt_int(unoff_by_tier[t]), L.fmt_int(tot_by_tier[t]),
          L.fmt_pct(off_by_tier[t] / tot_by_tier[t]) if tot_by_tier[t] else "n/a"]
         for t in tiers]))
    md += ["", f"![Official by tier]({plot_path})", ""]

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "n_official": n_off, "n_unofficial": n_unoff, "official_share": n_off / n,
        "official_median": float(np.median(off_installs)),
        "unofficial_median": float(np.median(unoff_installs)),
        "off_by_tier": dict(off_by_tier), "unoff_by_tier": dict(unoff_by_tier),
        "official_share_by_tier": {t: (off_by_tier[t] / tot_by_tier[t] if tot_by_tier[t] else None) for t in tiers},
    })
    print(f"[03] official={n_off} ({n_off/n:.1%}) "
          f"off_median={np.median(off_installs):,.0f} unoff_median={np.median(unoff_installs):,.0f} "
          f"off_share_by_tier={ {t: round(off_by_tier[t]/tot_by_tier[t],3) if tot_by_tier[t] else None for t in tiers} }")


if __name__ == "__main__":
    main()
