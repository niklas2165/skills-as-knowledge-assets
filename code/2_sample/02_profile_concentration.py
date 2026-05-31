"""02 - Repo and owner concentration.

How many skills come from how few repos / owners. Concentration is measured on
the *number of skills* (not installs); an install-weighted note is added for
context.

Outputs:
- plots/02_top_repos.png, plots/02_top_owners.png
- sections/02_concentration.md / .json

Run:  python3 scripts/02_profile_concentration.py
"""
from __future__ import annotations

from collections import Counter

import _lib as L

NUM, TOPIC = "02", "concentration"
TOPN = 20


def _bar(counts: Counter, n_total: int, title: str, fname: str, color: str) -> str:
    items = counts.most_common(TOPN)
    labels = [k for k, _ in items][::-1]
    vals = [c for _, c in items][::-1]
    fig, ax = L.plt.subplots(figsize=(8, 9))
    ax.barh(labels, vals, color=color, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Number of skills")
    ax.set_title(title)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v} ({v / n_total:.1%})", va="center", fontsize=8)
    ax.margins(x=0.15)
    return L.save_fig(fig, fname)


def main() -> None:
    _, skills = L.load()
    n = len(skills)
    repo_counts = Counter(r["repo"] for r in skills)
    owner_counts = Counter(L.owner_of(r["repo"]) for r in skills)

    # install-weighted share, for context only
    repo_installs = Counter()
    for r in skills:
        repo_installs[r["repo"]] += r["installs"]
    total_installs = sum(repo_installs.values())

    repo_bar = _bar(repo_counts, n, f"Top {TOPN} repos by skill count (of {len(repo_counts)} repos)",
                    f"{NUM}_top_repos", "#4C72B0")
    owner_bar = _bar(owner_counts, n, f"Top {TOPN} owners by skill count (of {len(owner_counts)} owners)",
                     f"{NUM}_top_owners", "#55A868")

    # ------------------------------------------------------------------ md
    md = [f"## {NUM} — Repo & owner concentration", ""]
    md.append(f"{L.fmt_int(n)} skills span **{L.fmt_int(len(repo_counts))} repos** "
              f"and **{L.fmt_int(len(owner_counts))} owners** "
              f"(mean {n/len(repo_counts):.1f} skills/repo, {n/len(owner_counts):.1f} skills/owner).")

    md += ["", "### Concentration ratios (share of all skills)", ""]
    md.append(L.md_table(
        ["Top-k", "Repos: share of skills", "Owners: share of skills"],
        [[f"top {k}", L.fmt_pct(L.top_share(repo_counts, k)), L.fmt_pct(L.top_share(owner_counts, k))]
         for k in (5, 10, 20)]))

    md += ["", f"### Top {TOPN} repos", ""]
    md.append(L.md_table(
        ["#", "Repo", "Skills", "Share of skills", "Total installs", "Share of installs"],
        [[i + 1, repo, L.fmt_int(c), L.fmt_pct(c / n),
          L.fmt_int(repo_installs[repo]), L.fmt_pct(repo_installs[repo] / total_installs)]
         for i, (repo, c) in enumerate(repo_counts.most_common(TOPN))]))

    md += ["", f"### Top {TOPN} owners", ""]
    md.append(L.md_table(
        ["#", "Owner", "Skills", "Share of skills"],
        [[i + 1, owner, L.fmt_int(c), L.fmt_pct(c / n)]
         for i, (owner, c) in enumerate(owner_counts.most_common(TOPN))]))

    n_singletons = sum(1 for c in repo_counts.values() if c == 1)
    md += ["", "### Long tail", ""]
    md.append(f"- Repos contributing exactly 1 skill: **{L.fmt_int(n_singletons)}** "
              f"({L.fmt_pct(n_singletons / len(repo_counts))} of repos, "
              f"{L.fmt_pct(n_singletons / n)} of skills).")
    md.append(f"- Largest single repo: **{repo_counts.most_common(1)[0][0]}** "
              f"with {L.fmt_int(repo_counts.most_common(1)[0][1])} skills.")
    md += ["", f"![Top repos]({repo_bar})", "", f"![Top owners]({owner_bar})", ""]

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "n_repos": len(repo_counts), "n_owners": len(owner_counts),
        "repo_top_share": {k: L.top_share(repo_counts, k) for k in (5, 10, 20)},
        "owner_top_share": {k: L.top_share(owner_counts, k) for k in (5, 10, 20)},
        "top_repos": repo_counts.most_common(TOPN),
        "top_owners": owner_counts.most_common(TOPN),
        "singleton_repos": n_singletons,
        "largest_repo": repo_counts.most_common(1)[0],
    })
    print(f"[02] repos={len(repo_counts)} owners={len(owner_counts)} "
          f"top5repos={L.top_share(repo_counts,5):.1%} top20repos={L.top_share(repo_counts,20):.1%} "
          f"singletons={n_singletons}")


if __name__ == "__main__":
    main()
