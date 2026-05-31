"""08 - Repo-name patterns.

An informal, descriptive signal of what kinds of things these repos are. Two
views:

1. Bottom-up: most frequent tokens across distinct repo names (emergent
   vocabulary, no imposed categories).
2. Keyword probes: how many repo names contain structural words ("skills",
   "agent", ...) and -- as a clearly-labelled cross-check only -- the six
   legacy Excel domain words. The legacy probe matches the repo-name STRING;
   it is NOT a content classification and must not be treated as one.

Outputs:
- plots/08_repo_tokens.png
- sections/08_reponames.md / .json

Run:  python3 scripts/08_profile_reponames.py
"""
from __future__ import annotations

import re
from collections import Counter

import _lib as L

NUM, TOPIC = "08", "reponames"

STRUCTURAL = ["skills", "agent-skills", "agent", "claude", "mcp", "plugin",
              "awesome", "cursor", "codex", "prompt", "ai", "tool"]
# Legacy six-domain probe — DESCRIPTIVE CROSS-CHECK ONLY, not a classification.
LEGACY_DOMAINS = {
    "coding": ["cod", "dev", "program", "engineer"],
    "legal": ["legal", "law", "compliance", "contract"],
    "hr": ["hr", "recruit", "hiring", "people-ops"],
    "finance": ["financ", "fintech", "account", "invoic", "tax"],
    "healthcare": ["health", "medic", "clinic", "patient"],
    "marketing": ["market", "seo", "campaign", "brand", "social"],
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def main() -> None:
    _, skills = L.load()
    n = len(skills)

    # distinct repos, and skills-per-repo for skill-level shares
    repo_to_skillcount = Counter(r["repo"] for r in skills)
    distinct_repos = list(repo_to_skillcount)
    n_repos = len(distinct_repos)
    repo_names = {rp: L.repo_name_of(rp).lower() for rp in distinct_repos}

    # 1. token frequency over distinct repo names
    tok_repo = Counter()
    for rp in distinct_repos:
        for t in set(TOKEN_RE.findall(repo_names[rp])):
            tok_repo[t] += 1
    top_tokens = tok_repo.most_common(30)

    # helper: repos & skills whose repo-name contains a substring
    def contains(sub: str):
        reps = [rp for rp in distinct_repos if sub in repo_names[rp]]
        sk = sum(repo_to_skillcount[rp] for rp in reps)
        return len(reps), sk

    structural_rows = []
    for kw in STRUCTURAL:
        nr, ns = contains(kw)
        structural_rows.append([kw, nr, L.fmt_pct(nr / n_repos), ns, L.fmt_pct(ns / n)])

    legacy_rows = []
    legacy_stats = {}
    for dom, subs in LEGACY_DOMAINS.items():
        reps = [rp for rp in distinct_repos if any(s in repo_names[rp] for s in subs)]
        sk = sum(repo_to_skillcount[rp] for rp in reps)
        legacy_rows.append([dom, " / ".join(subs), len(reps), L.fmt_int(sk), L.fmt_pct(sk / n)])
        legacy_stats[dom] = {"repos": len(reps), "skills": sk, "skill_share": sk / n}

    # ------------------------------------------------------------------ plot
    fig, ax = L.plt.subplots(figsize=(8, 9))
    labels = [t for t, _ in top_tokens][::-1]
    vals = [c for _, c in top_tokens][::-1]
    ax.barh(labels, vals, color="#937860", edgecolor="white", linewidth=0.3)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v}", va="center", fontsize=8)
    ax.set_xlabel(f"Number of distinct repos (of {n_repos}) whose name contains the token")
    ax.set_title("Top 30 tokens in repo names (bottom-up)")
    ax.margins(x=0.12)
    plot_path = L.save_fig(fig, f"{NUM}_repo_tokens")

    # ------------------------------------------------------------------ md
    md = [f"## {NUM} — Repo-name patterns", ""]
    md.append(f"Tokens and keywords are matched against the **repo-name string** "
              f"(the part after `owner/`), lower-cased, across {L.fmt_int(n_repos)} "
              "distinct repos. This is a naming signal only — it says nothing about "
              "skill content and is not a classification.")
    md += ["", "### Most frequent repo-name tokens (bottom-up)", ""]
    md.append(L.md_table(["Token", "Repos", "Share of repos"],
                         [[t, L.fmt_int(c), L.fmt_pct(c / n_repos)] for t, c in top_tokens]))
    md += ["", "### Structural keyword probe", ""]
    md.append(L.md_table(["Keyword in repo name", "Repos", "Share of repos", "Skills", "Share of skills"],
                         structural_rows))
    md += ["", "### Legacy six-domain keyword probe (cross-check only)", ""]
    md.append("> **Caveat:** This matches the repo-name string against substrings "
              "loosely associated with the six domains of the *previous* thesis attempt. "
              "It is a descriptive cross-check to see whether those domain words even "
              "surface in repo names — **not** a content classification, and explicitly "
              "**not** a sampling axis. Substrings are crude (e.g. `cod` also matches "
              "`code`, `coding`, `codex`).")
    md.append("")
    md.append(L.md_table(["Legacy domain", "Substrings matched", "Repos", "Skills", "Share of skills"],
                         legacy_rows))
    md += ["", f"![Repo-name tokens]({plot_path})", ""]

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "n_repos": n_repos, "top_tokens": top_tokens,
        "structural": {kw: {"repos": contains(kw)[0], "skills": contains(kw)[1]} for kw in STRUCTURAL},
        "legacy_domain_probe": legacy_stats,
    })
    print(f"[08] top_tokens={[t for t,_ in top_tokens[:8]]} "
          f"contains_skills_repos={contains('skills')[0]} ({contains('skills')[1]} skills)")


if __name__ == "__main__":
    main()
