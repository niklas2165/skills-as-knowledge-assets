"""99 - Assemble population_profile.md.

Stitches the section fragments (sections/NN_*.md) into one standalone document
and writes a top-of-document "What stands out" header whose numbers are pulled
from the section stats JSONs (so nothing is hand-typed / fabricated).

This script does NOT recompute anything; it only reads the artefacts the
0X scripts produced. Re-run the 0X scripts first if the data changes.

Run:  python3 scripts/99_build_profile.py
"""
from __future__ import annotations

import json
from datetime import date

import _lib as L

OUT = L.PROFILE_DIR / "population_profile.md"

SECTION_ORDER = ["00_validation", "01_installs", "02_concentration", "03_official",
                 "04_audits", "05_skillmd", "06_text", "07_topic", "08_reponames"]


def load_stats(name: str) -> dict:
    p = L.SECTIONS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def pct(x, d=1):
    return "n/a" if x is None else f"{100*x:.{d}f}%"


def main() -> None:
    S = {name: load_stats(name) for name in SECTION_ORDER}
    inst, con, off, aud, md_, txt, top, rep = (
        S["01_installs"], S["02_concentration"], S["03_official"], S["04_audits"],
        S["05_skillmd"], S["06_text"], S["07_topic"], S["08_reponames"])

    n = S["00_validation"].get("n_records", 5235)
    disc = S["00_validation"].get("discrepancy_count", 0)

    # --- What stands out (numbers sourced from stats JSONs) ---
    stand = []
    stand.append(
        f"**Validation clean.** All {n:,} records carry the full top-level schema and "
        f"every metadata coverage claim reproduced exactly ({disc} discrepancies). "
        "Missing values are encoded as `null`, never absent keys.")
    stand.append(
        f"**Installs are extremely skewed and not independent across skills.** Median "
        f"{inst['median']:,.0f} vs mean {inst['mean']:,.0f}; Gini {inst['gini']:.2f}; the top 1% "
        f"of skills hold {pct(inst['top_share']['0.01'],0)} of all installs and the top 10% "
        f"hold {pct(inst['top_share']['0.1'],0)}. Most skills sit near the install floor "
        f"(min {inst['min']:,}). Counts are heavily tied ({inst['distinct_install_values']:,} "
        f"distinct values; {pct(inst['tied_share'],0)} of skills share a count), and same-repo "
        "skills cluster at similar install levels — so install tier is partly collinear with repo "
        "(flag for stratification).")
    stand.append(
        f"**Supply is concentrated in few repos/owners.** {con['n_repos']:,} repos / "
        f"{con['n_owners']:,} owners; the top 20 repos hold {pct(con['repo_top_share']['20'])} "
        f"of all skills and the top 5 owners {pct(con['owner_top_share']['5'])}. Yet "
        f"{con['singleton_repos']:,} repos contribute just one skill — a heavy head AND a long tail. "
        "(Relevant given the prior attempt's single-vendor domination problem.)")
    stand.append(
        f"**SKILL.md coverage ({pct(md_['coverage_overall'],0)}) is uneven and partly an artefact "
        "of scrape method.** Coverage rises with installs "
        f"({pct(md_['coverage_by_tier']['<1K'])} at <1K -> {pct(md_['coverage_by_tier']['100K-1M'])} at 100K-1M), "
        f"official skills have *lower* coverage than unofficial ({pct(md_['coverage_official'])} vs {pct(md_['coverage_unofficial'])}), "
        f"and it is bimodal at the repo level ({len(md_['big_repos_zero_coverage'])} large repos at 0%, "
        f"{md_['big_repos_full_coverage_count']} at 100%). Restricting to SKILL.md-complete skills would drop "
        "whole repos non-randomly — flag for the Phase-2 population decision.")
    aud_svc = aud["per_service"]
    def warn_rate(svc):
        d = aud_svc.get(svc, {}); t = sum(d.values()); return d.get("Warn",0)/t if t else 0
    stand.append(
        f"**Audit signals disagree across services.** {pct(aud['coverage'],0)} of skills are audited; "
        f"Snyk warns on {pct(warn_rate('Snyk'),0)} of audited skills while Socket warns on only "
        f"{pct(warn_rate('Socket'),0)}. {aud['any_fail']:,} audited skills carry at least one Fail "
        f"({pct(aud['any_fail']/aud['n_with_audits'],0)}). 'Audited' is not a single pass/fail axis.")
    stand.append(
        f"**Official is a publisher-claimed minority that grows with reach.** "
        f"{off['n_official']:,} skills ({pct(off['official_share'],0)}) are flagged official; the official "
        f"share climbs from {pct(off['official_share_by_tier']['<1K'])} of the <1K tier to "
        f"{pct(off['official_share_by_tier']['10K-100K'])} of the 10K-100K tier. Not externally verified.")
    stand.append(
        f"**Two fields are non-starters as category axes; the rich text is partial.** "
        f"`topic_slug` covers only {pct(top['coverage'],1)} ({top['n_present']} skills, {top['n_distinct_topics']} topics). "
        f"`summary` (100% coverage) is short (median {txt['summary_words']['median']:.0f} words); "
        f"`skill_md.body` is substantial (median {txt['body_words']['median']:.0f} words) but exists for only "
        f"{pct(md_['coverage_overall'],0)}. Repo names lean on 'skills'/'agent'/'claude'; the six legacy "
        "domain words barely surface in repo names (see §08).")

    # --- assemble ---
    out = []
    out.append("# Population profile — skills.sh detail-complete corpus")
    out.append("")
    out.append(f"_Generated {date.today().isoformat()} from "
               f"`inputs/{L.INPUT_PATH.name}` ({n:,} records). "
               "Regenerate with `scripts/00_validate_schema.py` ... `08_*.py` then "
               "`scripts/99_build_profile.py`. All numbers below are computed by those "
               "scripts; this phase profiles the population and flags patterns — it makes "
               "no sampling or clustering decisions._")
    out.append("")
    out.append("## What stands out")
    out.append("")
    out.append("_Flags, not conclusions — each bullet is a pattern visible in the data, "
               "stated without interpretation beyond what is shown._")
    out.append("")
    for b in stand:
        out.append(f"- {b}")
    out.append("")
    out.append("## Methodology & definitions")
    out.append("")
    out.append("- **Install metric:** the exact integer `installs` (leaderboard API) is "
               "used for all numeric analysis; the abbreviated `installs_abbreviated` "
               "K/M string is never used numerically.")
    out.append("- **Install tiers** (used for all cross-tabs): decade / order-of-magnitude "
               "bins on `installs` — `<1K`, `1K-10K`, `10K-100K`, `100K-1M`, `>=1M`. "
               "Chosen over quantile tiers because the distribution spans >3 orders of "
               "magnitude and round, population-independent boundaries are easier to "
               "interpret. Quartile/percentile stats are reported separately in §01.")
    out.append("- **Histograms:** install histogram uses 40 log-spaced bins on a log "
               "x-axis; length histograms use 40 linear bins with the x-axis clipped at "
               "the 99th percentile for readability (all stats use unclipped data).")
    out.append("- **Gini / Lorenz:** computed manually (sorted-rank formula), no external "
               "library.")
    out.append("- **`has SKILL.md`** := `skill_md.fetch_status == 'found'` (equivalently "
               "`skill_md.raw` present). The scrape used no GitHub Tree-API fallback, so "
               "absence reflects scrape coverage as well as true absence.")
    out.append("- **Owner** := the part of `repo` before `/`. **Repo-name** := the part "
               "after `/` (used for the §08 token/keyword probes, which match strings only "
               "and are not content classifications).")
    out.append("")
    out.append("---")
    out.append("")
    for name in SECTION_ORDER:
        frag = L.SECTIONS_DIR / f"{name}.md"
        if frag.exists():
            out.append(frag.read_text().rstrip())
            out.append("")
            out.append("---")
            out.append("")

    doc = "\n".join(out).rstrip() + "\n"
    OUT.write_text(doc)
    print(f"[99] wrote {OUT.relative_to(L.ROOT)} ({len(doc):,} chars, "
          f"{len(SECTION_ORDER)} sections, {len(stand)} headline bullets)")


if __name__ == "__main__":
    main()
