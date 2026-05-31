"""
Step 1c (merge) — Combine the leaderboard (1b) and detail (1c) scrapes
into a single per-skill record set.

Inputs
------
- raw/skills_sh_leaderboard_<date>.json   (1b — full population, exact installs)
- raw/skills_sh_detail_<date>.json         (1c — detail for the scraped subset)

What it does
------------
1. Deduplicates each input by skill_url. Both scrapes contain a small
   number of duplicate skill_urls caused by pagination drift in 1b (the
   /api/skills/all-time endpoint paginates a live-ranked list; a skill
   can appear on two adjacent pages if installs shift mid-scrape). The
   duplicates have identical install counts; we keep the lowest-rank
   occurrence as canonical.
2. Inner-joins on skill_url: keeps ONLY skills that have BOTH a
   leaderboard entry and a successful (HTTP 200) detail record. Per the
   user's decision on 2026-05-20, the combined file is the detail-complete
   population, not the full leaderboard.
3. Writes one combined JSON with all relevant scraped fields per skill,
   plus provenance metadata.

Output
------
- processed/skills_combined_<date>.json

The two install figures (exact integer from 1b, abbreviated string from
the detail page) are BOTH kept and NOT reconciled — they're captured at
slightly different moments and can disagree (e.g. 431.6K vs 436.0K).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "raw"
PROCESSED_DIR = REPO_ROOT / "processed"


def dedup_by_url(skills: list[dict], rank_key: str) -> tuple[list[dict], int]:
    """Keep the lowest-`rank_key` occurrence of each skill_url. Returns (deduped, n_dupes_removed)."""
    best: dict[str, dict] = {}
    for s in skills:
        url = s["skill_url"]
        cur = best.get(url)
        if cur is None or (s.get(rank_key) is not None and s[rank_key] < cur.get(rank_key, float("inf"))):
            best[url] = s
    deduped = list(best.values())
    return deduped, len(skills) - len(deduped)


def build_combined_record(lb: dict, dt: dict, lb_scrape_date: str, dt_scrape_date: str) -> dict:
    return {
        # identity / join key
        "skill_url": lb["skill_url"],
        "skill_id": lb["skill_id"],
        "repo": lb["repo"],
        "repo_url": lb["repo_url"],
        "is_official": lb.get("is_official"),
        # popularity
        "leaderboard_rank": lb.get("rank"),
        "installs": lb.get("installs"),                       # exact integer (1b)
        "installs_abbreviated": dt.get("installs_abbreviated"),  # detail page (1c)
        "github_stars_abbreviated": dt.get("github_stars_abbreviated"),
        # descriptive
        "skill_name": dt.get("skill_name_h1"),
        "summary": dt.get("summary"),
        "topic_slug": dt.get("primary_topic_slug"),
        "topic_label": dt.get("primary_topic_label"),
        "install_command": dt.get("install_command"),
        "audits": dt.get("audits", []),
        # SKILL.md (from GitHub, grouped)
        "skill_md": {
            "fetch_status": dt.get("skill_md_fetch_status"),
            "path": dt.get("skill_md_path"),
            "branch": dt.get("skill_md_branch"),
            "raw": dt.get("skill_md_raw"),
            "frontmatter": dt.get("skill_md_frontmatter") or {},
            "body": dt.get("skill_md_body"),
        },
        # provenance
        "leaderboard_scrape_date": lb_scrape_date,
        "detail_scrape_date": dt_scrape_date,
        "source": "skills_sh",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-20", help="Date stamp of the raw inputs/output")
    args = ap.parse_args()
    d = args.date

    lb_path = RAW_DIR / f"skills_sh_leaderboard_{d}.json"
    dt_path = RAW_DIR / f"skills_sh_detail_{d}.json"
    for p in (lb_path, dt_path):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 2

    lb_doc = json.loads(lb_path.read_text())
    dt_doc = json.loads(dt_path.read_text())
    lb_scrape_date = lb_doc["metadata"].get("scrape_date", d)
    dt_scrape_date = dt_doc["metadata"].get("scrape_date", d)

    lb_skills, lb_dupes = dedup_by_url(lb_doc["skills"], rank_key="rank")
    # Detail records carry leaderboard_rank; dedup on that.
    dt_skills, dt_dupes = dedup_by_url(
        [s for s in dt_doc["skills"] if s.get("http_status") == 200],
        rank_key="leaderboard_rank",
    )
    dt_by_url = {s["skill_url"]: s for s in dt_skills}

    combined: list[dict] = []
    for lb in sorted(lb_skills, key=lambda s: s.get("rank", 1 << 30)):
        dt = dt_by_url.get(lb["skill_url"])
        if dt is None:
            continue  # inner join: skip skills with no detail
        combined.append(build_combined_record(lb, dt, lb_scrape_date, dt_scrape_date))

    # --- quality / profile summary ---
    n = len(combined)
    have = lambda f: sum(1 for r in combined if r.get(f) not in (None, ""))
    have_md = sum(1 for r in combined if r["skill_md"]["raw"])
    have_topic = sum(1 for r in combined if r.get("topic_slug"))
    have_audits = sum(1 for r in combined if r.get("audits"))
    official = sum(1 for r in combined if r.get("is_official"))
    unique_repos = len({r["repo"] for r in combined})
    unique_owners = len({r["repo"].split("/")[0] for r in combined})
    installs = sorted(r["installs"] for r in combined if r.get("installs") is not None)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"skills_combined_{d}.json"
    payload = {
        "metadata": {
            "created_date": date.today().isoformat(),
            "leaderboard_input": lb_path.name,
            "detail_input": dt_path.name,
            "join": "inner join on skill_url (detail-complete population)",
            "leaderboard_unique_skills": len(lb_skills),
            "leaderboard_duplicate_rows_removed": lb_dupes,
            "detail_unique_skills_http200": len(dt_skills),
            "detail_duplicate_rows_removed": dt_dupes,
            "combined_records": n,
            "field_coverage": {
                "skill_name": have("skill_name"),
                "summary": have("summary"),
                "topic_slug": have_topic,
                "installs_abbreviated": have("installs_abbreviated"),
                "github_stars_abbreviated": have("github_stars_abbreviated"),
                "install_command": have("install_command"),
                "audits_present": have_audits,
                "skill_md_raw": have_md,
            },
            "is_official_count": official,
            "unique_source_repos": unique_repos,
            "unique_owners": unique_owners,
            "installs_min": installs[0] if installs else None,
            "installs_median": installs[len(installs) // 2] if installs else None,
            "installs_max": installs[-1] if installs else None,
            "notes": (
                "Detail-complete subset of the skills.sh all-time leaderboard, "
                "per user decision 2026-05-20. Duplicate skill_urls (pagination "
                "drift in 1b) removed, keeping lowest-rank occurrence. 'installs' "
                "is the exact integer from the leaderboard API; "
                "'installs_abbreviated' is the K/M string from the detail page — "
                "both kept, not reconciled. topic_slug is sparse (~1.7%); not a "
                "viable stratification axis. skill_md.raw present for ~64% (no "
                "GitHub Tree-API fallback used). See decisions.md."
            ),
        },
        "skills": combined,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"  leaderboard: {len(lb_doc['skills'])} rows → {len(lb_skills)} unique ({lb_dupes} dupes removed)", file=sys.stderr)
    print(f"  detail(200): {len(dt_skills)} unique ({dt_dupes} dupes removed)", file=sys.stderr)
    print(f"  combined (inner join): {n} skills", file=sys.stderr)
    print(f"  unique repos={unique_repos}  owners={unique_owners}  official={official}", file=sys.stderr)
    print(f"  skill_md present: {have_md}/{n} ({have_md/n*100:.1f}%)", file=sys.stderr)
    print(f"  saved → {out_path.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
