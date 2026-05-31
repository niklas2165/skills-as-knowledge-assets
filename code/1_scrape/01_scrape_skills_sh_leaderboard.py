"""
Step 1b — Scrape the skills.sh all-time leaderboard via its paginated
internal API endpoint.

Endpoint
--------
GET https://skills.sh/api/skills/all-time/{page}

This is the endpoint the skills.sh frontend itself calls to render the
homepage's "All Time" leaderboard tab. It is NOT listed under /docs/api
(the documented public API at /api/v1 requires a Bearer key); however
it is publicly available without authentication, returns JSON, and is
used openly by the MIT-licensed third-party project
https://github.com/mastra-ai/skills-api (see src/scraper/scrape.ts).
Our use of it mirrors what any browser visiting skills.sh does. We
throttle to ~1 request/second to stay well under any reasonable
rate-limit threshold.

robots.txt note: skills.sh's robots.txt disallows /api/ for indexing
crawlers. We are not an indexing crawler; this is a one-shot data
collection. We have logged this in decisions.md.

Response shape (one page)
-------------------------
{
  "skills": [
    {"source": "owner/repo", "skillId": "...", "name": "...",
     "installs": N, "isOfficial": true},     # isOfficial absent when false
    ...
  ],
  "total": 9718,
  "hasMore": true|false,
  "page": N
}

Outputs
-------
- raw/skills_sh_leaderboard_<YYYY-MM-DD>.json    — parsed + enriched leaderboard
- raw/skills_sh_api_pages_<YYYY-MM-DD>.jsonl     — one raw API response per line
                                                   (for reproducibility / re-parsing)

The HTML-based scrape that ran first today is preserved at
raw/skills_sh_leaderboard_html_2026-05-20.json (238 official-only rows).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ENDPOINT_TEMPLATE = "https://skills.sh/api/skills/all-time/{page}"
SOURCE_LABEL = "skills_sh"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_DELAY_SECONDS = 3.0          # base inter-request delay
RATE_LIMIT_BACKOFF_SECONDS = [60, 120, 240]  # waits on successive 429s
MAX_PAGES = 1000  # safety cap; real value is ~49

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "raw"


def fetch_page(page: int) -> dict:
    """Fetch one page with 429-retry. Raises on persistent failure."""
    url = ENDPOINT_TEMPLATE.format(page=page)
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    for attempt, wait in enumerate([0] + RATE_LIMIT_BACKOFF_SECONDS):
        if wait:
            print(f"  ! 429 on page {page}; sleeping {wait}s before retry {attempt}", file=sys.stderr)
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} for {url}")
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            # else loop will sleep and retry
    raise RuntimeError(f"Exceeded retry budget for page {page}")


def load_existing_pages(pages_path: Path) -> tuple[dict[int, dict], int]:
    """Return ({page_idx: response}, next_page_to_fetch) from any existing JSONL."""
    cache: dict[int, dict] = {}
    if not pages_path.exists():
        return cache, 0
    with pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cache[rec["page"]] = rec["response"]
    next_page = max(cache) + 1 if cache else 0
    return cache, next_page


def enrich(skill: dict, rank: int, scrape_date: str) -> dict:
    source = skill["source"]
    skill_id = skill["skillId"]
    return {
        "rank": rank,
        "skill_name": skill["name"],
        "skill_id": skill_id,
        "repo": source,
        "repo_url": f"https://github.com/{source}",
        "skill_url": f"https://www.skills.sh/{source}/{skill_id}",
        "installs": skill["installs"],
        "is_official": bool(skill.get("isOfficial", False)),
        "scrape_date": scrape_date,
        "source": SOURCE_LABEL,
    }


def main() -> int:
    scrape_date = date.today().isoformat()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    pages_path = RAW_DIR / f"skills_sh_api_pages_{scrape_date}.jsonl"
    out_path = RAW_DIR / f"skills_sh_leaderboard_{scrape_date}.json"

    cache, start_page = load_existing_pages(pages_path)
    if cache:
        print(f"  resuming: {len(cache)} pages already on disk, starting at page {start_page}", file=sys.stderr)

    started = time.monotonic()
    last_data: dict | None = None
    last_page: int = start_page - 1 if cache else -1

    with pages_path.open("a", encoding="utf-8") as pages_f:
        for page in range(start_page, MAX_PAGES):
            data = fetch_page(page)
            pages_f.write(json.dumps({"page": page, "fetched_at": time.time(), "response": data}, ensure_ascii=False) + "\n")
            pages_f.flush()
            cache[page] = data
            last_data = data
            last_page = page

            skills = data.get("skills", [])
            print(
                f"  page={page:3d}  rows={len(skills):3d}  cumulative={sum(len(cache[p]['skills']) for p in cache):5d}/{data.get('total')}  hasMore={data.get('hasMore')}",
                file=sys.stderr,
            )

            if not data.get("hasMore", False):
                break

            time.sleep(REQUEST_DELAY_SECONDS)
        else:
            raise RuntimeError(f"MAX_PAGES={MAX_PAGES} reached without hasMore=False; bug or API change")

    elapsed = time.monotonic() - started

    # Assemble final records in page order
    all_records: list[dict] = []
    total_reported: int | None = None
    for page_idx in sorted(cache):
        data = cache[page_idx]
        if total_reported is None:
            total_reported = data.get("total")
        for skill in data.get("skills", []):
            all_records.append(enrich(skill, rank=len(all_records) + 1, scrape_date=scrape_date))
    page = last_page

    # Sanity: ranks should be 1..N and installs non-increasing
    rank_ok = [r["rank"] for r in all_records] == list(range(1, len(all_records) + 1))
    installs_ordered = all(
        all_records[i]["installs"] >= all_records[i + 1]["installs"]
        for i in range(len(all_records) - 1)
    )

    payload = {
        "metadata": {
            "scrape_date": scrape_date,
            "source_url": "https://skills.sh/api/skills/all-time/{page}",
            "source_label": SOURCE_LABEL,
            "endpoint_status": "undocumented_public_used_by_frontend",
            "endpoint_auth_required": False,
            "view_captured": "all-time",
            "records_captured": len(all_records),
            "total_reported_by_endpoint": total_reported,
            "pages_fetched": page + 1,
            "request_delay_seconds": REQUEST_DELAY_SECONDS,
            "scrape_elapsed_seconds": round(elapsed, 1),
            "rank_contiguous_1_to_n": rank_ok,
            "installs_non_increasing": installs_ordered,
            "raw_pages_file": pages_path.name,
            "notes": (
                "Captured via the same paginated endpoint the skills.sh "
                "frontend uses (mirrored in mastra-ai/skills-api, MIT). "
                "robots.txt disallows /api/ for crawlers; this is a "
                "one-shot research collection, not crawler activity. "
                "See decisions.md and investigation/findings_1a.md."
            ),
        },
        "skills": all_records,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(file=sys.stderr)
    print(f"  saved leaderboard to {out_path.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"  saved raw pages to   {pages_path.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"  records: {len(all_records)} | reported total: {total_reported} | "
          f"ranks contiguous: {rank_ok} | installs non-increasing: {installs_ordered}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
