"""
Step 1c — Scrape per-skill detail data: skills.sh detail page + SKILL.md from GitHub.

Two data sources per skill
--------------------------
1. skills.sh detail HTML at /{owner}/{repo}/{skillId} — yields topic,
   audit signals, install command, summary, install/star abbreviations.
2. raw.githubusercontent.com/{owner}/{repo}/{branch}/{path} — yields
   the SKILL.md raw markdown + parsed YAML frontmatter. Path patterns
   ported from mastra-ai/skills-api/src/github/fetch-skill.ts. We
   intentionally do NOT use the GitHub Tree API fallback (which would
   need auth at the 60/hr unauthenticated limit). About 90% of skills
   resolve through the simple path patterns + simplified-skillId rule.

Fields captured (per skill)
---------------------------
From skills.sh detail HTML:
  - skill_name_h1              : <h1> text
  - summary                    : <meta name="description">
  - primary_topic_slug/_label  : e.g. "design" / "Design & UI"
  - installs_abbreviated       : e.g. "436.0K"
  - github_stars_abbreviated   : e.g. "137.6K"
  - install_command            : from RSC payload
  - audits                     : list of {slug, name, status} (dynamic)

From GitHub:
  - skill_md_fetch_status      : "found" or "not_found_via_patterns"
  - skill_md_path              : path within repo where SKILL.md lives
  - skill_md_branch            : "main" or "master"
  - skill_md_raw               : full raw SKILL.md content
  - skill_md_frontmatter       : parsed YAML frontmatter (dict)
  - skill_md_body              : markdown body without frontmatter

Deliberately NOT captured (would need GH auth or non-trivial work)
-------------------------------------------------------------------
- Exact GitHub stars (only K/M visible on detail page)
- first_seen / last_updated dates (not on detail page)
- skills.sh-rendered SKILL.md preview/restHtml (React-streaming refs)
- Full skill directory dump
- GitHub Tree-API fallback for missed SKILL.md (needs PAT)

Outputs
-------
- raw/skills_sh_detail_<YYYY-MM-DD>.json            — parsed rows
- raw/skills_sh_detail_html_<YYYY-MM-DD>/*.html      — raw skills.sh HTML
- raw/skills_sh_skill_md_<YYYY-MM-DD>/*.md           — raw SKILL.md per skill

Resume / re-runs
----------------
Both per-skill caches are checked first; any cached file is reused.
Partial runs are recoverable; re-parsing is free.

Usage
-----
    python scripts/02_scrape_skills_sh_detail.py                  # default: 10 skills
    python scripts/02_scrape_skills_sh_detail.py --limit 1000
    python scripts/02_scrape_skills_sh_detail.py --limit all      # full population
    python scripts/02_scrape_skills_sh_detail.py --input raw/sample_<date>.json
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Detail HTML route confirmed tolerant at 0.05s sustained for 24 reqs (no 429).
# Going 0.1s leaves headroom; natural page-load latency (~0.8s) dominates anyway.
REQUEST_DELAY_SECONDS = 0.1
RATE_LIMIT_BACKOFF_SECONDS = [60, 120, 240]

# SKILL.md path patterns ported from mastra-ai/skills-api fetch-skill.ts.
SKILL_MD_PATH_PATTERNS = [
    "skills/{skillId}/SKILL.md",
    "{skillId}/SKILL.md",
    ".skills/{skillId}/SKILL.md",
    "agent-skills/{skillId}/SKILL.md",
]
SKILL_MD_BRANCHES = ["main", "master"]

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "raw"

# Audit entries are rendered as anchors with href="/security/<slug>" inside
# a panel. Each anchor contains a label span (class includes "truncate") and
# a status span (class includes "rounded"). We discover audits dynamically
# rather than hardcoding service names so new services are picked up too.
AUDIT_BLOCK_RE = re.compile(
    r'href="[^"]*/security/([a-z0-9-]+)"[^>]*>'
    r'.*?<span[^>]*truncate[^>]*>([^<]+)</span>'
    r'.*?<span[^>]*rounded[^>]*>([^<]+)</span>',
    re.S,
)


_TRANSIENT_NETWORK_ERRORS = (
    TimeoutError,
    urllib.error.URLError,  # also catches ConnectionResetError on most platforms
    ConnectionError,
    OSError,
)
_TRANSIENT_BACKOFFS = [5, 15, 45]  # short backoffs for transient net errors


def fetch_html(url: str) -> tuple[int, str]:
    """Return (http_status, html_body).

    Retries on:
      - HTTP 429 with exponential backoff (RATE_LIMIT_BACKOFF_SECONDS)
      - Transient network errors (timeouts, resets) with shorter backoff
    Returns ``(-1, "")`` if all retries are exhausted.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    rate_limit_attempts = 0
    transient_attempts = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if rate_limit_attempts >= len(RATE_LIMIT_BACKOFF_SECONDS):
                    print(f"  ! 429 retries exhausted for {url}", file=sys.stderr)
                    return -1, ""
                wait = RATE_LIMIT_BACKOFF_SECONDS[rate_limit_attempts]
                rate_limit_attempts += 1
                print(f"  ! 429 on {url}; sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            # Non-429 HTTP error — capture body for diagnostics, no retry
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            return e.code, body
        except _TRANSIENT_NETWORK_ERRORS as e:
            if transient_attempts >= len(_TRANSIENT_BACKOFFS):
                print(f"  ! transient errors exhausted on {url}: {type(e).__name__}: {e}", file=sys.stderr)
                return -1, ""
            wait = _TRANSIENT_BACKOFFS[transient_attempts]
            transient_attempts += 1
            print(f"  ! {type(e).__name__} on {url}: {e!s}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue


def safe_filename(skill_url: str, ext: str = ".html") -> str:
    # https://www.skills.sh/anthropics/skills/frontend-design → anthropics__skills__frontend-design{ext}
    path = skill_url.rstrip("/").split("skills.sh/", 1)[-1]
    return path.replace("/", "__") + ext


def simplified_skill_id(skill_id: str) -> str | None:
    """Strip first prefix segment if present (e.g. vercel-react-best-practices → react-best-practices)."""
    parts = skill_id.split("-", 1)
    return parts[1] if len(parts) == 2 and parts[1] else None


def fetch_skill_md(owner: str, repo: str, skill_id: str, delay: float) -> tuple[str | None, str | None, str | None, str | None]:
    """Try mastra's path patterns to locate SKILL.md on raw.githubusercontent.com.
    Returns (status, path, branch, raw_content)."""
    candidates: list[tuple[str, str]] = []
    for sid in [skill_id, simplified_skill_id(skill_id)]:
        if not sid:
            continue
        for branch in SKILL_MD_BRANCHES:
            for pat in SKILL_MD_PATH_PATTERNS:
                candidates.append((branch, pat.replace("{skillId}", sid)))

    for branch, path in candidates:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    return "found", path, branch, r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError:
            pass  # 404 from the pattern; try the next one
        except _TRANSIENT_NETWORK_ERRORS:
            pass  # transient — move on; don't burn the whole skill on one bad attempt
        except Exception:
            pass
        time.sleep(delay)
    return "not_found_via_patterns", None, None, None


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Minimal YAML-front-matter parser. Returns (metadata, body).
    Handles flat key:value pairs and 'key:' nested-block keys (preserved as a single string).
    """
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end < 0:
        return {}, raw
    fm_text = raw[4:end]
    body = raw[end + 4 :].lstrip()
    meta: dict = {}
    current_key = None
    nested_buf: list[str] = []
    for line in fm_text.splitlines():
        if not line.strip():
            if current_key:
                nested_buf.append("")
            continue
        # nested continuation (line starts with whitespace and we're inside a key)
        if current_key and (line.startswith(" ") or line.startswith("\t")):
            nested_buf.append(line)
            continue
        # finalize previous key
        if current_key is not None and nested_buf:
            meta[current_key] = "\n".join(nested_buf).strip()
            nested_buf = []
            current_key = None
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v == "":
                current_key = k
                nested_buf = []
            else:
                # strip matching surrounding quotes
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                meta[k] = v
        else:
            # unparseable line — ignore
            pass
    if current_key is not None and nested_buf:
        meta[current_key] = "\n".join(nested_buf).strip()
    return meta, body


def parse_detail(html: str) -> dict:
    out: dict = {
        "skill_name_h1": None,
        "summary": None,
        "primary_topic_slug": None,
        "primary_topic_label": None,
        "installs_abbreviated": None,
        "github_stars_abbreviated": None,
        "install_command": None,
        "audits": [],
    }

    # <h1 ...>name</h1>
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
    if m:
        out["skill_name_h1"] = html_mod.unescape(m.group(1)).strip()

    # <meta name="description" content="...">
    m = re.search(r'<meta name="description" content="([^"]*)"', html)
    if m:
        out["summary"] = html_mod.unescape(m.group(1)).strip()

    # Primary topic: pill/badge anchor in the article column. The nav sidebar
    # also lists all topics with plain "text-(--ds-gray-700) hover:text-foreground"
    # links — must skip those. Article topic always carries either the
    # "inline-flex ... rounded-full" pill class or the "underline decoration-dotted"
    # inline-link class. Skills without an assigned topic show neither.
    for m in re.finditer(r'<a\b([^>]*\bhref="/topic/([a-z0-9-]+)"[^>]*)>([^<]+)</a>', html):
        attrs, slug, label = m.group(1), m.group(2), m.group(3)
        if ("inline-flex" in attrs and "rounded-full" in attrs) \
                or ("underline" in attrs and "decoration-dotted" in attrs):
            out["primary_topic_slug"] = slug
            out["primary_topic_label"] = html_mod.unescape(label).strip()
            break

    # Sidebar "Installs" label → next K/M-styled value
    m = re.search(
        r'<span>Installs?</span></div>\s*<div[^>]*>(\d+(?:\.\d+)?[KM]?)</div>',
        html,
    )
    if m:
        out["installs_abbreviated"] = m.group(1)

    # Sidebar "GitHub Stars" label → next span value (may be K/M or integer)
    m = re.search(
        r'<span>GitHub Stars</span>.*?<span>([^<]+)</span>',
        html,
        re.S,
    )
    if m:
        out["github_stars_abbreviated"] = m.group(1).strip()

    # Install command from RSC: "command":"npx skills add ..."
    m = re.search(r'\\"command\\":\\"([^\\"]+)\\"', html)
    if m:
        out["install_command"] = m.group(1).replace("\\/", "/")

    # Audit entries — discovered dynamically from /security/{slug} anchors.
    seen = set()
    for m in AUDIT_BLOCK_RE.finditer(html):
        slug = m.group(1)
        if slug in seen:
            continue
        seen.add(slug)
        out["audits"].append({
            "slug": slug,
            "name": html_mod.unescape(m.group(2)).strip(),
            "status": html_mod.unescape(m.group(3)).strip(),
        })

    return out


def enrich_row(skill: dict, parsed: dict, http_status: int, scrape_date: str, md_info: dict) -> dict:
    return {
        "detail_scrape_date": scrape_date,
        "http_status": http_status,
        "skill_url": skill["skill_url"],
        "repo": skill["repo"],
        "skill_id": skill["skill_id"],
        "leaderboard_rank": skill.get("rank"),
        "leaderboard_installs": skill.get("installs"),
        **parsed,
        **md_info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Leaderboard JSON to read skills from (default: latest in raw/)")
    ap.add_argument("--limit", default="10", help="Number of top-ranked skills to scrape, or 'all'")
    ap.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS)
    ap.add_argument("--scrape-date", default=None,
                    help="Override the date stamp used for cache + output filenames "
                         "(default: today). Use this to resume a multi-day scrape "
                         "and preserve the existing cache directory.")
    ap.add_argument("--from-cache-only", action="store_true",
                    help="Do not make any network calls. Rebuild the detail JSON "
                         "from the per-skill cache, skipping skills with no cached "
                         "HTML. Used to regenerate output after an interrupted run.")
    args = ap.parse_args()

    scrape_date = args.scrape_date or date.today().isoformat()
    if args.input:
        input_path = Path(args.input)
    else:
        candidates = sorted(RAW_DIR.glob("skills_sh_leaderboard_*.json"))
        candidates = [p for p in candidates if "_html_" not in p.name]
        if not candidates:
            print("ERROR: no leaderboard file found in raw/", file=sys.stderr)
            return 2
        input_path = candidates[-1]
    print(f"  input: {input_path.relative_to(REPO_ROOT)}", file=sys.stderr)

    with input_path.open() as f:
        leaderboard = json.load(f)
    skills_in = leaderboard["skills"]
    if args.limit != "all":
        skills_in = skills_in[: int(args.limit)]

    html_dir = RAW_DIR / f"skills_sh_detail_html_{scrape_date}"
    md_dir = RAW_DIR / f"skills_sh_skill_md_{scrape_date}"
    html_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"skills_sh_detail_{scrape_date}.json"

    rows: list[dict] = []
    started = time.monotonic()

    n_total = len(skills_in)
    for i, skill in enumerate(skills_in, start=1):
        try:
            _process_skill(i, n_total, skill, args, scrape_date, html_dir, md_dir, rows)
        except Exception as e:
            # Last-resort guard: a bug or one-off failure on a single skill
            # should not nuke the whole run. Log and continue.
            print(f"  !! unhandled exception on rank {skill.get('rank')} "
                  f"({skill.get('repo')}/{skill.get('skill_id')}): "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            rows.append({
                "detail_scrape_date": scrape_date,
                "http_status": -1,
                "skill_url": skill["skill_url"],
                "repo": skill["repo"],
                "skill_id": skill["skill_id"],
                "leaderboard_rank": skill.get("rank"),
                "leaderboard_installs": skill.get("installs"),
                "scrape_error": f"{type(e).__name__}: {e}",
            })

    return _write_output(rows, scrape_date, started, input_path, html_dir, md_dir, out_path)


def _process_skill(i: int, n_total: int, skill: dict, args, scrape_date: str,
                   html_dir: Path, md_dir: Path, rows: list[dict]) -> None:
    """Body of the per-skill scrape — extracted so the outer loop can catch any failure."""
    url = skill["skill_url"]
    html_path = html_dir / safe_filename(url, ".html")
    md_path = md_dir / safe_filename(url, ".md")
    md_meta_path = md_dir / safe_filename(url, ".meta.json")

    if True:  # decorative block; kept for minimal-diff against earlier inline version
        # In cache-only mode, skip any skill we never successfully scraped
        # (no cached HTML) — used to rebuild the detail JSON from disk with
        # zero network calls.
        if getattr(args, "from_cache_only", False) and not html_path.exists():
            return

        # 1) skills.sh detail HTML — fetch or load from cache.
        # Only successful (HTTP 200) responses get cached. Re-runs will
        # retry any URL that 404'd previously, but failures are cheap
        # and idempotent. This avoids the trap of caching error pages
        # and later mistaking them for valid skill pages.
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
            http_status = 200
            cached_html = True
        else:
            http_status, html = fetch_html(url)
            if http_status == 200:
                html_path.write_text(html, encoding="utf-8")
            cached_html = False
            time.sleep(args.delay)

        parsed = parse_detail(html) if http_status == 200 else {
            "skill_name_h1": None, "summary": None, "primary_topic_slug": None,
            "primary_topic_label": None, "installs_abbreviated": None,
            "github_stars_abbreviated": None, "install_command": None, "audits": [],
        }

        # 2) SKILL.md from GitHub — fetch or load from cache.
        # 113/9718 skills come from non-GitHub sources (e.g. open.feishu.cn,
        # skills.volces.com, smithery.ai); their `repo` is a single host
        # token, not owner/repo. GitHub patterns don't apply — flag and skip.
        owner_repo = skill["repo"].split("/", 1)
        is_github_source = len(owner_repo) == 2
        if md_meta_path.exists():
            md_meta = json.loads(md_meta_path.read_text(encoding="utf-8"))
            raw_md = md_path.read_text(encoding="utf-8") if md_path.exists() else None
            cached_md = True
        elif not is_github_source:
            md_meta = {"status": "non_github_source", "path": None, "branch": None}
            md_meta_path.write_text(json.dumps(md_meta) + "\n", encoding="utf-8")
            raw_md = None
            cached_md = False
        elif getattr(args, "from_cache_only", False):
            # No cached MD and cache-only mode: don't fetch; mark as not attempted.
            md_meta = {"status": "not_attempted_cache_only", "path": None, "branch": None}
            raw_md = None
            cached_md = False
        else:
            owner, repo = owner_repo
            status, mpath, mbranch, raw_md = fetch_skill_md(owner, repo, skill["skill_id"], args.delay)
            md_meta = {"status": status, "path": mpath, "branch": mbranch}
            md_meta_path.write_text(json.dumps(md_meta) + "\n", encoding="utf-8")
            if raw_md is not None:
                md_path.write_text(raw_md, encoding="utf-8")
            cached_md = False

        frontmatter, body = ({}, "")
        if raw_md is not None:
            frontmatter, body = parse_frontmatter(raw_md)

        md_info = {
            "skill_md_fetch_status": md_meta["status"],
            "skill_md_path": md_meta["path"],
            "skill_md_branch": md_meta["branch"],
            "skill_md_raw": raw_md,
            "skill_md_frontmatter": frontmatter,
            "skill_md_body": body if raw_md is not None else None,
        }

        row = enrich_row(skill, parsed, http_status, scrape_date, md_info)
        rows.append(row)

        flags = []
        if cached_html: flags.append("Hc")
        if cached_md: flags.append("Mc")
        flag_str = f"({''.join(flags)})" if flags else ""
        n_audits = len(row['audits'])
        audit_summary = "/".join(a['status'][:1] for a in row['audits']) or "-"
        md_status = "FOUND" if raw_md is not None else "MISS"
        print(
            f"  [{i:4d}/{n_total:4d}] HTTP {http_status} "
            f"topic={row['primary_topic_slug']!s:<18} "
            f"installs={row['installs_abbreviated']!s:<8} "
            f"stars={row['github_stars_abbreviated']!s:<8} "
            f"audits={n_audits}[{audit_summary}] "
            f"md={md_status:<5} "
            f"{flag_str} {skill['repo']}/{skill['skill_id']}",
            file=sys.stderr,
        )


def _write_output(rows: list[dict], scrape_date: str, started: float, input_path: Path,
                  html_dir: Path, md_dir: Path, out_path: Path) -> int:
    from collections import Counter
    elapsed = time.monotonic() - started
    ok = sum(1 for r in rows if r["http_status"] == 200)
    miss = lambda field: sum(1 for r in rows if r["http_status"] == 200 and r.get(field) in (None, ""))
    audit_counts = Counter(len(r.get("audits", [])) for r in rows if r["http_status"] == 200)
    payload = {
        "metadata": {
            "scrape_date": scrape_date,
            "input_file": input_path.name,
            "records_attempted": len(rows),
            "records_http_200": ok,
            "scrape_elapsed_seconds": round(elapsed, 1),
            "html_dir": html_dir.name,
            "md_dir": md_dir.name,
            "field_coverage_among_200s": {
                "skill_name_h1": ok - miss("skill_name_h1"),
                "summary": ok - miss("summary"),
                "primary_topic_slug": ok - miss("primary_topic_slug"),
                "installs_abbreviated": ok - miss("installs_abbreviated"),
                "github_stars_abbreviated": ok - miss("github_stars_abbreviated"),
                "install_command": ok - miss("install_command"),
                "any_audits_captured": sum(1 for r in rows if r["http_status"] == 200 and r.get("audits")),
                "skill_md_found": sum(1 for r in rows if r.get("skill_md_raw")),
                "skill_md_frontmatter_nonempty": sum(1 for r in rows if r.get("skill_md_frontmatter")),
            },
            "audit_count_distribution": {str(k): v for k, v in sorted(audit_counts.items())},
            "notes": (
                "Detail page scrape via HTML route /{owner}/{repo}/{skillId} "
                "(robots.txt allows). SKILL.md fetched from raw.githubusercontent.com "
                "via the path patterns from mastra-ai/skills-api (skills/{id}/, "
                "{id}/, .skills/{id}/, agent-skills/{id}/) on branches main+master, "
                "plus simplified-skillId fallback. No GitHub Tree-API fallback (would "
                "need PAT for 60+/hr). Non-GitHub sources are flagged with "
                "skill_md_fetch_status=non_github_source and their skills.sh "
                "detail pages 404 (no detail data captured). Exact stars and "
                "first_seen/last_updated dates NOT captured — not on detail page. "
                "See decisions.md."
            ),
        },
        "skills": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  saved detail to {out_path.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
