"""40 - Enrich the 120-skill sample CSV with scraped data + full SKILL.md.

- Joins the partner's 120-row CSV to the scrape JSON on skill_url, with a
  (repo_name, skill_name) fallback for the 10 rows whose skill_url is a GitHub
  repo URL (decisions.md 2026-05-22). Fallback-matched rows get the correct
  skills.sh skill_url written into the output.
- Adds scrape ecosystem/audit signals + scraped summary; drops source,
  skill_type (already absent), function_description; adds empty
  coordination_burden; blanks the placeholder coding columns.
- Retrieves SKILL.md: Pass A = cached skill_md.raw; Pass B = GitHub Tree-API
  fallback. Writes content inline.

SECURITY: the GitHub token is loaded from .env, used only in request headers,
and never logged/echoed/written. The script prints only its length.

Inputs (read-only): sample/agent_skills_corpus 1.csv (cp1252),
                    inputs/skills_combined_2026-05-20.json
Outputs: sample/agent_skills_corpus_v2_2026-05-22.csv (utf-8, QUOTE_ALL)
         sample/corpus_enrichment_report.md

Run:  python3 scripts/40_enrich_corpus_csv.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
INPUT_CSV = ROOT / "sample" / "agent_skills_corpus 1.csv"   # NB: space, not underscore
JSON_PATH = ROOT / "inputs" / "skills_combined_2026-05-20.json"
OUT_CSV = ROOT / "sample" / "agent_skills_corpus_v2_2026-05-22.csv"
REPORT = ROOT / "sample" / "corpus_enrichment_report.md"

REQUEST_DELAY = 0.1
TIMEOUT = 30
AUDIT_COLS = {"snyk": "snyk_status", "socket": "socket_status",
              "agent-trust-hub": "agent_trust_hub_status"}

FINAL_COLUMNS = [
    # identity (from input CSV)
    "skill_id", "skill_name", "repo_name", "repo_url", "skill_url",
    # ecosystem signals (scrape)
    "installs", "leaderboard_rank", "is_official", "github_stars_abbreviated",
    # summary (scrape)
    "summary",
    # audit signals (scrape)
    "snyk_status", "socket_status", "agent_trust_hub_status",
    # classification
    "domain", "scope",
    # knowledge-type
    "procedural", "analytical", "orchestration", "compliance", "documentation",
    # codifiability
    "explicitness", "tacit_dependency", "context_sensitivity",
    # task horizon
    "coordination_burden",
    # evaluability
    "evaluation_type", "tests_present", "thresholds_defined", "output_verifiability", "error_handling",
    # architecture & dependency
    "requires_memory", "requires_tools", "requires_external_services", "dependency_complexity",
    # governance
    "documentation_quality", "provenance_clarity", "maintenance_signals", "safety_notes", "scope_limits",
    # audit/process
    "coding_status", "audit_confidence", "coding_notes",
    # derived
    "codifiability_score", "codifiability_rationale", "evaluability_score", "evaluability_rationale",
    # skill.md
    "has_skill_md", "skill_md_source", "skill_md_content",
]
# coding columns that carry over from the input but must be EMPTY in v2
EMPTY_CODING = [c for c in FINAL_COLUMNS if c in {
    "scope", "procedural", "analytical", "orchestration", "compliance", "documentation",
    "explicitness", "tacit_dependency", "context_sensitivity", "coordination_burden",
    "evaluation_type", "tests_present", "thresholds_defined", "output_verifiability", "error_handling",
    "requires_memory", "requires_tools", "requires_external_services", "dependency_complexity",
    "documentation_quality", "provenance_clarity", "maintenance_signals", "safety_notes", "scope_limits",
    "coding_status", "audit_confidence", "coding_notes",
    "codifiability_score", "codifiability_rationale", "evaluability_score", "evaluability_rationale",
}]


def stop(msg: str) -> "None":
    print(f"\nSTOP: {msg}")
    sys.exit(1)


def load_token() -> str:
    if not ENV.exists():
        stop(f".env not found at {ENV}")
    token = None
    for line in ENV.read_text().splitlines():
        s = line.strip()
        if s.startswith("GITHUB_TOKEN") and "=" in s:
            token = s.split("=", 1)[1].strip().strip('"').strip("'")
    if not token:
        stop("GITHUB_TOKEN line not found in .env")
    print(f"GITHUB_TOKEN loaded (length: {len(token)})")
    return token


# ---------------------------------------------------------------- GitHub API
class GitHub:
    def __init__(self, token: str):
        self.s = requests.Session()
        self._h = {"Authorization": f"Bearer {token}",
                   "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        self.branch_cache: dict[tuple, str | None] = {}
        self.branch_status: dict[tuple, int] = {}  # (owner,repo)->HTTP status of metadata call
        self.tree_cache: dict[tuple, tuple] = {}  # (owner,repo)->(paths,truncated)

    def _get(self, url: str, raw: bool = False):
        time.sleep(REQUEST_DELAY)
        r = self.s.get(url, headers=self._h, timeout=TIMEOUT)
        if r.status_code in (401, 403):
            stop(f"GitHub returned {r.status_code} (rate limit / bad token). "
                 "Aborting without retrying. URL path: " + url.split('github.com')[-1])
        return r

    def default_branch(self, owner: str, repo: str):
        key = (owner, repo)
        if key not in self.branch_cache:
            r = self._get(f"https://api.github.com/repos/{owner}/{repo}")
            self.branch_status[key] = r.status_code
            self.branch_cache[key] = r.json().get("default_branch") if r.status_code == 200 else None
        return self.branch_cache[key]

    def skillmd_paths(self, owner: str, repo: str, branch: str):
        key = (owner, repo)
        if key not in self.tree_cache:
            r = self._get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
            if r.status_code != 200:
                self.tree_cache[key] = ([], False)
            else:
                j = r.json()
                paths = [e["path"] for e in j.get("tree", [])
                         if e.get("type") == "blob" and e["path"].rsplit("/", 1)[-1] == "SKILL.md"]
                self.tree_cache[key] = (paths, bool(j.get("truncated")))
        return self.tree_cache[key]

    def raw(self, owner: str, repo: str, branch: str, path: str):
        from urllib.parse import quote
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/" + quote(path)
        r = self._get(url, raw=True)
        return r.text if r.status_code == 200 else None


def parse_owner_repo(repo_url: str):
    if not repo_url.startswith("https://github.com/"):
        return None
    parts = repo_url[len("https://github.com/"):].strip("/").split("/")
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def simplified_id(skill_slug: str) -> str:
    return skill_slug.split("-", 1)[1] if "-" in skill_slug else skill_slug


def main():
    token = load_token()
    gh = GitHub(token)

    # ---- load inputs ----
    with open(INPUT_CSV, encoding="cp1252", newline="") as f:
        in_rows = list(csv.DictReader(f))
    print(f"input CSV rows: {len(in_rows)}")
    data = json.loads(JSON_PATH.read_text())
    recs = data["skills"]
    by_url = {r["skill_url"]: r for r in recs}
    by_repo = defaultdict(list)
    for r in recs:
        by_repo[r["repo"]].append(r)
    print("JSON record top-level keys:", sorted(recs[0].keys()))

    # ---- join (skill_url, then (repo,skill_name) fallback) ----
    matched = []   # (in_row, json_rec, join_method, corrected_url_or_None)
    join_fail = []
    for x in in_rows:
        url = (x["skill_url"] or "").strip()
        if url in by_url:
            matched.append((x, by_url[url], "skill_url", None))
            continue
        # fallback on (repo_name, skill_name)
        cands = [r for r in by_repo.get(x["repo_name"], [])
                 if r["skill_name"].strip().lower() == x["skill_name"].strip().lower()]
        if len(cands) == 1:
            matched.append((x, cands[0], "repo+skill_name", cands[0]["skill_url"]))
        else:
            join_fail.append((x, len(cands)))
    if join_fail:
        print(f"\n{len(join_fail)} rows failed BOTH joins:")
        for x, nc in join_fail:
            print(f"   {x['skill_id']} {x['skill_name']!r} repo={x['repo_name']!r} candidates={nc}")
        stop("unresolved join failures — aborting before any writes.")
    print(f"join: {sum(1 for m in matched if m[2]=='skill_url')} by skill_url, "
          f"{sum(1 for m in matched if m[2]=='repo+skill_name')} by fallback")

    # repo counts in the sample (for the Pass-B +1 rule)
    sample_repo_count = Counter(rec["repo"] for _, rec, _, _ in matched)

    # ---- build base rows + audit fields + Pass A ----
    rows_out = []
    passA = passB = unrec = 0
    treeapi_paths = []          # (skill_id, repo, path) for report
    unrec_rows = []             # (skill_id, repo, reason)
    spot = {"cached": None, "tree_api": None, "unrecoverable": None}

    for x, rec, method, corrected_url in matched:
        o = {c: "" for c in FINAL_COLUMNS}
        # identity
        o["skill_id"] = x["skill_id"]
        o["skill_name"] = x["skill_name"]
        o["repo_name"] = x["repo_name"]
        o["repo_url"] = x["repo_url"]
        o["skill_url"] = corrected_url if corrected_url else (x["skill_url"] or "").strip()
        # ecosystem
        o["installs"] = rec.get("installs", "")
        o["leaderboard_rank"] = rec.get("leaderboard_rank", "")
        o["is_official"] = "Y" if rec.get("is_official") else "N"
        o["github_stars_abbreviated"] = rec.get("github_stars_abbreviated") or ""
        # summary
        o["summary"] = rec.get("summary") or ""
        # audits -> lowercase, missing if absent
        astat = {col: "missing" for col in AUDIT_COLS.values()}
        for a in (rec.get("audits") or []):
            col = AUDIT_COLS.get(a.get("slug"))
            if col:
                astat[col] = (a.get("status") or "missing").lower()
        o.update(astat)
        # classification
        o["domain"] = x.get("domain", "")
        # coding columns explicitly blank
        for c in EMPTY_CODING:
            o[c] = ""

        # ---- Pass A: cached skill_md.raw ----
        raw = (rec.get("skill_md") or {}).get("raw")
        if raw:
            o["skill_md_content"] = raw
            o["skill_md_source"] = "cached"
            o["has_skill_md"] = "Y"
            passA += 1
            if spot["cached"] is None:
                spot["cached"] = o
        else:
            o["_needs_passB"] = True
        rows_out.append((o, rec))

    # ---- Pass B: GitHub Tree API for rows still missing ----
    for o, rec in rows_out:
        if not o.pop("_needs_passB", False):
            continue
        repo_url = rec.get("repo_url", "")
        slug = rec.get("skill_id", "")
        owr = parse_owner_repo(repo_url)
        if not owr:
            o["skill_md_content"], o["skill_md_source"], o["has_skill_md"] = "", "unrecoverable", "N"
            unrec += 1; unrec_rows.append((o["skill_id"], o["repo_name"], "non-GitHub source"))
            if spot["unrecoverable"] is None: spot["unrecoverable"] = o
            continue
        owner, repo = owr
        branch = gh.default_branch(owner, repo)
        if not branch:
            st = gh.branch_status.get((owner, repo))
            reason = (f"repo not found on GitHub (HTTP 404 — deleted/renamed since the scrape)"
                      if st == 404 else f"repo metadata fetch failed (HTTP {st})")
            o["skill_md_content"], o["skill_md_source"], o["has_skill_md"] = "", "unrecoverable", "N"
            unrec += 1; unrec_rows.append((o["skill_id"], o["repo_name"], reason))
            if spot["unrecoverable"] is None: spot["unrecoverable"] = o
            continue
        paths, truncated = gh.skillmd_paths(owner, repo, branch)
        simp = simplified_id(slug)
        single_bonus = 1 if (len(paths) == 1 and sample_repo_count[rec["repo"]] == 1) else 0
        scored = []
        for p in paths:
            sc = (10 if slug and slug in p else 0) + (5 if simp and simp in p else 0) + single_bonus
            scored.append((sc, p))
        scored.sort(key=lambda t: -t[0])
        reason = None
        if not scored or scored[0][0] == 0:
            reason = "no SKILL.md candidate scored > 0" + (" (tree truncated)" if truncated else "")
        elif len(scored) > 1 and scored[0][0] == scored[1][0]:
            reason = f"ambiguous: {scored[0][0]}-point tie between {sum(1 for s,_ in scored if s==scored[0][0])} paths"
        if reason:
            o["skill_md_content"], o["skill_md_source"], o["has_skill_md"] = "", "unrecoverable", "N"
            unrec += 1; unrec_rows.append((o["skill_id"], o["repo_name"], reason))
            if spot["unrecoverable"] is None: spot["unrecoverable"] = o
            continue
        best_path = scored[0][1]
        content = gh.raw(owner, repo, branch, best_path)
        if not content:
            o["skill_md_content"], o["skill_md_source"], o["has_skill_md"] = "", "unrecoverable", "N"
            unrec += 1; unrec_rows.append((o["skill_id"], o["repo_name"], "raw fetch failed"))
            if spot["unrecoverable"] is None: spot["unrecoverable"] = o
            continue
        o["skill_md_content"], o["skill_md_source"], o["has_skill_md"] = content, "tree_api", "Y"
        passB += 1
        treeapi_paths.append((o["skill_id"], o["repo_name"], best_path))
        if spot["tree_api"] is None: spot["tree_api"] = o

    out_dicts = [o for o, _ in rows_out]

    # ---- write CSV (utf-8, QUOTE_ALL) ----
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_COLUMNS, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_dicts)

    write_report(in_rows, out_dicts, matched, passA, passB, unrec, treeapi_paths, unrec_rows, spot)
    print(f"\n[40] wrote {OUT_CSV.name} ({len(out_dicts)} rows) and {REPORT.name}")
    print(f"[40] SKILL.md: cached={passA} tree_api={passB} unrecoverable={unrec}")


def write_report(in_rows, out, matched, passA, passB, unrec, treeapi_paths, unrec_rows, spot):
    n = len(out)
    cat = Counter(o["domain"] for o in out)
    offc = Counter(o["is_official"] for o in out)
    fallback = [(x["skill_id"], x["skill_name"], rec["skill_url"])
                for x, rec, m, c in matched if m == "repo+skill_name"]

    def audit_dist(col):
        return dict(Counter(o[col] for o in out).most_common())

    L = []
    L.append("# Corpus enrichment report (v2)")
    L.append("")
    L.append(f"_Input: `sample/{INPUT_CSV.name}` → output: `sample/{OUT_CSV.name}`._")
    L.append("")
    # unrecoverable flagged at top
    L.append("## ⚠️ Unrecoverable SKILL.md (handle manually)")
    L.append("")
    if unrec_rows:
        L.append(f"**{len(unrec_rows)} skill(s)** could not get SKILL.md content:")
        L.append("")
        L.append("| skill_id | repo | reason |")
        L.append("| --- | --- | --- |")
        for sid, repo, reason in unrec_rows:
            L.append(f"| {sid} | {repo} | {reason} |")
    else:
        L.append("**None** — all 120 skills have SKILL.md content.")
    L.append("")
    L.append("## Counts")
    L.append("")
    L.append(f"- Input rows: **{len(in_rows)}** · Output rows: **{n}** (expected 120 / 120)")
    L.append(f"- skill_url join failures (after fallback): **0**")
    L.append(f"- Rescued via (repo, skill_name) fallback: **{len(fallback)}** "
             "(skill_url corrected to the matched skills.sh URL)")
    L.append("")
    L.append("### Per-category counts (expected 10 each)")
    L.append("")
    L.append("| category | n |")
    L.append("| --- | --- |")
    for c, k in sorted(cat.items()):
        L.append(f"| {c} | {k} |")
    L.append("")
    L.append("## New scrape fields — coverage")
    L.append("")
    pop = lambda col: sum(1 for o in out if str(o[col]).strip() != "")
    L.append("| field | non-empty / 120 |")
    L.append("| --- | --- |")
    for col in ["installs", "leaderboard_rank", "is_official", "github_stars_abbreviated", "summary"]:
        L.append(f"| {col} | {pop(col)} |")
    L.append("")
    L.append(f"**is_official:** " + ", ".join(f"{k}={v}" for k, v in offc.most_common()))
    L.append("")
    L.append("### Audit status distribution")
    L.append("")
    L.append("| service | distribution |")
    L.append("| --- | --- |")
    for col in ["snyk_status", "socket_status", "agent_trust_hub_status"]:
        L.append(f"| {col} | " + ", ".join(f"{k}={v}" for k, v in audit_dist(col).items()) + " |")
    L.append("")
    L.append("## SKILL.md retrieval")
    L.append("")
    L.append(f"- Pass A (cached `skill_md.raw`): **{passA}**")
    L.append(f"- Pass B (GitHub Tree API): **{passB}**")
    L.append(f"- Unrecoverable: **{unrec}**")
    L.append("")
    if treeapi_paths:
        L.append("### Pass B — recovered paths (spot-check)")
        L.append("")
        L.append("| skill_id | repo | path |")
        L.append("| --- | --- | --- |")
        for sid, repo, path in treeapi_paths:
            L.append(f"| {sid} | {repo} | `{path}` |")
        L.append("")
    L.append("## Spot-check rows")
    L.append("")
    for kind in ["cached", "tree_api", "unrecoverable"]:
        o = spot[kind]
        L.append(f"### first {kind}")
        if not o:
            L.append("_(none)_"); L.append(""); continue
        L.append(f"- skill_id={o['skill_id']} · skill_name={o['skill_name']} · repo={o['repo_name']}")
        L.append(f"- domain={o['domain']} · installs={o['installs']} · is_official={o['is_official']} "
                 f"· source={o['skill_md_source']} · has_skill_md={o['has_skill_md']}")
        L.append(f"- audits: snyk={o['snyk_status']} socket={o['socket_status']} "
                 f"agent_trust_hub={o['agent_trust_hub_status']}")
        if o["skill_md_content"]:
            snippet = o["skill_md_content"][:300].replace("\n", " ⏎ ")
            L.append(f"- skill_md_content[:300]: {snippet}")
        L.append("")
    REPORT.write_text("\n".join(L).rstrip() + "\n")


if __name__ == "__main__":
    main()