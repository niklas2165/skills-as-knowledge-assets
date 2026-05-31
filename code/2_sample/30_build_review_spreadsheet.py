"""30 - Build the Phase-4 review spreadsheet for sampling selection.

Joins the population with Method B category assignments (on skill_url), keeps
the 4,603 summary_usable=true records, and produces a per-category review
workbook plus a flat CSV mirror and a short report.

Method A is NOT used here (dropped from the sampling pipeline, decisions.md
2026-05-21).

Inputs (read-only):
  processed/skills_with_text_2026-05-20.json
  analysis/clustering/method_b_results.json

Outputs:
  sample/skills_review_2026-05-20.xlsx   (one tab per category + _summary)
  sample/skills_review_2026-05-20.csv    (flat mirror, all categories)
  analysis/profile/review_spreadsheet_report.md

Run:  python3 scripts/30_build_review_spreadsheet.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

import _lib as L

DATA_DATE = "2026-05-20"
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed" / f"skills_with_text_{DATA_DATE}.json"
METHOD_B = ROOT / "analysis" / "clustering" / "method_b_results.json"
SAMPLE_DIR = ROOT / "sample"
XLSX = SAMPLE_DIR / f"skills_review_{DATA_DATE}.xlsx"
CSV = SAMPLE_DIR / f"skills_review_{DATA_DATE}.csv"
REPORT = ROOT / "analysis" / "profile" / "review_spreadsheet_report.md"

MAX_PER_REPO = 2
NONE_CAT = "none"

COLUMNS = [
    "selected", "suggested_default", "category", "secondary_category",
    "skill_name", "repo", "leaderboard_rank", "installs", "is_official",
    "summary_clean", "snyk_status", "socket_status", "agent_trust_hub_status",
    "skill_url", "repo_url",
]
COL_WIDTHS = {
    "selected": 8, "suggested_default": 16, "category": 24, "secondary_category": 22,
    "skill_name": 28, "repo": 32, "leaderboard_rank": 16, "installs": 12,
    "is_official": 11, "summary_clean": 90, "snyk_status": 11, "socket_status": 12,
    "agent_trust_hub_status": 20, "skill_url": 50, "repo_url": 45,
}
AUDIT_SLUGS = {"snyk": "snyk_status", "socket": "socket_status",
               "agent-trust-hub": "agent_trust_hub_status"}


def audit_statuses(rec) -> dict:
    """Map each of the three services to Pass/Warn/Fail, else 'missing'."""
    out = {col: "missing" for col in AUDIT_SLUGS.values()}
    for a in (rec.get("audits") or []):
        col = AUDIT_SLUGS.get(a.get("slug"))
        if col:
            out[col] = a.get("status", "missing")
    return out


def excel_safe(title: str) -> str:
    """Excel sheet titles: <=31 chars, none of []:*?/\\ ."""
    for ch in "[]:*?/\\":
        title = title.replace(ch, "-")
    return title[:31]


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    pop = json.loads(PROCESSED.read_text())["skills"]
    by_url = {r["skill_url"]: r for r in pop}
    bres = json.loads(METHOD_B.read_text())
    assign = {a["skill_url"]: a for a in bres["assignments"]}

    # join: usable records that have a Method B assignment
    rows = []
    for url, a in assign.items():
        rec = by_url[url]
        if not rec.get("summary_usable"):
            continue  # defensive; assignments are only over usable records
        st = audit_statuses(rec)
        rows.append({
            "selected": "",
            "suggested_default": None,  # filled below
            "category": a["primary"],
            "secondary_category": a.get("secondary") or "",
            "skill_name": rec["skill_name"],
            "repo": rec["repo"],
            "leaderboard_rank": rec["leaderboard_rank"],
            "installs": rec["installs"],
            "is_official": rec["is_official"],
            "summary_clean": rec["summary_clean"],
            "snyk_status": st["snyk_status"],
            "socket_status": st["socket_status"],
            "agent_trust_hub_status": st["agent_trust_hub_status"],
            "skill_url": rec["skill_url"],
            "repo_url": rec["repo_url"],
        })

    # suggested_default: top-MAX_PER_REPO by leaderboard_rank within each
    # (category, repo); tie-break installs desc, then skill_name asc.
    groups = defaultdict(list)
    for r in rows:
        groups[(r["category"], r["repo"])].append(r)
    for grp in groups.values():
        grp.sort(key=lambda r: (r["leaderboard_rank"], -r["installs"], r["skill_name"]))
        for i, r in enumerate(grp):
            r["suggested_default"] = "Y" if i < MAX_PER_REPO else "N"

    df = pd.DataFrame(rows, columns=COLUMNS)

    # category order: by size desc, "none" forced last
    sizes = df["category"].value_counts()
    cats = [c for c in sizes.index if c != NONE_CAT]
    if NONE_CAT in sizes.index:
        cats.append(NONE_CAT)

    # ---- per-category summary ----
    summary_rows = []
    for c in cats:
        sub = df[df["category"] == c]
        summary_rows.append({
            "category": c,
            "total_skills": len(sub),
            "unique_repos": sub["repo"].nunique(),
            "suggested_defaults": int((sub["suggested_default"] == "Y").sum()),
            "median_installs": int(statistics.median(sub["installs"])),
            "median_leaderboard_rank": int(statistics.median(sub["leaderboard_rank"])),
        })
    summary_df = pd.DataFrame(summary_rows)
    total_default = int((df["suggested_default"] == "Y").sum())
    summary_df.loc[len(summary_df)] = {
        "category": "TOTAL", "total_skills": len(df),
        "unique_repos": df["repo"].nunique(), "suggested_defaults": total_default,
        "median_installs": "", "median_leaderboard_rank": "",
    }

    # ---- write xlsx ----
    with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
        summary_df.to_excel(xw, sheet_name="_summary", index=False)
        used_titles = {"_summary"}
        cat_to_sheet = {}
        for c in cats:
            title = excel_safe(c)
            base, k = title, 2
            while title in used_titles:
                title = excel_safe(f"{base}_{k}"); k += 1
            used_titles.add(title)
            cat_to_sheet[c] = title
            sub = df[df["category"] == c].sort_values("leaderboard_rank").reset_index(drop=True)
            sub.to_excel(xw, sheet_name=title, index=False)

        # formatting: freeze header, column widths
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            if ws.title == "_summary":
                for i, col in enumerate(summary_df.columns, start=1):
                    ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
                        max(14, len(col) + 2)
            else:
                for i, col in enumerate(COLUMNS, start=1):
                    ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
                        COL_WIDTHS.get(col, 16)

    # ---- write flat CSV mirror (ordered by category size, then rank) ----
    cat_order = {c: i for i, c in enumerate(cats)}
    df_csv = df.assign(_o=df["category"].map(cat_order)).sort_values(
        ["_o", "leaderboard_rank"]).drop(columns="_o")
    df_csv.to_csv(CSV, index=False)

    # ---- report ----
    n = len(df)
    low_repo = [r for r in summary_rows if r["unique_repos"] < 25]
    weak_rule = [r for r in summary_rows if r["total_skills"] == r["suggested_defaults"]]
    md = ["# Review-spreadsheet report — Phase 4", ""]
    md.append(f"_Generated by `scripts/30_build_review_spreadsheet.py` from "
              f"`processed/skills_with_text_{DATA_DATE}.json` + "
              f"`analysis/clustering/method_b_results.json`. "
              f"Outputs: `sample/{XLSX.name}` (+ `{CSV.name}`)._")
    md += ["", "## Totals", ""]
    md.append(f"- Records included (summary_usable=true, with a Method B category): **{n:,}** "
              "(632 broken-summary records excluded).")
    md.append(f"- `suggested_default = Y` (max {MAX_PER_REPO}/repo/category): "
              f"**{total_default:,}** ({total_default/n:.1%}) — the rough sample ceiling "
              "under the diversity rule.")
    md.append(f"- Categories: **{len(cats)}** (incl. `none`).")
    md += ["", "## Per-category counts", ""]
    md.append(L.md_table(
        ["Category", "Total", "Unique repos", "Suggested defaults", "Median installs", "Median rank"],
        [[r["category"], L.fmt_int(r["total_skills"]), L.fmt_int(r["unique_repos"]),
          L.fmt_int(r["suggested_defaults"]), L.fmt_int(r["median_installs"]),
          L.fmt_int(r["median_leaderboard_rank"])] for r in summary_rows]))
    md += ["", "## Anomalies / flags", ""]
    md.append(f"- **Sampling-ceiling check:** {total_default:,} suggested defaults is "
              f"{'ABOVE 1,000' if total_default >= 1000 else 'within range'} — the 100-skill "
              f"target is {'easily reachable' if total_default >= 100 else 'NOT reachable'} "
              "from the default set. The max-2-per-repo rule is a weak constraint at this "
              "scale (it only drops 3rd+ same-repo skills per category); per-category sample "
              "sizes still need to be chosen during review.")
    md.append(f"- **Low publisher diversity** (<25 unique repos): " +
              (", ".join(f"`{r['category']}` ({r['unique_repos']} repos, "
                         f"{r['total_skills']} skills)" for r in
                         sorted(low_repo, key=lambda r: r['unique_repos'])) or "none") + ".")
    md.append("  `business-strategy` is the most concentrated (121 skills from 21 repos), the "
              "kind of vendor concentration the diversity rule is meant to curb.")
    if weak_rule:
        md.append("- **Diversity rule changes nothing** (every skill is already a suggested "
                  "default) in: " + ", ".join(f"`{r['category']}`" for r in weak_rule) +
                  " — these categories have ≤2 skills per repo throughout.")
    else:
        md.append("- The diversity rule trims at least one skill in every category "
                  "(no category has ≤2 skills per repo throughout).")

    REPORT.write_text("\n".join(md).rstrip() + "\n")

    print(f"[30] rows={n} suggested_default_Y={total_default} ({total_default/n:.1%}) "
          f"categories={len(cats)}")
    print(f"[30] wrote {XLSX.name}, {CSV.name}, {REPORT.name}")


if __name__ == "__main__":
    main()