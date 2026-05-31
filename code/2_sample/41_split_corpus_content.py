"""41 - Split the enriched corpus into a clean CSV + a content-rich JSON.

Per user feedback (2026-05-22):
- The CSV must NOT carry the full SKILL.md text (it spills across rows when the
  CSV is opened in Excel). URLs are sufficient there. Drop `skill_md_content`
  and `skill_md_source`; keep `summary`, `has_skill_md`, and all 120 skills in
  corpus order.
- The full SKILL.md content lives in a NEW JSON file alongside the CSV.

Order of operations: read the current (content-bearing) CSV fully into memory,
write the JSON first (so content is preserved), then overwrite the CSV with the
cleaned version.

Inputs:  sample/agent_skills_corpus_v2_2026-05-22.csv  (current, with content)
Outputs: sample/agent_skills_corpus_v2_2026-05-22.json (all fields + full content)
         sample/agent_skills_corpus_v2_2026-05-22.csv  (cleaned, overwritten)

Run:  python3 scripts/41_split_corpus_content.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "sample" / "agent_skills_corpus_v2_2026-05-22.csv"
JSON_PATH = ROOT / "sample" / "agent_skills_corpus_v2_2026-05-22.json"

DROP_FROM_CSV = ["skill_md_content", "skill_md_source"]   # CSV only
INT_FIELDS = ["installs", "leaderboard_rank"]


def main():
    if not CSV_PATH.exists():
        sys.exit(f"STOP: {CSV_PATH} not found")
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    n = len(rows)
    if "skill_md_content" not in cols:
        sys.exit("STOP: CSV has no skill_md_content column — already split? Aborting "
                 "to avoid producing a JSON without content.")
    print(f"read {n} rows, {len(cols)} columns")

    # ---- build JSON records (all fields incl. full content; ints cast) ----
    json_records = []
    for r in rows:
        rec = {}
        for c in cols:
            v = r.get(c, "")
            if c in INT_FIELDS and str(v).strip() != "":
                try:
                    v = int(v)
                except ValueError:
                    pass
            rec[c] = v
        json_records.append(rec)

    n_with = sum(1 for r in rows if (r.get("has_skill_md") or "").strip().upper() == "Y")
    payload = {
        "metadata": {
            "generated": date.today().isoformat(),
            "source_csv": CSV_PATH.name,
            "n_records": n,
            "n_with_skill_md": n_with,
            "n_without_skill_md": n - n_with,
            "note": ("Content-rich corpus: every field from the cleaned review CSV plus the "
                     "full SKILL.md text in skill_md_content (empty for skills with "
                     "has_skill_md=N). skill_md_source kept here as provenance "
                     "(cached / tree_api / unrecoverable). The companion CSV omits "
                     "skill_md_content and skill_md_source."),
        },
        "skills": json_records,
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {JSON_PATH.name}: {n} records, {n_with} with full SKILL.md content")

    # ---- write cleaned CSV (drop content + source), preserve order ----
    keep_cols = [c for c in cols if c not in DROP_FROM_CSV]
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep_cols, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in keep_cols})
    print(f"updated {CSV_PATH.name}: dropped {DROP_FROM_CSV}, now {len(keep_cols)} columns, {n} rows")
    print(f"   kept summary={'summary' in keep_cols}  has_skill_md={'has_skill_md' in keep_cols}")


if __name__ == "__main__":
    main()