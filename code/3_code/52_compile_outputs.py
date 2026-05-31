#!/usr/bin/env python3
"""Compile coded outputs into a single analysis-ready flat CSV.

Reads coded/coded_<latest_date>.json (or the pilot output with --pilot) and
writes a flat CSV next to it in coded/, named <stem>_final.csv, with one row
per skill in the codebook's required_column_order — the layout the analysis
phase consumes.

A note on layout: the v3 codebook (inputs/System_Prompt.json) is a refined
schema. It differs from the legacy corpus CSV header on a few columns
(category vs domain; tests_quality vs tests_present; thresholds_quality vs
thresholds_defined; has_skill_md added; dependency_complexity dropped). This
script emits the v3 layout (authoritative for the coded data) and prints the
differences against the legacy corpus CSV so the analyst is aware, rather
than silently remapping semantically distinct columns.

NOT executed in the setup session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _llm  # noqa: E402

CODED_DIR = _llm.BASE_DIR / "coded"
PILOT_DIR = _llm.BASE_DIR / "pilot"


def _latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No coded JSON found to compile.")
    return sorted(paths)[-1]  # filenames carry an ISO date -> lexical == chronological


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="Compile the pilot output instead of the full run.")
    args = parser.parse_args()

    if args.pilot:
        src = _latest(list(PILOT_DIR.glob("pilot_coded_*.json")))
    else:
        src = _latest([p for p in CODED_DIR.glob("coded_*.json")
                       if not p.stem.endswith("_final")])

    data = json.loads(src.read_text())
    records = data.get("skills", [])
    print(f"Loaded {len(records)} coded records from {src.name}")

    system_prompt = _llm.load_system_prompt()
    columns = _llm.required_column_order(system_prompt)

    df = pd.DataFrame(records).reindex(columns=columns)
    out_path = CODED_DIR / f"{src.stem}_final.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows x {len(columns)} cols -> {out_path}")

    # Informational: how the v3 layout compares to the legacy corpus CSV.
    corpus_csv = sorted(_llm.INPUTS_DIR.glob("agent_skills_corpus*updated.csv"))
    if corpus_csv:
        legacy = list(pd.read_csv(corpus_csv[-1], nrows=0).columns)
        only_v3 = [c for c in columns if c not in legacy]
        only_legacy = [c for c in legacy if c not in columns]
        if only_v3 or only_legacy:
            print("\nColumn differences vs legacy corpus CSV "
                  f"({corpus_csv[-1].name}):")
            if only_v3:
                print(f"  only in v3 coded layout: {only_v3}")
            if only_legacy:
                print(f"  only in legacy corpus CSV: {only_legacy}")
            print("  (renames likely: category<-domain, tests_quality<-tests_present, "
                  "thresholds_quality<-thresholds_defined.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())