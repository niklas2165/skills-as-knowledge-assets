"""22 - Method B step 2: classify all 4,603 usable skills (GPT-4o-mini).

Reads the categories proposed in step 1 and assigns each of the 4,603
summary_usable=true skills a primary category (or "none") and an optional
secondary. Chunked (CHUNK summaries/call) + concurrent for speed/cost; every
skill is validated to receive an assignment (missing indices retried).

Outputs: analysis/clustering/method_b_results.json
         analysis/clustering/method_b_summary.md
         analysis/clustering/usage_method_b_classify.json

Run:  python3 scripts/22_llm_classify.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import _lib as L
import _llm

MODEL = "gpt-4o-mini"
CHUNK = 25
MAX_WORKERS = 8
CHUNK_RETRIES = 2
SEED = 42
N_EXAMPLES = 5
CLUSTER_DIR = Path(__file__).resolve().parent.parent / "analysis" / "clustering"
PROCESSED = Path(__file__).resolve().parent.parent / "processed" / "skills_with_text_2026-05-20.json"

SYSTEM = ("You classify AI agent skills into a fixed set of categories by what each skill "
          "is for. Respond with a single valid JSON object and nothing else.")


def build_user(cats_block: str, chunk_rows) -> str:
    summaries = "\n".join(f"{i}. {r['summary_clean']}" for i, r in enumerate(chunk_rows))
    return (
        "Categories (name: definition):\n" + cats_block + "\n\n"
        "For each summary below, assign exactly one PRIMARY category by its name. "
        "Optionally add ONE SECONDARY category name if a second clearly applies, else null. "
        'If no category fits, use "none" as the primary. '
        'Return JSON: {"results": [{"index": <int>, "primary": "<name|none>", '
        '"secondary": "<name|null>"}]} with exactly one entry per summary, using the '
        "indices shown.\n\nSummaries:\n" + summaries
    )


def classify_chunk(client, cats_block, valid_names, chunk_rows, usage):
    """Return {local_index: (primary, secondary)} for the chunk, validated."""
    want = set(range(len(chunk_rows)))
    got: dict[int, tuple] = {}
    for _ in range(CHUNK_RETRIES + 1):
        missing_rows = [chunk_rows[i] for i in sorted(want - set(got))]
        if not missing_rows:
            break
        # re-index missing rows locally for the retry call
        idx_map = sorted(want - set(got))
        obj = _llm.chat_json(client, MODEL, SYSTEM, build_user(cats_block, missing_rows), usage=usage)
        for item in obj.get("results", []):
            li = item.get("index")
            if not isinstance(li, int) or li < 0 or li >= len(missing_rows):
                continue
            gi = idx_map[li]
            primary = (item.get("primary") or "none").strip()
            if primary not in valid_names and primary != "none":
                primary = "none"
            secondary = item.get("secondary")
            if secondary not in valid_names:
                secondary = None
            got[gi] = (primary, secondary)
    # any still missing -> "none" (recorded as invalid/unparsed)
    for i in want - set(got):
        got[i] = ("none", None)
    return got


def main():
    with open(PROCESSED) as fh:
        skills = json.load(fh)["skills"]
    rows = [r for r in skills if r.get("summary_usable")]
    n = len(rows)

    cat_doc = json.loads((CLUSTER_DIR / "method_b_categories.json").read_text())
    cats = cat_doc["categories"]
    valid_names = {c["name"] for c in cats}
    cats_block = "\n".join(f"- {c['name']}: {c['definition']}" for c in cats)
    print(f"[22] classifying {n:,} skills into {len(cats)} categories with {MODEL} "
          f"({CHUNK}/call, {MAX_WORKERS} workers)...")

    chunks = [(s, rows[s:s + CHUNK]) for s in range(0, n, CHUNK)]
    usage = _llm.Usage()
    client = _llm.get_client()
    assign: dict[int, tuple] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(classify_chunk, client, cats_block, valid_names, rws, usage): start
                for start, rws in chunks}
        for fut in as_completed(futs):
            start = futs[fut]
            for li, pair in fut.result().items():
                assign[start + li] = pair
            done += 1
            if done % 25 == 0 or done == len(chunks):
                print(f"   {done}/{len(chunks)} chunks | {usage.summary()}")

    primary = [assign[i][0] for i in range(n)]
    secondary = [assign[i][1] for i in range(n)]
    pc = Counter(primary)
    n_none = pc.get("none", 0)
    n_secondary = sum(1 for s in secondary if s)

    assignments = [{"skill_url": rows[i]["skill_url"], "repo": rows[i]["repo"],
                    "skill_id": rows[i]["skill_id"],
                    "primary": primary[i], "secondary": secondary[i]} for i in range(n)]
    results = {
        "method": "B step 2: LLM classification", "model": MODEL,
        "chunk_size": CHUNK, "n_records": n, "n_categories": len(cats),
        "category_names": sorted(valid_names),
        "primary_counts": dict(pc.most_common()),
        "n_none": n_none, "n_with_secondary": n_secondary,
        "assignments": assignments,
    }
    (CLUSTER_DIR / "method_b_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    usage.save(CLUSTER_DIR / "usage_method_b_classify.json")

    # ---- summary markdown ----
    cat_def = {c["name"]: c["definition"] for c in cats}
    sec_counts = Counter(s for s in secondary if s)
    rng = random.Random(SEED)
    md = ["# Method B — LLM-derived categories", ""]
    md.append(f"_GPT-4o proposed categories from 200 sampled summaries; GPT-4o-mini "
              f"classified all **{n:,}** usable skills. \"none\" allowed; one optional "
              "secondary category per skill._")
    md += ["", "## Categories & sizes (by primary assignment)", ""]
    md.append(L.md_table(
        ["Category", "Primary n", "Share", "As secondary", "Definition"],
        [[name, L.fmt_int(pc.get(name, 0)), L.fmt_pct(pc.get(name, 0) / n),
          L.fmt_int(sec_counts.get(name, 0)), cat_def.get(name, "")]
         for name in sorted(valid_names, key=lambda x: -pc.get(x, 0))]
        + [["none", L.fmt_int(n_none), L.fmt_pct(n_none / n), "—", "no proposed category fit"]]))
    md += ["", "## Totals", ""]
    md.append(f"- Categories proposed: **{len(cats)}**")
    md.append(f"- Skills classified \"none of these\": **{L.fmt_int(n_none)}** ({n_none / n:.1%})")
    md.append(f"- Skills given a secondary category: **{L.fmt_int(n_secondary)}** ({n_secondary / n:.1%})")
    md += ["", "## Example skills per category (up to {} random each)".format(N_EXAMPLES), ""]
    by_cat = {}
    for i in range(n):
        by_cat.setdefault(primary[i], []).append(i)
    for name in sorted(valid_names, key=lambda x: -pc.get(x, 0)):
        members = by_cat.get(name, [])
        md.append(f"**{name}** (n={len(members)}) — _{cat_def.get(name, '')}_")
        for i in rng.sample(members, min(N_EXAMPLES, len(members))):
            md.append(f"  - `{rows[i]['repo']}` — {rows[i]['summary_clean'][:140]}")
        md.append("")
    (CLUSTER_DIR / "method_b_summary.md").write_text("\n".join(md).rstrip() + "\n")

    print(f"[22] done. none={n_none} ({n_none/n:.1%}) secondary={n_secondary} "
          f"| usage: {usage.summary()}")
    print(f"[22] wrote method_b_results.json, method_b_summary.md")


if __name__ == "__main__":
    main()