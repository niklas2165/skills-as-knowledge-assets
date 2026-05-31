"""21 - Method B step 1: LLM proposes categories from a random sample (GPT-4o).

Takes a fixed random sample of 200 cleaned summaries (seed 42) from the 4,603
usable records and asks GPT-4o to propose 8-25 bottom-up categories.

Output: analysis/clustering/method_b_categories.json
        analysis/clustering/usage_method_b_propose.json

Run:  python3 scripts/21_llm_propose_categories.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import _llm

SEED = 42
N_SAMPLE = 200
MODEL = "gpt-4o"
CLUSTER_DIR = Path(__file__).resolve().parent.parent / "analysis" / "clustering"
PROCESSED = Path(__file__).resolve().parent.parent / "processed" / "skills_with_text_2026-05-20.json"

SYSTEM = (
    "You analyse a corpus of AI agent skills to derive bottom-up functional categories. "
    "Work only from the summaries provided. Do not impose any predefined industry taxonomy. "
    "Respond with a single valid JSON object and nothing else."
)

INSTRUCTION = (
    "Read these {n} agent-skill summaries and propose a set of categories that describe "
    "what they're for. Aim for between 8 and 25 categories. Categories should be "
    "distinguishable from each other and each should fit multiple skills. "
    'Return a JSON object with one key "categories", whose value is a list. Each list '
    "item must have: "
    '"name" (short, kebab-case), '
    '"definition" (one sentence), '
    '"example_summary_indices" (a list of 3-5 indices, from the input list, of example '
    "skills that fit). Use only indices that appear in the input."
)


def main():
    with open(PROCESSED) as fh:
        skills = json.load(fh)["skills"]
    usable = [r for r in skills if r.get("summary_usable")]
    rng = random.Random(SEED)
    sample = rng.sample(usable, N_SAMPLE)

    numbered = "\n".join(f"{i}. {r['summary_clean']}" for i, r in enumerate(sample))
    user = (INSTRUCTION.format(n=N_SAMPLE)
            + f"\n\nHere are the {N_SAMPLE} summaries, indexed 0-{N_SAMPLE - 1}:\n" + numbered)

    usage = _llm.Usage()
    client = _llm.get_client()
    print(f"[21] proposing categories from {N_SAMPLE} summaries with {MODEL}...")
    obj = _llm.chat_json(client, MODEL, SYSTEM, user, usage=usage)
    cats = obj.get("categories", [])
    print(f"[21] {len(cats)} categories proposed")
    for c in cats:
        print(f"   - {c.get('name')}: {c.get('definition')}")

    out = {
        "method": "B step 1: LLM category proposal",
        "model": MODEL, "seed": SEED, "n_sample": N_SAMPLE,
        "categories": cats,
        "sample": [{"index": i, "skill_url": r["skill_url"], "repo": r["repo"],
                    "summary_clean": r["summary_clean"]} for i, r in enumerate(sample)],
    }
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    (CLUSTER_DIR / "method_b_categories.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    usage.save(CLUSTER_DIR / "usage_method_b_propose.json")
    print(f"[21] wrote method_b_categories.json | usage: {usage.summary()}")


if __name__ == "__main__":
    main()