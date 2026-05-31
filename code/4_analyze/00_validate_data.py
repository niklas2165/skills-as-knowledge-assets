#!/usr/bin/env python3
"""
00_validate_data.py — sanity gate for the consolidated working datasets.

Self-contained: reads ONLY from Coding/analysis/data/. Run this at the start of
every analysis session. It asserts the structural invariants the downstream
analyses rely on, then prints a short descriptive report. Any failed assertion
raises and stops the run (non-zero exit).

Usage:
    python3 scripts/00_validate_data.py
"""

import sys

import pandas as pd

ROOT = "/Users/Niklas/Master Thesis Claude Code "
DATA = ROOT + "/Coding/analysis/data"

EXPECTED_POPULATION = 5235
EXPECTED_CORPUS = 120
EXPECTED_DOUBLE_CODED = 24
EXPECTED_PER_CATEGORY = 10
EXPECTED_CATEGORIES = 12

# Fields permitted to be null (free-text rationales / notes left blank by coders).
CORPUS_NULLABLE = {"codifiability_rationale", "evaluability_rationale", "coding_notes"}
DC_NULLABLE_BASES = {"codifiability_rationale", "evaluability_rationale", "coding_notes"}
DC_NULLABLE_EXTRA = {"codifiability_score_h2_provided", "evaluability_score_h2_provided"}


def hr(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    failures = []

    def check(cond, msg):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    # ---- load ----
    hr("LOAD")
    population = pd.read_parquet(DATA + "/population.parquet")
    corpus = pd.read_parquet(DATA + "/corpus.parquet")
    dc = pd.read_parquet(DATA + "/double_coded.parquet")
    cat_map = pd.read_csv(DATA + "/category_mapping.csv")
    codebook = pd.read_csv(DATA + "/codebook.csv")
    print(f"  population:      {population.shape[0]} rows x {population.shape[1]} cols")
    print(f"  corpus:          {corpus.shape[0]} rows x {corpus.shape[1]} cols")
    print(f"  double_coded:    {dc.shape[0]} rows x {dc.shape[1]} cols")
    print(f"  category_mapping:{cat_map.shape[0]} rows")
    print(f"  codebook:        {codebook.shape[0]} variables")

    # ---- row counts ----
    hr("ROW COUNTS")
    check(len(population) == EXPECTED_POPULATION,
          f"population has {len(population)} rows (expected {EXPECTED_POPULATION})")
    check(len(corpus) == EXPECTED_CORPUS,
          f"corpus has {len(corpus)} rows (expected {EXPECTED_CORPUS})")
    check(len(dc) == EXPECTED_DOUBLE_CODED,
          f"double_coded has {len(dc)} rows (expected {EXPECTED_DOUBLE_CODED})")
    check(len(cat_map) == EXPECTED_CATEGORIES,
          f"category_mapping has {len(cat_map)} rows (expected {EXPECTED_CATEGORIES})")

    # ---- referential integrity ----
    hr("REFERENTIAL INTEGRITY")
    pop_urls = set(population["skill_url"])
    corpus_urls = set(corpus["skill_url"])
    missing_in_pop = corpus_urls - pop_urls
    check(not missing_in_pop,
          f"all {len(corpus)} corpus skills exist in population by skill_url "
          f"(missing: {sorted(missing_in_pop) if missing_in_pop else 'none'})")

    corpus_ids = set(corpus["skill_id"])
    dc_ids = set(dc["skill_id"])
    missing_in_corpus = dc_ids - corpus_ids
    check(not missing_in_corpus,
          f"all {len(dc)} double_coded skills exist in corpus by skill_id "
          f"(missing: {sorted(missing_in_corpus) if missing_in_corpus else 'none'})")

    flagged = set(corpus.loc[corpus["in_double_coded_subset"], "skill_id"])
    check(flagged == dc_ids,
          f"in_double_coded_subset flag ({len(flagged)}) matches double_coded ids "
          f"(symmetric_diff: {sorted(flagged ^ dc_ids) if flagged ^ dc_ids else 'none'})")

    # ---- category coverage ----
    hr("CATEGORY COVERAGE")
    map_cats = set(cat_map["category"])
    corpus_cats = set(corpus["category"])
    uncovered = corpus_cats - map_cats
    check(not uncovered,
          f"every corpus category appears in category_mapping "
          f"(uncovered: {sorted(uncovered) if uncovered else 'none'})")
    check(corpus_cats == map_cats,
          f"corpus categories == category_mapping categories ({len(corpus_cats)} each)")

    # ---- nullity ----
    hr("REQUIRED-FIELD NULLITY")
    corpus_req = [c for c in corpus.columns if c not in CORPUS_NULLABLE]
    corpus_bad = {c: int(corpus[c].isna().sum()) for c in corpus_req if corpus[c].isna().any()}
    check(not corpus_bad,
          f"corpus has no nulls in required fields "
          f"({'offenders: ' + str(corpus_bad) if corpus_bad else 'clean'})")

    dc_nullable = set(DC_NULLABLE_EXTRA)
    for base in DC_NULLABLE_BASES:
        for suf in ("_h1", "_h2", "_llm"):
            dc_nullable.add(base + suf)
    dc_req = [c for c in dc.columns if c not in dc_nullable]
    dc_bad = {c: int(dc[c].isna().sum()) for c in dc_req if dc[c].isna().any()}
    check(not dc_bad,
          f"double_coded has no nulls in required fields "
          f"({'offenders: ' + str(dc_bad) if dc_bad else 'clean'})")

    # ====================================================================
    # DESCRIPTIVE REPORT
    # ====================================================================
    hr("REPORT — corpus category distribution")
    vc = corpus["category"].value_counts().sort_index()
    for cat, n in vc.items():
        mark = "" if n == EXPECTED_PER_CATEGORY else "  <-- not 10"
        print(f"  {cat:24} {n}{mark}")
    check((vc == EXPECTED_PER_CATEGORY).all(),
          f"all {EXPECTED_CATEGORIES} categories have exactly "
          f"{EXPECTED_PER_CATEGORY} skills")

    hr("REPORT — corpus audit-status distribution (lowercased)")
    for col in ["snyk_status", "socket_status", "agent_trust_hub_status"]:
        dist = dict(corpus[col].value_counts(dropna=False).sort_index())
        print(f"  {col:24} {dist}")

    hr("REPORT — corpus coding_status")
    print(f"  {dict(corpus['coding_status'].value_counts())}")
    print(f"  audit_confidence: {dict(corpus['audit_confidence'].value_counts().sort_index())}")

    hr("REPORT — population Method B coverage & audit presence")
    n_no_mb = int(population["method_b_category"].isna().sum())
    print(f"  population rows without a Method B category: {n_no_mb} "
          f"({100*n_no_mb/len(population):.1f}%)")
    for col in ["snyk_status", "socket_status", "agent_trust_hub_status"]:
        n_null = int(population[col].isna().sum())
        print(f"  population {col:24} null (no audit): {n_null}")
    print(f"  population has_skill_md: {dict(population['has_skill_md'].value_counts())}")

    hr("REPORT — H1 vs H2 vs LLM derived scores (side-by-side spot-check)")
    print("  skill  category                 | codif: h1 h2 llm (h2_prov) | eval: h1 h2 llm (h2_prov)")
    cols = ["skill_id", "category",
            "codifiability_score_h1", "codifiability_score_h2", "codifiability_score_llm",
            "codifiability_score_h2_provided",
            "evaluability_score_h1", "evaluability_score_h2", "evaluability_score_llm",
            "evaluability_score_h2_provided"]
    for _, r in dc[cols].iterrows():
        cp = r["codifiability_score_h2_provided"]
        ep = r["evaluability_score_h2_provided"]
        cflag = " *" if cp != r["codifiability_score_h2"] else ""
        eflag = " *" if ep != r["evaluability_score_h2"] else ""
        print(f"  {r['skill_id']:5}  {r['category']:24} |"
              f"       {int(r['codifiability_score_h1'])}  {int(r['codifiability_score_h2'])}  "
              f"{int(r['codifiability_score_llm'])}  ({cp}){cflag:2} |"
              f"      {int(r['evaluability_score_h1'])}  {int(r['evaluability_score_h2'])}  "
              f"{int(r['evaluability_score_llm'])}  ({ep}){eflag}")
    print("  ('*' marks where H2's hand-entered score differed from the round-half-up recompute)")

    # ---- verdict ----
    hr("VERDICT")
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        print("\n  STOP: working data is NOT valid. Do not proceed to analyses.")
        return 1
    print("  ALL CHECKS PASSED. Working data is valid and ready for analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
