# Skills as Codified Knowledge Assets — Data and Code Companion

This repository is the data-and-code companion to a master's thesis investigating when codified know-how in public agent skills becomes reliably executable. The thesis treats the public agent-skill format as a knowledge asset and asks under what conditions such an asset is codifiable, evaluable, and bounded enough for an AI agent to act on it. The companion covers the full pipeline — scrape → sample → code → analyse — on a stratified corpus of 120 skills drawn from a population of ~5,235 skills scraped from skills.sh.

## Structure

- [`README.md`](README.md) — this file.
- [`codebook.md`](codebook.md) — the coding frame: every variable, its 0–4 anchors, and the two derived-score formulas (`codifiability_score`, `evaluability_score`).
- [`data/`](data/) — analysis-ready datasets.
  - [`corpus_120.json`](data/corpus_120.json) — the 120-skill input corpus with metadata and full `skill_md_content`; the artefact actually shown to the LLM coder.
  - [`coded_120.csv`](data/coded_120.csv) — one row per skill: all coded variables, derived codifiability and evaluability scores, and the two rationale fields.
  - [`reliability_24.csv`](data/reliability_24.csv) — the 24-skill double-coding subset; each variable appears three times with `_h1`, `_h2`, `_llm` suffixes (two human coders plus the LLM).
  - [`population_frame.csv`](data/population_frame.csv) — the ~5,235-skill population frame, metadata only (no SKILL.md content): identifiers, URLs, installs, official flag, the three external audit statuses (Snyk, Socket, AgentTrustHub), and the induced category label.
- [`code/`](code/) — the pipeline scripts. Paths and imports are preserved from the working tree; nothing here is meant to be runnable end-to-end out of this directory.
  - [`1_scrape/`](code/1_scrape/) — build the population frame from skills.sh: leaderboard scrape, detail scrape, and the merge that produces the combined population JSON.
  - [`2_sample/`](code/2_sample/) — population profiling, text preparation, LLM-based category induction and classification, and the stratified-sample build with the post-sampling enrichment/correction scripts.
  - [`3_code/`](code/3_code/) — the LLM-assisted coding pipeline (`50_run_pilot.py`, `51_run_full.py`, `52_compile_outputs.py`, `53_descriptive_summary.py`, `_llm.py`) and the coding [`System_Prompt.json`](code/3_code/System_Prompt.json), which defines every variable the LLM emits and is the source of truth for the codebook.
  - [`4_analyze/`](code/4_analyze/) — `_build_datasets.py` and `_style.py` plus the numbered analysis scripts (`02_…` through `09_…`) that consume the coded corpus and emit the tables and figures in `outputs/`.
- [`outputs/`](outputs/) — generated artefacts referenced in the thesis.
  - [`tables/`](outputs/tables/) — CSV result tables, prefixed `02_…` through `09_…` to match the analysis script that produced them.
  - [`figures/`](outputs/figures/) — PNG figures with the same `02_…` to `09_…` prefix scheme (SVG duplicates are excluded).
