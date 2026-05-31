# Codebook

Coding frame for the 120-skill corpus. Each skill is one row; each variable is one column in [`data/coded_120.csv`](data/coded_120.csv). Source of truth for variable definitions is the LLM coding system prompt at [`code/3_code/System_Prompt.json`](code/3_code/System_Prompt.json); derived-score formulas live in [`code/3_code/_llm.py`](code/3_code/_llm.py).

Most variables are coded on a 0–4 ordinal scale. Two variables are categorical strings, three are binary, and two are derived in Python from component scores. The two rationale fields are short free-text justifications referencing evidence in `skill_md_content`.

## Scope and knowledge-type mix

**`scope`** — categorical string. Breadth of work the skill covers.
Allowed values: `micro_task`, `single_bounded_task`, `multi_step_workflow`, `end_to_end_process`, `unclear`.

**`procedural`** — 0–4. Intensity of procedural know-how (how-to-do-something steps, scripted operations).
- 0: No procedural content
- 1: Marginal — brief mention of steps without detail
- 2: Present but secondary
- 3: Substantially procedural
- 4: Dominant — the skill is fundamentally a procedure

**`analytical`** — 0–4. Intensity of analytical know-how (interpretation, reasoning, evaluation of evidence). Anchors mirror `procedural`.

**`orchestration`** — 0–4. Intensity of orchestration know-how (coordinating across tools, files, services, sub-tasks). Anchors mirror `procedural`.

**`compliance`** — 0–4. Intensity of compliance know-how (rule-following, policy adherence, regulatory or process discipline). Anchors mirror `procedural`.

**`documentation`** — 0–4. Intensity of documentation know-how (producing or organising documentation as the primary function of the skill).
*Note:* this measures whether the skill IS a documentation skill, not whether the skill is well-documented. The latter is `documentation_quality`.

## Codifiability components

**`explicitness`** — 0–4. Clarity of the CORE instructions in SKILL.md — what to do, when, in what order, with what decisions. Focus on the main instructional body, not on supporting material.
- 0: No clear instructions; only describes purpose or context
- 1: Brief and vague; an agent could not reliably execute
- 2: Some explicit steps with significant gaps or ambiguity
- 3: Clear instructions covering the main flow
- 4: Comprehensive, unambiguous instructions covering main paths and key decisions

**`documentation_quality`** — 0–4. Quality of SUPPORTING material beyond the core instructions: examples, templates, references, helper scripts, README context, inline comments. NOT the clarity of the instructions themselves (that is `explicitness`).
- 0: No supporting material
- 1: Minimal (one line of context or a single brief example)
- 2: Some but thin (occasional examples or partial templates)
- 3: Good supporting material across multiple places
- 4: Comprehensive: working examples, ready-to-use templates, references, helper scripts

*Distinction:* a skill can have clear instructions but no examples (high `explicitness`, low `documentation_quality`), or extensive context but vague instructions (low `explicitness`, high `documentation_quality`).

**`tacit_dependency`** — 0–4. Extent to which the skill relies on unstated human judgement, domain expertise, or implicit assumptions. **Higher = MORE tacit dependence = LOWER codifiability.**
- 0: No tacit dependence; everything an agent needs is stated
- 1: Minor unstated assumptions; mostly self-contained
- 2: Some implicit knowledge required (e.g., a domain convention)
- 3: Substantial unstated judgement or expertise required
- 4: Heavy dependence on tacit human judgement, domain expertise, or contextual reading

**`context_sensitivity`** — 0–4. Extent to which the skill depends on a specific organisation, repository, toolchain, legal setting, or data environment. **Higher = MORE context-bound = LOWER codifiability / transferability.**
- 0: Fully portable; works in any environment
- 1: Minor environmental assumptions (a common tool, standard format)
- 2: Some specific dependencies (a particular framework or platform)
- 3: Strong dependence on a specific stack or organisational setting
- 4: Tightly coupled to a single environment; would not function elsewhere without rework

**`codifiability_score`** — 0–4 derived integer (computed in Python after API response, NOT emitted by the LLM).

```
codifiability_score = round_half_up(
    ( explicitness + documentation_quality
    + (4 - tacit_dependency) + (4 - context_sensitivity) ) / 4
)
```

`round_half_up` is standard rounding (0.5→1, 1.5→2, 2.5→3, 3.5→4), not Python's banker's `round()`.

**`codifiability_rationale`** — string, 1–3 sentences citing specific evidence from `skill_md_content`. Does not restate the score.

## Task horizon

**`coordination_burden`** — 0–4. Extent to which the workflow requires coordination across multiple steps, tools, files, or transitions. Higher = longer task horizon.
- 0: Single-step task; no coordination
- 1: Two or three sequential steps with no branching
- 2: Multi-step but mostly linear
- 3: Multi-step with branching, parallel paths, or significant tool transitions
- 4: Complex orchestration across many steps, tools, decision points

## Evaluability components

**`evaluation_type`** — categorical string (descriptive only; NOT in the formula).
Allowed values: `objective` (deterministic rules/tests/schemas/thresholds), `hybrid` (formal checks + human review), `holistic` (mainly human judgement), `none_visible`, `unclear`.

**`output_verifiability`** — 0–4. How verifiable the output is in principle, regardless of whether tests are present.
- 0: No clear correctness criterion
- 1: Rough criteria but mostly subjective
- 2: Partially verifiable
- 3: Verifiable against stated criteria
- 4: Fully and deterministically verifiable

**`tests_quality`** — 0–4. Presence and quality of tests or test-like validation.
- 0: No mention of tests
- 1: Mentioned/recommended but none provided
- 2: Stub or partial references (function names, planned tests)
- 3: Some working tests visible (test files, validation scripts)
- 4: Comprehensive test coverage of main flows

**`thresholds_quality`** — 0–4. Presence and quality of thresholds, pass/fail criteria, or acceptance bounds.
- 0: None
- 1: Mentioned but vague ("should be good enough")
- 2: Some defined but partial coverage
- 3: Clear thresholds for the main outcomes
- 4: Comprehensive, parameterised thresholds covering edge cases

**`error_handling`** — 0–4. How well the skill describes failure modes and recovery or escalation.
- 0: None
- 1: Vague mention of possible failures
- 2: Some modes acknowledged with partial recovery guidance
- 3: Main failure modes described with recovery or escalation procedures
- 4: Comprehensive: failure modes, recovery paths, escalation rules

**`evaluability_score`** — 0–4 derived integer (computed in Python; `evaluation_type` is NOT part of the formula).

```
evaluability_score = round_half_up(
    ( output_verifiability + tests_quality + thresholds_quality + error_handling ) / 4
)
```

**`evaluability_rationale`** — string, 1–3 sentences citing specific evidence.

## Dependencies (binary)

**`requires_memory`** — 0/1. Skill requires persistent memory or state across invocations.
**`requires_tools`** — 0/1. Skill requires specific tools beyond the base agent (CLIs, libraries, scripts).
**`requires_external_services`** — 0/1. Skill requires external APIs, services, or credentials.

## Governance

**`provenance_clarity`** — 0–4. How clearly the skill identifies author, publisher, version, source.
- 0: None / 1: Bare minimum (filename or repo only) / 2: Author or version / 3: Author + version + source / 4: Comprehensive with version, author, license, source links

**`maintenance_signals`** — 0–4. Visible signs of ongoing maintenance: changelogs, version history, update notes, deprecation flags, last-updated references.
- 0: None / 1: Single date or version / 2: Versioning or brief changelog / 3: Changelog + version history / 4: Comprehensive evidence including deprecation handling

**`safety_notes`** — 0–4. Presence of safety, limitation, or trust-boundary notes.
- 0: None / 1: Brief mention of a limitation / 2: Some notes but partial / 3: Clear guidance / 4: Comprehensive notes, trust boundaries, risk disclosures

**`scope_limits`** — 0–4. Clarity of OUT-OF-SCOPE boundaries and escalation rules. Measures what the skill states it does NOT do.
- 0: None / 1: Vague mention / 2: Partial / 3: Clear in/out-of-scope with some escalation guidance / 4: Comprehensive boundaries with explicit escalation and handoff points

*Distinction:* `scope_limits` measures BOUNDARY clarity, not in-scope description clarity. In-scope clarity is `explicitness`.

## Coding audit

**`coding_status`** — categorical string. Status of this row.
Allowed: `complete`, `partial`, `ambiguous`, `insufficiently_evidenced`, `excluded`.

**`audit_confidence`** — 0–4. Coder's confidence in this row overall.
- 0: Very low — coded against thin evidence / 1: Low / 2: Moderate / 3: High / 4: Very high

**`coding_notes`** — free-text. Key evidence, uncertainties, limitations, exclusions, anomalies. Used whenever `audit_confidence` < 4 or any variable was difficult.
