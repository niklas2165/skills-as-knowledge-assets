"""10 - Text preparation for clustering (Phase 2).

Cleans the `summary` field for the full 5,235-skill population (no restriction;
`skill_md.body` is NOT used at this stage — see decisions.md 2026-05-21).

Per record, adds three fields (all original fields kept unchanged):
  - summary_clean       : cleaned, ORIGINAL case  (human reading / qualitative review)
  - summary_normalised  : cleaned, lower-cased     (embedding / clustering input)
  - summary_usable      : bool. False for the 632 records whose `summary` is only a
                          YAML block-scalar indicator (`>`, `|`, `>-`, `|-`) — the
                          description text was lost during scraping. Flagged (not
                          altered/dropped) so Phase-3 clustering can exclude/handle
                          them; recovery is deferred to post-sampling SKILL.md
                          retrieval (decisions.md 2026-05-21).

Cleaning pipeline (see decisions.md for the rule-level choices):
  1. Strip install commands: remove from `npx skills add` to end of line.
     (Confirmed 0 matches in summaries — install commands live in the separate
     `install_command` field. Rule applied anyway for correctness/auditability.)
  2. Strip URLs:
       a. markdown links `[text](http...)` -> keep `text`, drop URL + brackets.
       b. bare URLs `http(s)://...`         -> removed entirely.
     (Prose CLI mentions like `npx netlify` are NOT install commands and are kept.)
  3. Normalise whitespace: collapse runs to a single space, trim ends.
  4. Lower-case the result -> summary_normalised.

Outputs:
  - processed/skills_with_text_2026-05-20.json   (mirrors the input snapshot date)
  - analysis/profile/text_prep_report.md

Run:  python3 scripts/10_prepare_text.py
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from datetime import date

import _lib as L

DATA_DATE = "2026-05-20"  # match the input snapshot's date, not the processing date
PROCESSED_DIR = L.ROOT / "processed"
OUT_JSON = PROCESSED_DIR / f"skills_with_text_{DATA_DATE}.json"
REPORT = L.PROFILE_DIR / "text_prep_report.md"
SHORT_THRESHOLD = 20  # chars; rows below this are listed for spot-check
N_EXAMPLES = 8
SEED = 42

# --- regexes (compiled once) ---
RE_INSTALL = re.compile(r"npx\s+skills\s+add.*$", re.IGNORECASE | re.MULTILINE)
RE_HAS_INSTALL = re.compile(r"npx\s+skills\s+add", re.IGNORECASE)
RE_HAS_URL = re.compile(r"https?://", re.IGNORECASE)
RE_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*https?://[^)]*\)", re.IGNORECASE)
RE_BARE_URL = re.compile(r"https?://[^\s<>()\[\]]+", re.IGNORECASE)
RE_WS = re.compile(r"\s+")
# A summary that is ONLY a YAML block-scalar header (e.g. ">", "|", ">-", "|2"):
# the description text was lost in the scrape and only the indicator survived.
RE_YAML_PLACEHOLDER = re.compile(r"^[|>][0-9]*[+-]?$")


def is_yaml_placeholder(raw: str) -> bool:
    return bool(RE_YAML_PLACEHOLDER.match(raw.strip()))


def clean_summary(raw: str) -> tuple[str, str, bool, bool]:
    """Return (clean, normalised, had_url, had_install)."""
    had_install = bool(RE_HAS_INSTALL.search(raw))
    had_url = bool(RE_HAS_URL.search(raw))

    s = RE_INSTALL.sub("", raw)        # 1. install command -> end of line
    s = RE_MD_LINK.sub(r"\1", s)        # 2a. markdown link -> link text
    s = RE_BARE_URL.sub("", s)          # 2b. bare URL -> removed
    s = RE_WS.sub(" ", s).strip()       # 3. whitespace
    return s, s.lower(), had_url, had_install


def length_stats(values: list[int]) -> dict:
    vs = sorted(values)
    n = len(vs)

    def pct(p):
        if n == 1:
            return vs[0]
        k = (n - 1) * p / 100
        lo = int(k)
        frac = k - lo
        return vs[lo] + (vs[min(lo + 1, n - 1)] - vs[lo]) * frac

    return {"count": n, "min": vs[0], "median": statistics.median(vs),
            "mean": sum(vs) / n, "p90": pct(90), "p99": pct(99), "max": vs[-1]}


def stat_row(label, sc, sw):
    def f(d, k):
        v = d[k]
        return L.fmt_int(round(v)) if k != "count" else L.fmt_int(v)
    return [label, f(sc, "min"), f(sc, "median"), f(sc, "mean"), f(sc, "p90"), f(sc, "p99"), f(sc, "max")]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(L.INPUT_PATH) as fh:
        data = json.load(fh)
    meta, skills = data["metadata"], data["skills"]
    n = len(skills)

    n_url = n_install = n_empty = 0
    empties: list[dict] = []
    shorts: list[dict] = []
    raw_chars, raw_words, clean_chars, clean_words = [], [], [], []
    url_rows: list[dict] = []          # before/after for the (few) rows that had URLs
    ph_recoverable: list[dict] = []    # YAML-placeholder summaries WITH a body
    ph_unrecoverable: list[dict] = []  # YAML-placeholder summaries with no SKILL.md
    ph_indicators = Counter()          # which indicator (">", "|", ...)

    for r in skills:
        raw = r["summary"]
        clean, norm, had_url, had_install = clean_summary(raw)
        is_ph = is_yaml_placeholder(raw)
        usable = (clean != "") and not is_ph
        r["summary_clean"] = clean
        r["summary_normalised"] = norm
        r["summary_usable"] = usable

        n_url += had_url
        n_install += had_install
        raw_chars.append(len(raw)); raw_words.append(len(raw.split()))
        clean_chars.append(len(clean)); clean_words.append(len(clean.split()))

        if had_url:
            url_rows.append({"repo": r["repo"], "raw": raw, "clean": clean})

        if clean == "":
            n_empty += 1
            empties.append({"skill_url": r["skill_url"], "repo": r["repo"], "raw": raw})
        elif is_ph:
            ph_indicators[raw.strip()] += 1
            has_body = bool((r.get("skill_md") or {}).get("body"))
            row = {"repo": r["repo"], "skill_name": r["skill_name"], "raw": raw}
            (ph_recoverable if has_body else ph_unrecoverable).append(row)
        elif len(clean) < SHORT_THRESHOLD:
            shorts.append({"skill_url": r["skill_url"], "repo": r["repo"], "clean": clean, "chars": len(clean)})

    n_unusable = len(ph_recoverable) + len(ph_unrecoverable)

    # --- write processed JSON (preserve structure; add provenance) ---
    out_meta = dict(meta)
    out_meta["text_prep"] = {
        "processed_date": date.today().isoformat(),
        "source_file": L.INPUT_PATH.name,
        "fields_added": ["summary_clean", "summary_normalised", "summary_usable"],
        "input_field": "summary",
        "population": "full 5,235 (no SKILL.md restriction)",
        "cleaning_rules": [
            "strip 'npx skills add' to end of line",
            "markdown links -> link text (drop URL + brackets)",
            "bare http(s):// URLs removed",
            "whitespace collapsed + trimmed",
            "summary_normalised = lower-case of summary_clean",
            "summary_usable = false iff summary is only a YAML block-scalar indicator",
        ],
        "counts": {"records": n, "url_stripped": n_url, "install_stripped": n_install,
                   "empty_after_clean": n_empty, "short_usable_lt_%d" % SHORT_THRESHOLD: len(shorts),
                   "summary_unusable": n_unusable,
                   "unusable_recoverable_from_body": len(ph_recoverable),
                   "unusable_unrecoverable": len(ph_unrecoverable)},
    }
    OUT_JSON.write_text(json.dumps({"metadata": out_meta, "skills": skills}, ensure_ascii=False, indent=2))

    # --- report ---
    rng = random.Random(SEED)
    sample_idx = sorted(rng.sample(range(n), N_EXAMPLES))

    sc_raw, sw_raw = length_stats(raw_chars), length_stats(raw_words)
    sc_cl, sw_cl = length_stats(clean_chars), length_stats(clean_words)

    md = ["# Text-prep report — Phase 2", ""]
    md.append(f"_Generated {date.today().isoformat()} by `scripts/10_prepare_text.py` "
              f"from `inputs/{L.INPUT_PATH.name}`. Output: "
              f"`processed/{OUT_JSON.name}`._")
    md += ["", "## Summary", ""]
    md.append(L.md_table(["Metric", "Value"], [
        ["Records processed", L.fmt_int(n)],
        ["Summaries with URLs stripped", L.fmt_int(n_url)],
        ["Summaries with install commands stripped", L.fmt_int(n_install)],
        ["Empty after cleaning (**problem if >0**)", L.fmt_int(n_empty)],
        ["Non-informative summaries (YAML placeholder), flagged `summary_usable=false`",
         f"{L.fmt_int(n_unusable)} ({n_unusable / n:.1%})"],
        ["Usable summaries (`summary_usable=true`)", L.fmt_int(n - n_unusable)],
        [f"Genuinely short usable summaries (<{SHORT_THRESHOLD} chars, spot-check)", L.fmt_int(len(shorts))],
    ]))
    md.append("")
    md.append("**Install-command stripping matched 0 summaries.** Install commands are "
              "stored in the separate `install_command` field, not embedded in summary "
              "text. The rule is applied for auditability; prose CLI mentions (e.g. "
              "`npx netlify`) are intentionally kept as descriptive content.")
    md.append("")
    md.append(f"**{n_unusable / n:.1%} of summaries ({L.fmt_int(n_unusable)}) are "
              "non-informative** — see the anomaly section below. They are flagged via "
              "`summary_usable=false`, not altered or dropped.")

    md += ["", "## Data anomaly: broken summaries (YAML block-scalar indicators)", ""]
    md.append(f"**{L.fmt_int(n_unusable)} records ({n_unusable / n:.1%}) have a `summary` "
              "that is only a YAML block-scalar indicator** — the description text was lost "
              "during the detail-page scrape and only the indicator character survived. "
              "`skill_md.frontmatter.description` is broken identically (also `>`), so it is "
              "not a recovery source.")
    md.append("")
    md.append("Indicator breakdown (raw `summary` value):")
    md.append("")
    md.append(L.md_table(["Indicator", "Records"],
                         [[f"`{ind}`", L.fmt_int(c)] for ind, c in ph_indicators.most_common()]))
    md.append("")
    md.append("Recoverability of the lost text:")
    md.append("")
    md.append(L.md_table(["", "Records", "Note"],
        [["Recoverable from `skill_md.body`", L.fmt_int(len(ph_recoverable)),
          "real description present in body"],
         ["Unrecoverable from this dataset", L.fmt_int(len(ph_unrecoverable)),
          "no SKILL.md fetched; would need a GitHub fetch"]]))
    md.append("")
    md.append("**Handling (decisions.md 2026-05-21):** flag only. `summary_usable=false` "
              "marks these so Phase-3 clustering can exclude/handle them; the cleaned text "
              "is kept faithful (`summary_clean` = the indicator). Text recovery is deferred "
              "to the post-sampling SKILL.md retrieval, where the body-recoverable ones can "
              "be repaired and the unrecoverable ones fetched directly from GitHub — without "
              "mixing body-derived text into the clustering input now.")
    md.append("")
    md.append("First 15 unrecoverable records (no SKILL.md, no text anywhere in the data):")
    md.append("")
    md.append(L.md_table(["repo", "skill_name", "raw summary"],
                         [[u["repo"], f"`{u['skill_name']}`", repr(u["raw"])]
                          for u in ph_unrecoverable[:15]]))

    md += ["", "## Length distribution: before vs after cleaning", ""]
    md.append("Cleaning only alters the few rows that contained URLs, so before/after "
              "are nearly identical; the table documents that cleaning did not distort "
              "lengths. (No plot — it would show two overlapping distributions.)")
    md.append("")
    md.append("**Characters:**")
    md.append("")
    md.append(L.md_table(["", "min", "median", "mean", "p90", "p99", "max"],
                         [stat_row("raw summary", sc_raw, None), stat_row("summary_clean", sc_cl, None)]))
    md.append("")
    md.append("**Words:**")
    md.append("")
    md.append(L.md_table(["", "min", "median", "mean", "p90", "p99", "max"],
                         [stat_row("raw summary", sw_raw, None), stat_row("summary_clean", sw_cl, None)]))

    md += ["", "## Empty after cleaning", ""]
    if empties:
        md.append(f"**{len(empties)} row(s) became empty** — these need attention "
                  "(not silently dropped):")
        md.append("")
        md.append(L.md_table(["repo", "skill_url", "raw summary"],
                             [[e["repo"], e["skill_url"], repr(e["raw"])] for e in empties]))
    else:
        md.append("None. No summary was reduced to an empty string by cleaning.")

    md += ["", f"## Genuinely short usable summaries (<{SHORT_THRESHOLD} chars, {len(shorts)} rows)", ""]
    md.append("Usable summaries (placeholders excluded) that are still short — pre-existing, "
              "not created by cleaning. Listed for spot-check:")
    md.append("")
    if shorts:
        shorts_sorted = sorted(shorts, key=lambda x: x["chars"])
        md.append(L.md_table(["chars", "repo", "summary_clean"],
                             [[s["chars"], s["repo"], repr(s["clean"])] for s in shorts_sorted]))
    else:
        md.append("None.")

    md += ["", f"## URL-stripped rows ({len(url_rows)}) — before / after", ""]
    md.append("Every row where a URL was removed, for direct inspection:")
    md.append("")
    for u in url_rows:
        md.append(f"- **{u['repo']}**")
        md.append(f"  - raw:   {u['raw']!r}")
        md.append(f"  - clean: {u['clean']!r}")

    md += ["", f"## {N_EXAMPLES} random examples (seed={SEED})", ""]
    for i in sample_idx:
        r = skills[i]
        md.append(f"- **{r['repo']}** — `{r['skill_name']}`")
        md.append(f"  - raw:        {r['summary']!r}")
        md.append(f"  - clean:      {r['summary_clean']!r}")
        md.append(f"  - normalised: {r['summary_normalised']!r}")

    REPORT.write_text("\n".join(md).rstrip() + "\n")

    print(f"[10] records={n} url_stripped={n_url} install_stripped={n_install} "
          f"empty={n_empty} short(<{SHORT_THRESHOLD})={len(shorts)}")
    print(f"     wrote {OUT_JSON.relative_to(L.ROOT)} and {REPORT.relative_to(L.ROOT)}")


if __name__ == "__main__":
    main()