#!/usr/bin/env python3
"""
_style.py — shared visual style for every Phase-E findings figure.

This is imported by all section scripts (02_, 03_, ...) so the whole findings
chapter shares one consistent look. Section scripts NEVER pick their own
colours; they pull everything from here.

Design choices (one line + rationale each):
  - Font: DejaVu Sans, modest sizes — a clean sans that ships with matplotlib,
    so figures render identically on any machine (no missing-font fallbacks).
  - Spines: top and right hidden — less chart-junk, keeps the eye on the data.
  - Gridlines: light HORIZONTAL only — supports value reading on bar/hist charts
    without boxing the plot; vertical grid is off to avoid clutter.
  - Figure size: 7.0 x 4.2 in default — fits a single-column thesis page at
    300 DPI with room for axis labels; individual figures override as needed.
  - DPI: 100 on screen / 300 on save — crisp print output, light interactive use.
  - Bounding box: tight on save — trims whitespace so figures drop cleanly into
    the document.
  - CATEGORY_COLORS: read from data/category_mapping.csv's display_color column
    (NOT hardcoded here) so the CSV stays the single source of truth for both
    category labels and colours. Tableau-derived, perceptually distinguishable,
    colour-blind-aware; adjacencies avoid saturated red next to saturated green.
  - AUDIT_COLORS: green=pass, amber=warn, red=fail, grey=missing — the
    conventional traffic-light mapping, with a neutral grey for "no record".
  - BINARY_COLORS: a darker accent for the positive/"yes" level and a lighter
    neutral for the negative/"no" level, for official-vs-unofficial and
    has_skill_md-style contrasts.

save_fig(fig, name) writes BOTH figures/<name>.png and figures/<name>.svg.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------
# Paths (absolute; the project root has a TRAILING SPACE — keep it quoted).
# --------------------------------------------------------------------------
_ANALYSIS_DIR = Path("/Users/Niklas/Master Thesis Claude Code /Coding/analysis")
DATA_DIR = _ANALYSIS_DIR / "data"
FIG_DIR = _ANALYSIS_DIR / "figures"
TBL_DIR = _ANALYSIS_DIR / "tables"

# --------------------------------------------------------------------------
# (a) THESIS_RC — consistent matplotlib rcParams for every figure.
# --------------------------------------------------------------------------
THESIS_RC = {
    # fonts
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.titlesize":   13,
    "figure.titleweight": "bold",
    # spines — hide top and right
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     "#333333",
    "axes.linewidth":     0.8,
    # gridlines — light, horizontal only
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#D9D9D9",
    "grid.linewidth":     0.6,
    "grid.alpha":         0.8,
    "axes.axisbelow":     True,
    # figure geometry / output
    "figure.figsize":     (7.0, 4.2),
    "figure.dpi":         100,            # display
    "savefig.dpi":        300,            # print
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    # misc legibility
    "axes.titlepad":      10,
    "legend.frameon":     False,
}


# --------------------------------------------------------------------------
# (b) CATEGORY_COLORS — read from the CSV (single source of truth).
# --------------------------------------------------------------------------
def _load_category_colors():
    """category -> hex colour, read from data/category_mapping.csv. Also exposes
    a label->colour view and the category->display_label lookup so section
    scripts share one labelling/colour convention."""
    cm = pd.read_csv(DATA_DIR / "category_mapping.csv")
    if "display_color" not in cm.columns:
        raise RuntimeError(
            "category_mapping.csv is missing the 'display_color' column — "
            "re-run scripts/_build_datasets.py to regenerate it."
        )
    colors = dict(zip(cm["category"], cm["display_color"]))
    labels = dict(zip(cm["category"], cm["display_label"]))
    return colors, labels, cm


CATEGORY_COLORS, CATEGORY_LABELS, CATEGORY_MAPPING = _load_category_colors()

# Neutral greys for non-retained / missing buckets in population-wide figures.
EXCLUDED_COLOR = "#C7C7C7"   # excluded (but real) Method B categories
NONE_COLOR = "#9E9E9E"       # Method B explicitly assigned "none" (no cluster)
UNCATEGORIZED_COLOR = "#6E6E6E"  # no Method B record at all (null)

# --------------------------------------------------------------------------
# (c) AUDIT_COLORS — traffic-light mapping for the three external audits.
# --------------------------------------------------------------------------
AUDIT_COLORS = {
    "pass":    "#4E9F3D",   # green
    "warn":    "#E8A33D",   # amber
    "fail":    "#D1495B",   # red
    "missing": "#BFBFBF",   # grey (null / no record)
}
# Fixed stacking order for stacked audit bars.
AUDIT_ORDER = ["pass", "warn", "fail", "missing"]

# --------------------------------------------------------------------------
# (d) BINARY_COLORS — darker accent = positive/"yes", lighter neutral = "no".
# --------------------------------------------------------------------------
BINARY_COLORS = {
    "positive":   "#2C5F8A",   # dark blue accent  (yes / official / has SKILL.md)
    "negative":   "#BBC7D1",   # light neutral     (no / unofficial / missing)
    # convenience aliases (same two colours, named for common contrasts)
    "official":   "#2C5F8A",
    "unofficial": "#BBC7D1",
    "yes":        "#2C5F8A",
    "no":         "#BBC7D1",
}

# A single neutral accent for "one-colour" plots (e.g. per-knowledge-type
# distribution histograms) where colour should carry no categorical meaning.
NEUTRAL_ACCENT = "#2C5F8A"   # same dark blue as BINARY_COLORS["positive"]

# --------------------------------------------------------------------------
# (g) KNOWLEDGE_TYPE_COLORS — one colour per knowledge type (Section 3).
# --------------------------------------------------------------------------
# Used for the five-segment knowledge-profile stacked bars and any later
# section that needs a stable colour per knowledge type. Rationale:
#   - The five colours are the ColorBrewer "Dark2" qualitative palette (with
#     the fifth slot taken as Dark2's gold rather than its second green, so no
#     two segments are both green). Dark2 is more saturated than the muted
#     Tableau category palette, so a reader never confuses a knowledge-type
#     segment colour with a CATEGORY_COLORS hue — the two never share a figure,
#     but the contrast in saturation keeps them visually separate regardless.
#   - The five are pairwise distinguishable (teal-green / orange / indigo /
#     magenta / gold span the hue wheel) and Dark2 is colour-blind-aware.
#   - The fixed dict order is also the fixed segment-stacking order, chosen so
#     the two warm hues (orange=analytical, gold=documentation) are never
#     adjacent in the stack.
KNOWLEDGE_TYPE_COLORS = {
    "procedural":    "#1B9E77",   # teal-green
    "analytical":    "#D95F02",   # orange
    "orchestration": "#7570B3",   # indigo
    "compliance":    "#E7298A",   # magenta
    "documentation": "#E6AB02",   # gold
}
# Display labels (Title Case) for legends/axes.
KNOWLEDGE_TYPE_LABELS = {
    "procedural":    "Procedural",
    "analytical":    "Analytical",
    "orchestration": "Orchestration",
    "compliance":    "Compliance",
    "documentation": "Documentation",
}

# --------------------------------------------------------------------------
# (h) EVALUATION_TYPE_COLORS — one colour per evaluation_type (Section 5).
# --------------------------------------------------------------------------
# evaluation_type is a descriptive categorical (objective / hybrid / holistic /
# none_visible / unclear) — NOT a numeric input to the evaluability_score
# formula. Its five allowed values carry an ordinal reading from "most formal /
# checkable" (objective) to "no evaluation visible" (none_visible / unclear), so
# the palette is built to read as an ordered ramp rather than five arbitrary
# hues:
#   - the three substantive types (objective, hybrid, holistic) take a 3-step
#     ColorBrewer "Blues" ramp, dark→light, so darker = more formal/checkable;
#   - the two "absence" types (none_visible, unclear) take two neutral greys
#     (light then dark) so they read as outside the blue evaluation ramp.
# Blues + greys is colour-blind-safe and is deliberately distinct in hue from
# both CATEGORY_COLORS (Tableau) and KNOWLEDGE_TYPE_COLORS (Dark2), which never
# share a figure with it. The dict order is also the fixed stacking order.
EVALUATION_TYPE_COLORS = {
    "objective":    "#08519C",   # dark blue   — deterministic checks / pass-fail
    "hybrid":       "#3182BD",   # medium blue — formal checks + human review
    "holistic":     "#6BAED6",   # light blue  — mainly human judgement
    "none_visible": "#BDBDBD",   # light grey  — no evaluation visible
    "unclear":      "#737373",   # dark grey   — evaluation approach unclear
}
# Fixed value order (most-formal → absence), used for tables and stacked bars.
EVALUATION_TYPE_ORDER = ["objective", "hybrid", "holistic", "none_visible", "unclear"]
# Display labels for legends/axes.
EVALUATION_TYPE_LABELS = {
    "objective":    "Objective",
    "hybrid":       "Hybrid",
    "holistic":     "Holistic",
    "none_visible": "None visible",
    "unclear":      "Unclear",
}

# --------------------------------------------------------------------------
# (i) SCOPE_COLORS — one colour per scope value (Section 6, task horizon).
# --------------------------------------------------------------------------
# scope is the breadth-of-work categorical: micro_task / single_bounded_task /
# multi_step_workflow / end_to_end_process / unclear. Its four substantive
# values carry an ordinal reading from SHORTEST horizon (micro_task) to LONGEST
# (end_to_end_process), so the palette is built to read as an ordered ramp:
#   - the four substantive values take a 4-step ColorBrewer "Purples" ramp,
#     light→dark, so darker = longer task horizon;
#   - the one "absence" value (unclear) takes a neutral grey, the same logic
#     used for none_visible/unclear in EVALUATION_TYPE_COLORS, so it reads as
#     outside the purple horizon ramp.
# Purples is deliberately a different HUE from EVALUATION_TYPE_COLORS (Blues +
# greys): both are ordered ramps and could in principle sit in adjacent figures,
# so giving scope a purple ramp keeps the two from being mistaken for each
# other. It is also hue-distinct from CATEGORY_COLORS (Tableau); the only Dark2
# tone it sits near is the indigo knowledge-type colour, but SCOPE_COLORS and
# KNOWLEDGE_TYPE_COLORS never share a figure. The dict order is the fixed
# shortest→longest stacking order.
SCOPE_COLORS = {
    "micro_task":          "#DADAEB",   # lightest purple — shortest horizon
    "single_bounded_task": "#9E9AC8",   # light-medium purple
    "multi_step_workflow": "#756BB1",   # medium-dark purple
    "end_to_end_process":  "#54278F",   # darkest purple — longest horizon
    "unclear":             "#9E9E9E",   # neutral grey — scope not determinable
}
# Fixed value order (shortest → longest, then the absence bucket), used for
# tables, the scope-distribution bar, and the stacked-bar stacking order.
SCOPE_ORDER = ["micro_task", "single_bounded_task", "multi_step_workflow",
               "end_to_end_process", "unclear"]
# Ordinal encoding of the four substantive scope values for rank statistics
# (Spearman); "unclear" is intentionally excluded (mapped to NaN by callers).
SCOPE_ORDINAL = {
    "micro_task":          1,
    "single_bounded_task": 2,
    "multi_step_workflow": 3,
    "end_to_end_process":  4,
}
# Display labels for legends/axes.
SCOPE_LABELS = {
    "micro_task":          "Micro task",
    "single_bounded_task": "Single bounded task",
    "multi_step_workflow": "Multi-step workflow",
    "end_to_end_process":  "End-to-end process",
    "unclear":             "Unclear",
}


# --------------------------------------------------------------------------
# (j) CODER_PAIR_COLORS — one colour per coder pair (Section 9, reliability).
# --------------------------------------------------------------------------
# The reliability section compares three parallel codings of the 24-skill
# double-coded subset (two humans H1/H2 + the LLM) through their three pairwise
# agreements: H1-H2 (the human-human reference), H1-LLM, and H2-LLM. The three
# pairs get three distinct colours from the Okabe-Ito colour-blind-safe palette
# (blue / vermillion / bluish-green), maximally separable for the eight common
# colour-vision types. H1-H2 is given the blue so the human-human reference
# reads as the "anchor" colour wherever it appears (the dot plot in fig (b) and
# the human-human series in fig (g)); the two human-LLM pairs take the warm
# vermillion and the cool green. These hues are distinct from CATEGORY_COLORS
# (Tableau), KNOWLEDGE_TYPE_COLORS (Dark2), the Blues/Purples ramps and the
# traffic-light AUDIT_COLORS; Section 9 figures do not share a frame with those.
CODER_PAIR_COLORS = {
    "h1_h2":  "#0072B2",   # blue       — human-human reference pair
    "h1_llm": "#D55E00",   # vermillion — human(H1)-LLM pair
    "h2_llm": "#009E73",   # bluish-green — human(H2)-LLM pair
}
# Display labels for legends/axes (en-dash between the two coder labels).
CODER_PAIR_LABELS = {
    "h1_h2":  "H1–H2",
    "h1_llm": "H1–LLM",
    "h2_llm": "H2–LLM",
}


# --------------------------------------------------------------------------
# (e) apply_style() — call once at the top of every section script.
# --------------------------------------------------------------------------
def apply_style():
    """Push THESIS_RC into matplotlib's global rcParams."""
    plt.rcParams.update(THESIS_RC)


# --------------------------------------------------------------------------
# (e2) text_color_for_bg() — pick black/white overlay text on a coloured cell.
# --------------------------------------------------------------------------
# Shared helper for heatmap cell labels (Section 3 defined this locally; from
# Section 4 on it lives here so the threshold is set in one place). Returns
# "black" on light backgrounds and "white" on dark ones, using the WCAG
# relative-luminance of the cell colour.
#
# Threshold: 0.35 (was 0.40 in Section 3's local copy). On a viridis 0–4 scale
# the 0.40 cutoff fell at value ≈ 2.85, so the bright green cells at values
# ~2.7–2.8 (WCAG luminance ≈ 0.37–0.40) were still getting white text, which
# reads poorly. Lowering to 0.35 flips those cells to black while leaving the
# darker teal/blue/purple cells (value ≤ ~2.6) on white, and keeps the whole
# 3.0–3.7 range (and the bright-yellow top) on black. Section 3's already-
# produced figures are NOT regenerated; they keep their local 0.40 helper.
def text_color_for_bg(rgba, threshold=0.35):
    """Black on light cells, white on dark cells, by WCAG relative luminance."""
    r, g, b = rgba[0], rgba[1], rgba[2]

    def _lin(ch):
        return ch / 12.92 if ch <= 0.03928 else ((ch + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return "black" if lum > threshold else "white"


# --------------------------------------------------------------------------
# (f) save_fig() — write both .png and .svg with consistent settings.
# --------------------------------------------------------------------------
def save_fig(fig, name):
    """Save fig to figures/<name>.png and figures/<name>.svg. `name` is the
    stem (no extension). Returns the two paths written."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    svg = FIG_DIR / f"{name}.svg"
    fig.savefig(png)
    fig.savefig(svg)
    return png, svg


if __name__ == "__main__":
    # Quick self-check: print the loaded palette.
    apply_style()
    print("CATEGORY_COLORS (from CSV):")
    for c, h in CATEGORY_COLORS.items():
        print(f"  {c:<26} {h}  '{CATEGORY_LABELS[c]}'")
    print("\nAUDIT_COLORS:", AUDIT_COLORS)
    print("BINARY_COLORS:", BINARY_COLORS)
    print("NEUTRAL_ACCENT:", NEUTRAL_ACCENT)
    print("KNOWLEDGE_TYPE_COLORS:", KNOWLEDGE_TYPE_COLORS)
    print("EVALUATION_TYPE_COLORS:", EVALUATION_TYPE_COLORS)
    print("SCOPE_COLORS:", SCOPE_COLORS)
    print("CODER_PAIR_COLORS:", CODER_PAIR_COLORS)
