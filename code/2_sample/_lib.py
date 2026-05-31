"""Shared helpers for the population-profile scripts (phase 1).

All profile scripts import from here so that data loading, the install-tier
definition, concentration maths, and plot/section output are defined once.

Conventions
-----------
- Input JSON is treated as READ-ONLY. Nothing here writes to inputs/.
- Plots -> analysis/profile/plots/<name>.png at 150 dpi.
- Each profile script writes a markdown fragment to
  analysis/profile/sections/NN_<topic>.md and a machine-readable stats dict
  to analysis/profile/sections/NN_<topic>.json. 99_build_profile.py stitches
  the fragments into population_profile.md and reads the JSONs to populate the
  "what stands out" header with real (non-fabricated) numbers.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # static output only; no interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so scripts run from any cwd)
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
INPUT_PATH = ROOT / "inputs" / "skills_combined_2026-05-20.json"
PROFILE_DIR = ROOT / "analysis" / "profile"
PLOTS_DIR = PROFILE_DIR / "plots"
SECTIONS_DIR = PROFILE_DIR / "sections"

for _d in (PLOTS_DIR, SECTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
FIG_DPI = 150


def save_fig(fig, name: str) -> str:
    """Save a figure to the plots dir as PNG; return the relative path used in md."""
    out = PLOTS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return f"plots/{name}.png"  # relative to population_profile.md location


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load() -> tuple[dict, list[dict]]:
    """Return (metadata, skills). Read-only."""
    with open(INPUT_PATH) as fh:
        data = json.load(fh)
    return data["metadata"], data["skills"]


def owner_of(repo: str) -> str:
    """'owner/name' -> 'owner'. Falls back to the whole string if no slash."""
    return repo.split("/", 1)[0] if "/" in repo else repo


def repo_name_of(repo: str) -> str:
    """'owner/name' -> 'name'."""
    return repo.split("/", 1)[1] if "/" in repo else repo


# ---------------------------------------------------------------------------
# Install tiers
# ---------------------------------------------------------------------------
# Decade (order-of-magnitude) bins on the exact integer install count. The
# install distribution spans ~717 -> 1.61M (>3 orders of magnitude), so log
# decades give interpretable, round, fixed boundaries that do not depend on the
# population's own quantiles. Quartile-based stats are reported separately in
# the install section; these labelled tiers are the axis used for cross-tabs
# (official status, SKILL.md coverage).
INSTALL_TIERS = [
    ("<1K", 0, 1_000),
    ("1K-10K", 1_000, 10_000),
    ("10K-100K", 10_000, 100_000),
    ("100K-1M", 100_000, 1_000_000),
    (">=1M", 1_000_000, math.inf),
]
INSTALL_TIER_LABELS = [t[0] for t in INSTALL_TIERS]


def tier_of(installs: int) -> str:
    for label, lo, hi in INSTALL_TIERS:
        if lo <= installs < hi:
            return label
    return INSTALL_TIERS[-1][0]  # safety; should not hit


# ---------------------------------------------------------------------------
# Concentration maths (computed manually; no sklearn dependency)
# ---------------------------------------------------------------------------
def gini(values: list[float]) -> float:
    """Gini coefficient of a list of non-negative values.

    0 = perfect equality, ->1 = maximal concentration. Uses the sorted-rank
    formula G = (2*sum(i*x_i) / (n*sum(x))) - (n+1)/n, i 1-indexed ascending.
    """
    xs = sorted(v for v in values if v is not None)
    n = len(xs)
    if n == 0:
        return float("nan")
    total = sum(xs)
    if total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def lorenz_points(values: list[float]) -> tuple[list[float], list[float]]:
    """Return (cum_pop_share, cum_value_share), both starting at (0,0).

    Sorted ascending so the curve bows below the 45-degree line.
    """
    xs = sorted(v for v in values if v is not None)
    n = len(xs)
    total = sum(xs)
    pop = [0.0]
    val = [0.0]
    running = 0.0
    for i, x in enumerate(xs):
        running += x
        pop.append((i + 1) / n)
        val.append(running / total if total else 0.0)
    return pop, val


def top_share(counts: Counter, k: int) -> float:
    """Share of the grand total held by the top-k keys of a Counter."""
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    topk = sum(c for _, c in counts.most_common(k))
    return topk / total


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_section(num: str, topic: str, markdown: str) -> None:
    (SECTIONS_DIR / f"{num}_{topic}.md").write_text(markdown.rstrip() + "\n")


def write_stats(num: str, topic: str, stats: dict) -> None:
    (SECTIONS_DIR / f"{num}_{topic}.json").write_text(json.dumps(stats, indent=2))


def md_table(headers: list[str], rows: list[list]) -> str:
    """Render a GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt_int(n) -> str:
    return f"{int(n):,}"


def fmt_pct(x, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"
