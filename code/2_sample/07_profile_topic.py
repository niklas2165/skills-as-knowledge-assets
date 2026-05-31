"""07 - topic_slug coverage and per-topic counts.

Known to be sparse (~1.7%). Confirm and characterise; do NOT build analysis on
it. Output exists to document why topic_slug is not a stratification axis.

Outputs:
- plots/07_topic_counts.png
- sections/07_topic.md / .json

Run:  python3 scripts/07_profile_topic.py
"""
from __future__ import annotations

from collections import Counter

import _lib as L

NUM, TOPIC = "07", "topic"


def main() -> None:
    _, skills = L.load()
    n = len(skills)
    present = [r for r in skills if r.get("topic_slug")]
    n_present = len(present)

    label_counts = Counter(r.get("topic_label") or r["topic_slug"] for r in present)

    # ------------------------------------------------------------------ plot
    items = label_counts.most_common()
    fig, ax = L.plt.subplots(figsize=(8, max(3, 0.45 * len(items) + 1)))
    labels = [k for k, _ in items][::-1]
    vals = [c for _, c in items][::-1]
    ax.barh(labels, vals, color="#8172B3", edgecolor="white", linewidth=0.3)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v}", va="center", fontsize=8)
    ax.set_xlabel("Number of skills (of {} with a topic)".format(n_present))
    ax.set_title(f"topic_label counts — only {n_present}/{n:,} skills ({n_present/n:.1%}) have a topic")
    ax.margins(x=0.12)
    plot_path = L.save_fig(fig, f"{NUM}_topic_counts")

    # ------------------------------------------------------------------ md
    md = [f"## {NUM} — topic_slug coverage", ""]
    md.append(f"Only **{L.fmt_int(n_present)}** of {L.fmt_int(n)} skills "
              f"({L.fmt_pct(n_present / n)}) carry a `topic_slug`/`topic_label`; "
              f"the other {L.fmt_int(n - n_present)} are `null`. "
              "**This field is far too sparse to use as a category axis** and is "
              "characterised here only to document that.")
    md += ["", f"### Per-topic counts ({len(label_counts)} distinct topics)", ""]
    md.append(L.md_table(["Topic", "Skills", "Share of the {} with a topic".format(n_present)],
                         [[lbl, L.fmt_int(c), L.fmt_pct(c / n_present)] for lbl, c in items]))
    md += ["", f"![Topic counts]({plot_path})", ""]

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "n_present": n_present, "coverage": n_present / n,
        "n_distinct_topics": len(label_counts),
        "topic_counts": label_counts.most_common(),
    })
    print(f"[07] topic present={n_present} ({n_present/n:.2%}) distinct_topics={len(label_counts)}")


if __name__ == "__main__":
    main()
