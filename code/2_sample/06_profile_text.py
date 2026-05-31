"""06 - Text length distributions.

- summary: chars and words, all 5,235 skills.
- skill_md.body: chars and words, the 3,345 skills that have a body.

These inform Phase 2 (what text to feed the clustering).

Outputs:
- plots/06_summary_length.png, plots/06_body_length.png
- sections/06_text.md / .json

Run:  python3 scripts/06_profile_text.py
"""
from __future__ import annotations

import numpy as np

import _lib as L

NUM, TOPIC = "06", "text"
PCTS = [10, 25, 50, 75, 90, 99]


def stats(arr) -> dict:
    a = np.asarray(arr, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "median": float(np.median(a)),
            **{f"p{p}": float(np.percentile(a, p)) for p in PCTS}}


def stat_rows(s) -> list[list]:
    return [["count", L.fmt_int(s["n"])], ["min", L.fmt_int(round(s["min"]))],
            ["median", L.fmt_int(round(s["median"]))], ["mean", L.fmt_int(round(s["mean"]))],
            ["p90", L.fmt_int(round(s["p90"]))], ["p99", L.fmt_int(round(s["p99"]))],
            ["max", L.fmt_int(round(s["max"]))]]


def main() -> None:
    _, skills = L.load()

    summ_chars = [len(r["summary"]) for r in skills]
    summ_words = [len(r["summary"].split()) for r in skills]
    bodies = [(r["skill_md"] or {}).get("body") for r in skills]
    bodies = [b for b in bodies if b]
    body_chars = [len(b) for b in bodies]
    body_words = [len(b.split()) for b in bodies]

    s_sc, s_sw = stats(summ_chars), stats(summ_words)
    s_bc, s_bw = stats(body_chars), stats(body_words)

    # Summary appears hard-capped near ~160 chars (meta-description style):
    sc = np.asarray(summ_chars)
    cap_le160 = float((sc <= 160).mean())
    cap_band = float(((sc >= 150) & (sc <= 161)).mean())
    short_lt50 = float((sc < 50).mean())

    # ------------------------------------------------------------------ plots
    fig, axes = L.plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].hist(summ_chars, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[0].axvline(s_sc["median"], color="#C44E52", ls="--", lw=1, label=f"median {s_sc['median']:.0f}")
    axes[0].set_xlabel("Summary length (chars)"); axes[0].set_ylabel("Skills"); axes[0].legend()
    axes[1].hist(summ_words, bins=40, color="#55A868", edgecolor="white", linewidth=0.3)
    axes[1].axvline(s_sw["median"], color="#C44E52", ls="--", lw=1, label=f"median {s_sw['median']:.0f}")
    axes[1].set_xlabel("Summary length (words)"); axes[1].legend()
    fig.suptitle(f"Summary length distribution (n={len(summ_chars):,})")
    summ_path = L.save_fig(fig, f"{NUM}_summary_length")

    fig, axes = L.plt.subplots(1, 2, figsize=(11, 4.4))
    # body lengths are long-tailed; clip x for readability but keep stats on full data
    axes[0].hist(np.clip(body_chars, 0, np.percentile(body_chars, 99)), bins=40,
                 color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[0].axvline(s_bc["median"], color="#C44E52", ls="--", lw=1, label=f"median {s_bc['median']:.0f}")
    axes[0].set_xlabel("Body length (chars, clipped at p99)"); axes[0].set_ylabel("Skills"); axes[0].legend()
    axes[1].hist(np.clip(body_words, 0, np.percentile(body_words, 99)), bins=40,
                 color="#55A868", edgecolor="white", linewidth=0.3)
    axes[1].axvline(s_bw["median"], color="#C44E52", ls="--", lw=1, label=f"median {s_bw['median']:.0f}")
    axes[1].set_xlabel("Body length (words, clipped at p99)"); axes[1].legend()
    fig.suptitle(f"SKILL.md body length distribution (n={len(body_chars):,} with body)")
    body_path = L.save_fig(fig, f"{NUM}_body_length")

    # ------------------------------------------------------------------ md
    md = [f"## {NUM} — Text length distributions", ""]
    md.append(f"`summary` exists for all {L.fmt_int(len(summ_chars))} skills; "
              f"`skill_md.body` exists for {L.fmt_int(len(body_chars))} "
              f"({L.fmt_pct(len(body_chars) / len(skills))}).")
    md += ["", "### Summary length", ""]
    md.append("Chars and words, side by side:")
    md.append("")
    md.append(L.md_table(["Statistic", "Chars", "Words"],
                         [[r_c[0], r_c[1], r_w[1]] for r_c, r_w in zip(stat_rows(s_sc), stat_rows(s_sw))]))
    md.append("")
    md.append(f"**`summary` is effectively hard-capped near ~160 characters** "
              f"(max {s_sc['max']:.0f}; {L.fmt_pct(cap_le160)} are <= 160 chars, "
              f"{L.fmt_pct(cap_band)} fall in 150-161 chars), with a left tail of short "
              f"entries ({L.fmt_pct(short_lt50)} under 50 chars). This is the shape of a "
              "truncated meta-description, not free-form text — the median (153) sits "
              "above the mean (120) for exactly this reason. Relevant for Phase 2: "
              "`summary` alone carries limited text; `skill_md.body` (where present) is "
              "an order of magnitude richer.")
    md += ["", f"![Summary length]({summ_path})", ""]
    md += ["### SKILL.md body length (the {} with a body)".format(L.fmt_int(len(body_chars))), ""]
    md.append(L.md_table(["Statistic", "Chars", "Words"],
                         [[r_c[0], r_c[1], r_w[1]] for r_c, r_w in zip(stat_rows(s_bc), stat_rows(s_bw))]))
    md += ["", f"![Body length]({body_path})", ""]
    md.append("Histograms clip the x-axis at the 99th percentile for readability; "
              "all reported statistics use the full (unclipped) data.")

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "summary_chars": s_sc, "summary_words": s_sw,
        "body_chars": s_bc, "body_words": s_bw,
        "n_with_body": len(body_chars),
        "summary_cap": {"share_le_160": cap_le160, "share_150_161": cap_band, "share_lt_50": short_lt50},
    })
    print(f"[06] summary chars med={s_sc['median']:.0f} words med={s_sw['median']:.0f} | "
          f"body chars med={s_bc['median']:.0f} words med={s_bw['median']:.0f} (n={len(body_chars)})")


if __name__ == "__main__":
    main()
