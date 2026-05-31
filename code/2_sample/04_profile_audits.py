"""04 - Audit coverage and pass/warn/fail patterns.

Three audit services appear in the data: Gen Agent Trust Hub, Snyk, Socket.
Reports coverage (skills with any audits) and the pass/warn/fail breakdown per
service, plus a per-skill worst-status summary.

Outputs:
- plots/04_audit_status_by_service.png
- sections/04_audits.md / .json

Run:  python3 scripts/04_profile_audits.py
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

import _lib as L

NUM, TOPIC = "04", "audits"
STATUS_ORDER = ["Pass", "Warn", "Fail"]
STATUS_COLOR = {"Pass": "#55A868", "Warn": "#DD8452", "Fail": "#C44E52"}
WORST_RANK = {"Pass": 0, "Warn": 1, "Fail": 2}


def main() -> None:
    _, skills = L.load()
    n = len(skills)

    with_audits = [r for r in skills if r.get("audits")]
    n_with = len(with_audits)
    n_without = n - n_with

    # service -> Counter(status)
    svc_status: dict[str, Counter] = defaultdict(Counter)
    svc_names: dict[str, str] = {}
    for r in with_audits:
        for a in r["audits"]:
            svc_status[a["slug"]][a["status"]] += 1
            svc_names[a["slug"]] = a["name"]

    services = sorted(svc_status, key=lambda s: -sum(svc_status[s].values()))

    # per-skill worst status across its services (only for skills with audits)
    worst = Counter()
    for r in with_audits:
        ranks = [WORST_RANK.get(a["status"], 0) for a in r["audits"]]
        worst_status = STATUS_ORDER[max(ranks)] if ranks else None
        worst[worst_status] += 1
    any_fail = sum(1 for r in with_audits if any(a["status"] == "Fail" for a in r["audits"]))
    all_pass = sum(1 for r in with_audits if all(a["status"] == "Pass" for a in r["audits"]))

    # ------------------------------------------------------------------ plot
    fig, ax = L.plt.subplots(figsize=(8, 5))
    x = np.arange(len(services))
    width = 0.26
    for j, status in enumerate(STATUS_ORDER):
        vals = [svc_status[s].get(status, 0) for s in services]
        ax.bar(x + (j - 1) * width, vals, width, color=STATUS_COLOR[status], label=status)
        for i, v in enumerate(vals):
            if v:
                ax.text(x[i] + (j - 1) * width, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([svc_names[s] for s in services])
    ax.set_ylabel("Number of audited skills")
    ax.set_title("Audit status by service (skills with audits)")
    ax.legend(title="Status")
    plot_path = L.save_fig(fig, f"{NUM}_audit_status_by_service")

    # ------------------------------------------------------------------ md
    md = [f"## {NUM} — Audit coverage & pass/warn/fail", ""]
    md.append(f"**{L.fmt_int(n_with)}** skills ({L.fmt_pct(n_with / n)}) carry audit results; "
              f"**{L.fmt_int(n_without)}** ({L.fmt_pct(n_without / n)}) have an empty audits list.")
    md += ["", "### Per-service status breakdown (among the {} audited skills)".format(L.fmt_int(n_with)), ""]
    rows = []
    for s in services:
        c = svc_status[s]
        tot = sum(c.values())
        rows.append([svc_names[s], L.fmt_int(tot),
                     f"{c.get('Pass',0)} ({c.get('Pass',0)/tot:.1%})",
                     f"{c.get('Warn',0)} ({c.get('Warn',0)/tot:.1%})",
                     f"{c.get('Fail',0)} ({c.get('Fail',0)/tot:.1%})"])
    md.append(L.md_table(["Service", "Audited", "Pass", "Warn", "Fail"], rows))
    md.append("")
    md.append("Note: Socket is missing for a small number of audited skills; the "
              "'Audited' column is each service's own denominator.")

    md += ["", "### Per-skill worst status across its services", ""]
    md.append(L.md_table(["Worst status", "Skills", "Share of audited"],
                         [[st, L.fmt_int(worst.get(st, 0)), L.fmt_pct(worst.get(st, 0) / n_with)]
                          for st in STATUS_ORDER]))
    md.append("")
    md.append(f"- All three services Pass: **{L.fmt_int(all_pass)}** "
              f"({L.fmt_pct(all_pass / n_with)} of audited).")
    md.append(f"- At least one Fail: **{L.fmt_int(any_fail)}** "
              f"({L.fmt_pct(any_fail / n_with)} of audited).")
    md += ["", f"![Audit status by service]({plot_path})", ""]

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "n_with_audits": n_with, "n_without_audits": n_without,
        "coverage": n_with / n,
        "per_service": {svc_names[s]: dict(svc_status[s]) for s in services},
        "worst_status": dict(worst), "all_pass": all_pass, "any_fail": any_fail,
    })
    print(f"[04] audited={n_with} ({n_with/n:.1%}) empty={n_without} "
          f"all_pass={all_pass} any_fail={any_fail} "
          f"per_service={ {svc_names[s]: dict(svc_status[s]) for s in services} }")


if __name__ == "__main__":
    main()