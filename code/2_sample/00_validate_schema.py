"""00 - Validate the input against its own metadata block.

Confirms record count, top-level schema, and field coverage match what the
metadata claims. Recomputes every number in the metadata block from the raw
records and flags any discrepancy. Produces no plots.

Run:  python3 scripts/00_validate_schema.py
"""
from __future__ import annotations

import statistics
from collections import Counter

import _lib as L

NUM, TOPIC = "00", "validation"


def is_present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    return True


def main() -> None:
    meta, skills = L.load()
    n = len(skills)
    checks: list[tuple[str, object, object, bool]] = []  # (label, claimed, observed, ok)

    # --- record count ---
    checks.append(("combined_records", meta["combined_records"], n,
                   meta["combined_records"] == n))

    # --- schema: which top-level keys appear, and in how many records ---
    key_presence = Counter()
    for r in skills:
        key_presence.update(r.keys())
    all_keys = sorted(key_presence)
    keys_in_all = [k for k in all_keys if key_presence[k] == n]
    keys_partial = {k: key_presence[k] for k in all_keys if key_presence[k] != n}

    # --- field coverage (recomputed) ---
    cov = {
        "skill_name": sum(is_present(r.get("skill_name")) for r in skills),
        "summary": sum(is_present(r.get("summary")) for r in skills),
        "topic_slug": sum(is_present(r.get("topic_slug")) for r in skills),
        "installs_abbreviated": sum(is_present(r.get("installs_abbreviated")) for r in skills),
        "github_stars_abbreviated": sum(is_present(r.get("github_stars_abbreviated")) for r in skills),
        "install_command": sum(is_present(r.get("install_command")) for r in skills),
        "audits_present": sum(1 for r in skills if r.get("audits")),
        "skill_md_raw": sum(1 for r in skills if (r.get("skill_md") or {}).get("raw")),
    }
    for field, claimed in meta["field_coverage"].items():
        observed = cov.get(field)
        checks.append((f"field_coverage.{field}", claimed, observed, claimed == observed))

    # --- scalar metadata claims ---
    installs = [r["installs"] for r in skills]
    n_official = sum(1 for r in skills if r.get("is_official") is True)
    repos = {r["repo"] for r in skills}
    owners = {L.owner_of(r["repo"]) for r in skills}
    obs_median = statistics.median(installs)
    checks += [
        ("is_official_count", meta["is_official_count"], n_official, meta["is_official_count"] == n_official),
        ("unique_source_repos", meta["unique_source_repos"], len(repos), meta["unique_source_repos"] == len(repos)),
        ("unique_owners", meta["unique_owners"], len(owners), meta["unique_owners"] == len(owners)),
        ("installs_min", meta["installs_min"], min(installs), meta["installs_min"] == min(installs)),
        ("installs_max", meta["installs_max"], max(installs), meta["installs_max"] == max(installs)),
        ("installs_median", meta["installs_median"], obs_median, float(meta["installs_median"]) == float(obs_median)),
    ]

    # --- supporting detail not in metadata but worth recording ---
    fetch_status = Counter((r.get("skill_md") or {}).get("fetch_status", "<null>") for r in skills)
    audit_empty = sum(1 for r in skills if isinstance(r.get("audits"), list) and len(r["audits"]) == 0)
    fm_parsed = sum(1 for r in skills if (r.get("skill_md") or {}).get("frontmatter"))
    body_present = sum(1 for r in skills if (r.get("skill_md") or {}).get("body"))

    discrepancies = [c for c in checks if not c[3]]

    # ------------------------------------------------------------------ output
    rows = [[lbl, L.fmt_int(claimed) if isinstance(claimed, (int, float)) else claimed,
             L.fmt_int(obs) if isinstance(obs, (int, float)) else obs,
             "OK" if ok else "**MISMATCH**"] for lbl, claimed, obs, ok in checks]

    md = [f"## {NUM} — Schema & metadata validation", ""]
    md.append(f"Input: `inputs/{L.INPUT_PATH.name}`  |  records: **{L.fmt_int(n)}**")
    md.append("")
    if discrepancies:
        md.append(f"**{len(discrepancies)} discrepancy(ies) found vs the metadata block** "
                  "— see the MISMATCH rows below.")
    else:
        md.append("Every metadata claim was reproduced exactly from the raw records "
                  "(**no discrepancies**).")
    md += ["", "### Metadata claim vs recomputed", ""]
    md.append(L.md_table(["Metadata field", "Claimed", "Observed", "Match"], rows))

    md += ["", "### Top-level schema", ""]
    md.append(f"All records carry the same {len(keys_in_all)} top-level keys "
              "(none missing in any record):")
    md.append("")
    md.append("`" + "`, `".join(keys_in_all) + "`")
    if keys_partial:
        md += ["", "Keys not present in every record:"]
        for k, c in keys_partial.items():
            md.append(f"- `{k}`: {L.fmt_int(c)}/{L.fmt_int(n)}")
    else:
        md += ["", "No top-level key is partially present — missing values are "
               "encoded as JSON `null`, never as absent keys or empty strings "
               "(except where noted)."]

    md += ["", "### Supporting detail (not claimed in metadata)", ""]
    md.append(L.md_table(
        ["Item", "Value"],
        [["skill_md.fetch_status = found", L.fmt_int(fetch_status.get("found", 0))],
         ["skill_md.fetch_status = not_found_via_patterns", L.fmt_int(fetch_status.get("not_found_via_patterns", 0))],
         ["skill_md.fetch_status = not_attempted_cache_only", L.fmt_int(fetch_status.get("not_attempted_cache_only", 0))],
         ["skill_md.body present", L.fmt_int(body_present)],
         ["skill_md.frontmatter parsed", L.fmt_int(fm_parsed)],
         ["records with empty audits list", L.fmt_int(audit_empty)]]))
    md.append("")
    md.append(f"Note: {L.fmt_int(cov['skill_md_raw'])} records have `skill_md.raw`/`body`, "
              f"but only {L.fmt_int(fm_parsed)} have parseable `frontmatter` — "
              f"{cov['skill_md_raw'] - fm_parsed} SKILL.md files were fetched but their "
              "YAML frontmatter did not parse.")

    L.write_section(NUM, TOPIC, "\n".join(md))
    L.write_stats(NUM, TOPIC, {
        "n_records": n,
        "discrepancy_count": len(discrepancies),
        "discrepancies": [{"field": d[0], "claimed": d[1], "observed": d[2]} for d in discrepancies],
        "toplevel_keys": keys_in_all,
        "keys_partial": keys_partial,
        "field_coverage_observed": cov,
        "fetch_status": dict(fetch_status),
        "audit_empty_list": audit_empty,
        "frontmatter_parsed": fm_parsed,
        "body_present": body_present,
        "installs_median_observed": obs_median,
    })

    # console summary
    print(f"[00] records={n}  discrepancies={len(discrepancies)}")
    for d in discrepancies:
        print(f"   MISMATCH {d[0]}: claimed={d[1]} observed={d[2]}")
    print(f"   fetch_status={dict(fetch_status)}  frontmatter_parsed={fm_parsed}")


if __name__ == "__main__":
    main()
