"""Compare the primary confirmatory family recomputed from MSC data regenerated in the audit with the archived family.

    python compare_confirm.py ARCHIVED_confirm_primary.json REPLAYED_confirm_primary.json OUT.json
"""
import json
import sys

FIELDS = ("auc_a", "auc_b", "diff", "ci_lo", "ci_hi", "holm_p")


def main() -> None:
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    b = json.load(open(sys.argv[2], encoding="utf-8"))
    decisions, at_3dp, max_abs = [], [], 0.0

    def compare(ra, rb, tag):
        nonlocal max_abs
        assert ra["id"] == rb["id"], (ra["id"], rb["id"])
        if ra["result"] != rb["result"]:
            decisions.append(f"{tag}:{ra['id']}")
        for k in FIELDS:
            if ra.get(k) is None:
                continue
            max_abs = max(max_abs, abs(ra[k] - rb[k]))
            if round(ra[k], 3) != round(rb[k], 3):
                at_3dp.append(f"{tag}:{ra['id']}.{k} {ra[k]:.5f} -> {rb[k]:.5f}")

    for ra, rb in zip(a["primary_family"], b["primary_family"]):
        compare(ra, rb, "family")
    for split in a["primary_by_split"]:
        for ra, rb in zip(a["primary_by_split"][split], b["primary_by_split"][split]):
            compare(ra, rb, split)
    confirmed_a = sum(r["result"] == "confirmed" for r in a["primary_family"])
    confirmed_b = sum(r["result"] == "confirmed" for r in b["primary_family"])
    out = {"facts_archived": a["facts"], "facts_replayed": b["facts"], "confirmed_archived": confirmed_a,
           "confirmed_replayed": confirmed_b, "family_size": len(b["primary_family"]),
           "decisions_changed": decisions, "values_changed_at_3dp": at_3dp, "max_abs_difference": max_abs}
    out["summary"] = (f"{confirmed_b} of {len(b['primary_family'])} comparisons confirmed, as archived ({confirmed_a}); "
                      f"{len(decisions)} decisions changed overall or per split; largest difference in any AUROC, effect, "
                      f"interval bound or Holm p {max_abs:.1e}; {len(at_3dp)} values change at three decimals.")
    json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
