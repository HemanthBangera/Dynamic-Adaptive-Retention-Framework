import json, collections
from pathlib import Path
H = Path.home()
a = [json.loads(l) for l in open(H / "replay_audit/pristine/benchmark_runs/revision/addendum/msc_test_s3/facts.jsonl")]
b = [json.loads(l) for l in open(H / "replay_audit/out/msc_test_s3/facts.jsonl")]
count_diff = [i for i, (x, y) in enumerate(zip(a, b)) if x["frequency"] != y["frequency"] or x["success"] != y["success"]]
print("facts with count differences:", count_diff, "dialogues:", sorted({a[i]["dialogue"] for i in count_diff}))
write_fields = ["mentions", "mention_sessions", "last_write_time", "opportunities", "created_session", "created_at", "text", "labels"]
print("write/label fields differing:", sum(1 for x, y in zip(a, b) for f in write_fields if x.get(f) != y.get(f)))
pa = [json.loads(l) for l in open(H / "replay_audit/pristine/benchmark_runs/revision/test/E1/ruler_qa1_197K/per_question.jsonl")]
pb = [json.loads(l) for l in open(H / "replay_audit/out/E1/ruler_qa1_197K/per_question.jsonl")]
c = collections.Counter()
def walk(x, y, path):
    if isinstance(x, dict):
        for k in set(x) | set(y):
            walk(x.get(k), y.get(k), path + "." + k)
    elif isinstance(x, list):
        if len(x) != len(y): c[path + " (length)"] += 1
        for u, v in zip(x, y): walk(u, v, path + "[]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)):
        pass
    elif x != y:
        c[path] += 1
for x, y in zip(pa, pb): walk(x, y, "")
print("E1 non-numeric differences by field:", dict(c))
keys = [k for k in pa[0] if "rank" in k.lower() or k in ("shown", "ranked")]
print("rank-like keys:", keys, "identical:", all(x.get(k) == y.get(k) for x, y in zip(pa, pb) for k in keys))
out = {"msc_facts_with_count_differences": len(count_diff), "msc_count_difference_dialogues": sorted({a[i]["dialogue"] for i in count_diff}),
       "msc_write_or_label_fields_differing": sum(1 for x, y in zip(a, b) for f in write_fields if x.get(f) != y.get(f)),
       "e1_non_numeric_difference_fields": dict(c), "e1_ranked_lists_identical": all(x.get(k) == y.get(k) for x, y in zip(pa, pb) for k in keys)}
json.dump(out, open(H / "replay_audit/tie_and_ranking_checks.json", "w"), indent=1)
