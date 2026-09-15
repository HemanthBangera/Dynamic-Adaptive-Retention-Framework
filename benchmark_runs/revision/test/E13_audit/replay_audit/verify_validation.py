import json, collections
from pathlib import Path
H = Path.home()
a = [json.loads(l) for l in open(H / "replay_audit/pristine/benchmark_runs/revision/addendum/msc_validation_s3/facts.jsonl")]
b = [json.loads(l) for l in open(H / "replay_audit/out/msc_validation_s3/facts.jsonl")]
count_diff = [i for i, (x, y) in enumerate(zip(a, b)) if x["frequency"] != y["frequency"] or x["success"] != y["success"] or x["failure"] != y["failure"]]
write_fields = ["mentions", "mention_sessions", "last_write_time", "opportunities", "created_session", "created_at", "text", "labels", "dialogue"]
comp_max = 0.0; comp_n = 0
for x, y in zip(a, b):
    for k, v in x["components"].items():
        if v != y["components"][k]:
            comp_n += 1; comp_max = max(comp_max, abs(v - y["components"][k]))
out = {"facts": [len(a), len(b)], "facts_with_count_differences": len(count_diff),
       "dialogues": sorted({a[i]["dialogue"] for i in count_diff}),
       "write_or_label_fields_differing": sum(1 for x, y in zip(a, b) for f in write_fields if x.get(f) != y.get(f)),
       "component_values_differing": comp_n, "component_max_abs": comp_max}
json.dump(out, open(H / "replay_audit/validation_s3_checks.json", "w"), indent=1)
print(json.dumps(out))
