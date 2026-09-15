"""ALFWorld walkthroughs parse on every split, including the simulator-id out-of-distribution one."""

import json

from benchmarks.dars_eval.run_alfworld import evaluate
from data.groupB.extractor import _canonical_name, _normalize_step, _parse_walkthrough

OUT_OF_DISTRIBUTION = [
    "go to desk_bar__minus_00_dot_57_bar__plus_00_dot_00_bar__minus_01_dot_35",
    "take cd_bar__minus_00_dot_40_bar__plus_00_dot_86_bar__minus_00_dot_66 "
    "from desk_bar__minus_00_dot_57_bar__plus_00_dot_00_bar__minus_01_dot_35",
    "use desklamp_bar__minus_02_dot_30_bar__plus_00_dot_87_bar__plus_00_dot_75",
]
READABLE = ["go to desk 1", "take cd 1 from desk 1", "go to shelf 2", "use desklamp 1"]


def test_canonical_name_prefers_a_trailing_type_segment():
    assert _canonical_name("cd_bar__minus_00_dot_40_bar__plus_00_dot_86") == "cd"
    assert _canonical_name("desk_bar__minus_00_dot_57_bar__plus_00_dot_00") == "desk"
    # the train split calls this receptacle "sinkbasin", and the id says so at the end
    assert _canonical_name("sink_bar__minus_00_dot_30_bar__plus_03_dot_26_bar_sinkbasin") == "sinkbasin"


def test_normalize_step_rewrites_ids_and_leaves_readable_steps_alone():
    assert _normalize_step(OUT_OF_DISTRIBUTION[1]) == "take cd 1 from desk 1"
    for step in READABLE:
        assert _normalize_step(step) == step


def test_out_of_distribution_walkthrough_yields_the_same_fields_as_the_readable_one():
    raw = _parse_walkthrough(OUT_OF_DISTRIBUTION)
    assert (raw["target_obj"], raw["source"]) == ("Cd", "Desk")
    assert (raw["tool"], raw["action_type"]) == ("Desklamp", "examine")
    readable = _parse_walkthrough(READABLE)
    assert (raw["target_obj"], raw["source"], raw["tool"]) == \
           (readable["target_obj"], readable["source"], readable["tool"])


def test_tool_actions_parse_through_the_sinkbasin_alias():
    wt = _parse_walkthrough([
        "clean cloth_bar__minus_00_dot_26_bar__plus_02_dot_84 "
        "with sink_bar__minus_00_dot_30_bar__plus_03_dot_26_bar_sinkbasin"])
    assert wt["action_type"] == "clean" and wt["tool"] == "Sinkbasin"


def test_unscorable_split_is_reported_rather_than_raising(tmp_path):
    (tmp_path / "eval_test_out.jsonl").write_text(json.dumps({
        "task_id": "t1", "split": "test_out", "task_type": "pick_and_place_simple",
        "truth": {"obj": None, "source": None, "tool": None, "action": None,
                  "task_type": "pick_and_place_simple"},
        "reachable": False, "candidates": [], "needed": ["p1"], "query": "q"}) + "\n", encoding="utf-8")
    (tmp_path / "memories.jsonl").write_text(json.dumps({
        "pid": "p1", "recency": 0.0, "created_at": 0.0, "frequency": 0,
        "F": 0.0, "U": 0.5, "P": 0.0}) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"now": 10.0}), encoding="utf-8")

    report = evaluate(tmp_path, "test_out", (0.0005, "score_only", 0.0, (0, 1, 0, 0)),
                      (0.0005, (0, 0, 0, 1)), n_boot=20)
    assert report["h3"]["not_evaluable"]
    assert "mrr" not in report["h3"]
    assert report["h5"]["harmful_deletion"]["selected"]["mean"] == 0.0
