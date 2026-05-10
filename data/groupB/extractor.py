"""
Group B – ALFWorld Memory Extractor
======================================
Parses PDDL game_content into DARS-compatible memories with four layers:

  Layer 1  Instance   – trial-specific object-location facts
  Layer 2  Concept    – generalized object-receptacle rules (Memory Templating)
  Layer 3  Strategy   – task-type-level action templates
  Layer 4  Goal       – per-task natural-language goal description

Each memory carries a ``concept_id`` for cross-trial aggregation.
Concept memories share the same ``concept_id`` as their source instances,
enabling DARS to measure concept-level utility convergence.

Run standalone:  python -m data.groupB.extractor
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

FEEDBACK_THRESHOLD = 0.45
NOISE_FACTS_PER_TASK = 5

STRATEGY_TEMPLATES: Dict[str, str] = {
    "pick_heat_then_place_in_recep": (
        "To heat an object: find the target object, take it to a Microwave, "
        "heat it with the Microwave, then go to the destination receptacle "
        "and place it there"
    ),
    "pick_clean_then_place_in_recep": (
        "To clean an object: find the target object, take it to a SinkBasin, "
        "clean it with the SinkBasin, then go to the destination receptacle "
        "and place it there"
    ),
    "pick_cool_then_place_in_recep": (
        "To cool an object: find the target object, take it to a Fridge, "
        "cool it with the Fridge, then go to the destination receptacle "
        "and place it there"
    ),
    "pick_and_place_simple": (
        "To place an object: find the target object, pick it up, "
        "go to the destination receptacle, and place it there"
    ),
    "pick_two_obj_and_place": (
        "To place two objects: find each target object one at a time, "
        "pick it up, go to the destination receptacle, place it, "
        "then repeat for the second object"
    ),
    "look_at_obj_in_light": (
        "To examine an object in light: find the target object, pick it up, "
        "go to a DeskLamp or FloorLamp, and use the lamp to examine it"
    ),
}

ACTION_KEYWORDS: Dict[str, List[str]] = {
    "heat": ["heat", "hot", "microwave", "warm"],
    "clean": ["clean", "wash", "sinkbasin", "sink"],
    "cool": ["cool", "cold", "fridge", "chill"],
    "examine": ["examine", "look", "lamp", "light", "toggle"],
    "place": ["place", "put", "move"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessedTask:
    """Fully extracted task ready for DARS training and evaluation."""

    task_id: str
    task_type: str
    memories: List[Dict[str, Any]]
    goal_description: str
    goal_predicates: List[str]
    goal_objects: Set[str]
    goal_actions: Set[str]
    walkthrough: List[str]
    object_facts: List[Dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  PDDL Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_coords(name: str) -> str:
    """Strip coordinate suffixes from PDDL object names.

    ``Bowl_bar__minus_00_dot_42_bar__plus_00_dot_08_bar__minus_01_dot_76``
    → ``Bowl``
    """
    return name.split("_bar__")[0] if "_bar__" in name else name


def _strip_type_suffix(type_name: str) -> str:
    """``AppleType`` → ``Apple``"""
    return type_name[:-4] if type_name.endswith("Type") else type_name


def _capitalize(name: str) -> str:
    """Capitalize first letter: ``apple`` → ``Apple``."""
    return name[0].upper() + name[1:] if name else name


# ═══════════════════════════════════════════════════════════════════════════════
#  Walkthrough Parser
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_walkthrough(walkthrough: List[str]) -> Dict[str, Optional[str]]:
    """Extract structured actors from the walkthrough action sequence.

    Returns dict with keys: target_obj, source, tool, destination, action_type.
    """
    result: Dict[str, Optional[str]] = {
        "target_obj": None,
        "source": None,
        "tool": None,
        "destination": None,
        "action_type": None,
    }

    for step in walkthrough:
        sl = step.strip().lower()

        take = re.match(r"take\s+(\w+)\s+\d+\s+from\s+(\w+)\s+\d+", sl)
        if take:
            result["target_obj"] = _capitalize(take.group(1))
            result["source"] = _capitalize(take.group(2))
            continue

        action = re.match(r"(heat|clean|cool)\s+(\w+)\s+\d+\s+with\s+(\w+)\s+\d+", sl)
        if action:
            result["action_type"] = action.group(1)
            result["tool"] = _capitalize(action.group(3))
            continue

        move = re.match(r"(?:move|put)\s+(\w+)\s+\d+\s+to\s+(\w+)\s+\d+", sl)
        if move:
            result["destination"] = _capitalize(move.group(2))
            continue

        use = re.match(r"use\s+(\w+)\s+\d+", sl)
        if use:
            result["tool"] = _capitalize(use.group(1))
            result["action_type"] = "examine"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  PDDL Goal Parser
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_goal_block(pddl: str) -> str:
    """Extract the raw `(:goal ...)` block from PDDL."""
    m = re.search(r"\(:goal\s*(.*?)(?:\)\s*\(:metric|\)\s*$)", pddl, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_goal_details(goal_block: str) -> Tuple[Set[str], Set[str]]:
    """Parse goal block into (goal_objects, goal_actions).

    goal_objects includes both target object types and destination receptacle types.
    goal_actions includes action verbs derived from state predicates.
    """
    objects: Set[str] = set()

    for m in re.finditer(r"objectType\s+\?\w+\s+(\w+)", goal_block):
        objects.add(_strip_type_suffix(m.group(1)))
    for m in re.finditer(r"receptacleType\s+\?\w+\s+(\w+)", goal_block):
        objects.add(_strip_type_suffix(m.group(1)))

    actions: Set[str] = set()
    if re.search(r"isHot", goal_block):
        actions.update(["heat", "microwave"])
    if re.search(r"isClean", goal_block):
        actions.update(["clean", "sinkbasin", "sink"])
    if re.search(r"isCool", goal_block):
        actions.update(["cool", "fridge"])
    if re.search(r"isToggled", goal_block):
        actions.update(["examine", "lamp", "desklamp", "floorlamp"])

    return objects, actions


def _parse_in_receptacle(pddl: str) -> List[Dict[str, str]]:
    """Extract all ``(inReceptacle obj recep)`` from the ``:init`` block."""
    init_m = re.search(r"\(:init(.*?)\(:goal", pddl, re.DOTALL)
    if not init_m:
        return []
    init_block = init_m.group(1)
    pairs = re.findall(r"\(inReceptacle\s+(\S+)\s+(\S+)\)", init_block)
    results: List[Dict[str, str]] = []
    for obj_raw, recep_raw in pairs:
        obj_t = _strip_coords(obj_raw)
        rec_t = _strip_coords(recep_raw)
        results.append({
            "obj_type": obj_t,
            "recep_type": rec_t,
            "concept_id": f"{obj_t.lower()}_{rec_t.lower()}",
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Goal Description Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_goal_nl(
    task_type: str,
    goal_objects: Set[str],
    wt_info: Dict[str, Optional[str]],
) -> str:
    """Build a natural-language goal description from parsed data."""
    obj_name = wt_info.get("target_obj") or "the target object"
    dest = wt_info.get("destination") or "the destination"

    if task_type == "pick_heat_then_place_in_recep":
        return f"Heat the {obj_name} and place it on the {dest}"
    if task_type == "pick_clean_then_place_in_recep":
        return f"Clean the {obj_name} and place it on the {dest}"
    if task_type == "pick_cool_then_place_in_recep":
        return f"Cool the {obj_name} and place it on the {dest}"
    if task_type == "look_at_obj_in_light":
        return f"Examine the {obj_name} under a lamp"
    if task_type == "pick_two_obj_and_place":
        objs = sorted(goal_objects - {dest})
        obj_str = " and ".join(objs) if objs else obj_name
        return f"Place the {obj_str} in the {dest}"
    return f"Place the {obj_name} in the {dest}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_task(row: Dict[str, Any]) -> Optional[ProcessedTask]:
    """Extract a single ALFWorld task into DARS-ready structures.

    Returns None if the game_content cannot be parsed.
    """
    task_id = row["id"]
    task_type = row["task_type"]

    try:
        gc = json.loads(row["game_content"])
    except (json.JSONDecodeError, TypeError):
        logger.warning("Skipping %s: invalid game_content JSON", task_id)
        return None

    pddl = gc.get("pddl_problem", "")
    walkthrough = gc.get("walkthrough", [])
    if not pddl:
        logger.warning("Skipping %s: empty pddl_problem", task_id)
        return None

    wt_info = _parse_walkthrough(walkthrough)
    goal_block = _extract_goal_block(pddl)
    goal_objects, goal_actions = _extract_goal_details(goal_block)
    in_receps = _parse_in_receptacle(pddl)

    goal_desc = _build_goal_nl(task_type, goal_objects, wt_info)

    # Raw goal predicates for PDDL-grounded relevance checking
    goal_preds = re.findall(r"\((\w+)\s+[^)]+\)", goal_block)

    memories: List[Dict[str, Any]] = []
    object_facts: List[Dict[str, Any]] = []

    # ── Layer 1 & 2: Location Memories (from walkthrough) ─────────────
    if wt_info["target_obj"] and wt_info["source"]:
        obj = wt_info["target_obj"]
        src = wt_info["source"]
        cid = f"{obj.lower()}_{src.lower()}"
        object_facts.append({
            "obj_type": obj,
            "recep_type": src,
            "concept_id": cid,
            "instance_text": f"{obj} is located in {src}",
            "concept_text": f"{obj} objects are typically found in {src} receptacles",
        })
        memories.append({
            "text": f"{obj} is located in {src}",
            "mem_type": "instance",
            "concept_id": cid,
            "tags": [
                "mem_type:instance",
                f"task_type:{task_type}",
                f"task_id:{task_id}",
                f"concept_id:{cid}",
            ],
        })
        memories.append({
            "text": f"{obj} objects are typically found in {src} receptacles",
            "mem_type": "concept",
            "concept_id": cid,
            "tags": [
                "mem_type:concept",
                f"task_type:{task_type}",
                f"concept_id:{cid}",
            ],
        })

    # ── Layer 1 & 2: Tool Memories (from walkthrough) ─────────────────
    if wt_info["tool"] and wt_info["action_type"]:
        tool = wt_info["tool"]
        act = wt_info["action_type"]
        cid = f"{tool.lower()}_{act.lower()}"
        memories.append({
            "text": f"{tool} is used to {act} objects",
            "mem_type": "instance",
            "concept_id": cid,
            "tags": [
                "mem_type:instance",
                f"task_type:{task_type}",
                f"task_id:{task_id}",
                f"concept_id:{cid}",
            ],
        })
        memories.append({
            "text": f"{tool} receptacles are commonly used for {act} operations",
            "mem_type": "concept",
            "concept_id": cid,
            "tags": [
                "mem_type:concept",
                f"task_type:{task_type}",
                f"concept_id:{cid}",
            ],
        })

    # ── Layer 1: Sampled Environment Facts (noise) ────────────────────
    goal_obj_lower = {o.lower() for o in goal_objects}
    irrelevant = [
        f for f in in_receps if f["obj_type"].lower() not in goal_obj_lower
    ]
    sampled = random.sample(irrelevant, min(NOISE_FACTS_PER_TASK, len(irrelevant)))
    for fact in sampled:
        cid = fact["concept_id"]
        memories.append({
            "text": f"{fact['obj_type']} is located in {fact['recep_type']}",
            "mem_type": "instance",
            "concept_id": cid,
            "tags": [
                "mem_type:instance",
                f"task_type:{task_type}",
                f"task_id:{task_id}",
                f"concept_id:{cid}",
            ],
        })

    # ── Layer 3: Strategy Memory ──────────────────────────────────────
    strategy_text = STRATEGY_TEMPLATES.get(
        task_type, "Complete the household task efficiently"
    )
    memories.append({
        "text": strategy_text,
        "mem_type": "strategy",
        "concept_id": f"strategy_{task_type}",
        "tags": [
            "mem_type:strategy",
            f"task_type:{task_type}",
            f"concept_id:strategy_{task_type}",
        ],
    })

    # ── Layer 4: Goal Memory ──────────────────────────────────────────
    memories.append({
        "text": goal_desc,
        "mem_type": "goal",
        "concept_id": f"goal_{task_id}",
        "tags": [
            "mem_type:goal",
            f"task_type:{task_type}",
            f"task_id:{task_id}",
        ],
    })

    return ProcessedTask(
        task_id=task_id,
        task_type=task_type,
        memories=memories,
        goal_description=goal_desc,
        goal_predicates=goal_preds,
        goal_objects=goal_objects,
        goal_actions=goal_actions,
        walkthrough=walkthrough,
        object_facts=object_facts,
    )


def extract_all(
    rows: List[Dict[str, Any]],
    max_per_type: Optional[int] = None,
) -> Dict[str, List[ProcessedTask]]:
    """Extract tasks grouped by task_type.

    Args:
        rows: raw dataset rows (one split).
        max_per_type: cap per task_type (None = all).

    Returns:
        ``{task_type: [ProcessedTask, ...]}``
    """
    from collections import defaultdict

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["task_type"]].append(row)

    result: Dict[str, List[ProcessedTask]] = {}
    for ttype, type_rows in sorted(by_type.items()):
        subset = type_rows[:max_per_type] if max_per_type else type_rows
        tasks: List[ProcessedTask] = []
        for r in subset:
            pt = extract_task(r)
            if pt is not None:
                tasks.append(pt)
        result[ttype] = tasks
        logger.info("Extracted %d/%d tasks for %s", len(tasks), len(subset), ttype)

    return result


def print_extraction_report(
    extracted: Dict[str, List[ProcessedTask]],
) -> None:
    """Print summary statistics of the extraction."""
    total_tasks = sum(len(v) for v in extracted.values())
    total_mems = sum(len(t.memories) for ts in extracted.values() for t in ts)

    mem_type_counts: Dict[str, int] = {}
    concept_ids: set = set()
    for tasks in extracted.values():
        for t in tasks:
            for m in t.memories:
                mt = m["mem_type"]
                mem_type_counts[mt] = mem_type_counts.get(mt, 0) + 1
                concept_ids.add(m.get("concept_id", ""))

    print("\n" + "=" * 70)
    print("  ALFWORLD EXTRACTION REPORT")
    print("=" * 70)
    print(f"  Task types:     {len(extracted)}")
    print(f"  Total tasks:    {total_tasks}")
    print(f"  Total memories: {total_mems}")
    print(f"  Unique concepts: {len(concept_ids)}")

    print(f"\n  --- Memories by Type ---")
    for mt, cnt in sorted(mem_type_counts.items()):
        print(f"    {mt}: {cnt}")

    print(f"\n  --- Per Task Type ---")
    for ttype, tasks in sorted(extracted.items()):
        mems = sum(len(t.memories) for t in tasks)
        print(f"    {ttype}: {len(tasks)} tasks, {mems} memories")

    if extracted:
        sample_type = next(iter(extracted))
        sample = extracted[sample_type][0]
        print(f"\n  --- Sample: {sample.task_id} ---")
        print(f"    Goal: {sample.goal_description}")
        print(f"    Goal objects: {sample.goal_objects}")
        print(f"    Goal actions: {sample.goal_actions}")
        print(f"    Walkthrough: {sample.walkthrough[:4]}")
        for m in sample.memories[:6]:
            print(f"    [{m['mem_type']:10s}] {m['text'][:70]}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import json as _json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cache = Path(__file__).parent / "cache" / "alfworld_train.json"
    with open(cache, "r", encoding="utf-8") as f:
        rows = _json.load(f)
    extracted = extract_all(rows, max_per_type=3)
    print_extraction_report(extracted)
