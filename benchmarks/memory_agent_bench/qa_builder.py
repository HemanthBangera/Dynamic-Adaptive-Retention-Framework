"""
Build (formatted_query, answer, qa_pair_id) tuples like MemoryAgentBench
`ConversationCreator._create_query_answer_pairs`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from third_party.memoryagentbench_eval import get_template

# Agent name must contain `Agentic_memory` for upstream `normalize_agent_name`.
MAB_AGENT_TEMPLATE_KEY = "dars_Agentic_memory"


def _get_field_value(
    question_data: Dict[str, Any], field_name: str, question_index: int
) -> Any:
    field_value = question_data.get(field_name) or (
        question_data.get("metadata") or {}
    ).get(field_name)
    if field_value is None:
        return None
    if isinstance(field_value, list) and question_index < len(field_value):
        return field_value[question_index]
    return field_value


def _create_qa_metadata(
    question_data: Dict[str, Any],
    question: str,
    answer: Any,
    question_index: int,
) -> Dict[str, Any]:
    qa_metadata = dict(question_data)
    qa_metadata.update({"question": question, "answer": answer})
    for field_name in (
        "question_dates",
        "question_types",
        "question_ids",
        "previous_events",
        "qa_pair_ids",
    ):
        val = _get_field_value(question_data, field_name, question_index)
        if val is not None:
            qa_metadata[field_name] = val
    if "source" not in qa_metadata:
        qa_metadata["source"] = (question_data.get("metadata") or {}).get("source", "")
    return qa_metadata


def _create_single_qa_pair(
    question_data: Dict[str, Any],
    question: str,
    answer: Any,
    question_index: int,
    sub_dataset: str,
) -> Tuple[str, Any, Any]:
    qa_metadata = _create_qa_metadata(question_data, question, answer, question_index)
    query_template = get_template(sub_dataset, "query", MAB_AGENT_TEMPLATE_KEY)
    formatted_query = query_template.format(**qa_metadata)
    qa_pair_id = qa_metadata.get("qa_pair_ids")
    return formatted_query, answer, qa_pair_id


def build_qa_pairs(row: Dict[str, Any], sub_dataset: str) -> List[Tuple[str, Any, Any]]:
    """Return list of (formatted_query, answer, qa_pair_id)."""
    question_data = {k: v for k, v in row.items() if k != "context"}
    questions = question_data.get("questions") or []
    answers = question_data.get("answers") or []
    if not isinstance(questions, list):
        questions = [questions]
    if not isinstance(answers, list):
        answers = [answers]

    if len(questions) > 1 and len(answers) > 1:
        return [
            _create_single_qa_pair(question_data, q, a, i, sub_dataset)
            for i, (q, a) in enumerate(zip(questions, answers))
        ]
    q0 = questions[0] if questions else ""
    a0 = answers[0] if len(answers) == 1 else answers
    return [_create_single_qa_pair(question_data, q0, a0, 0, sub_dataset)]


def min_context_chars() -> int:
    """Upstream ConversationCreator asserts context length > 2000."""
    return 2001
