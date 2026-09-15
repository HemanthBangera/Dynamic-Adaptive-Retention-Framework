"""E11 sources: family/split mapping, ICL example units and DetectiveQA retrieval queries (no LLM)."""

from benchmarks.dars_eval.datasets import family_of, icl_units, retrieval_query, split_name_for
from benchmarks.dars_eval.memory_units import WindowGuard


def test_family_and_split_mapping():
    assert family_of("icl_banking77_5900shot_balance") == "icl"
    assert split_name_for("icl_banking77_5900shot_balance") == "Test_Time_Learning"
    assert family_of("detective_qa") == "detective_qa"
    assert split_name_for("detective_qa") == "Long_Range_Understanding"
    assert split_name_for("factconsolidation_sh_32k") == "Conflict_Resolution"
    assert split_name_for("ruler_qa1_197K") == "Accurate_Retrieval"
    assert family_of("eventqa_full") == "eventqa"


def test_icl_units_hold_one_labelled_example_each():
    ctx = "How do I top up?\nlabel: 3\n\nMy card is lost\nlabel: 7\n\n\nWhy was I charged twice?\nlabel: 3"
    units = icl_units(ctx, WindowGuard())
    assert [u.text for u in units] == [
        "How do I top up?\nlabel: 3", "My card is lost\nlabel: 7", "Why was I charged twice?\nlabel: 3"]
    assert [u.meta["example"] for u in units] == [0, 1, 2]


def test_detective_query_keeps_only_the_question_and_options():
    q = ('\n For example:\n [story text]\n Question: x ( )\n A. a\n Output: {"answer":"A. a"}\n'
         ' Now Answer the Question:  Who took the key?\nA. Anna\nB. Ben\n Output:')
    assert retrieval_query("detective_qa", q, WindowGuard()) == "Who took the key?\nA. Anna\nB. Ben"
