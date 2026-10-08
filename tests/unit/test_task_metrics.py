from treepc.evaluation.humaneval import evaluate_humaneval
from treepc.evaluation.task_metrics import gsm8k_exact_match


def test_gsm8k_exact_match_prefers_final_answer() -> None:
    assert gsm8k_exact_match("2 + 3 = 5\nFinal answer: 5", "5")
    assert not gsm8k_exact_match("Final answer: 6", "5")


def test_humaneval_executes_completion() -> None:
    result = evaluate_humaneval(
        "def add(a, b):\n",
        "    return a + b",
        "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        "add",
    )
    assert result["passed"]
