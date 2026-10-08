"""Four-task grading. Code execution needs an externally isolated worker."""

import os
import re
import subprocess
import sys

from treepc.evaluation.humaneval import evaluate_humaneval
from treepc.evaluation.task_metrics import gsm8k_exact_match

MAX_NEW_TOKENS = {"gsm8k": 256, "humaneval": 512, "math500": 512, "mbpp": 512}
if os.environ.get("TREEPC_MAX_NEW_TOKENS"):
    MAX_NEW_TOKENS = dict.fromkeys(MAX_NEW_TOKENS, int(os.environ["TREEPC_MAX_NEW_TOKENS"]))


def task_score(dataset, sample, output):
    if dataset == "gsm8k":
        return gsm8k_exact_match(output, sample["answer"]), None
    if dataset == "math500":
        from math_verify import parse, verify

        return bool(verify(parse("$" + sample["answer"] + "$"), parse(output))), None
    if dataset == "humaneval":
        result = evaluate_humaneval(sample["prompt"], output, sample["test"], sample["entry_point"])
        return bool(result["passed"]), result["error"]
    if dataset != "mbpp":
        raise ValueError(f"Unsupported task {dataset}")
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", output, re.S)
    code = fenced[0] if fenced else output
    program = "\n".join([*sample.get("test_setup_code", "").splitlines(), code, *sample["test_list"]])
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-"], input=program, text=True, capture_output=True, timeout=5, check=False
        )
        return result.returncode == 0, None if result.returncode == 0 else result.stderr[-2000:]
    except subprocess.TimeoutExpired:
        return False, "timeout after 5s"
