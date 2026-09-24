from __future__ import annotations

import re
import subprocess
import sys
from typing import Any


def extract_python_completion(prompt: str, output: str, entry_point: str) -> str:
    text = output.strip("\r\n")
    fenced = re.findall(r"```(?:python)?[ \t]*\r?\n?(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced[0].strip("\r\n")
    if text.startswith(prompt):
        text = text[len(prompt) :]
    for stop in ("\n\n#", "\n\nif __name__", "\n\nExplanation:", "\n\nThe code"):
        if stop in text:
            text = text.split(stop, 1)[0]
    first = next((line for line in text.splitlines() if line.strip()), "")
    contains_definition = bool(re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", text))
    if text and first == first.lstrip() and not contains_definition:
        text = "\n".join(("    " + line) if line else line for line in text.splitlines())
    return text.rstrip() + "\n"


def evaluate_humaneval(
    prompt: str,
    model_output: str,
    test: str,
    entry_point: str,
    timeout_seconds: int = 5,
) -> dict[str, Any]:
    completion = extract_python_completion(prompt, model_output, entry_point)
    code = prompt + "\n" + completion + "\n" + test + f"\ncheck({entry_point})\n"
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-"],
            input=code,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        error = None if result.returncode == 0 else (result.stderr or result.stdout)[-2000:]
        return {"passed": result.returncode == 0, "completion": completion, "error": error}
    except subprocess.TimeoutExpired:
        return {"passed": False, "completion": completion, "error": f"timeout after {timeout_seconds}s"}
