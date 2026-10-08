#!/usr/bin/env python
"""Download official evaluation sources; never manufacture benchmark training splits."""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/" "master/grade_school_math/data/test.jsonl"
)
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "TreePC-artifact/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return response.read()


def jsonl_rows(payload: bytes) -> Iterable[dict[str, Any]]:
    for line in payload.decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def prepare_gsm8k(payload: bytes) -> list[dict[str, Any]]:
    rows = []
    for index, raw in enumerate(jsonl_rows(payload)):
        rationale = str(raw["answer"])
        answer = rationale.rsplit("####", maxsplit=1)[-1].strip().replace(",", "")
        rows.append(
            {
                "sample_id": str(index),
                "dataset": "gsm8k",
                "task_type": "math",
                "question": raw["question"],
                "answer": answer,
                "rationale": rationale,
                "raw": raw,
                "source_split": "test",
            }
        )
    return rows


def prepare_humaneval(payload: bytes) -> list[dict[str, Any]]:
    rows = []
    for raw in jsonl_rows(gzip.decompress(payload)):
        rows.append(
            {
                "sample_id": raw["task_id"],
                "dataset": "humaneval",
                "task_type": "code",
                "prompt": raw["prompt"],
                "entry_point": raw["entry_point"],
                "test": raw["test"],
                "canonical_solution": raw["canonical_solution"],
                "raw": raw,
                "source_split": "test",
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="data/processed/track1_general",
        help="Output directory containing gsm8k/ and humaneval/ (default: %(default)s)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["gsm8k", "humaneval", "math500", "mbpp"],
        default=["gsm8k", "humaneval"],
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser()
    if "gsm8k" in args.datasets:
        write_rows(output_root / "gsm8k/samples.jsonl", prepare_gsm8k(download(GSM8K_URL)), args.force)
    if "humaneval" in args.datasets:
        write_rows(
            output_root / "humaneval/samples.jsonl", prepare_humaneval(download(HUMANEVAL_URL)), args.force
        )
    if "math500" in args.datasets:
        from datasets import load_dataset

        rows = [
            dict(row, sample_id=str(row["unique_id"]), dataset="math500", source_split="test")
            for row in load_dataset("HuggingFaceH4/MATH-500", split="test")
        ]
        write_rows(output_root / "math500/samples.jsonl", rows, args.force)
    if "mbpp" in args.datasets:
        url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
        rows = [
            dict(row, sample_id=str(row["task_id"]), dataset="mbpp", source_split="test")
            for row in jsonl_rows(download(url))
            if 11 <= int(row["task_id"]) <= 510
        ]
        write_rows(output_root / "mbpp/samples.jsonl", rows, args.force)


if __name__ == "__main__":
    main()
