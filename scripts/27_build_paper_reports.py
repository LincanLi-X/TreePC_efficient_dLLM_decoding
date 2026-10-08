#!/usr/bin/env python
"""RQ3/task curves, validation-selected matched-quality RQ4, and labeled proxies."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from treepc.utils.io import write_json


def aggregate(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = tuple(
            row[k]
            for k in (
                "backbone",
                "dataset",
                "teacher_nfe",
                "split",
                "method",
                "steps",
                "gpu_name",
                "max_new_tokens",
                "profile_aux",
                "manifest_fingerprint",
            )
        )
        identity = (*key, row["sample_id"], row["repeat"])
        if identity in seen:
            raise ValueError(f"Duplicate evaluation row: {identity}")
        seen.add(identity)
        groups[key].append(row)
    summaries = []
    for key, values in groups.items():
        samples = defaultdict(list)
        for row in values:
            samples[row["sample_id"]].append(row)
        if any(len({r["passed"] for r in v}) != 1 for v in samples.values()):
            raise ValueError("Timing repeats with the same seed changed task outcomes")
        result = dict(
            zip(
                (
                    "backbone",
                    "dataset",
                    "teacher_nfe",
                    "split",
                    "method",
                    "steps",
                    "gpu_name",
                    "max_new_tokens",
                    "profile_aux",
                    "manifest_fingerprint",
                ),
                key,
            )
        )
        result.update(
            score=statistics.mean(float(v[0]["passed"]) for v in samples.values()),
            sample_count=len(samples),
            repeat_count=len(values),
            actual_nfe=statistics.mean(r["actual_nfe"] for r in values),
            latency_s=statistics.mean(r["latency_s"] for r in values),
            peak_gpu_memory_mib=max(r["peak_gpu_memory_mib"] for r in values),
            throughput=statistics.mean(r["output_tokens_per_s"] for r in values),
            timing_source="measured",
            sample_ids=sorted(samples),
        )
        for metric in ("dependency_head_s", "mst_s", "correction_head_s"):
            result[metric] = statistics.mean(r.get("timings", {}).get(metric, 0.0) for r in values)
        summaries.append(result)
    return summaries


def matched_quality(rows, tolerance):
    groups = defaultdict(list)
    for row in rows:
        if not row["profile_aux"]:
            groups[
                tuple(
                    row[k]
                    for k in (
                        "backbone",
                        "dataset",
                        "teacher_nfe",
                        "gpu_name",
                        "max_new_tokens",
                        "manifest_fingerprint",
                    )
                )
            ].append(row)
    result = []
    for values in groups.values():
        validation = [r for r in values if r["split"] == "validation"]
        test = [r for r in values if r["split"] == "test"]
        refs = [r for r in validation if r["method"] == "vanilla" and r["steps"] == r["teacher_nfe"]]
        if not refs:
            continue
        ref = refs[0]
        test_lookup = {(r["method"], r["steps"]): r for r in test}
        baseline = test_lookup.get(("vanilla", ref["steps"]))
        if baseline is None:
            continue
        for method in sorted({r["method"] for r in validation} - {"vanilla"}):
            candidates = [
                r for r in validation if r["method"] == method and r["score"] >= ref["score"] - tolerance
            ]
            if not candidates:
                continue
            selected = min(candidates, key=lambda r: (r["latency_s"], r["steps"]))
            target = test_lookup.get((method, selected["steps"]))
            if target is None:
                continue
            if target["sample_ids"] != baseline["sample_ids"]:
                raise ValueError("Matched-quality comparison has different test prompts")
            if selected["sample_ids"] != ref["sample_ids"]:
                raise ValueError("Budget selection has different validation prompts")
            result.append(
                {k: v for k, v in target.items() if k != "sample_ids"}
                | {
                    "reference_steps": ref["steps"],
                    "reference_test_score": baseline["score"],
                    "reference_latency_s": baseline["latency_s"],
                    "speedup": baseline["latency_s"] / target["latency_s"],
                    "selection_split": "validation",
                    "selection_tolerance": tolerance,
                    "test_quality_within_tolerance": target["score"] >= baseline["score"] - tolerance,
                }
            )
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row if key != "sample_ids"})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument("--proxy-spec", help="Optional JSON list: anchor fields + target_nfe/target_dataset")
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.inputs for line in Path(path).read_text().splitlines() if line]
    summaries = aggregate(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "task_curves.json", summaries)
    write_csv(output / "task_curves.csv", summaries)
    matched = matched_quality(summaries, args.quality_tolerance)
    write_json(
        output / "rq4_matched_quality.json",
        {
            "rows": matched,
            "status": "complete" if matched else "missing_validation_or_quality_matched_candidates",
            "timing_source": "measured",
            "selection_split": "validation",
        },
    )
    write_csv(output / "rq4_matched_quality.csv", matched)
    write_csv(
        output / "rq3_test.csv", [r for r in summaries if r["split"] == "test" and not r["profile_aux"]]
    )
    write_csv(output / "rq4_overhead.csv", [r for r in summaries if r["profile_aux"]])
    if args.proxy_spec:
        proxies = []
        for spec in json.loads(Path(args.proxy_spec).read_text()):
            anchors = [r for r in summaries if all(r.get(k) == v for k, v in spec["anchor"].items())]
            if len(anchors) != 1:
                raise ValueError("Proxy specification must resolve exactly one measured anchor")
            anchor = anchors[0]
            proxies.append(
                {
                    "target_dataset": spec["target_dataset"],
                    "target_nfe": spec["target_nfe"],
                    "latency_s": anchor["latency_s"] * spec["target_nfe"] / anchor["actual_nfe"],
                    "timing_source": "estimated",
                    "formula": "anchor_latency * target_nfe / anchor_actual_nfe",
                    "anchor": anchor,
                    "limitations": "Linear proxy, not a target-task hardware measurement",
                }
            )
        write_json(output / "rq4_estimates.json", proxies)


if __name__ == "__main__":
    main()
