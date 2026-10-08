# Dataset assets

Dataset contents are intentionally excluded. Official sources and the expected local layout are:

| Dataset | Official source | Expected local location |
|---|---|---|
| GSM8K | [OpenAI GSM8K](https://github.com/openai/grade-school-math) | `data/processed/track1_general/gsm8k/samples.jsonl` |
| HumanEval | [OpenAI HumanEval](https://github.com/openai/human-eval) | `data/processed/track1_general/humaneval/samples.jsonl` |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `data/processed/track1_general/math500/samples.jsonl` |
| MBPP | [Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp) | `data/processed/track1_general/mbpp/samples.jsonl` |

All four task adapters are implemented. Optional preparation:
`python scripts/prepare_data.py --datasets gsm8k humaneval math500 mbpp --output-root /external/data`.
No records are included. Every JSONL row needs a globally unique `sample_id` within its dataset.

- GSM8K: `question`, numeric `answer` (without rationale).
- HumanEval: `prompt`, `test`, `entry_point`.
- MATH-500 (`math500`): `problem`, LaTeX `answer`.
- MBPP: `text`, `test_list`, optional `test_setup_code`.

Standard mode requires `source_split: train|test` and genuinely separate training sources.
Downloaded rows are official test records; internal repurposing must never be called leaderboard evaluation.
