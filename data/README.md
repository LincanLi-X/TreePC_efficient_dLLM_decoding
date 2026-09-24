# Dataset assets

Dataset contents are intentionally excluded. Official sources and the expected local layout are:

| Dataset | Official source | Expected local location |
|---|---|---|
| GSM8K | [OpenAI GSM8K](https://github.com/openai/grade-school-math) | `data/processed/track1_general/gsm8k/samples.jsonl` |
| HumanEval | [OpenAI HumanEval](https://github.com/openai/human-eval) | `data/processed/track1_general/humaneval/samples.jsonl` |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `data/processed/track1_general/math500/samples.jsonl` |
| MBPP | [Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp) | `data/processed/track1_general/mbpp/samples.jsonl` |

Run `python scripts/prepare_data.py` to download and normalize the GSM8K and HumanEval files used
by the checked-in `large_v2` pipeline. MATH-500 and MBPP are listed for the broader benchmark
suite; they require a task adapter before use with this snapshot.
