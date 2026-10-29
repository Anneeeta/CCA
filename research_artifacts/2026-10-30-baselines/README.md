# Additional baseline research artifacts

This supplement accompanies the two-group CCA evaluation. It contains the selected runs of OmniThink and OpenDeepSearch + ReAct on 100 WildSeek tasks and the accepted Group 2 MiMo evaluations of CCA and both baselines. It complements the original experimental archive, which contains Group 1, the component ablations, and the de-identified human study.

## Contents

- `generation/omnithink.jsonl.xz`: 100 selected generation records, including articles, tree/concepts, retrieval records, model requests/responses and usage.
- `generation/opendeepsearch.jsonl.xz`: 100 selected generation records, including final answers, search contexts, tool records, model requests/responses and usage.
- `evaluations/requests.jsonl.xz`: 200 accepted evaluation inputs, one per task and rubric. Each input contains the three anonymous system records and the executed rubric.
- `evaluations/results.jsonl.xz`: the corresponding 200 accepted responses, scores, model usage and stored anonymous mappings.
- `evaluations/accepted.json`: task/rubric mappings and source hashes, including the verified first-ten evaluations reused in the 100-task analysis.
- `analysis/scores.csv`: 900 task/criterion rows, with the three system scores on each row.
- `analysis/summary_original.json` and `analysis/recompute.py`: archived summary and offline reproduction of its means and 27 corrected paired comparisons.
- `analysis/recompute_omnibus.py` and `analysis/omnibus_holm9.json`: post-hoc Group 2 Friedman tests and Holm-9 adjustment.
- `analysis/recompute_usage.py`, `analysis/group1_usage_summary.json` and `analysis/generation_usage_all_systems.json`: reproducible provider-reported generation-token totals for all five systems.
- `provenance/`: selected-generation and evaluation indexes. Original source hashes are retained; the packaged records remove credentials and absolute local workspace paths.
- `scripts/`: the generation/evaluation orchestration scripts used locally. Archived prompts and requests retain the actual model, tool and parameter settings used by each selected run.

The generation archives contain one JSON object per line. Each object's `files` mapping records the source-file hash, format, and parsed content. For JSONL files, the content is a list of their original records; text artifacts retain their text. The final information tree and complete call traces are retained; duplicate intermediate tree snapshots are omitted.

## Reproduce statistics

Use Python with NumPy and SciPy, then run:

```
python analysis/recompute.py
python analysis/recompute_omnibus.py
python analysis/recompute_usage.py
```

The program recomputes sample standard deviations, two-sided Wilcoxon signed-rank tests using Pratt's zero handling and automatic inference, and one Holm family covering all 27 criterion/pair comparisons. It asserts agreement with the archived summary and writes `analysis/summary_reproduced.json`. It makes no paid model or search calls.

Group 2 contains CCA, OmniThink and OpenDeepSearch + ReAct. Candidate order was shuffled by task. CCA scores were elicited together with the other two records and were not fixed to Group 1 scores. Comparisons and corrected significance are interpreted within Group 2. The manuscript's ablation anchors remain the Group 1 CCA scores.

## Run selection and use

Successes from the original batches and permitted failure-only retries were selected using the saved accepted indexes. Successful task records were preserved. OmniThink's selected runs include non-polishing output ceilings of 2,000 and 8,192 tokens; actual requests document which setting was used. Selected final answers and the evaluation inputs are provided together so users can inspect what the judge received.

Source titles, URLs and retrieved text are retained for research verification. Rights in third-party web content remain with its respective owners. The original WildSeek task set is distributed with Co-STORM; cite Jiang et al. (2024) when using it. Cite the OmniThink and OpenDeepSearch papers for the two baseline systems. This package is supplied for research review and reproducibility; it contains no API credentials or manuscript PDF.
