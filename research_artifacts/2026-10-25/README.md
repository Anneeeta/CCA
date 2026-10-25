# Experimental data for CCA

This release supports **A Learner-State-Driven Multi-Agent Architecture for
Exploratory Information Seeking**. It contains the accepted records used for
the manuscript's results and offline code for reproducing the statistical tables.
The release does not run models or make search requests.

## Contents

| Content | Coverage | Location |
| --- | --- | --- |
| Task inputs | 100 common WildSeek tasks | `tasks/wildseek_tasks.json` |
| Generation trajectories | CCA, Co-STORM and RAG, 100 tasks each | `archives/generation_*.jsonl.xz` |
| Ablation trajectories | Five variants, 100 tasks each | `archives/generation_<variant>.jsonl.xz` |
| Accepted benchmark responses | 100 tasks × two rubrics = 200 joint evaluations | `evaluations/benchmark/` |
| Accepted ablation responses | 100 tasks × five variants × two rubrics = 1,000 evaluations | `evaluations/ablations/` |
| Archived evaluation inputs | The corresponding 1,200 message payloads | `archives/*evaluation_inputs*.jsonl.xz` |
| Numerical scores | Benchmark, ablation differences and generation usage | `scores/` |
| Paired human ratings | 40 paired sessions, 20 for each baseline; nine metrics | `human/` |
| Executed scoring prompts | Four benchmark and reference-guided ablation prompts | `protocols/` |
| Statistical reproduction | Offline analysis and saved manuscript statistics | `analysis/` |
| Provenance | Source hashes, released-record paths and archive counts | `provenance/`, `manifest.json`, `SHA256SUMS.txt` |

Each compressed archive is UTF-8 JSON Lines with XZ compression. A line is one
evaluation request or one system-task generation record. This avoids a large,
uncompressed checkout while preserving full message contents and trajectories.
Source webpages are represented by the retrieval material retained in the run;
the release is not an archive of complete webpages.

## Systems and ablations

The benchmark compares CCA, Co-STORM and RAG robot on the same 100 tasks.
`LLM Baseline` is the historical internal label for RAG robot in saved evaluation
mappings. Score tables normalize that label to `RAG`.

| Stored variant | Manuscript label |
| --- | --- |
| `no_ms` | w/o Learner State |
| `random_selection` | w/o Question Selection |
| `no_shared_context` | w/o Shared Knowledge Context |
| `no_patches` | w/o Cross-Round Adaptation |
| `no_expert_instructor_separation` | w/o Expert–Instructor Separation |

The generation model is DeepSeek-V4.1-Flash (recorded API identifier `deepseek-flash`); the judge is
`mimo-v2.6-flash`. Generation records preserve their effective settings when
available. The human study used DeepSeek-Flash-V4.1-Preview and Tavily.

## Scoring protocols

General-quality metrics are **Clarity**, **Engagement**, **Depth Appropriateness**,
and **Breadth**. Exploratory metrics are **Inquiry Relevance**, **Perspective
Expansion**, **Evidence-Based Progress**, **Incremental Value**, and **Adaptive
Coherence**.

For the benchmark, each request jointly presents one task's three anonymous
system records, using a task-seeded shuffle. The accepted response supplies
0–10 ratings at one-decimal resolution. The two rubrics are evaluated separately.
The raw response, anonymous mapping, protocol label and accepted scores are kept
together, so the system assignment can be checked.

For an ablation, the same task's Full CCA benchmark scores are supplied as fixed
references. Full material is presented first and the target variant second.
The judge returns a metric-specific difference `delta = target - reference`.
The files retain `reference_scores`, `deltas`, and `calibrated_target_scores`.
The last field equals reference plus delta and is **not** an independently
obtained absolute rating. Reference scores are not a constant 8.0. The five
variants reuse each task's reference vector, and those repetitions are not
additional independent Full observations.

Archived request messages preserve the protocol actually sent, including its
wording. They are the authoritative record of each evaluation's inputs.
Generation files also contain internal state updates and candidate utility
scores. Their presence in a generation archive does not mean they were all
included as user-visible evidence in scoring; consult the corresponding
archived evaluation input.

## Human data and privacy

`human/paired_sessions.json` contains the 40 retained paired sessions with
assigned topics and ratings. `human/ratings.csv` provides their long form.
`collected_score` uses the original values 2, 4, 6, 8 and 10;
`reported_score` divides these values by two for the manuscript's 1–5 scale.
The auxiliary Overall rating is outside the nine-metric analysis.

`session_id` identifies a paired session. The human-results files contain only
anonymous session IDs, predefined topics, comparison systems and ratings.
Names, submitted participant identifiers, P codes, cross-session participant
linkage, original filenames, timestamps and free-text dialogue are not released.
No identifier lookup table is published. Session counts define the paired
analyses. See [the human-results README](human/README.md) for the field definitions.
The 2026-10-26 update preserves the same 40 sessions and all rating values.

## Reproduce the statistics

Python 3.11 or later is required. The original numerical audit used NumPy 2.1.2
and SciPy 1.16.2; dependency versions are pinned for reproducibility.

From the repository root:

```bash
python -m pip install -r research_artifacts/2026-10-25/analysis/requirements.txt
python research_artifacts/2026-10-25/analysis/reproduce.py --output reproduced_statistics
python research_artifacts/2026-10-25/analysis/reproduce_human.py --output reproduced_human
```

`analysis/expected/` contains the saved statistics used for the manuscript.
The benchmark pairs observations by task, applies Friedman tests and two-sided
Wilcoxon signed-rank tests, and uses Holm correction across nine omnibus tests
and 18 planned CCA–baseline contrasts. The supplementary all-pair family contains
27 contrasts. Ablation correction covers 25 exploratory or 20 general-quality
contrasts. The automatic analyses use Pratt handling of zero differences and
20,000 task-bootstrap samples, seed 20261023. Human-rating tests enumerate sign
assignments of nonzero paired differences and correct the 18 contrasts jointly.
The reproduction scripts write to separate output directories.

Read a compressed archive with the Python standard library:

```python
import json
import lzma

with lzma.open(
    "research_artifacts/2026-10-25/archives/generation_CCA.jsonl.xz",
    "rt", encoding="utf-8"
) as archive:
    for line in archive:
        record = json.loads(line)
        print(record["task"], record["artifacts"].keys())
```

Generation usage sums successful generation workflows, including initialization,
agent calls, state maintenance and final synthesis. It excludes judge calls,
superseded attempts and search-service charges, and therefore is not a complete
monetary billing total. The three published generation totals are CCA 7,934,915,
Co-STORM 6,693,479, and RAG 1,489,089 input-plus-output tokens.

## Provenance and release processing

Accepted evaluation indexes point to the released response file and its SHA-256.
Provenance files additionally retain hashes of the original response, request
and generation files before release processing. Credentials, identifying human
fields and local workstation paths are excluded or removed; numerical ratings,
reference vectors and deltas are preserved. Individual model-call logs, worker
logs, unsuccessful attempts, model weights, manuscript source files, test
programs and paid experiment drivers are outside this release.

`SHA256SUMS.txt` checks every released file except itself. `manifest.json` records
the archive counts and sizes. The original local experiment files were not
edited when preparing this release.

## Attribution and terms

WildSeek task inputs originate from the
[Co-STORM project](https://github.com/stanford-oval/storm/tree/main/WildSeek).
Its MIT notice is retained in `tasks/STORM_LICENSE.txt`. Cite Co-STORM's paper,
*Into the Unknown Unknowns: Engaged Human Learning through Participation in
Language Model Agent Conversations*, when using these tasks.

Author-produced code and curated records follow the repository's
[MIT License](../../LICENSE). Third-party source passages and referenced works
retain their original rights; inclusion as retrieved experimental evidence does
not relicense the underlying webpages. A data license does not imply that
generated claims or judge assessments have been independently fact-checked.
