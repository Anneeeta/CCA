# Cooperative Cognitive Agents (CCA)

CCA supports exploratory information seeking through successive questions,
evidence analysis, explanations, and persistent state. Four roles coordinate
the investigation: Student, Expert, Instructor, and Arbiter.

## Core workflow

1. The Instructor creates an initial topic overview; the Arbiter seeds shared
   knowledge from this bootstrap exchange.
2. The Student generates five candidate questions with `question`, `type`, and
   `rationale` fields. Types distinguish articulated gaps, conflicting accounts,
   and potential blind spots.
3. Automatic selection combines learner-state relevance, guidance relevance,
   and novelty with weights 0.3, 0.3, and 0.4. It selects up to two questions and
   applies a cosine-similarity diversity check. Users can select questions manually.
4. The Instructor routes the questions, the Expert retrieves and analyzes
   evidence, and the Instructor formulates an explanation.
5. The Student updates learner state (`Ms`); the Arbiter then maintains shared
   knowledge (`K`) and can propose prompt or sampling changes for later calls.
6. Round summaries provide bounded context for subsequent calls. The Instructor
   uses the retained context and both terminal pools for final synthesis.

During exploration, the Student directly accesses `Ms`; the Expert and Instructor
receive `K`. The Arbiter can read both pools for knowledge maintenance. The
Instructor receives both terminal pools only for final synthesis.

### State and context

- Pool updates are incremental. Existing IDs revise entries; omitted entries
  remain stored, and explicit removal IDs govern deletion.
- Exact-content deduplication normalizes Unicode and whitespace while preserving
  case, punctuation, numerical conditions, and negation. It does not merge
  semantically similar paraphrases automatically.
- Context limits apply to views sent to agents, rather than deleting stored
  dialogue or pool entries. Pool views use deterministic lexical relevance.
- `summary_mode: arbiter` uses an Arbiter-proposed summary when valid, with a local
  extractive fallback. `summary_mode: extractive` uses local excerpts for every
  round. The supplied Flash configuration selects the extractive policy.
- Model calls track provider-reported input/output token usage. Retrieval records,
  pool-change sidecars, and complete exchanges remain in local runtime files.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Anneeeta/CCA.git
cd CCA
python -m venv .venv
# Activate the virtual environment using your operating system's command.
python -m pip install -e .
```

Copy `.env.example` to `.env` and supply the credentials for the provider you use:

```dotenv
DEEPSEEK_API_KEY=your_official_deepseek_key
```

The public defaults use DeepSeek's official API and the `deepseek-flash` model
identifier. The local encoder is `sentence-transformers/all-MiniLM-L6-v2`.
Its first use may download the model; for an offline installation, configure
`orchestrator.embedding_model` to a downloaded model directory. If encoder loading
fails, the runtime logs a hashing fallback; that fallback changes question scoring.

## Run CCA

Use the supplied generation and retrieval configuration:

```bash
cca --config configs/deepseek_flash_official.yaml --topic "Financial-analysis chatbots"
```

The equivalent entry points are:

```bash
python -m cca_cli --config configs/deepseek_flash_official.yaml --topic "Your topic"
python main.py --config configs/deepseek_flash_official.yaml --topic "Your topic"
```

Choose candidate questions manually:

```bash
cca --config configs/deepseek_flash_official.yaml --topic "Your topic" --manual-select
```

Reset the configured pools and round log before starting a fresh investigation:

```bash
cca --config configs/deepseek_flash_official.yaml --topic "Your topic" --reset
```

`--reset-logs` additionally clears the configured `cca.log` when used with `--reset`.
Without a YAML configuration, `cca --topic "Your topic"` runs with default settings
and retrieval disabled. Run `cca --help` to inspect the interactive CLI options.

## Retrieval and provider configuration

Available retrieval backends are `deepseek`, `tavily`, and `duckduckgo`:

```bash
cca --topic "Your topic" --retriever deepseek
cca --topic "Your topic" --retriever tavily
cca --topic "Your topic" --retrieval --retriever duckduckgo
```

- DeepSeek Search uses `DEEPSEEK_SEARCH_API_KEY`, falling back to the official
  `DEEPSEEK_API_KEY`. A custom answer-model key is not reused for this service.
- Tavily uses its official endpoint by default. Set `TAVILY_API_KEY`, or provide
  the JSON-list environment variable `TAVILY_API_KEYS` for account-key rotation.
- DuckDuckGo uses the installed `ddgs` package without another API key.
- MiMo, SiliconFlow, and OpenRouter model routes use separate `MIMO_API_KEY`,
  `SILICONFLOW_API_KEY`, and `OPENROUTER_API_KEY` credentials. Select their model
  identifiers explicitly in YAML; changing a custom endpoint does not redirect
  known provider-pinned models.
- For other OpenAI-compatible models, set `CCA_BASE_URL` and `CCA_API_KEY`, and
  explicitly select the model in YAML. Legacy endpoint aliases remain supported.

`configs/deepseek_search.yaml` enables DeepSeek retrieval while retaining the
chosen answer-model settings. `configs/deepseek_flash_official.yaml` explicitly
sets the Flash model, role temperatures, retrieval settings, local summary policy,
and separate runtime paths. YAML fields override their corresponding environment
settings. Credentials belong in the local environment, not in committed YAML.

## Local files

Runtime files are created under the configured `data/` and `logs/` directories.
The Flash configuration uses `data/deepseek_flash_official/` and
`logs/deepseek_flash_official/`. Retained files can include dialogue, retrieval
results, state pools, bootstrap material, and pool-change records.

Use separate configured paths for independent investigations. `--reset` clears
the selected paths; it is not a selector for a different run.

## Repository scope

This update publishes the core interactive program, runtime configurations, and
documentation. Experiment runners, batch drivers, evaluators, test programs,
datasets, raw results, model weights, manuscript sources, and populated `.env`
files are outside this repository's release scope.

## License

CCA is released under the MIT License. See [LICENSE](LICENSE).
