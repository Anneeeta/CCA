# Cooperative Cognitive Agents (CCA)

CCA is a multi-agent system for iterative, learner-guided topic exploration. It
coordinates four role-specialized agents and maintains persistent knowledge
state across rounds.

## Architecture

- **Student** generates candidate questions from learner state, dialogue
  history, and optional user guidance.
- **Instructor** routes each question and turns researched material into a
  structured teaching response.
- **Expert** formulates search queries and develops an evidence-oriented
  viewpoint.
- **Arbiter** updates shared knowledge and proposes bounded prompt or parameter
  adaptations for later rounds.
- **Shared knowledge pool (`K`)** stores normalized knowledge available to the
  Instructor, Expert, and Arbiter.
- **Student memory pool (`Ms`)** stores learner-state entries across four
  categories: known, known-unknown, conflict, and unknown-unknown.

At each round, CCA scores candidate questions for learner-state relevance,
guidance relevance, and novelty. It then selects up to two questions, optionally
retrieves web evidence, produces an Instructor response, updates both pools,
and applies Arbiter adaptations. The Instructor synthesizes the final report.

## Requirements

- Python 3.10 or newer
- An API endpoint compatible with the OpenAI Chat Completions API
- An API key for that endpoint

The default endpoint and model are DeepSeek's official API and
`deepseek-chat`. Both can be overridden through environment variables.

## Installation

```bash
git clone https://github.com/Anneeeta/CCA.git
cd CCA
python -m venv .venv
```

Activate the virtual environment, then install CCA:

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

Create a local `.env` from `.env.example` and set at least `CCA_API_KEY`:

```dotenv
CCA_API_KEY=your_api_key_here
CCA_BASE_URL=https://api.deepseek.com/v1
```

`DEEPSEEK_API_KEY` and `OPENAI_API_KEY` are accepted as API-key aliases.
`DEEPSEEK_BASE_URL` and `OPENAI_BASE_URL` are accepted as endpoint aliases.
Never commit a populated `.env` file.

Optional retrieval modes:

- DuckDuckGo retrieval requires no additional API key.
- Tavily retrieval requires `TAVILY_API_KEY`.

## Usage

Start an interactive run:

```bash
cca --topic "retrieval-augmented generation"
```

The module entry points are equivalent:

```bash
python -m cca_cli --topic "retrieval-augmented generation"
python main.py --topic "retrieval-augmented generation"
```

Enable retrieval:

```bash
cca --topic "retrieval-augmented generation" --retrieval
cca --topic "retrieval-augmented generation" --retrieval --retriever tavily
```

Select one or two candidate questions manually in each round:

```bash
cca --topic "retrieval-augmented generation" --manual-select
```

Reset persistent pools before a run:

```bash
cca --topic "retrieval-augmented generation" --reset
```

Run `cca --help` for all options.

## Runtime Data

CCA creates `data/` and `logs/` locally. These directories may contain prompts,
model responses, retrieved text, and learner-state information. They are ignored
by Git and should be reviewed before being shared.

## Repository Scope

This repository contains the core interactive CCA implementation only. Research
datasets, evaluation code, automated judges, experiment runners, model weights,
and result artifacts are intentionally excluded.

## License

CCA is released under the MIT License. See [LICENSE](LICENSE).
