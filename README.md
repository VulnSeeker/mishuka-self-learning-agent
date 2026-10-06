# Onyx Self-Learning Agent

Onyx is a Python AI agent that learns a skill from web research, stores structured knowledge in a local vector database, and uses that knowledge to execute future tasks with grounded answers.

## What it does

- Bootstraps skills from the web (`onyx learn <skill>`)
- Builds a structured per-skill knowledge base (ChromaDB + SQLite metadata)
- Runs tasks against learned skills (`onyx run <skill_id> "<task>"`)
- Optionally generates and executes Python code in a restricted sandbox
- Detects runtime knowledge gaps and can absorb new knowledge
- Includes a Streamlit dashboard for interactive use
- Includes a small-model training workflow (`onyx train "<task>"`)

## Architecture

Onyx is organized as composable modules behind a single facade (`Onyx`):

- **`onyx.agent.Onyx`**: top-level orchestrator for CLI, web, and library users
- **`onyx.bootstrapper.SkillBootstrapper`**: skill-spec generation, source discovery, extraction, dedupe, persistence
- **`onyx.runtime.RuntimeAgent`**: retrieval, planning, optional code execution, answer synthesis, gap detection
- **`onyx.model_builder.ModelBuilder`**: LLM-planned, sandbox-executed ML baseline experiments
- **`onyx.storage`**:
  - `Registry` (SQLite): skills and knowledge metadata
  - `VectorStore` (ChromaDB): embeddings and semantic retrieval
- **`onyx.llm.LLMClient`**: OpenAI-compatible chat/JSON/embedding wrapper with retries
- **`onyx.search.SearchClient`**: Tavily (if configured) with DuckDuckGo fallback
- **`onyx.crawler.Crawler`**: robots-aware HTML-to-text extractor with code-block preservation
- **`onyx.sandbox.Sandbox`**: isolated subprocess execution for generated Python snippets

## Repository structure

```text
src/onyx/
  agent.py         # Onyx facade
  bootstrapper.py  # Skill learning pipeline
  runtime.py       # Task execution pipeline
  model_builder.py # ML experiment planner/runner
  storage.py       # SQLite registry + ChromaDB wrapper
  llm.py           # OpenAI-compatible client helpers
  search.py        # Tavily/DDG search backend
  crawler.py       # HTML crawler/cleaner
  sandbox.py       # Constrained Python execution
  cli.py           # CLI commands (entrypoint: onyx)
  web.py           # Streamlit dashboard
  config.py        # Environment-driven runtime config
  schemas.py       # Pydantic contracts
demo/              # Example JSON outputs from real runs
```

## Installation

### Requirements

- Python 3.10+
- pip

### Install from source

```bash
git clone https://github.com/VulnSeeker/onyx-self-learning-agent.git
cd onyx-self-learning-agent
pip install -e .
```

### Optional extras

```bash
# dashboard + dev + docs + optional integrations
pip install -e ".[all]"
```

## Configuration

Copy and edit environment settings:

```bash
cp .env.example .env
```

Minimum required setup is an OpenAI-compatible endpoint and key (or local Ollama-style config as shown in `.env.example`).

Important variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `LLM_MODEL`
- `EMBED_MODEL`
- `TAVILY_API_KEY` (optional; otherwise DuckDuckGo is used)
- `DATA_DIR`, `SKILLS_DIR`
- `MAX_SOURCES_PER_SKILL`, `REQUEST_TIMEOUT`, `CODE_TIMEOUT`

## CLI usage

```bash
onyx learn "Python asyncio"
onyx skills
onyx run python_asyncio "Write a basic asyncio example that runs 3 tasks concurrently"
onyx show python_asyncio --limit 20
onyx train "Train a text classifier to detect toxic comments"
onyx info
onyx version
```

### CLI commands

- `learn <skill>`: research and build/update a skill knowledge base
- `run <skill_id> <task>`: answer a task using learned knowledge
- `skills`: list learned skills
- `show <skill_id>`: inspect skill metadata and recent entries
- `delete <skill_id>`: remove a skill and its vector collection
- `train <task>`: plan/train a small model in sandbox
- `web`: launch Streamlit UI
- `info`: print effective runtime configuration
- `version`: print package version

## Web dashboard

Launch:

```bash
onyx web
```

The dashboard includes pages for:

- Learn
- Run Task
- Skills
- Train
- Settings

## Data and persistence

By default, Onyx stores runtime state under `./data`:

- `data/registry.sqlite3` — skill registry + knowledge metadata
- `data/chroma/` — persistent vector collections
- `data/logs/` — runtime logs

## Development

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Common checks:

```bash
ruff check src
ruff format --check src
mypy src
pytest
```

## Notes and limitations

- Generated code executes in a constrained subprocess sandbox, but it is **not** a full security boundary.
- Web knowledge quality depends on source quality and search results.
- For sensitive or regulated workflows, add domain-specific review and guardrails.

## License

MIT — see [LICENSE](LICENSE).
