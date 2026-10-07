# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Render / Fly.io deployment
- Approval queue for knowledge updates
- Graph memory for entity relations
- Multi-agent collaboration
- Expanded test coverage (currently 24%)

## [0.1.0] - 2026-10-06

### Added
- Initial release of Onyx self-learning AI agent.
- Web research to knowledge base pipeline with per-skill ChromaDB collections.
- Task orchestrator: analyze, match skills, auto-learn, execute, compose.
- Skill-scoped runtime with sandboxed Python execution.
- Small-model builder for ML training tasks.
- FastAPI REST service with async job queue.
- CLI with subcommands: `task`, `learn`, `run`, `skills`, `show`, `delete`, `train`, `serve`, `web`, `info`, `version`.
- Streamlit dashboard.
- Docker support with multi-stage build and healthchecks.
- GitHub Actions workflows for CI and Docker build plus smoke test.
- Pytest suite with 40 tests covering schemas, storage, sandbox, and parser.
- Support for multiple LLM providers: OpenAI, Groq, Together, Ollama, OpenRouter.
- MIT License.
