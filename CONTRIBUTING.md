
```markdown
# Contributing to Onyx

Thanks for considering contributing to Onyx. This document outlines the
process for submitting issues and pull requests.

## Code of Conduct

Be respectful. Constructive criticism is welcome; personal attacks are not.

## Ways to Contribute

- **Report bugs** via GitHub Issues
- **Suggest features** via GitHub Issues
- **Improve documentation** (README, docstrings, examples)
- **Add tests** to increase coverage
- **Fix bugs** and submit pull requests
- **Add new skills** to the `demo/` folder
- **Report security issues** (see SECURITY.md if present)

## Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/onyx-self-learning-agent.git
cd onyx-self-learning-agent

# Install with dev dependencies
pip install -e ".[api,dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API key

# Verify installation
onyx info
```

## Running Checks Locally

Before submitting a pull request, run:

```bash
# Lint
ruff check src tests

# Format
ruff format src tests

# Type check
mypy src

# Tests
pytest
```

All four must pass before a PR can be merged.

## Branch Naming

- `feat/short-description` — new features
- `fix/short-description` — bug fixes
- `docs/short-description` — documentation
- `refactor/short-description` — refactors
- `test/short-description` — test improvements
- `chore/short-description` — tooling, dependencies

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add graph memory backend
fix: correct dedupe threshold for short entries
docs: update quick start examples
refactor: split storage layer into separate modules
test: add coverage for orchestrator
chore: bump ruff to 0.6.0
```

Keep the subject line under 72 characters. Use the body for details if needed.

## Pull Request Process

1. Fork the repository
2. Create a branch from `main` with a clear name
3. Make your changes with focused commits
4. Run all checks locally (`ruff`, `mypy`, `pytest`)
5. Push your branch to your fork
6. Open a pull request against `main`
7. Fill in the PR template
8. Address review feedback promptly

## What Gets Merged

- Bug fixes with tests
- New features with tests and documentation
- Performance improvements with benchmarks
- Documentation improvements
- Dependency updates that pass CI

## What Doesn't Get Merged

- Changes that break existing tests
- Large refactors without prior discussion
- Features that duplicate existing functionality
- Code without tests for new behavior
- PRs that fail CI

## Questions

Open an issue with the `question` label, or start a discussion.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License (see [LICENSE](LICENSE)).
```
