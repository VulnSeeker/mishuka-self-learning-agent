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
