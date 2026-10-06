
"""
Onyx — the top-level facade.

Wires together:
  - LLMClient
  - SearchClient
  - Crawler
  - Sandbox
  - Registry + VectorStore
  - SkillBootstrapper
  - RuntimeAgent
  - ModelBuilder
  - TaskOrchestrator

External callers (CLI, web dashboard, library users) should only import
`Onyx` and its exceptions from this module.

Example (skill-scoped):
    >>> from onyx import Onyx
    >>> ox = Onyx()
    >>> ox.learn("OSINT")
    >>> result = ox.run("osint", "Enumerate emails for example.com")
    >>> print(result.answer)

Example (task-driven, auto-orchestration):
    >>> from onyx import Onyx
    >>> ox = Onyx()
    >>> result = ox.task("Find exposed emails for example.com and write a validator")
    >>> print(result.answer)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from onyx.bootstrapper import SkillBootstrapper
from onyx.config import CONFIG, Config
from onyx.crawler import Crawler
from onyx.llm import LLMClient
from onyx.model_builder import ModelBuilder, ModelBuilderError
from onyx.orchestrator import TaskOrchestrator, TaskResult
from onyx.runtime import RuntimeAgent, SkillNotFoundError
from onyx.sandbox import Sandbox
from onyx.schemas import RuntimeResult, SkillRecord
from onyx.search import SearchClient
from onyx.storage import Registry, VectorStore

log = logging.getLogger("onyx.agent")

ProgressCB = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Exceptions (re-exported from onyx/__init__.py)
# ---------------------------------------------------------------------------

class AgentError(RuntimeError):
    """Base exception for Onyx-level failures."""


# SkillNotFoundError is defined in runtime.py and re-exported here so
# `from onyx.agent import SkillNotFoundError` also works.


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class Onyx:
    """
    Self-learning skill agent.

    Two ways to use it:

    1. Skill-scoped (explicit):
        ox = Onyx()
        ox.learn("OSINT")
        result = ox.run("osint", "Find emails for example.com")

    2. Task-driven (automatic):
        ox = Onyx()
        result = ox.task("Find emails for example.com and write a validator")
        # Onyx analyzes the task, matches existing skills, auto-learns
        # missing ones, combines knowledge, and returns a final answer.
    """

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg

        # Infrastructure
        self.llm = LLMClient(cfg)
        self.search = SearchClient(cfg)
        self.crawler = Crawler(cfg)
        self.sandbox = Sandbox(cfg)

        # Storage
        self.registry = Registry(cfg)
        self.vectors = VectorStore(cfg)

        # Domain logic
        self.bootstrapper = SkillBootstrapper(
            self.llm, self.registry, self.vectors,
            self.search, self.crawler, cfg,
        )
        self.runtime = RuntimeAgent(
            self.llm, self.registry, self.vectors, self.bootstrapper,
            self.search, self.crawler, self.sandbox, cfg,
        )
        self.model_builder = ModelBuilder(self.llm, self.sandbox, cfg)

        # Top-level orchestrator (task → skills → execution → answer)
        self.orchestrator = TaskOrchestrator(
            llm=self.llm,
            registry=self.registry,
            vectors=self.vectors,
            bootstrapper=self.bootstrapper,
            runtime=self.runtime,
            search=self.search,
            crawler=self.crawler,
            sandbox=self.sandbox,
            model_builder=self.model_builder,
            cfg=cfg,
        )

        log.debug("Onyx initialised: %s", cfg.summary())

    # ------------------------------------------------------------------
    # Task-driven entry point (auto-orchestration)
    # ------------------------------------------------------------------

    def task(
        self,
        task: str,
        *,
        auto_learn: bool = True,
        progress: ProgressCB = None,
    ) -> TaskResult:
        """
        Execute a natural-language task end-to-end.

        Pipeline:
          1. Analyze task → required skills
          2. Match against existing registry
          3. Auto-learn any missing skills (if auto_learn=True)
          4. Retrieve knowledge from all matched skills
          5. Execute via the primary skill (with sandboxed code if needed)
          6. Optionally train a small model if the task demands it
          7. Compose the final answer

        Args:
            task: Natural-language description of what to accomplish.
            auto_learn: If True, bootstrap missing skills automatically.
            progress: Optional callback for streaming progress messages.

        Returns:
            TaskResult with answer, skill matches, code output, plan, etc.
        """
        task = (task or "").strip()
        if not task:
            raise AgentError("task is required")

        try:
            return self.orchestrator.run(
                task,
                auto_learn=auto_learn,
                progress=progress,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("task failed: %s", task[:120])
            raise AgentError(f"task failed: {e}") from e

    # ------------------------------------------------------------------
    # Skills (explicit control)
    # ------------------------------------------------------------------

    def learn(self, skill_name: str, progress: ProgressCB = None) -> dict[str, Any]:
        """Bootstrap a new skill (or append to an existing one)."""
        skill_name = (skill_name or "").strip()
        if not skill_name:
            raise AgentError("skill_name is required")
        try:
            return self.bootstrapper.learn(skill_name, progress=progress)
        except Exception as e:  # noqa: BLE001
            log.exception("learn failed for %s", skill_name)
            raise AgentError(f"learn failed: {e}") from e

    def run(
        self,
        skill_id: str,
        task: str,
        progress: ProgressCB = None,
    ) -> RuntimeResult:
        """Execute a task using a specific learned skill."""
        skill_id = (skill_id or "").strip()
        task = (task or "").strip()
        if not skill_id:
            raise AgentError("skill_id is required")
        if not task:
            raise AgentError("task is required")

        # SkillNotFoundError intentionally propagates — caller should
        # learn the skill first.
        return self.runtime.run(skill_id, task, progress=progress)

    def skills(self) -> list[SkillRecord]:
        """List all learned skills, newest first."""
        return self.registry.list()

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        """Return a skill record or None."""
        return self.registry.get(skill_id)

    def has_skill(self, skill_id: str) -> bool:
        return self.registry.has(skill_id)

    def delete(self, skill_id: str) -> bool:
        """Delete a skill: vectors + registry + metadata."""
        skill_id = (skill_id or "").strip()
        if not skill_id:
            return False
        self.vectors.drop(skill_id)
        return self.registry.delete(skill_id)

    def skill_entries(self, skill_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """List knowledge metadata for a skill."""
        return self.registry.list_knowledge_meta(skill_id, limit=limit)

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def train_model(
        self,
        task: str,
        progress: ProgressCB = None,
    ) -> dict[str, Any]:
        """Plan and train a small ML model. Returns plan + result."""
        try:
            return self.model_builder.build(task, progress=progress)
        except ModelBuilderError as e:
            raise AgentError(f"model build failed: {e}") from e

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release HTTP sessions. Idempotent."""
        try:
            self.crawler.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.search._session.close()  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "Onyx":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["Onyx", "AgentError", "SkillNotFoundError"]
