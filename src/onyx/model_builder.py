"""
Small-model builder for Onyx.

Given a task, this module:

  1. Asks the LLM to plan a small, self-contained ML experiment.
  2. Generates runnable training code (scikit-learn / transformers / torch).
  3. Runs it in the sandbox.
  4. Parses the final metric line.

Not a replacement for real MLOps. It's for quick baselines and small
proof-of-concept models on public datasets.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from onyx.config import CONFIG, Config
from onyx.llm import LLMClient, LLMResponseError
from onyx.sandbox import Sandbox
from onyx.schemas import TrainingPlan

log = logging.getLogger("onyx.model_builder")

ProgressCB = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You plan a small, self-contained machine learning experiment.

Return JSON:
{
  "dataset_name": "human-readable name",
  "dataset_url": "https://... (HuggingFace or stable URL)",
  "approach": "one-line description of the approach",
  "library": "sklearn" | "transformers" | "pytorch",
  "train_code": "full python source as a single string"
}

Hard rules for train_code:
  - Must be fully self-contained Python (no local file imports).
  - May use: sklearn, datasets (HuggingFace), transformers, torch, pandas, numpy.
  - Must download or load the dataset from the URL or a public loader.
  - Must train a small model.
  - Must run in under 5 minutes on CPU.
  - Must print a final line exactly of the form:
        METRIC <name>=<value>
    for example:  METRIC accuracy=0.87
  - Keep it under 80 lines.
  - No plots, no interactive input, no network calls beyond dataset loading.
  - No markdown fences in train_code; just plain source.
"""


# ---------------------------------------------------------------------------
# ModelBuilder
# ---------------------------------------------------------------------------

class ModelBuilder:
    """Plans and runs small ML experiments in the sandbox."""

    def __init__(
        self,
        llm: LLMClient,
        sandbox: Sandbox,
        cfg: Config = CONFIG,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, task: str) -> TrainingPlan:
        """Ask the LLM for a training plan."""
        task = (task or "").strip()
        if not task:
            raise ValueError("task is required")

        try:
            data = self.llm.chat_json(_PLAN_SYSTEM, f"Task: {task}")
        except LLMResponseError as e:
            raise ModelBuilderError(f"planning failed: {e}") from e

        if not isinstance(data, dict):
            raise ModelBuilderError("planner returned non-object JSON")

        try:
            plan = TrainingPlan(**data)
        except Exception as e:  # noqa: BLE001
            raise ModelBuilderError(f"plan validation failed: {e}") from e

        return plan

    def train(
        self,
        plan: TrainingPlan,
        progress: ProgressCB = None,
        timeout: int = 600,
    ) -> dict[str, Any]:
        """Run the training code in the sandbox and parse the metric."""
        cb = progress or (lambda _msg: None)

        code = (plan.train_code or "").strip()
        if not code:
            return {
                "ok": False,
                "metric": None,
                "stdout": "",
                "stderr": "empty training code",
                "plan": plan.model_dump(),
            }

        cb("Running training in sandbox...")
        output = self.sandbox.run_python(code, timeout=timeout)

        metric = _parse_metric(output)

        return {
            "ok": metric is not None,
            "metric": metric,
            "stdout": output,
            "stderr": "",
            "plan": plan.model_dump(),
        }

    def build(
        self,
        task: str,
        progress: ProgressCB = None,
    ) -> dict[str, Any]:
        """Convenience: plan + train in one call."""
        cb = progress or (lambda _msg: None)
        cb("Planning training...")
        plan = self.plan(task)
        cb(f"Plan: {plan.library} on {plan.dataset_name}")
        result = self.train(plan, progress=cb)
        result["task"] = task
        return result


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ModelBuilderError(RuntimeError):
    """Raised when planning or training fails at the orchestration level."""


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

_METRIC_RE = re.compile(r"^\s*METRIC\s+([A-Za-z_][\w\-]*)\s*=\s*([\d.eE+\-]+)\s*$")


def _parse_metric(output: str) -> str | None:
    """Find the last METRIC line in the sandbox output."""
    if not output:
        return None
    last: str | None = None
    for line in output.splitlines():
        m = _METRIC_RE.match(line)
        if m:
            last = line.strip()
    return last


__all__ = ["ModelBuilder", "ModelBuilderError"]
