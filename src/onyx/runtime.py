"""
Runtime agent for Onyx.

Given a task and a skill id, this module:

  1. Retrieves relevant knowledge from the skill DB.
  2. Asks the LLM to plan (steps, needs_code, needs_web).
  3. Optionally generates + runs Python in the sandbox.
  4. Synthesizes an answer grounded in retrieved knowledge.
  5. Detects knowledge gaps and updates the skill DB if new info emerged.

Returns a RuntimeResult for callers (CLI, web, library).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from onyx.bootstrapper import SkillBootstrapper
from onyx.config import CONFIG, Config
from onyx.crawler import Crawler
from onyx.llm import LLMClient, LLMResponseError
from onyx.sandbox import Sandbox
from onyx.schemas import KnowledgeEntry, RuntimePlan, RuntimeResult, SkillSpec
from onyx.search import SearchClient
from onyx.storage import Registry, VectorStore

log = logging.getLogger("onyx.runtime")

ProgressCB = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You are a task planner for a skill agent.

Given a task and retrieved skill knowledge, produce JSON:
{
  "steps": ["short step 1", "short step 2", ...],
  "needs_code": true|false,
  "needs_web": true|false,
  "notes": "anything relevant"
}

Rules:
  - Use ONLY the retrieved knowledge as ground truth for the plan.
  - needs_code=true only if running Python would produce a materially 
    better or verified answer.
  - needs_web=true only if retrieved knowledge is clearly insufficient.
  - Keep steps to 3 to 8 short items.
"""

_CODE_SYSTEM = """\
Write a self-contained Python program that accomplishes the task.

Rules:
  - Output ONLY Python source. No markdown fences. No commentary.
  - Use only the standard library unless the task explicitly requires more.
  - Print any results to stdout.
  - Never prompt for input.
  - Keep it under 60 lines.
"""

_ANSWER_SYSTEM = """\
You are the runtime of a skill agent.

Answer the user's task using ONLY:
  - the provided skill knowledge
  - the plan
  - the code output (if any)

Rules:
  - Be concrete. Show commands, snippets, or steps when relevant.
  - Cite the tool or procedure names you used, drawn from the context.
  - If knowledge is missing, say so explicitly instead of guessing.
"""

_GAP_SYSTEM = """\
Decide whether the agent discovered NEW knowledge during this task that is
NOT covered by the retrieved context and would be valuable to store.

Return JSON:
{
  "gap": true|false,
  "reason": "short explanation",
  "summary": "one-line summary of the new knowledge (empty if gap=false)"
}
"""


# ---------------------------------------------------------------------------
# RuntimeAgent
# ---------------------------------------------------------------------------

class RuntimeAgent:
    """Executes tasks against a learned skill."""

    def __init__(
        self,
        llm: LLMClient,
        registry: Registry,
        vectors: VectorStore,
        bootstrapper: SkillBootstrapper,
        search: SearchClient,
        crawler: Crawler,
        sandbox: Sandbox,
        cfg: Config = CONFIG,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.vectors = vectors
        self.bootstrapper = bootstrapper
        self.search = search
        self.crawler = crawler
        self.sandbox = sandbox
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        skill_id: str,
        task: str,
        progress: ProgressCB = None,
    ) -> RuntimeResult:
        cb = progress or (lambda _msg: None)

        skill_id = (skill_id or "").strip()
        task = (task or "").strip()
        if not skill_id:
            raise ValueError("skill_id is required")
        if not task:
            raise ValueError("task is required")

        if not self.registry.has(skill_id):
            raise SkillNotFoundError(
                f"Skill '{skill_id}' not found. Learn it first with: onyx learn {skill_id}"
            )

        # 1. Retrieve
        cb("Retrieving knowledge...")
        context = self._retrieve(skill_id, task, k=8)
        ctx_text = self._format_context(context)

        # 2. Plan
        cb("Planning...")
        plan = self._plan(task, ctx_text)

        # 3. Optionally execute code
        code_output = ""
        if plan.needs_code:
            cb("Generating code...")
            code = self._generate_code(task, ctx_text, plan)
            if code:
                cb("Running code in sandbox...")
                code_output = self.sandbox.run_python(code)
                cb(f"Sandbox output: {_truncate(code_output, 160)}")

        # 4. Optional web fallback when knowledge is thin
        web_supplement = ""
        if plan.needs_web and len(context) < 3:
            cb("Searching web (sparse local knowledge)...")
            for src in self.search.search(task, max_results=3):
                text = self.crawler.fetch(src.url)
                if text:
                    web_supplement += f"\n\n[source: {src.url}]\n{text[:3000]}"

        # 5. Compose the final answer
        cb("Composing answer...")
        answer = self._compose_answer(task, ctx_text, plan, code_output, web_supplement)

        # 6. Detect gap and update DB
        cb("Checking for knowledge gaps...")
        gap = self._detect_gap(task, answer, ctx_text)

        new_entries = 0
        if gap.get("gap") and web_supplement:
            cb("Gap detected — extracting new knowledge...")
            try:
                new_entries = self._absorb_gap(skill_id, web_supplement)
                if new_entries:
                    self.registry.set_entry_count(skill_id, self.vectors.count(skill_id))
            except Exception as e:  # noqa: BLE001
                log.warning("gap update failed: %s", e)

        return RuntimeResult(
            answer=answer,
            plan=plan.model_dump(),
            code_output=code_output,
            gap=gap,
            new_entries=new_entries,
            context_used=len(context),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, skill_id: str, task: str, k: int) -> list[dict[str, Any]]:
        try:
            emb = self.llm.embed_one(task)
        except Exception as e:  # noqa: BLE001
            log.error("retrieval embedding failed: %s", e)
            return []

        res = self.vectors.query(skill_id, emb, k=k)
        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]

        out: list[dict[str, Any]] = []
        if not ids[0]:
            return out

        for i, doc_id in enumerate(ids[0]):
            out.append({
                "id": doc_id,
                "document": docs[0][i] if i < len(docs[0]) else "",
                "metadata": metas[0][i] if i < len(metas[0]) else {},
                "distance": dists[0][i] if dists and i < len(dists[0]) else None,
            })
        return out

    @staticmethod
    def _format_context(context: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for c in context:
            m = c.get("metadata") or {}
            parts.append(
                f"[{m.get('type', '?')}] {m.get('title', '')}\n"
                f"source: {m.get('source_url', '')}\n"
                f"{c.get('document', '')}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self, task: str, ctx: str) -> RuntimePlan:
        try:
            data = self.llm.chat_json(
                _PLAN_SYSTEM,
                f"Task: {task}\n\nRetrieved knowledge:\n{ctx[:8000] or '(none)'}",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("planning failed: %s", e)
            return RuntimePlan(steps=[], needs_code=False, needs_web=True, notes="fallback")

        if not isinstance(data, dict):
            return RuntimePlan(steps=[], needs_code=False, needs_web=True, notes="fallback")

        try:
            return RuntimePlan(**data)
        except Exception as e:  # noqa: BLE001
            log.warning("plan validation failed: %s", e)
            return RuntimePlan(steps=[], needs_code=False, needs_web=True, notes="fallback")

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _generate_code(self, task: str, ctx: str, plan: RuntimePlan) -> str:
        try:
            code = self.llm.chat(
                _CODE_SYSTEM,
                f"Task: {task}\n\nPlan: {plan.model_dump()}\n\nKnowledge:\n{ctx[:4000]}",
                temperature=0.1,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("code generation failed: %s", e)
            return ""

        return _strip_code_fences(code)

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def _compose_answer(
        self,
        task: str,
        ctx: str,
        plan: RuntimePlan,
        code_output: str,
        web_supplement: str,
    ) -> str:
        try:
            return self.llm.chat(
                _ANSWER_SYSTEM,
                f"Task: {task}\n\n"
                f"Plan: {plan.model_dump()}\n\n"
                f"Skill knowledge:\n{ctx[:6000] or '(none)'}\n\n"
                f"Code output:\n{code_output[:2000] or '(none)'}\n\n"
                f"Web supplement:\n{web_supplement[:3000] or '(none)'}",
                temperature=0.2,
            )
        except Exception as e:  # noqa: BLE001
            log.error("answer composition failed: %s", e)
            return f"[error composing answer: {e}]"

    # ------------------------------------------------------------------
    # Gap detection + absorption
    # ------------------------------------------------------------------

    def _detect_gap(self, task: str, answer: str, ctx: str) -> dict[str, Any]:
        try:
            data = self.llm.chat_json(
                _GAP_SYSTEM,
                f"Task: {task}\n\nAnswer: {answer[:2000]}\n\n"
                f"Retrieved context:\n{ctx[:2000] or '(none)'}",
            )
        except Exception as e:  # noqa: BLE001
            log.debug("gap detection failed: %s", e)
            return {"gap": False, "reason": "classifier error", "summary": ""}

        if not isinstance(data, dict):
            return {"gap": False, "reason": "invalid classifier output", "summary": ""}

        return {
            "gap": bool(data.get("gap", False)),
            "reason": str(data.get("reason", ""))[:500],
            "summary": str(data.get("summary", ""))[:500],
        }

    def _absorb_gap(self, skill_id: str, web_supplement: str) -> int:
        pseudo_spec = SkillSpec(name=skill_id)
        entries = self.bootstrapper._extract(
            pseudo_spec, web_supplement, "runtime_gap"
        )
        if not entries:
            return 0
        added, updated, _ = self.bootstrapper._dedupe_and_store(skill_id, entries)
        return added + updated


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SkillNotFoundError(ValueError):
    """Raised when a skill has not been learned yet."""


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(code: str) -> str:
    if not code:
        return ""
    m = _FENCE_RE.search(code)
    if m:
        return m.group(1).strip()
    if code.lstrip().startswith("```"):
        lines = code.split("```")
        if len(lines) >= 2:
            body = lines[1]
            if body.lstrip().lower().startswith("python"):
                body = body.lstrip()[6:]
            return body.strip()
    return code.strip()


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


__all__ = ["RuntimeAgent", "SkillNotFoundError"]
