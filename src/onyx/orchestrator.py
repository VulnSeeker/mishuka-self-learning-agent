"""
Task Orchestrator for Onyx.

Top-level entry point that turns a natural-language task into a completed
result by:

  1. Analyzing the task to determine which skills are required.
  2. Checking the registry for existing skills.
  3. Auto-learning missing skills from the web.
  4. Retrieving relevant knowledge from ALL matched skills.
  5. Executing the task (with sandboxed code when needed).
  6. Optionally training a small ML model if the task demands one.
  7. Composing the final answer grounded in everything above.

Everything is LLM-driven and idempotent: re-running the same task will
reuse existing skill DBs and only learn genuinely missing ones.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from onyx.bootstrapper import SkillBootstrapper
from onyx.config import CONFIG, Config
from onyx.crawler import Crawler
from onyx.llm import LLMClient, LLMResponseError
from onyx.model_builder import ModelBuilder, ModelBuilderError
from onyx.runtime import RuntimeAgent
from onyx.sandbox import Sandbox
from onyx.schemas import RuntimeResult, SkillRecord
from onyx.search import SearchClient
from onyx.storage import Registry, VectorStore

log = logging.getLogger("onyx.orchestrator")

ProgressCB = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ANALYZE_SYSTEM = """\
You are a task analyzer for an autonomous agent.

Given a user task, decide which skills the agent needs to complete it.

Return JSON:
{
  "required_skills": ["short skill name 1", "short skill name 2"],
  "reasoning": "one sentence explaining why these skills",
  "needs_code": true|false,
  "needs_model_training": true|false,
  "primary_skill": "the single most important skill"
}

Rules:
  - required_skills: 1 to 4 short names. Use the most specific names you can.
    Examples: "OSINT", "Python asyncio", "Python decorators", "Docker",
              "Linux command line", "Dockerfile", "Regex", "Git", "Kubernetes"
  - needs_code: true if the task asks for a script, function, or program
  - needs_model_training: true ONLY if the task explicitly asks to train
    a machine-learning model. Otherwise false.
  - primary_skill: the one skill that would be used first.
"""

_MATCH_SYSTEM = """\
You match a required skill name against a list of existing skill records.

Return JSON:
{
  "match_id": "existing_skill_id or null",
  "confidence": 0.0 to 1.0,
  "reasoning": "short explanation"
}

Rules:
  - Match semantically, not literally. "asyncio" matches "python_asyncio".
  - "OSINT" matches "open_source_intelligence_osint".
  - Only return a match if confidence >= 0.7.
  - If no existing skill is a good match, return match_id=null.
"""

_COMPOSE_SYSTEM = """\
You are the final answer composer for an autonomous agent.

You are given:
  - the original task
  - the list of skills that were used
  - knowledge retrieved from each skill
  - plan and code output (if any)
  - a small-model training result (if any)

Produce the final, comprehensive answer for the user.

Rules:
  - Be concrete. Show code, commands, and steps.
  - Cite which skill each piece of information came from.
  - If code was executed, include the actual output.
  - If knowledge was missing, say so explicitly.
  - Keep the answer under 2500 words.
"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TaskAnalysis:
    required_skills: list[str] = field(default_factory=list)
    reasoning: str = ""
    needs_code: bool = False
    needs_model_training: bool = False
    primary_skill: str = ""


@dataclass
class SkillMatch:
    required_name: str
    existing_id: Optional[str]
    existing_name: Optional[str]
    confidence: float
    status: str  # "matched" | "learned" | "failed"


@dataclass
class TaskResult:
    task: str
    analysis: TaskAnalysis
    skill_matches: list[SkillMatch]
    answer: str
    code_output: str
    plan: dict[str, Any]
    context_used: int
    new_entries: int
    model_result: Optional[dict[str, Any]]
    elapsed_seconds: float
    progress_log: list[str]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TaskOrchestrator:
    """Turns a natural-language task into a completed result."""

    def __init__(
        self,
        llm: LLMClient,
        registry: Registry,
        vectors: VectorStore,
        bootstrapper: SkillBootstrapper,
        runtime: RuntimeAgent,
        search: SearchClient,
        crawler: Crawler,
        sandbox: Sandbox,
        model_builder: ModelBuilder,
        cfg: Config = CONFIG,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.vectors = vectors
        self.bootstrapper = bootstrapper
        self.runtime = runtime
        self.search = search
        self.crawler = crawler
        self.sandbox = sandbox
        self.model_builder = model_builder
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        auto_learn: bool = True,
        progress: ProgressCB = None,
        learn_cooldown: int = 15,
    ) -> TaskResult:
        """Execute a task end-to-end."""
        cb = progress or (lambda _m: None)
        task = (task or "").strip()
        if not task:
            raise ValueError("task is required")

        started = time.time()
        progress_log: list[str] = []

        def _log(msg: str) -> None:
            progress_log.append(msg)
            cb(msg)

        # 1. Analyze
        _log("Analyzing task...")
        analysis = self._analyze_task(task)
        _log(f"  Required skills: {', '.join(analysis.required_skills) or '(none)'}")
        if analysis.needs_model_training:
            _log("  Task also requires model training")

        # 2. Match + auto-learn
        matches = self._resolve_skills(
            analysis.required_skills,
            auto_learn=auto_learn,
            cooldown=learn_cooldown,
            log=_log,
        )
        matched_ids = [m.existing_id for m in matches if m.existing_id]

        if not matched_ids:
            _log("❌ No usable skills available — cannot proceed")
            return TaskResult(
                task=task,
                analysis=analysis,
                skill_matches=matches,
                answer="[error: no usable skills after resolution]",
                code_output="",
                plan={},
                context_used=0,
                new_entries=0,
                model_result=None,
                elapsed_seconds=time.time() - started,
                progress_log=progress_log,
            )

        _log(f"  Using skills: {', '.join(matched_ids)}")

        # 3. Retrieve combined context
        _log("Retrieving combined knowledge...")
        combined_context = self._retrieve_multi(matched_ids, task, k_per_skill=5)

        # 4. Optional model training
        model_result: Optional[dict[str, Any]] = None
        if analysis.needs_model_training:
            _log("Training a small model (this may take minutes)...")
            try:
                model_result = self.model_builder.build(
                    task, progress=lambda m: _log(f"  [model] {m}")
                )
                _log(f"  Model result: {model_result.get('metric') or 'no metric'}")
            except ModelBuilderError as e:
                _log(f"  ❌ Model training failed: {e}")

        # 5. Execute using the primary matched skill (runtime)
        primary_id = matched_ids[0]
        if analysis.primary_skill:
            for m in matches:
                if m.required_name.lower() == analysis.primary_skill.lower() and m.existing_id:
                    primary_id = m.existing_id
                    break

        _log(f"Executing via primary skill: {primary_id}")
        runtime_result = self._execute(
            primary_id=primary_id,
            task=task,
            extra_context=combined_context,
            log=_log,
        )

        # 6. Compose final answer
        _log("Composing final answer...")
        answer = self._compose(
            task=task,
            analysis=analysis,
            matches=matches,
            context=combined_context,
            runtime=runtime_result,
            model_result=model_result,
        )

        elapsed = time.time() - started
        _log(f"Done in {elapsed:.1f}s")

        return TaskResult(
            task=task,
            analysis=analysis,
            skill_matches=matches,
            answer=answer,
            code_output=runtime_result.code_output,
            plan=runtime_result.plan,
            context_used=runtime_result.context_used,
            new_entries=runtime_result.new_entries,
            model_result=model_result,
            elapsed_seconds=elapsed,
            progress_log=progress_log,
        )

    # ------------------------------------------------------------------
    # Step 1 — analyze
    # ------------------------------------------------------------------

    def _analyze_task(self, task: str) -> TaskAnalysis:
        try:
            data = self.llm.chat_json(_ANALYZE_SYSTEM, f"Task: {task}")
        except LLMResponseError as e:
            log.warning("task analysis failed: %s", e)
            return TaskAnalysis()

        if not isinstance(data, dict):
            return TaskAnalysis()

        skills = [
            str(s).strip()
            for s in (data.get("required_skills") or [])
            if str(s).strip()
        ][:4]

        return TaskAnalysis(
            required_skills=skills,
            reasoning=str(data.get("reasoning", ""))[:500],
            needs_code=bool(data.get("needs_code", False)),
            needs_model_training=bool(data.get("needs_model_training", False)),
            primary_skill=str(data.get("primary_skill", "")).strip(),
        )

    # ------------------------------------------------------------------
    # Step 2 — match + auto-learn
    # ------------------------------------------------------------------

    def _resolve_skills(
        self,
        required: list[str],
        *,
        auto_learn: bool,
        cooldown: int,
        log: Callable[[str], None],
    ) -> list[SkillMatch]:
        matches: list[SkillMatch] = []
        existing = self.registry.list()

        for req in required:
            log(f"  Matching: {req}")
            match = self._match_one(req, existing)
            if match and match.existing_id:
                log(f"    ✅ Matched -> {match.existing_id} (conf {match.confidence:.2f})")
                matches.append(match)
                continue

            # No existing match -> try to learn
            if not auto_learn:
                log("    ⚠️  No match and auto_learn disabled")
                matches.append(SkillMatch(
                    required_name=req, existing_id=None, existing_name=None,
                    confidence=0.0, status="failed",
                ))
                continue

            log(f"    📚 Learning new skill: {req}")
            try:
                result = self.bootstrapper.learn(
                    req, progress=lambda m: log(f"      {m}")
                )
                new_id = result["skill_id"]
                log(f"    ✅ Learned -> {new_id} ({result['total']} entries)")
                matches.append(SkillMatch(
                    required_name=req,
                    existing_id=new_id,
                    existing_name=result["spec"].get("name", req),
                    confidence=1.0,
                    status="learned",
                ))
                # refresh existing list for subsequent matches
                existing = self.registry.list()
                if cooldown > 0:
                    log(f"    ⏸️  Cooldown {cooldown}s")
                    time.sleep(cooldown)
            except Exception as e:
                log(f"    ❌ Learning failed: {e}")
                matches.append(SkillMatch(
                    required_name=req, existing_id=None, existing_name=None,
                    confidence=0.0, status="failed",
                ))

        return matches

    def _match_one(
        self,
        required: str,
        existing: list[SkillRecord],
    ) -> Optional[SkillMatch]:
        if not existing:
            return None

        # Fast path: exact or substring match
        req_norm = re.sub(r"[^a-z0-9]+", "_", required.lower()).strip("_")
        for s in existing:
            if s.skill_id == req_norm:
                return SkillMatch(
                    required_name=required, existing_id=s.skill_id,
                    existing_name=s.name, confidence=1.0, status="matched",
                )

        # LLM-based fuzzy matching
        listing = "\n".join(f"- id={s.skill_id}  name={s.name}" for s in existing)
        try:
            verdict = self.llm.chat_json(
                _MATCH_SYSTEM,
                f"Required skill: {required}\n\nExisting skills:\n{listing}",
            )
        except Exception as e:
            log.debug("match LLM failed: %s", e)
            return None

        if not isinstance(verdict, dict):
            return None

        match_id = verdict.get("match_id")
        confidence = float(verdict.get("confidence", 0.0) or 0.0)

        if not match_id or confidence < 0.7:
            return None

        for s in existing:
            if s.skill_id == match_id:
                return SkillMatch(
                    required_name=required, existing_id=s.skill_id,
                    existing_name=s.name, confidence=confidence, status="matched",
                )

        return None

    # ------------------------------------------------------------------
    # Step 3 — multi-skill retrieval
    # ------------------------------------------------------------------

    def _retrieve_multi(
        self,
        skill_ids: list[str],
        task: str,
        k_per_skill: int = 5,
    ) -> list[dict[str, Any]]:
        try:
            emb = self.llm.embed_one(task)
        except Exception as e:
            log.error("embedding failed: %s", e)
            return []

        combined: list[dict[str, Any]] = []
        for skill_id in skill_ids:
            res = self.vectors.query(skill_id, emb, k=k_per_skill)
            ids = res.get("ids") or [[]]
            docs = res.get("documents") or [[]]
            metas = res.get("metadatas") or [[]]
            dists = res.get("distances") or [[]]
            if not ids[0]:
                continue
            for i, doc_id in enumerate(ids[0]):
                combined.append({
                    "skill_id": skill_id,
                    "id": doc_id,
                    "document": docs[0][i] if i < len(docs[0]) else "",
                    "metadata": metas[0][i] if i < len(metas[0]) else {},
                    "distance": dists[0][i] if dists and i < len(dists[0]) else 1.0,
                })

        # sort by distance (lower = more relevant), keep top overall
        combined.sort(key=lambda r: r.get("distance", 1.0))
        return combined[:12]

    # ------------------------------------------------------------------
    # Step 4 — execute via runtime
    # ------------------------------------------------------------------

    def _execute(
        self,
        primary_id: str,
        task: str,
        extra_context: list[dict[str, Any]],
        log: Callable[[str], None],
    ) -> RuntimeResult:
        try:
            return self.runtime.run(
                skill_id=primary_id,
                task=task,
                progress=lambda m: log(f"  {m}"),
            )
        except Exception as e:
            log(f"  ❌ Runtime execution failed: {e}")
            return RuntimeResult(
                answer=f"[runtime error: {e}]",
                plan={}, code_output="", gap={}, new_entries=0, context_used=0,
            )

    # ------------------------------------------------------------------
    # Step 5 — compose
    # ------------------------------------------------------------------

    def _compose(
        self,
        task: str,
        analysis: TaskAnalysis,
        matches: list[SkillMatch],
        context: list[dict[str, Any]],
        runtime: RuntimeResult,
        model_result: Optional[dict[str, Any]],
    ) -> str:
        # Format context, grouped by skill
        by_skill: dict[str, list[str]] = {}
        for c in context:
            sid = c.get("skill_id", "?")
            meta = c.get("metadata") or {}
            block = (
                f"[{meta.get('type','?')}] {meta.get('title','')}\n"
                f"{c.get('document','')[:1200]}"
            )
            by_skill.setdefault(sid, []).append(block)

        context_text = ""
        for sid, blocks in by_skill.items():
            context_text += f"\n### Skill: {sid}\n" + "\n---\n".join(blocks) + "\n"

        skills_used = ", ".join(
            f"{m.required_name} -> {m.existing_id or 'unmatched'}"
            for m in matches
        )

        model_summary = ""
        if model_result:
            model_summary = (
                f"Model trained: {model_result.get('plan', {}).get('library', '?')} "
                f"on {model_result.get('plan', {}).get('dataset_name', '?')}. "
                f"Metric: {model_result.get('metric', 'n/a')}"
            )

        prompt = (
            f"Original task:\n{task}\n\n"
            f"Skills used:\n{skills_used}\n\n"
            f"Runtime plan:\n{runtime.plan}\n\n"
            f"Runtime answer (draft):\n{runtime.answer[:2500]}\n\n"
            f"Code output:\n{runtime.code_output[:1500]}\n\n"
            f"Model training:\n{model_summary or '(none)'}\n\n"
            f"Knowledge retrieved across skills:\n{context_text[:6000]}"
        )

        try:
            return self.llm.chat(_COMPOSE_SYSTEM, prompt, temperature=0.2)
        except Exception as e:
            log.error("composition failed: %s", e)
            return runtime.answer or f"[composition failed: {e}]"


__all__ = ["TaskOrchestrator", "TaskResult", "TaskAnalysis", "SkillMatch"]
