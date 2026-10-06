"""
FastAPI service layer for Onyx.

Exposes the agent over HTTP so it can be called from anywhere:

    POST /task                  Submit a task, returns job ID
    GET  /task/{job_id}         Poll task status/result
    GET  /skills                List all learned skills
    GET  /skills/{skill_id}     Skill detail + knowledge entries
    POST /skills/{skill_id}/delete  Admin: delete a skill
    GET  /health                Liveness + readiness probe
    GET  /version               Version info
    GET  /docs                  Auto-generated OpenAPI docs

Authentication:
    Set ONYX_API_TOKEN in the environment to require
    `Authorization: Bearer <token>` on all /task and /skills endpoints.
    Leave it empty for local development (no auth).

Run with:
    uvicorn onyx.api:app --host 0.0.0.0 --port 8000

Or via the CLI:
    onyx serve
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from onyx.agent import AgentError, Onyx, SkillNotFoundError
from onyx.config import CONFIG
from onyx.version import __version__

log = logging.getLogger("onyx.api")


# ===========================================================================
# Configuration
# ===========================================================================

API_TOKEN = (os.getenv("ONYX_API_TOKEN") or "").strip()
JOB_TTL_SECONDS = int(os.getenv("ONYX_JOB_TTL_SECONDS", "3600"))  # 1 hour
MAX_TASK_LENGTH = int(os.getenv("ONYX_MAX_TASK_LENGTH", "4000"))
CORS_ORIGINS = [
    o.strip()
    for o in (os.getenv("ONYX_CORS_ORIGINS") or "*").split(",")
    if o.strip()
]


# ===========================================================================
# App + middleware
# ===========================================================================

app = FastAPI(
    title="Onyx API",
    description="Self-learning AI agent — submit a task, get a result.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# ===========================================================================
# Auth
# ===========================================================================

_bearer = HTTPBearer(auto_error=False)


async def _require_auth(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """Enforce bearer token auth when ONYX_API_TOKEN is set."""
    if not API_TOKEN:
        return  # auth disabled
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if creds.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API token",
        )


# ===========================================================================
# Agent singleton
# ===========================================================================

_agent_lock = threading.Lock()
_agent_instance: Optional[Onyx] = None


def _get_agent() -> Onyx:
    """Lazy thread-safe singleton for the Onyx agent."""
    global _agent_instance
    if _agent_instance is None:
        with _agent_lock:
            if _agent_instance is None:
                log.info("Initialising Onyx agent")
                _agent_instance = Onyx()
    return _agent_instance


@app.on_event("startup")
async def _on_startup() -> None:
    log.info(
        "Onyx API starting: version=%s auth=%s jobs_ttl=%ds",
        __version__,
        "enabled" if API_TOKEN else "disabled",
        JOB_TTL_SECONDS,
    )
    # Eager-init so the first request is fast
    try:
        _get_agent()
    except Exception as e:  # noqa: BLE001
        log.error("Agent init failed at startup: %s", e)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global _agent_instance
    if _agent_instance is not None:
        try:
            _agent_instance.close()
        except Exception:  # noqa: BLE001
            pass
    log.info("Onyx API shut down cleanly")


# ===========================================================================
# Job store (in-memory; replace with Redis for multi-worker)
# ===========================================================================

class _Job:
    __slots__ = ("id", "task", "status", "created_at", "started_at",
                 "finished_at", "result", "error", "progress_log")

    def __init__(self, job_id: str, task: str) -> None:
        self.id = job_id
        self.task = task
        self.status = "queued"  # queued | running | done | failed
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress_log: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "task": self.task,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": (
                (self.finished_at or time.time()) - self.started_at
                if self.started_at else None
            ),
            "result": self.result,
            "error": self.error,
            "progress_log_tail": self.progress_log[-20:],
        }


_jobs: dict[str, _Job] = {}
_jobs_lock = threading.Lock()


def _gc_jobs() -> None:
    """Drop jobs older than JOB_TTL_SECONDS."""
    now = time.time()
    with _jobs_lock:
        stale = [
            jid for jid, job in _jobs.items()
            if (now - job.created_at) > JOB_TTL_SECONDS
        ]
        for jid in stale:
            del _jobs[jid]


# ===========================================================================
# Schemas
# ===========================================================================

class TaskRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=MAX_TASK_LENGTH)
    auto_learn: bool = True

    @field_validator("task")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("task cannot be empty")
        return v


class TaskSubmitResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


class SkillSummary(BaseModel):
    skill_id: str
    name: str
    description: str
    tools: list[str]
    entry_count: int
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    skills_count: int
    auth_enabled: bool


_STARTED_AT = time.time()


# ===========================================================================
# Background execution
# ===========================================================================

def _run_task_sync(job: _Job) -> None:
    """Runs in a background thread. Updates the job in place."""
    job.status = "running"
    job.started_at = time.time()

    def _cb(msg: str) -> None:
        job.progress_log.append(msg)
        log.debug("[job %s] %s", job.id, msg)

    try:
        agent = _get_agent()
        result = agent.task(job.task, progress=_cb)
        job.result = {
            "task": result.task,
            "answer": result.answer,
            "code_output": result.code_output,
            "plan": result.plan,
            "context_used": result.context_used,
            "new_entries": result.new_entries,
            "model_result": result.model_result,
            "elapsed_seconds": result.elapsed_seconds,
            "analysis": {
                "required_skills": result.analysis.required_skills,
                "reasoning": result.analysis.reasoning,
                "needs_code": result.analysis.needs_code,
                "needs_model_training": result.analysis.needs_model_training,
                "primary_skill": result.analysis.primary_skill,
            },
            "skill_matches": [
                {
                    "required_name": m.required_name,
                    "existing_id": m.existing_id,
                    "existing_name": m.existing_name,
                    "confidence": m.confidence,
                    "status": m.status,
                }
                for m in result.skill_matches
            ],
        }
        job.status = "done"
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job.id)
        job.error = str(e)
        job.status = "failed"
    finally:
        job.finished_at = time.time()


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/", include_in_schema=False)
async def _root() -> dict[str, str]:
    return {
        "service": "Onyx",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def _health() -> HealthResponse:
    try:
        skills = _get_agent().skills()
        count = len(skills)
    except Exception:  # noqa: BLE001
        count = -1

    return HealthResponse(
        status="ok" if count >= 0 else "degraded",
        version=__version__,
        uptime_seconds=round(time.time() - _STARTED_AT, 1),
        skills_count=count,
        auth_enabled=bool(API_TOKEN),
    )


@app.get("/version", include_in_schema=False)
async def _version() -> dict[str, str]:
    return {"version": __version__}


@app.post(
    "/task",
    response_model=TaskSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_require_auth)],
    summary="Submit a task for asynchronous execution",
)
async def _submit_task(req: TaskRequest, request: Request) -> TaskSubmitResponse:
    _gc_jobs()

    job_id = uuid.uuid4().hex[:16]
    job = _Job(job_id, req.task)

    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_task_sync,
        args=(job,),
        daemon=True,
        name=f"onyx-job-{job_id}",
    )
    thread.start()

    base = str(request.base_url).rstrip("/")
    return TaskSubmitResponse(
        job_id=job_id,
        status="queued",
        poll_url=f"{base}/task/{job_id}",
    )


@app.get(
    "/task/{job_id}",
    dependencies=[Depends(_require_auth)],
    summary="Poll task status and retrieve result",
)
async def _get_task(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"job '{job_id}' not found or expired",
        )
    return job.to_dict()


@app.get(
    "/skills",
    response_model=list[SkillSummary],
    dependencies=[Depends(_require_auth)],
    summary="List all learned skills",
)
async def _list_skills() -> list[SkillSummary]:
    try:
        skills = _get_agent().skills()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return [
        SkillSummary(
            skill_id=s.skill_id,
            name=s.name,
            description=s.description,
            tools=s.tools,
            entry_count=s.entry_count,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in skills
    ]


@app.get(
    "/skills/{skill_id}",
    dependencies=[Depends(_require_auth)],
    summary="Get details of a specific skill",
)
async def _get_skill(skill_id: str, limit: int = 50) -> dict[str, Any]:
    agent = _get_agent()
    rec = agent.get_skill(skill_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"skill '{skill_id}' not found")
    entries = agent.skill_entries(skill_id, limit=limit)
    return {
        "skill_id": rec.skill_id,
        "name": rec.name,
        "description": rec.description,
        "tools": rec.tools,
        "entry_count": rec.entry_count,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
        "entries": entries,
    }


@app.delete(
    "/skills/{skill_id}",
    dependencies=[Depends(_require_auth)],
    summary="Delete a skill and its knowledge base",
)
async def _delete_skill(skill_id: str) -> dict[str, Any]:
    ok = _get_agent().delete(skill_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"skill '{skill_id}' not found")
    return {"deleted": True, "skill_id": skill_id}


# ===========================================================================
# Exception handlers
# ===========================================================================

@app.exception_handler(AgentError)
async def _agent_error(_: Request, exc: AgentError) -> Any:
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(SkillNotFoundError)
async def _skill_not_found(_: Request, exc: SkillNotFoundError) -> Any:
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={"detail": str(exc)})
