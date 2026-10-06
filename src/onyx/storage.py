"""
Storage layer for Onyx.

Two complementary stores, exposed behind small classes:

  Registry     — SQLite-backed skill registry + knowledge metadata.
                 Thread-safe via per-call connections.
                 WAL mode for concurrent reads.

  VectorStore  — ChromaDB persistent client.
                 One collection per skill, cosine similarity.
                 Thread-safe via internal locking.

Nothing else in Onyx touches SQLite or Chroma directly.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from onyx.config import CONFIG, Config
from onyx.schemas import KnowledgeEntry, SkillRecord

log = logging.getLogger("onyx.storage")


# ===========================================================================
# Registry — SQLite
# ===========================================================================

_REGISTRY_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS skills (
    skill_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    tools_json    TEXT NOT NULL DEFAULT '[]',
    spec_json     TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    entry_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS knowledge_meta (
    id           TEXT PRIMARY KEY,
    skill_id     TEXT NOT NULL,
    title        TEXT,
    type         TEXT,
    source_url   TEXT,
    source_type  TEXT,
    confidence   REAL,
    status       TEXT,
    timestamp    TEXT,
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_knowledge_skill
    ON knowledge_meta(skill_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_status
    ON knowledge_meta(status);
"""


class Registry:
    """SQLite-backed skill registry and knowledge metadata store."""

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self.db_path: Path = cfg.registry_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
            isolation_level=None,   # autocommit; we manage transactions manually
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_REGISTRY_SCHEMA)
        log.debug("Registry ready at %s", self.db_path)

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def has(self, skill_id: str) -> bool:
        skill_id = (skill_id or "").strip()
        if not skill_id:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM skills WHERE skill_id = ? LIMIT 1",
                (skill_id,),
            ).fetchone()
        return row is not None

    def get(self, skill_id: str) -> Optional[SkillRecord]:
        skill_id = (skill_id or "").strip()
        if not skill_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(self) -> list[SkillRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM skills ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def upsert(self, rec: SkillRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skills
                    (skill_id, name, description, tools_json, spec_json,
                     created_at, updated_at, entry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(skill_id) DO UPDATE SET
                    name         = excluded.name,
                    description  = excluded.description,
                    tools_json   = excluded.tools_json,
                    spec_json    = excluded.spec_json,
                    updated_at   = excluded.updated_at,
                    entry_count  = excluded.entry_count
                """,
                (
                    rec.skill_id,
                    rec.name,
                    rec.description,
                    json.dumps(rec.tools),
                    json.dumps(rec.spec),
                    rec.created_at,
                    rec.updated_at,
                    rec.entry_count,
                ),
            )

    def set_entry_count(self, skill_id: str, count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE skills SET entry_count = ?, updated_at = ? WHERE skill_id = ?",
                (int(count), _utcnow_iso(), skill_id),
            )

    def delete(self, skill_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                cur = conn.execute(
                    "DELETE FROM skills WHERE skill_id = ?", (skill_id,)
                )
                conn.execute(
                    "DELETE FROM knowledge_meta WHERE skill_id = ?", (skill_id,)
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Knowledge metadata
    # ------------------------------------------------------------------

    def add_knowledge_meta(
        self,
        skill_id: str,
        entries: list[KnowledgeEntry],
        ids: list[str],
    ) -> None:
        if not entries or len(entries) != len(ids):
            return
        now = _utcnow_iso()
        rows = [
            (
                ids[i],
                skill_id,
                e.title,
                e.type,
                e.source_url,
                e.source_type,
                float(e.confidence),
                "approved",
                now,
            )
            for i, e in enumerate(entries)
        ]
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO knowledge_meta
                        (id, skill_id, title, type, source_url,
                         source_type, confidence, status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def count_knowledge(self, skill_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM knowledge_meta WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
        return int(row["c"]) if row else 0

    def list_knowledge_meta(
        self, skill_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, type, source_url, confidence, status, timestamp
                FROM knowledge_meta
                WHERE skill_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (skill_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> SkillRecord:
        return SkillRecord(
            skill_id=row["skill_id"],
            name=row["name"],
            description=row["description"] or "",
            tools=_safe_json_list(row["tools_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            entry_count=int(row["entry_count"] or 0),
            spec=_safe_json_dict(row["spec_json"]),
        )


# ===========================================================================
# VectorStore — ChromaDB
# ===========================================================================

class VectorStore:
    """ChromaDB wrapper. One collection per skill, cosine space."""

    _client_lock = threading.Lock()
    _client: Optional[chromadb.PersistentClient] = None

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self.path: Path = cfg.chroma_dir
        self.path.mkdir(parents=True, exist_ok=True)

        with VectorStore._client_lock:
            if VectorStore._client is None:
                VectorStore._client = chromadb.PersistentClient(
                    path=str(self.path),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=False,
                    ),
                )
        self.client = VectorStore._client

    # ------------------------------------------------------------------
    # Collection access
    # ------------------------------------------------------------------

    def collection(self, skill_id: str):
        """Return (creating if needed) the collection for a skill."""
        return self.client.get_or_create_collection(
            name=f"skill_{skill_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def drop(self, skill_id: str) -> None:
        """Delete the collection. Silent if missing."""
        try:
            self.client.delete_collection(name=f"skill_{skill_id}")
            log.debug("Dropped collection skill_%s", skill_id)
        except Exception as e:  # noqa: BLE001
            log.debug("drop collection skill_%s: %s", skill_id, e)

    def count(self, skill_id: str) -> int:
        try:
            return int(self.collection(skill_id).count())
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        skill_id: str,
        embedding: list[float],
        k: int = 8,
    ) -> dict[str, Any]:
        """Return Chroma's raw query dict, or an empty shape on failure."""
        empty: dict[str, Any] = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        if not embedding:
            return empty
        try:
            coll = self.collection(skill_id)
            if coll.count() == 0:
                return empty
            return coll.query(
                query_embeddings=[embedding],
                n_results=min(k, coll.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("vector query failed for %s: %s", skill_id, e)
            return empty

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_batch(
        self,
        skill_id: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents in one shot. Skips empty batches."""
        if not ids:
            return
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("add_batch: all lists must be the same length")
        coll = self.collection(skill_id)
        coll.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        try:
            return [c.name for c in self.client.list_collections()]
        except Exception:  # noqa: BLE001
            return []


# ===========================================================================
# Shared helpers
# ===========================================================================

def _safe_json_list(raw: Any) -> list:
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
        return list(val) if isinstance(val, list) else []
    except Exception:  # noqa: BLE001
        return []


def _safe_json_dict(raw: Any) -> dict:
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
        return dict(val) if isinstance(val, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


__all__ = ["Registry", "VectorStore"]
