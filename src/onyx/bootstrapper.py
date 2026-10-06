"""
Skill bootstrapper for Onyx.

Given a skill name, this module:

  1. Generates a structured SkillSpec (subtopics, tools, keywords).
  2. Finds source URLs via web search.
  3. Crawls each source to clean text.
  4. Extracts KnowledgeEntry objects via the LLM.
  5. Deduplicates against the existing skill DB.
  6. Persists to ChromaDB + SQLite registry.

The result is a per-skill knowledge base the runtime can query.

Everything is idempotent: re-learning a skill appends only new knowledge.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from onyx.config import CONFIG, Config
from onyx.crawler import Crawler
from onyx.llm import LLMClient, LLMResponseError
from onyx.schemas import (
    DedupeAction,
    KnowledgeEntry,
    SkillRecord,
    SkillSpec,
    Source,
)
from onyx.search import SearchClient
from onyx.storage import Registry, VectorStore

log = logging.getLogger("onyx.bootstrapper")

ProgressCB = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SPEC_SYSTEM = """\
You are a curriculum designer for an autonomous learning agent.

Given a skill name, produce a JSON skill specification with exactly these keys:

  - name:        string, canonical skill name
  - description: one-sentence summary
  - subtopics:   5 to 10 concrete subtopics to cover
  - tools:       real tool / library / API names (concrete, not vague)
  - keywords:    search keywords a researcher would use
  - success_criteria: 3 to 6 measurable outcomes
  - legal_notes: any ethical/legal constraints (empty string if none)

Rules:
  - Be specific and practical. Avoid marketing language.
  - Do not invent tools that don't exist.
  - If the skill has legal implications (e.g. OSINT, security), fill legal_notes.
"""

_EXTRACT_SYSTEM = """\
You extract structured, self-contained knowledge from raw web text for one skill.

Return JSON:
{
  "entries": [
    {
      "title": "short, descriptive title",
      "type": "fact | procedure | code | tool | failure",
      "content": "actionable, self-contained content",
      "tags": ["tag1", "tag2"],
      "confidence": 0.0
    }
  ]
}

Rules:
  - Extract 3 to 8 high-quality entries. Skip navigation, ads, boilerplate.
  - content must be self-contained. Include code blocks verbatim if present.
  - type "procedure" = step-by-step. "code" = snippet. "tool" = software. 
    "failure" = common pitfall / error / gotcha. "fact" = everything else.
  - confidence in [0,1] reflecting the source's authority.
  - NEVER invent information not present in the text.
  - If the text is not relevant to the skill, return {"entries": []}.
"""

_DEDUPE_SYSTEM = """\
You compare a NEW knowledge entry to an EXISTING similar entry.

Return JSON: {"action": "add" | "update" | "duplicate" | "conflict", "reason": "..."}

Definitions:
  - add:       genuinely new information, non-overlapping with EXISTING
  - update:    refines or extends EXISTING with new details
  - duplicate: essentially the same information
  - conflict:  contradicts EXISTING

Be strict. When in doubt, prefer "duplicate".
"""


# ---------------------------------------------------------------------------
# Bootstrapper
# ---------------------------------------------------------------------------

class SkillBootstrapper:
    """Builds a new skill database by researching the web."""

    def __init__(
        self,
        llm: LLMClient,
        registry: Registry,
        vectors: VectorStore,
        search: SearchClient,
        crawler: Crawler,
        cfg: Config = CONFIG,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.vectors = vectors
        self.search = search
        self.crawler = crawler
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(self, skill_name: str, progress: ProgressCB = None) -> dict:
        """Learn a skill. Returns a summary dict."""
        cb = progress or (lambda _msg: None)

        skill_name = (skill_name or "").strip()
        if not skill_name:
            raise ValueError("skill_name is required")

        # 1. Build the spec
        spec = self._build_spec(skill_name)
        skill_id = _slugify(spec.name)
        cb(f"Spec ready: {spec.name} ({len(spec.subtopics)} subtopics)")

        if self.registry.has(skill_id):
            cb(f"Skill '{skill_id}' exists — appending new knowledge")

        # 2. Find sources
        sources = self._gather_sources(spec)
        cb(f"Found {len(sources)} candidate sources")

        if not sources:
            cb("No sources found — aborting")
            return {
                "skill_id": skill_id,
                "spec": spec.model_dump(),
                "added": 0,
                "updated": 0,
                "skipped": 0,
                "total": self.vectors.count(skill_id),
            }

        # 3. Crawl + extract
        raw_entries: list[KnowledgeEntry] = []
        for i, src in enumerate(sources, 1):
            cb(f"[{i}/{len(sources)}] {_truncate(src.url, 90)}")
            text = self.crawler.fetch(src.url)
            if not text:
                continue
            entries = self._extract(spec, text, src.url)
            raw_entries.extend(entries)

        if not raw_entries:
            cb("No knowledge extracted from sources")
            self._ensure_skill_record(skill_id, spec)
            return {
                "skill_id": skill_id,
                "spec": spec.model_dump(),
                "added": 0,
                "updated": 0,
                "skipped": 0,
                "total": self.vectors.count(skill_id),
            }

        # 4. Dedupe + persist
        cb(f"Extracted {len(raw_entries)} raw entries; deduplicating...")
        added, updated, skipped = self._dedupe_and_store(skill_id, raw_entries)

        # 5. Ensure the skill record exists and counts are fresh
        total = self.vectors.count(skill_id)
        self._ensure_skill_record(skill_id, spec, entry_count=total)

        cb(f"Done — added={added} updated={updated} skipped={skipped} total={total}")
        return {
            "skill_id": skill_id,
            "spec": spec.model_dump(),
            "added": added,
            "updated": updated,
            "skipped": skipped,
            "total": total,
        }

    # ------------------------------------------------------------------
    # Spec
    # ------------------------------------------------------------------

    def _build_spec(self, skill_name: str) -> SkillSpec:
        try:
            data = self.llm.chat_json(_SPEC_SYSTEM, f"Skill: {skill_name}")
        except LLMResponseError as e:
            log.warning("spec generation failed: %s — using fallback", e)
            return SkillSpec(name=skill_name, description=f"Auto-spec for {skill_name}")

        if not isinstance(data, dict):
            return SkillSpec(name=skill_name, description=f"Auto-spec for {skill_name}")

        data.setdefault("name", skill_name)
        try:
            return SkillSpec(**data)
        except Exception as e:  # noqa: BLE001
            log.warning("SkillSpec validation failed: %s", e)
            return SkillSpec(name=skill_name, description=f"Auto-spec for {skill_name}")

    # ------------------------------------------------------------------
    # Source discovery
    # ------------------------------------------------------------------

    def _gather_sources(self, spec: SkillSpec) -> list[Source]:
        queries: list[str] = []
        queries.append(f"{spec.name} tutorial documentation")
        queries.extend(f"{spec.name} {t}" for t in spec.subtopics[:6])
        queries.extend(spec.keywords[:6])

        seen: set[str] = set()
        out: list[Source] = []

        for q in queries:
            if len(out) >= self.cfg.max_sources_per_skill:
                break
            results = self.search.search(q, max_results=4)
            for r in results:
                if not r.url or r.url in seen:
                    continue
                seen.add(r.url)
                out.append(r)
                if len(out) >= self.cfg.max_sources_per_skill:
                    break
            time.sleep(0.3)  # be polite

        return out

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract(
        self,
        spec: SkillSpec,
        page_text: str,
        url: str,
    ) -> list[KnowledgeEntry]:
        if not page_text:
            return []

        prompt = (
            f"Skill: {spec.name}\n"
            f"Subtopics: {', '.join(spec.subtopics)}\n"
            f"Source URL: {url}\n\n"
            f"Text:\n{page_text}"
        )

        try:
            data = self.llm.chat_json(_EXTRACT_SYSTEM, prompt)
        except LLMResponseError as e:
            log.warning("extraction failed for %s: %s", url, e)
            return []
        except Exception as e:  # noqa: BLE001
            log.warning("extraction crashed for %s: %s", url, e)
            return []

        if not isinstance(data, dict):
            return []

        entries: list[KnowledgeEntry] = []
        for raw in data.get("entries", []) or []:
            if not isinstance(raw, dict):
                continue
            try:
                entry = KnowledgeEntry(
                    title=str(raw.get("title", "Untitled")),
                    type=raw.get("type", "fact"),
                    content=str(raw.get("content", "")),
                    source_url=url,
                    source_type="web",
                    confidence=float(raw.get("confidence", 0.7)),
                    tags=list(raw.get("tags", []) or []),
                )
            except Exception as e:  # noqa: BLE001
                log.debug("bad entry from %s: %s", url, e)
                continue
            entries.append(entry)

        return entries

    # ------------------------------------------------------------------
    # Dedupe + store
    # ------------------------------------------------------------------

    def _dedupe_and_store(
        self,
        skill_id: str,
        entries: list[KnowledgeEntry],
    ) -> tuple[int, int, int]:
        if not entries:
            return 0, 0, 0

        texts = [f"{e.title}\n\n{e.content}" for e in entries]
        try:
            embeddings = self.llm.embed(texts)
        except Exception as e:  # noqa: BLE001
            log.error("embedding failed: %s", e)
            return 0, 0, 0

        if len(embeddings) != len(entries):
            log.error("embedding count mismatch: %d vs %d", len(embeddings), len(entries))
            return 0, 0, 0

        coll = self.vectors.collection(skill_id)
        existing_count = coll.count()

        add_ids: list[str] = []
        add_embs: list[list[float]] = []
        add_docs: list[str] = []
        add_metas: list[dict] = []
        add_entries: list[KnowledgeEntry] = []

        added = updated = skipped = 0

        for entry, emb, text in zip(entries, embeddings, texts):
            decision: DedupeAction = "add"

            if existing_count > 0:
                decision = self._classify(entry, emb, skill_id)

            if decision == "duplicate":
                skipped += 1
                continue

            doc_id = _stable_id(skill_id, entry.title, entry.content)
            add_ids.append(doc_id)
            add_embs.append(emb)
            add_docs.append(text)
            add_metas.append({
                "title": entry.title,
                "type": entry.type,
                "source_url": entry.source_url,
                "source_type": entry.source_type,
                "confidence": float(entry.confidence),
                "tags": ",".join(entry.tags),
                "status": "approved",
                "timestamp": _utcnow_iso(),
            })
            add_entries.append(entry)

            if decision == "update":
                updated += 1
            else:
                added += 1

            existing_count += 1  # keep local count roughly in sync

        if add_ids:
            try:
                self.vectors.add_batch(
                    skill_id, add_ids, add_embs, add_docs, add_metas
                )
                self.registry.add_knowledge_meta(skill_id, add_entries, add_ids)
            except Exception as e:  # noqa: BLE001
                log.error("persist failed for %s: %s", skill_id, e)
                return 0, 0, skipped

        return added, updated, skipped

    def _classify(
        self,
        entry: KnowledgeEntry,
        emb: list[float],
        skill_id: str,
    ) -> DedupeAction:
        try:
            res = self.vectors.query(skill_id, emb, k=1)
        except Exception as e:  # noqa: BLE001
            log.debug("query failed: %s", e)
            return "add"

        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        dists = res.get("distances") or [[]]

        if not ids[0] or not docs[0]:
            return "add"

        top_dist = float(dists[0][0]) if dists[0] else 1.0
        top_doc = docs[0][0]

        # Very close → duplicate, no LLM call
        if top_dist < self.cfg.dedupe_distance:
            return "duplicate"

        # Ask the LLM
        try:
            verdict = self.llm.chat_json(
                _DEDUPE_SYSTEM,
                f"NEW:\n{entry.title}\n{entry.content[:1500]}\n\n"
                f"EXISTING:\n{top_doc[:1500]}",
            )
        except Exception as e:  # noqa: BLE001
            log.debug("dedupe LLM failed: %s", e)
            return "add"

        if not isinstance(verdict, dict):
            return "add"

        action = str(verdict.get("action", "add")).lower()
        if action not in {"add", "update", "duplicate", "conflict"}:
            return "add"
        return action  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def _ensure_skill_record(
        self,
        skill_id: str,
        spec: SkillSpec,
        entry_count: Optional[int] = None,
    ) -> None:
        now = _utcnow_iso()
        existing = self.registry.get(skill_id)
        rec = SkillRecord(
            skill_id=skill_id,
            name=spec.name,
            description=spec.description,
            tools=spec.tools,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            entry_count=entry_count if entry_count is not None else self.vectors.count(skill_id),
            spec=spec.model_dump(),
        )
        self.registry.upsert(rec)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

_SLUG_RE = __import__("re").compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    s = _SLUG_RE.sub("_", name.lower()).strip("_")
    return s or "skill"


def _stable_id(*parts: str) -> str:
    import hashlib
    h = hashlib.sha1("||".join(parts).encode("utf-8"))
    return h.hexdigest()[:24]


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


__all__ = ["SkillBootstrapper"]
