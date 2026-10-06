"""Tests for onyx.storage — Registry (SQLite) and VectorStore (Chroma)."""

from __future__ import annotations

from onyx.schemas import KnowledgeEntry, SkillRecord
from onyx.storage import Registry


class TestRegistry:
    def test_empty_initially(self, temp_config):
        reg = Registry(temp_config)
        assert reg.list() == []
        assert not reg.has("anything")

    def test_upsert_and_get(self, temp_config):
        reg = Registry(temp_config)
        rec = SkillRecord(
            skill_id="test_skill",
            name="Test Skill",
            description="A test",
            tools=["tool1"],
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            entry_count=0,
            spec={"name": "Test Skill"},
        )
        reg.upsert(rec)
        assert reg.has("test_skill")

        got = reg.get("test_skill")
        assert got is not None
        assert got.name == "Test Skill"
        assert got.tools == ["tool1"]

    def test_upsert_updates(self, temp_config):
        reg = Registry(temp_config)
        rec = SkillRecord(
            skill_id="s", name="v1", description="",
            tools=[], created_at="t", updated_at="t",
        )
        reg.upsert(rec)
        rec.name = "v2"
        rec.entry_count = 5
        reg.upsert(rec)

        got = reg.get("s")
        assert got is not None
        assert got.name == "v2"
        assert got.entry_count == 5

    def test_set_entry_count(self, temp_config):
        reg = Registry(temp_config)
        rec = SkillRecord(
            skill_id="s", name="S", description="",
            tools=[], created_at="t", updated_at="t",
        )
        reg.upsert(rec)
        reg.set_entry_count("s", 42)
        got = reg.get("s")
        assert got is not None
        assert got.entry_count == 42

    def test_delete(self, temp_config):
        reg = Registry(temp_config)
        rec = SkillRecord(
            skill_id="s", name="S", description="",
            tools=[], created_at="t", updated_at="t",
        )
        reg.upsert(rec)
        assert reg.delete("s") is True
        assert not reg.has("s")
        assert reg.delete("nonexistent") is False

    def test_add_knowledge_meta(self, temp_config):
        reg = Registry(temp_config)
        rec = SkillRecord(
            skill_id="s", name="S", description="",
            tools=[], created_at="t", updated_at="t",
        )
        reg.upsert(rec)

        entries = [
            KnowledgeEntry(title="A", content="x" * 100),
            KnowledgeEntry(title="B", content="y" * 100),
        ]
        ids = ["id_a", "id_b"]

        reg.add_knowledge_meta("s", entries, ids)
        assert reg.count_knowledge("s") == 2

        metas = reg.list_knowledge_meta("s")
        assert len(metas) == 2
        titles = {m["title"] for m in metas}
        assert "A" in titles and "B" in titles

    def test_get_nonexistent(self, temp_config):
        reg = Registry(temp_config)
        assert reg.get("nope") is None
