
"""Tests for onyx.schemas — pydantic validation."""

from __future__ import annotations

import pytest

from onyx.schemas import (
    KnowledgeEntry,
    RuntimePlan,
    SkillSpec,
    TrainingPlan,
)


class TestSkillSpec:
    def test_minimal(self):
        spec = SkillSpec(name="OSINT")
        assert spec.name == "OSINT"
        assert spec.subtopics == []
        assert spec.tools == []

    def test_full(self):
        spec = SkillSpec(
            name="Python asyncio",
            description="Async programming",
            subtopics=["tasks", "gather", "loop"],
            tools=["asyncio", "aiohttp"],
            keywords=["async", "await"],
            success_criteria=["write a working coroutine"],
            legal_notes="",
        )
        assert spec.description == "Async programming"
        assert len(spec.subtopics) == 3

    def test_name_stripped_and_required(self):
        with pytest.raises(Exception):
            SkillSpec(name="")
        with pytest.raises(Exception):
            SkillSpec(name="   ")

    def test_lists_deduplicated(self):
        spec = SkillSpec(
            name="test",
            subtopics=["a", "A", "b", "B", "b"],
        )
        assert spec.subtopics == ["a", "b"]


class TestKnowledgeEntry:
    def test_minimal_valid(self):
        entry = KnowledgeEntry(
            title="Title",
            content="x" * 100,
        )
        assert entry.type == "fact"
        assert entry.confidence == 0.7

    def test_content_too_short_rejected(self):
        with pytest.raises(Exception):
            KnowledgeEntry(title="T", content="short")

    def test_invalid_type_rejected(self):
        """Invalid type values must be rejected by the schema."""
        with pytest.raises(Exception):
            KnowledgeEntry(
                title="T",
                content="x" * 100,
                type="nonsense",
            )

    def test_confidence_clamped(self):
        e1 = KnowledgeEntry(title="T", content="x" * 100, confidence=2.0)
        assert e1.confidence == 1.0
        e2 = KnowledgeEntry(title="T", content="x" * 100, confidence=-1.0)
        assert e2.confidence == 0.0

    def test_tags_cleaned(self):
        entry = KnowledgeEntry(
            title="T", content="x" * 100,
            tags=["  a  ", "", "b", "a"],
        )
        # empty string stripped, whitespace stripped
        assert "a" in entry.tags
        assert "b" in entry.tags
        assert "" not in entry.tags


class TestRuntimePlan:
    def test_defaults(self):
        plan = RuntimePlan()
        assert plan.steps == []
        assert plan.needs_code is False
        assert plan.needs_web is False

    def test_with_steps(self):
        plan = RuntimePlan(
            steps=["step 1", "step 2"],
            needs_code=True,
        )
        assert len(plan.steps) == 2


class TestTrainingPlan:
    def test_minimal(self):
        plan = TrainingPlan(train_code="print('x')")
        assert plan.library == "sklearn"

    def test_empty_code_rejected(self):
        with pytest.raises(Exception):
            TrainingPlan(train_code="")
