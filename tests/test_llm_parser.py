"""Tests for the JSON parser used by the LLM layer."""

from __future__ import annotations

import pytest

from onyx.llm import LLMResponseError, parse_json_loose


class TestParseJsonLoose:
    def test_pure_json_object(self):
        assert parse_json_loose('{"a": 1, "b": 2}') == {"a": 1, "b": 2}

    def test_pure_json_array(self):
        assert parse_json_loose('[1, 2, 3]') == [1, 2, 3]

    def test_fenced_json(self):
        raw = '```json\n{"a": 1}\n```'
        assert parse_json_loose(raw) == {"a": 1}

    def test_fenced_no_lang(self):
        raw = '```\n{"a": 1}\n```'
        assert parse_json_loose(raw) == {"a": 1}

    def test_prose_before_and_after(self):
        raw = 'Here is the result:\n{"a": 1, "b": 2}\nHope this helps!'
        assert parse_json_loose(raw) == {"a": 1, "b": 2}

    def test_nested_object(self):
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        assert parse_json_loose(raw) == {"outer": {"inner": [1, 2, 3]}}

    def test_prose_with_array(self):
        raw = 'The list is [1, 2, 3] as requested.'
        assert parse_json_loose(raw) == [1, 2, 3]

    def test_empty_raises(self):
        with pytest.raises(LLMResponseError):
            parse_json_loose("")

    def test_whitespace_only_raises(self):
        with pytest.raises(LLMResponseError):
            parse_json_loose("   \n\t  ")

    def test_no_json_raises(self):
        with pytest.raises(LLMResponseError):
            parse_json_loose("just some random text with no json")

    def test_truncated_json_in_fence(self):
        # Common case: LLM cuts off. We should still try to parse.
        raw = '```json\n{"a": 1, "b": {"c": 2}}\n```'
        assert parse_json_loose(raw) == {"a": 1, "b": {"c": 2}}
