"""Tests for onyx.sandbox — subprocess Python execution."""

from __future__ import annotations

from onyx.sandbox import Sandbox


class TestSandbox:
    def test_hello_world(self, temp_config):
        sbx = Sandbox(temp_config)
        out = sbx.run_python('print("hello")')
        assert "hello" in out

    def test_stderr_captured(self, temp_config):
        sbx = Sandbox(temp_config)
        code = 'import sys\nprint("out")\nprint("err", file=sys.stderr)'
        out = sbx.run_python(code)
        assert "out" in out
        assert "err" in out
        assert "[stderr]" in out

    def test_empty_code(self, temp_config):
        sbx = Sandbox(temp_config)
        out = sbx.run_python("")
        assert "[empty code]" in out

    def test_exception_captured(self, temp_config):
        sbx = Sandbox(temp_config)
        out = sbx.run_python("raise ValueError('boom')")
        assert "ValueError" in out
        assert "boom" in out

    def test_timeout(self, temp_config):
        # Override timeout to something small
        sbx = Sandbox(temp_config)
        code = "import time\ntime.sleep(60)"
        out = sbx.run_python(code, timeout=2)
        assert "timeout" in out.lower()

    def test_no_host_env_leak(self, temp_config):
        sbx = Sandbox(temp_config)
        # Environment variables should not include our own secrets
        code = "import os\nprint(os.environ.get('OPENAI_API_KEY', 'NOT_SET'))"
        out = sbx.run_python(code)
        assert "NOT_SET" in out
