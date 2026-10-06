"""
Sandboxed Python execution for Onyx.

Runs generated Python code in a temp directory with resource limits
(CPU, memory, process group) on POSIX systems. Not a security boundary —
for truly untrusted code, plug in a Docker backend via `integrations/`.

Design goals:
  - No network to the host filesystem
  - Hard timeout on wall clock AND CPU
  - Memory cap on POSIX
  - Kills the whole process group on timeout (no orphaned children)
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from onyx.config import CONFIG, Config

log = logging.getLogger("onyx.sandbox")

# Memory cap for the child (bytes). 512 MiB by default.
_MEM_LIMIT_BYTES = 512 * 1024 * 1024


class Sandbox:
    """Run short-lived Python snippets in isolation."""

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_python(self, code: str, timeout: int | None = None) -> str:
        """
        Execute `code` and return combined stdout+stderr (truncated).
        Never raises — always returns a string.
        """
        timeout = timeout or self.cfg.code_timeout
        code = (code or "").strip()
        if not code:
            return "[empty code]"

        with tempfile.TemporaryDirectory(prefix="onyx_sbx_") as tmp:
            script = Path(tmp) / "snippet.py"
            try:
                script.write_text(code, encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                return f"[sandbox write error] {e}"

            preexec = self._build_preexec(timeout)

            # Minimal environment — no secrets leak into the child.
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": tmp,
                "TMPDIR": tmp,
                "LANG": "C.UTF-8",
            }

            # -I: isolated mode (ignore PYTHON* env, no user site)
            # -S: skip site.py
            cmd = [sys.executable, "-I", "-S", str(script)]

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=tmp,
                    env=env,
                    preexec_fn=preexec,
                    start_new_session=(preexec is None),  # POSIX: preexec does setsid
                )
            except Exception as e:  # noqa: BLE001
                return f"[sandbox spawn error] {e}"

            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._kill_tree(proc)
                try:
                    proc.communicate(timeout=2)
                except Exception:  # noqa: BLE001
                    pass
                return f"[timeout after {timeout}s]"

            stdout = out or ""
            stderr = err or ""
            combined = stdout + (f"\n[stderr]\n{stderr}" if stderr else "")
            combined = combined.strip() or "[no output]"
            return combined[:6000]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_preexec(timeout: int):
        """Return a callable to run in the child before exec, or None."""
        if os.name != "posix":
            return None
        try:
            import resource  # noqa: WPS433
        except ImportError:
            return None

        def _limits() -> None:  # pragma: no cover - POSIX only
            try:
                # Memory cap (address space)
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES),
                )
                # CPU seconds
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (timeout, timeout + 2),
                )
                # File size: 16 MiB
                resource.setrlimit(
                    resource.RLIMIT_FSIZE,
                    (16 * 1024 * 1024, 16 * 1024 * 1024),
                )
                # No core dumps
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                # New session so we can kill the whole tree
                os.setsid()
            except Exception:  # noqa: BLE001
                pass

        return _limits

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["Sandbox"]
