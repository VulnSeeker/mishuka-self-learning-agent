"""
Command-line interface for Onyx.

Subcommands:
    onyx learn <skill>       Bootstrap a new skill from web research
    onyx run <skill> <task>  Execute a task using a learned skill
    onyx skills              List learned skills
    onyx show <skill>        Show details of a learned skill
    onyx delete <skill>      Delete a skill and its knowledge base
    onyx train <task>        Plan and train a small ML model
    onyx web                 Launch the Streamlit dashboard
    onyx info                Show current configuration
    onyx version             Show version

Entry point declared in pyproject.toml:
    [project.scripts]
    onyx = "onyx.cli:main"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from onyx.agent import AgentError, Onyx, SkillNotFoundError
from onyx.config import CONFIG
from onyx.version import __version__

# ---------------------------------------------------------------------------
# Rich is optional at import time; fall back to plain print.
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    _console = Console()
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _console = None  # type: ignore[assignment]
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print(msg: str = "") -> None:
    if _HAS_RICH and _console is not None:
        _console.print(msg)
    else:
        print(msg)


def _print_err(msg: str) -> None:
    if _HAS_RICH and _console is not None:
        _console.print(f"[bold red]error:[/bold red] {msg}")
    else:
        print(f"error: {msg}", file=sys.stderr)


def _print_ok(msg: str) -> None:
    if _HAS_RICH and _console is not None:
        _console.print(f"[bold green]✓[/bold green] {msg}")
    else:
        print(f"ok: {msg}")


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

def _make_progress_cb():
    """Return a callable(str) that updates a Rich spinner, or a plain print."""
    if not _HAS_RICH:
        def _plain(msg: str) -> None:
            print(f"  {msg}", flush=True)
        return _plain, None

    assert _console is not None
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=_console,
        transient=True,
    )
    progress.start()
    task_id = progress.add_task("starting...", total=None)

    def _cb(msg: str) -> None:
        progress.update(task_id, description=msg)

    return _cb, progress


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_learn(args: argparse.Namespace) -> int:
    cb, progress = _make_progress_cb()
    try:
        with Onyx() as ox:
            result = ox.learn(args.skill, progress=cb)
    except AgentError as e:
        if progress:
            progress.stop()
        _print_err(str(e))
        return 1
    finally:
        if progress:
            progress.stop()

    _print()
    _print_ok(f"Skill: {result['skill_id']}")
    _print(f"  entries added  : {result['added']}")
    _print(f"  entries updated: {result['updated']}")
    _print(f"  entries skipped: {result['skipped']}")
    _print(f"  total entries  : {result['total']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cb, progress = _make_progress_cb()
    try:
        with Onyx() as ox:
            result = ox.run(args.skill, args.task, progress=cb)
    except SkillNotFoundError as e:
        if progress:
            progress.stop()
        _print_err(str(e))
        return 2
    except AgentError as e:
        if progress:
            progress.stop()
        _print_err(str(e))
        return 1
    finally:
        if progress:
            progress.stop()

    _print()
    if _HAS_RICH and _console is not None:
        _console.rule("[bold]answer")
    else:
        print("=" * 60)
    _print(result.answer)

    _print()
    if _HAS_RICH and _console is not None:
        _console.rule("[bold]stats")
    else:
        print("-" * 60)
    _print(f"  context entries : {result.context_used}")
    _print(f"  new entries     : {result.new_entries}")
    _print(f"  gap detected    : {result.gap.get('gap', False)}")

    if args.show_code and result.code_output:
        _print()
        if _HAS_RICH and _console is not None:
            _console.rule("[bold]code output")
        else:
            print("-" * 60)
        _print(result.code_output)

    if args.show_plan and result.plan:
        _print()
        if _HAS_RICH and _console is not None:
            _console.rule("[bold]plan")
        else:
            print("-" * 60)
        _print(json.dumps(result.plan, indent=2))

    return 0


def _cmd_skills(_args: argparse.Namespace) -> int:
    with Onyx() as ox:
        skills = ox.skills()

    if not skills:
        _print("[yellow]No skills learned yet.[/yellow]" if _HAS_RICH else "No skills learned yet.")
        return 0

    if _HAS_RICH and _console is not None:
        table = Table(title="Learned skills")
        table.add_column("skill_id", style="cyan", no_wrap=True)
        table.add_column("name", style="bold")
        table.add_column("entries", justify="right")
        table.add_column("updated", style="dim")
        for s in skills:
            table.add_row(s.skill_id, s.name, str(s.entry_count), s.updated_at)
        _console.print(table)
    else:
        print(f"{'skill_id':<25} {'name':<30} {'entries':>8}  updated")
        for s in skills:
            print(f"{s.skill_id:<25} {s.name:<30} {s.entry_count:>8}  {s.updated_at}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with Onyx() as ox:
        rec = ox.get_skill(args.skill)
        if rec is None:
            _print_err(f"skill '{args.skill}' not found")
            return 2
        entries = ox.skill_entries(args.skill, limit=args.limit)

    _print(f"[bold]Skill:[/bold] {rec.name}" if _HAS_RICH else f"Skill: {rec.name}")
    _print(f"  id          : {rec.skill_id}")
    _print(f"  description : {rec.description}")
    _print(f"  tools       : {', '.join(rec.tools) or '(none)'}")
    _print(f"  entries     : {rec.entry_count}")
    _print(f"  created_at  : {rec.created_at}")
    _print(f"  updated_at  : {rec.updated_at}")

    if entries:
        _print()
        if _HAS_RICH and _console is not None:
            table = Table(title=f"Recent knowledge ({len(entries)} shown)")
            table.add_column("type", style="magenta", no_wrap=True)
            table.add_column("title", style="bold")
            table.add_column("confidence", justify="right")
            table.add_column("source", style="dim", overflow="fold")
            for e in entries:
                table.add_row(
                    str(e.get("type") or ""),
                    str(e.get("title") or ""),
                    f"{float(e.get('confidence') or 0):.2f}",
                    str(e.get("source_url") or ""),
                )
            _console.print(table)
        else:
            print()
            for e in entries:
                print(f"  [{e.get('type')}] {e.get('title')}  "
                      f"(conf={e.get('confidence')})  {e.get('source_url')}")

    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    with Onyx() as ox:
        ok = ox.delete(args.skill)
    if ok:
        _print_ok(f"deleted '{args.skill}'")
        return 0
    _print_err(f"skill '{args.skill}' not found")
    return 2


def _cmd_train(args: argparse.Namespace) -> int:
    cb, progress = _make_progress_cb()
    try:
        with Onyx() as ox:
            out = ox.train_model(args.task, progress=cb)
    except AgentError as e:
        if progress:
            progress.stop()
        _print_err(str(e))
        return 1
    finally:
        if progress:
            progress.stop()

    plan = out.get("plan") or {}
    _print()
    _print_ok(f"Training complete: {plan.get('library', '?')} on {plan.get('dataset_name', '?')}")
    if out.get("metric"):
        _print(f"  metric      : {out['metric']}")
    _print(f"  approach    : {plan.get('approach', '')}")
    _print(f"  dataset_url : {plan.get('dataset_url', '')}")

    if args.show_output:
        _print()
        _print("[bold]sandbox output:[/bold]" if _HAS_RICH else "sandbox output:")
        _print(out.get("stdout", ""))

    return 0 if out.get("ok") else 1


def _cmd_web(_args: argparse.Namespace) -> int:
    """Launch Streamlit pointing at onyx/web.py."""
    web_file = Path(__file__).parent / "web.py"
    if not web_file.exists():
        _print_err(f"web.py not found at {web_file}")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(web_file),
        "--server.headless=true",
    ]
    _print(f"[cyan]Launching dashboard:[/cyan] {' '.join(cmd)}" if _HAS_RICH else " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError:
        _print_err("streamlit is not installed. Run: pip install 'onyx-self-learning-agent[web]'")
        return 1


def _cmd_info(_args: argparse.Namespace) -> int:
    _print(f"[bold]Onyx[/bold] v{__version__}" if _HAS_RICH else f"Onyx v{__version__}")
    _print(f"  {CONFIG.summary()}")
    _print(f"  data_dir   : {CONFIG.data_dir}")
    _print(f"  skills_dir : {CONFIG.skills_dir}")
    _print(f"  chroma_dir : {CONFIG.chroma_dir}")
    _print(f"  registry   : {CONFIG.registry_db}")
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onyx",
        description="Onyx — self-learning AI agent.",
    )
    parser.add_argument("--version", action="version", version=f"onyx {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # learn
    p = sub.add_parser("learn", help="Bootstrap a new skill from web research")
    p.add_argument("skill", help="Skill name, e.g. 'OSINT'")
    p.set_defaults(func=_cmd_learn)

    # run
    p = sub.add_parser("run", help="Execute a task using a learned skill")
    p.add_argument("skill", help="Skill id (see 'onyx skills')")
    p.add_argument("task", help="Natural-language task")
    p.add_argument("--show-code", action="store_true", help="Print sandbox code output")
    p.add_argument("--show-plan", action="store_true", help="Print the runtime plan JSON")
    p.set_defaults(func=_cmd_run)

    # skills
    p = sub.add_parser("skills", help="List learned skills")
    p.set_defaults(func=_cmd_skills)

    # show
    p = sub.add_parser("show", help="Show details of a learned skill")
    p.add_argument("skill", help="Skill id")
    p.add_argument("--limit", type=int, default=30, help="Max knowledge entries to show")
    p.set_defaults(func=_cmd_show)

    # delete
    p = sub.add_parser("delete", help="Delete a skill and its knowledge base")
    p.add_argument("skill", help="Skill id")
    p.set_defaults(func=_cmd_delete)

    # train
    p = sub.add_parser("train", help="Plan and train a small ML model")
    p.add_argument("task", help="Natural-language model task")
    p.add_argument("--show-output", action="store_true", help="Print sandbox output")
    p.set_defaults(func=_cmd_train)

    # web
    p = sub.add_parser("web", help="Launch the Streamlit dashboard")
    p.set_defaults(func=_cmd_web)

    # info
    p = sub.add_parser("info", help="Show current configuration")
    p.set_defaults(func=_cmd_info)

    # version
    p = sub.add_parser("version", help="Show version")
    p.set_defaults(func=_cmd_version)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _print()
        _print_err("interrupted")
        return 130
    except Exception as e:  # noqa: BLE001
        _print_err(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
