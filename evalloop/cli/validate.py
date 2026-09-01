"""`evalloop validate` - the command people run fifty times a day.

Rendering is deliberate. A validation error that says "field required" without
saying where is a scavenger hunt, so every problem prints as file:line, the
source line itself, a caret under the offending token, and a suggestion when one
can be derived.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from evalloop.config import ConfigKind, LoadedConfig, Problem, validate_paths

__all__ = ["render_problems", "validate_command"]

_MAX_SNIPPET = 100


def validate_command(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Config files to validate. Globs are expanded by the shell."),
    ],
    as_kind: Annotated[
        ConfigKind | None,
        typer.Option(
            "--as",
            help="Force a schema instead of inferring it from the filename or contents.",
        ),
    ] = None,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Print nothing; use the exit code.")
    ] = False,
) -> None:
    """Validate configuration files. Exit code 1 if anything is wrong."""
    console = Console(stderr=True)
    loaded, problems = validate_paths(list(paths), kind=as_kind)

    if not quiet:
        render_problems(console, loaded, problems)

    raise typer.Exit(1 if problems else 0)


def render_problems(
    console: Console,
    loaded: list[LoadedConfig],
    problems: list[Problem],
) -> None:
    if not problems:
        for config in loaded:
            console.print(f"[green]✓[/green] {config.source.filename}  [dim]({config.kind})[/dim]")
        count = len(loaded)
        console.print(f"\n[green]{count} file{'s' if count != 1 else ''} valid[/green]")
        return

    by_file: dict[str, list[Problem]] = {}
    for problem in problems:
        by_file.setdefault(problem.file, []).append(problem)

    for file, file_problems in by_file.items():
        console.print(f"\n[bold]{file}[/bold]")
        for problem in file_problems:
            _render_one(console, problem)

    total = len(problems)
    console.print(f"\n[red]{total} problem{'s' if total != 1 else ''} found[/red]")


def _render_one(console: Console, problem: Problem) -> None:
    location = f":{problem.line}" if problem.line else ""
    where = f"[dim]{location}[/dim] " if location else ""
    path = f"[cyan]{problem.path}[/cyan]  " if problem.path else ""

    approximate = problem.position is not None and not problem.position.exact
    prefix = "near " if approximate else ""
    console.print(f"  [red]✗[/red] {where}{path}{prefix}{problem.message}")

    if problem.snippet.strip() and problem.position is not None:
        console.print(_snippet_with_caret(problem))

    if problem.hint:
        console.print(f"      [yellow]→[/yellow] [dim]{problem.hint}[/dim]")


def _snippet_with_caret(problem: Problem) -> Text:
    assert problem.position is not None
    snippet = problem.snippet.rstrip()
    # Long lines are truncated rather than wrapped, so the caret column stays
    # aligned with the token it is pointing at.
    truncated = snippet[:_MAX_SNIPPET] + ("…" if len(snippet) > _MAX_SNIPPET else "")

    text = Text()
    text.append("      ")
    text.append(truncated, style="dim")
    caret_column = min(problem.position.column, len(truncated) + 1)
    text.append("\n      " + " " * (caret_column - 1) + "^", style="red")
    return text
