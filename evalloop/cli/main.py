"""EvalLoop command-line entry point.

Subcommands are registered here as each phase lands. P0 wires up the app itself;
`validate` arrives in P0.5, `ingest` in P0.6, and `evaluate` in P0.7.
"""

from __future__ import annotations

import typer
from rich.console import Console

from evalloop import __version__

app = typer.Typer(
    name="evalloop",
    help="Evaluate, improve, and promote AI models against your own production traces.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"evalloop {__version__}")
        raise typer.Exit


@app.callback()
def main(
    # Consumed by the eager callback above; the parameter itself is unused.
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the installed version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """EvalLoop."""


if __name__ == "__main__":
    app()
