"""EvalLoop command-line entry point.

Subcommands are registered here as each phase lands. `evaluate` arrives in
P0.7.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from evalloop import __version__
from evalloop.cli.ingest import ingest_command, snapshot_app
from evalloop.cli.validate import validate_command

app = typer.Typer(
    name="evalloop",
    help="Evaluate, improve, and promote AI models against your own production traces.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()

app.command("validate")(validate_command)
app.command("ingest")(ingest_command)
app.add_typer(snapshot_app, name="snapshot")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"evalloop {__version__}")
        raise typer.Exit


@app.callback()
def main(
    # Consumed by the eager callback above; the parameter itself is unused.
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the installed version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """EvalLoop."""


if __name__ == "__main__":
    app()
