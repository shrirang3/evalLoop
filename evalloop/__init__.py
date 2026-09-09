"""EvalLoop — a configurable evaluation and improvement control plane for AI products.

Two ways in, same engine.

**Python**, for finding out whether this works on your traces this afternoon:

    from evalloop import EvalLoop

    report = EvalLoop(
        judge="anthropic:claude-sonnet-5",
        tools="tools.yaml",
        traces="traces.jsonl",
    ).run()
    report.print()

**CLI**, for a pipeline that has to answer "did this metric move, or did the
check change underneath me?" six months later — versioned snapshots, hashed
judges, results in Postgres.

    evalloop ingest project.yaml && evalloop evaluate eval-suite.yaml
"""

from importlib.metadata import PackageNotFoundError, version

from evalloop.api import (
    DEFAULT_CHECKS,
    Dataset,
    EvalLoop,
    Report,
    llm_question,
    registry_check,
    text_matches_tools,
    tool_call_outcome,
    tool_selection,
)

try:
    __version__ = version("evalloop")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "DEFAULT_CHECKS",
    "Dataset",
    "EvalLoop",
    "Report",
    "__version__",
    "llm_question",
    "registry_check",
    "text_matches_tools",
    "tool_call_outcome",
    "tool_selection",
]
