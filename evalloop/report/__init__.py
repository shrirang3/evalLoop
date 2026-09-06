"""Rollups over stored results.

Evaluation writes one row per (trace, evaluator). That is the right shape to
store and the wrong shape to read: nobody wants four thousand verdicts, they
want the handful of patterns inside them.
"""

from evalloop.report.tool_calls import (
    SelectionRow,
    ToolCallReport,
    ViolationRow,
    build_tool_call_report,
)

__all__ = [
    "SelectionRow",
    "ToolCallReport",
    "ViolationRow",
    "build_tool_call_report",
]
