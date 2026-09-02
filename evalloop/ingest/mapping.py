"""Turn source rows into traces.

A customer's table is theirs. Rather than ask them to migrate, they write a
mapping - `input.user_request: user_transcript` - and this module moves the
values. That is the whole reason EvalLoop can be pointed at a production
database without a schema change.

P0 does direct field-to-path moves. Transforms (`json_parse`, `template`,
`python:`) and per-field missing-source policy are P1; the two defaults that
matter are implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from evalloop.contracts.paths import set_path
from evalloop.contracts.trace import Trace

__all__ = ["MappingResult", "apply_mapping", "map_row"]

_REQUIRED_CONTAINERS = ("input", "output")


@dataclass
class MappingResult:
    """Everything one pass produced, including what it could not."""

    traces: list[Trace] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unmapped_fields: set[str] = field(default_factory=set)
    """Source columns no mapping mentioned. Reported rather than silently
    dropped - a forgotten mapping line is otherwise invisible until someone
    wonders why a slice is empty."""

    skipped: int = 0


def map_row(
    row: dict[str, Any],
    mapping: dict[str, str],
    *,
    source_id: str | None = None,
) -> Trace:
    """Build one Trace from one source row.

    An empty mapping means the row is already in trace shape, which is the
    common case for a JSONL export produced for EvalLoop rather than harvested
    from a product database.

    A source field the row does not have is left unset rather than written as
    None, so an optional column that is simply absent does not become an
    explicit null - `ground_truth.tool_calls: null` means "no tool should have
    been called", which is a claim, not an absence.
    """
    if not mapping:
        payload: dict[str, Any] = dict(row)
    else:
        payload = {}
        for target, source in mapping.items():
            if source not in row:
                continue
            set_path(payload, target, row[source])

    for container in _REQUIRED_CONTAINERS:
        payload.setdefault(container, {})
    if source_id is not None:
        payload.setdefault("source_id", source_id)

    return Trace.model_validate(payload)


def apply_mapping(
    rows: list[tuple[dict[str, Any], str]],
    mapping: dict[str, str],
) -> MappingResult:
    """Map many rows, collecting failures instead of stopping at the first.

    One unmappable row out of ten thousand should cost you that row and a line
    of output, not the whole ingest.
    """
    result = MappingResult()
    mapped_sources = set(mapping.values())

    for row, source_id in rows:
        result.unmapped_fields |= set(row) - mapped_sources if mapping else set()
        try:
            result.traces.append(map_row(row, mapping, source_id=source_id))
        except (ValidationError, ValueError, TypeError) as exc:
            result.skipped += 1
            result.errors.append(f"{source_id}: {_first_line(exc)}")

    return result


def _first_line(exc: Exception) -> str:
    """Pydantic renders multi-line errors; one row needs one line of summary."""
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first["loc"]) or "(root)"
            return f"{location}: {first['msg']}"
    return str(exc).splitlines()[0]
