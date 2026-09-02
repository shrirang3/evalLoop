"""JSONL source connector.

Yields one dict per line. Malformed lines are collected rather than raised: a
100k-line export with three bad rows should ingest 99,997 traces and tell you
about the three, not stop at line 12 and leave you re-running.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["JsonlConnector", "SourceRow"]


@dataclass(frozen=True, slots=True)
class SourceRow:
    """One source record, with where it came from."""

    data: dict[str, Any]
    source_id: str
    """Line number for JSONL. Recorded on the trace so a mapping problem can be
    traced back to the exact input row."""


@dataclass
class JsonlConnector:
    """Read-only reader over a newline-delimited JSON file."""

    path: Path
    errors: list[str] = field(default_factory=list)

    def rows(self, *, limit: int | None = None) -> Iterator[SourceRow]:
        if not self.path.exists():
            raise FileNotFoundError(f"source file not found: {self.path}")

        emitted = 0
        with self.path.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                if limit is not None and emitted >= limit:
                    return

                line = raw.strip()
                if not line or line.startswith("//"):
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.errors.append(f"line {number}: invalid JSON ({exc.msg})")
                    continue

                if not isinstance(parsed, dict):
                    self.errors.append(
                        f"line {number}: expected an object, got {type(parsed).__name__}"
                    )
                    continue

                emitted += 1
                yield SourceRow(data=parsed, source_id=f"line:{number}")

    def fingerprint_config(self) -> dict[str, Any]:
        """Connector identity for the snapshot fingerprint.

        The resolved path, not the file contents - contents are already covered
        by the trace content hashes the fingerprint folds in.
        """
        return {"type": "jsonl", "path": str(self.path.resolve())}
