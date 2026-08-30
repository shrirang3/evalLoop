"""Dotted-path lookup into nested trace data.

One resolver, shared by every part of EvalLoop that lets a user name a field in
YAML: `ground_truth.has(...)`, the ingest mapping engine (P0.6), evaluator
`inputs` maps (P0.7), and feedback target sources (P4). Keeping it in one place
means `output.artifacts[0].uri` means exactly the same thing everywhere.

Grammar is deliberately small:

    a.b.c        - dict keys
    a[0].b       - list index (negative allowed)
    a.0.b        - list index, dotted form

Anything else is a malformed path and raises, rather than silently missing.
"""

from __future__ import annotations

import re
from typing import Any, Final

__all__ = ["MISSING", "Missing", "path_exists", "resolve_path", "split_path"]


class Missing:
    """Sentinel for "this path is not present", distinct from a stored ``None``.

    A trace may legitimately record ``ground_truth.tone = None``. That is not the
    same as having no ``tone`` key at all, and the feedback compiler has to tell
    those apart before it decides whether a target exists.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Final = Missing()

_SEGMENT: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INDEXED: Final = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[-?\d+\])+)$")
_INDEX: Final = re.compile(r"\[(-?\d+)\]")


def split_path(path: str) -> list[str | int]:
    """Parse a dotted path into a list of dict keys (str) and list indices (int).

    >>> split_path("output.artifacts[0].uri")
    ['output', 'artifacts', 0, 'uri']
    """
    if not path or not path.strip():
        raise ValueError("path is empty")

    parts: list[str | int] = []
    for raw in path.split("."):
        segment = raw.strip()
        if not segment:
            raise ValueError(f"malformed path {path!r}: empty segment")

        # Bare integer segment: the dotted form of a list index.
        if segment.lstrip("-").isdigit():
            parts.append(int(segment))
            continue

        if _SEGMENT.match(segment):
            parts.append(segment)
            continue

        indexed = _INDEXED.match(segment)
        if indexed is None:
            raise ValueError(
                f"malformed path {path!r}: segment {segment!r} is not a key, "
                "an index, or key[index]"
            )
        parts.append(indexed.group(1))
        parts.extend(int(i) for i in _INDEX.findall(indexed.group(2)))

    return parts


def resolve_path(data: Any, path: str) -> Any:
    """Return the value at ``path``, or :data:`MISSING` if it is not present.

    Never raises for absent data - only for a path that is itself malformed.
    Walking into a value that cannot be indexed (a string, an int) returns
    MISSING rather than raising, since that is a data shape problem for the
    caller to report, not a crash.
    """
    current = data
    for part in split_path(path):
        if isinstance(part, int):
            if not isinstance(current, (list, tuple)):
                return MISSING
            try:
                current = current[part]
            except IndexError:
                return MISSING
        else:
            if isinstance(current, dict):
                if part not in current:
                    return MISSING
                current = current[part]
            elif hasattr(current, part) and not isinstance(current, (str, bytes)):
                # Pydantic models and plain objects resolve by attribute, so a
                # path can cross from typed fields into free-form dicts:
                #   "output.tool_calls[0].name"  ->  model, list, model, str
                current = getattr(current, part)
            else:
                return MISSING
    return current


def path_exists(data: Any, path: str) -> bool:
    """True if ``path`` resolves, including to a stored ``None``."""
    return not isinstance(resolve_path(data, path), Missing)
