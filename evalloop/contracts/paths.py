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

__all__ = ["MISSING", "Missing", "path_exists", "resolve_path", "set_path", "split_path"]


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


def set_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Write `value` at `path`, creating intermediate containers as needed.

    The inverse of :func:`resolve_path`, and the reason the ingest mapping can
    be written the way a user thinks about it - `input.user_request:
    user_transcript` reads as "our field comes from their column" without the
    user having to nest anything by hand.

    Lists grow to fit an index rather than raising, since `output.artifacts[0]`
    is a perfectly reasonable thing to map into an empty trace. Gaps are filled
    with None, which then fails Trace validation with a real message if the
    caller skipped an element they needed.
    """
    parts = split_path(path)
    if isinstance(parts[0], int):
        raise ValueError(f"path {path!r} cannot start with a list index")

    container: Any = target
    for index, part in enumerate(parts[:-1]):
        nxt = parts[index + 1]
        child = _ensure_child(container, part, want_list=isinstance(nxt, int))
        container = child

    _assign(container, parts[-1], value)


def _ensure_child(container: Any, part: str | int, *, want_list: bool) -> Any:
    """Fetch container[part], creating it as a dict or list if absent."""
    empty: Any = [] if want_list else {}

    if isinstance(part, int):
        if not isinstance(container, list):
            raise TypeError(f"cannot index into {type(container).__name__} with [{part}]")
        _grow(container, part)
        if container[part] is None:
            container[part] = empty
        return container[part]

    if not isinstance(container, dict):
        raise TypeError(f"cannot set key {part!r} on {type(container).__name__}")
    existing = container.get(part)
    if existing is None:
        container[part] = empty
        return container[part]
    return existing


def _assign(container: Any, part: str | int, value: Any) -> None:
    if isinstance(part, int):
        if not isinstance(container, list):
            raise TypeError(f"cannot index into {type(container).__name__} with [{part}]")
        _grow(container, part)
        container[part] = value
        return
    if not isinstance(container, dict):
        raise TypeError(f"cannot set key {part!r} on {type(container).__name__}")
    container[part] = value


def _grow(items: list[Any], index: int) -> None:
    if index < 0:
        raise ValueError("negative list indices cannot be written to")
    while len(items) <= index:
        items.append(None)
