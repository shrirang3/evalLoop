"""Map a path inside parsed YAML back to the line it came from.

Pydantic reports errors as a location tuple - `("evaluators", 0, "expcted")` -
which is accurate and useless to a human staring at a 200-line file. This module
turns that tuple into a line and column.

The approach is `yaml.compose()`, which returns the node tree with source marks
still attached, rather than `safe_load()`, which discards them. Walking that tree
along the error path finds the exact token. No extra dependency, and the marks
come from the same parser that produced the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

__all__ = ["Position", "YamlSource", "locate"]


@dataclass(frozen=True, slots=True)
class Position:
    """A 1-indexed source location, as a human counts lines."""

    line: int
    column: int
    exact: bool
    """False when the full path could not be resolved and this is the closest
    enclosing node. A missing key has no line of its own, so the best available
    answer is the block it should have been in - said plainly rather than
    presented as precise."""

    matched: tuple[str | int, ...] = ()
    """The segments that actually exist in the file, with Pydantic's synthetic
    ones dropped. What gets shown to the reader, so the path they are told about
    is one they can find."""


class YamlSource:
    """A YAML document plus the machinery to point back into its text."""

    def __init__(self, text: str, filename: str | None = None) -> None:
        self.text = text
        self.filename = filename
        self.lines = text.splitlines()
        self._root: yaml.Node | None = yaml.compose(text)

    @classmethod
    def from_path(cls, path: str) -> YamlSource:
        with open(path, encoding="utf-8") as handle:
            return cls(handle.read(), filename=path)

    def data(self) -> Any:
        """The parsed document. `None` for an empty file."""
        return yaml.safe_load(self.text)

    def locate(self, path: tuple[Any, ...]) -> Position | None:
        """Position of `path`, or the closest enclosing node it could reach."""
        if self._root is None:
            return None
        return locate(self._root, path)

    def line_text(self, line: int) -> str:
        """Source text of a 1-indexed line, or empty past the end."""
        index = line - 1
        return self.lines[index] if 0 <= index < len(self.lines) else ""


def locate(root: yaml.Node, path: tuple[Any, ...]) -> Position | None:
    """Walk `root` along `path`, returning the deepest position reached.

    Three deliberate behaviours:

    - For a mapping key the *key* token is returned, not its value, so a caret
      lands under the offending name rather than under whatever it was set to.
    - A path that runs out mid-walk yields the last node that did resolve, with
      `exact=False`. Half an answer located in the file beats a precise answer
      the reader has to go looking for.
    - A segment that does not resolve is skipped when a later one does. Pydantic
      inserts synthetic segments that have no YAML counterpart - the tag of a
      discriminated union appears as `("evaluators", 0, "json_match", "expcted")`
      where only `evaluators`, `0`, and `expcted` exist in the file. Without the
      skip, every union error would point at the top of its block instead of the
      offending key.
    """
    node: yaml.Node = root
    marked: yaml.Node = root
    exact = True
    matched: list[str | int] = []

    index = 0
    while index < len(path):
        segment = path[index]
        resolved = _descend(node, segment)
        if resolved is not None:
            marked, node = resolved
            matched.append(segment)
            index += 1
            continue

        # Synthetic segment, but only if something after it does resolve here.
        # A trailing segment that fails is a genuinely absent key, which is
        # approximate and must be reported as such.
        if (
            isinstance(segment, str)
            and index + 1 < len(path)
            and _descend(node, path[index + 1]) is not None
        ):
            index += 1
            continue

        exact = False
        break

    return Position(
        line=marked.start_mark.line + 1,
        column=marked.start_mark.column + 1,
        exact=exact,
        matched=tuple(matched),
    )


def _descend(node: yaml.Node, segment: Any) -> tuple[yaml.Node, yaml.Node] | None:
    """One step down. Returns (node to mark, node to continue from)."""
    if isinstance(node, yaml.MappingNode):
        wanted = str(segment)
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.value == wanted:
                # Mark the key; continue from the value.
                return key_node, value_node
        return None

    if isinstance(node, yaml.SequenceNode):
        if not isinstance(segment, int):
            return None
        items = node.value
        index = segment if segment >= 0 else len(items) + segment
        if not 0 <= index < len(items):
            return None
        item = items[index]
        return item, item

    return None
