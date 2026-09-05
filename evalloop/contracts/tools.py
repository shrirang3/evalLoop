"""The tool registry: what the agent is allowed to call, and where.

This is the artifact that removes ground truth from tool correctness
(`plan/002-tool-registry-and-selection.md`). A team shipping a tool-calling
agent already wrote these definitions - it is the schema they hand the model on
every request - so asking for `tools.yaml` is asking for an export, not for
annotation. That distinction is the whole reason this file exists.

Two decisions worth reading before the code:

**Node scoping.** A tool being permitted somewhere in the graph is not the same
as being permitted here. `open_warranty_claim` at the `warranty` node is correct
and at the `triage` node is a scope violation, and only a per-node allowlist can
tell those apart. Flat single-node agents declare no nodes at all and every
check degrades to global membership rather than failing.

**`side_effecting` is three-valued.** `None` means the author has not said, and
duplicate detection is skipped for that tool with the reason recorded. Defaulting
it to True would flag a repeated `lookup_order` as a fault - a false failure, the
expensive direction (`plan/001` section 5.3). Defaulting it to False would
silently pass a double refund, which is the failure this check exists to catch.
Neither default is honest, so unset stays unset and says so.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalloop.contracts.trace import canonical_json

__all__ = [
    "NONE_CHOICE",
    "ArgumentSpec",
    "NodeSpec",
    "ToolRegistry",
    "ToolSpec",
]

_STRICT = ConfigDict(extra="forbid", frozen=True)

NONE_CHOICE = "none"
"""The answer a judge gives when no tool should have been called.

First-class on purpose. Calling nothing is frequently correct - refusing an
out-of-window refund needs no tool at all - and a judge whose answer space
excludes it is forced to invent a call, manufacturing a disagreement out of
correct behaviour.
"""

JsonType = Literal["string", "number", "integer", "boolean", "array", "object"]

_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


class ArgumentSpec(BaseModel):
    """One argument of one tool."""

    model_config = _STRICT

    type: JsonType
    required: bool = False
    enum: list[Any] | None = None
    description: str | None = None

    def type_matches(self, value: Any) -> bool:
        # bool is a subclass of int in Python and is not an integer here: a tool
        # declaring `count: integer` and receiving `True` has been called wrong.
        if isinstance(value, bool) and self.type != "boolean":
            return False
        if self.type == "number" and isinstance(value, bool):
            return False
        return isinstance(value, _PYTHON_TYPES[self.type])

    def render(self, name: str) -> str:
        """Signature fragment for the judge catalogue: `reason: "damaged"|"faulty"`."""
        shown = "|".join(canonical_json(v) for v in self.enum) if self.enum else self.type
        return f"{name}: {shown}"


class ToolSpec(BaseModel):
    """One tool the agent can call."""

    model_config = _STRICT

    description: str = Field(min_length=1)
    """Shown to the judge verbatim. Editing it changes the measurement, which is
    why it is inside `registry_hash`."""

    arguments: dict[str, ArgumentSpec] = Field(default_factory=dict)

    side_effecting: bool | None = None
    """Whether calling this twice does something twice. See the module docstring."""

    allow_extra_arguments: bool = False
    """A model passing an argument the registry does not declare is usually a
    hallucinated parameter. Teams with a passthrough argument set this True."""

    preconditions: list[str] = Field(default_factory=list)
    """Expressions that must hold for this call to be correct, e.g.
    `order.age_days <= 30`. Stored and hashed now; the expression engine is not
    implemented, so `tool_registry_check` refuses to be configured with
    `check_preconditions: true` rather than silently passing everything."""

    @model_validator(mode="after")
    def _preconditions_are_nonempty(self) -> ToolSpec:
        for expression in self.preconditions:
            if not expression.strip():
                raise ValueError("preconditions must not contain an empty expression")
        return self

    def signature(self, name: str) -> str:
        """`issue_refund(order_id: string, amount: number)`."""
        args = ", ".join(spec.render(arg) for arg, spec in self.arguments.items())
        return f"{name}({args})"


class NodeSpec(BaseModel):
    """One node of the agent graph and the tools it may call."""

    model_config = _STRICT

    tools: list[str] = Field(min_length=1)
    description: str | None = None


class ToolRegistry(BaseModel):
    """One `tools.yaml`.

    Hashed into the suite version. A description edit changes what the judge was
    shown, therefore changes the measurement, therefore must change the hash -
    otherwise a run from before and a run from after compare as though nothing
    moved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tools: dict[str, ToolSpec] = Field(min_length=1)
    nodes: dict[str, NodeSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _nodes_reference_known_tools(self) -> ToolRegistry:
        for node, spec in self.nodes.items():
            for tool in spec.tools:
                if tool not in self.tools:
                    known = ", ".join(sorted(self.tools))
                    raise ValueError(
                        f"node '{node}' allows tool '{tool}', which is not declared "
                        f"under `tools`; declared: {known}"
                    )
        return self

    @model_validator(mode="after")
    def _none_is_not_a_tool_name(self) -> ToolRegistry:
        # The judge's answer space is the allowlist plus NONE_CHOICE. A tool
        # actually called `none` would make "no tool" and "the none tool"
        # indistinguishable in every result row.
        if NONE_CHOICE in self.tools:
            raise ValueError(
                f"'{NONE_CHOICE}' is reserved: it is the judge's answer for "
                "'no tool should have been called' and cannot also be a tool name"
            )
        return self

    def has(self, tool: str) -> bool:
        return tool in self.tools

    def allowed(self, node: str | None) -> frozenset[str]:
        """Tools callable at `node`, or every declared tool when node is unset.

        An unknown node is not silently permissive - callers check `knows_node`
        first and report it, since a node name that is not in the registry is a
        config error or a routing bug, and both deserve to be visible.
        """
        if node is None or node not in self.nodes:
            return frozenset(self.tools)
        return frozenset(self.nodes[node].tools)

    def knows_node(self, node: str | None) -> bool:
        return node is None or node in self.nodes

    def choices(self, node: str | None) -> list[str]:
        """The judge's closed answer space: allowed tools, sorted, plus `none`."""
        return [*sorted(self.allowed(node)), NONE_CHOICE]

    def catalogue(self, node: str | None = None) -> str:
        """The tool list as the judge sees it.

        Sorted, so two runs of the same registry render byte-identical prompts
        and the LLM cache actually hits.
        """
        lines: list[str] = []
        for name in sorted(self.allowed(node)):
            spec = self.tools[name]
            lines.append(f"  {spec.signature(name)}")
            lines.append(f"      {spec.description}")
        return "\n".join(lines)

    def check_arguments(self, tool: str, arguments: dict[str, Any]) -> list[str]:
        """Validate one call's arguments. Returns reasons, empty when valid."""
        spec = self.tools[tool]
        problems: list[str] = []

        for name, argument in spec.arguments.items():
            if name not in arguments:
                if argument.required:
                    problems.append(f"missing required argument '{name}'")
                continue
            value = arguments[name]
            if value is None:
                if argument.required:
                    problems.append(f"required argument '{name}' is null")
                continue
            if not argument.type_matches(value):
                problems.append(
                    f"argument '{name}' should be {argument.type}, got "
                    f"{type(value).__name__} ({value!r})"
                )
                continue
            if argument.enum is not None and value not in argument.enum:
                allowed = ", ".join(canonical_json(v) for v in argument.enum)
                problems.append(f"argument '{name}' is {value!r}, not one of [{allowed}]")

        if not spec.allow_extra_arguments:
            for name in arguments:
                if name not in spec.arguments:
                    problems.append(f"unexpected argument '{name}'")

        return problems

    def registry_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()
