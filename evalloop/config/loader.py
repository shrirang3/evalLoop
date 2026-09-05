"""Load a YAML config, work out which schema it is, and validate it.

The output is a list of `Problem`s rather than a raised exception, because a
config with six mistakes should report six mistakes. Failing on the first one
turns fixing a file into six edit-run cycles.
"""

from __future__ import annotations

import difflib
import types
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from evalloop.config.yaml_positions import Position, YamlSource
from evalloop.contracts.judgeconf import JudgeConfig
from evalloop.contracts.project import ProjectConfig, check_integrity
from evalloop.contracts.promotion import PromotionConfig
from evalloop.contracts.suite import EvalSuite
from evalloop.contracts.tools import ToolRegistry
from evalloop.contracts.training import TrainingConfig

__all__ = [
    "SCHEMAS",
    "ConfigKind",
    "LoadedConfig",
    "Problem",
    "detect_kind",
    "load_config",
    "validate_paths",
]


class ConfigKind(StrEnum):
    PROJECT = "project"
    SUITE = "eval-suite"
    JUDGES = "judges"
    TOOLS = "tools"
    TRAINING = "training"
    PROMOTION = "promotion"


SCHEMAS: dict[ConfigKind, type[BaseModel]] = {
    ConfigKind.PROJECT: ProjectConfig,
    ConfigKind.SUITE: EvalSuite,
    ConfigKind.TOOLS: ToolRegistry,
    ConfigKind.TRAINING: TrainingConfig,
    ConfigKind.PROMOTION: PromotionConfig,
    # JUDGES is a mapping of name -> JudgeConfig rather than a single model, so
    # it is validated separately in _validate_judges.
}

_FILENAME_HINTS: dict[str, ConfigKind] = {
    "project": ConfigKind.PROJECT,
    "eval-suite": ConfigKind.SUITE,
    "eval_suite": ConfigKind.SUITE,
    "suite": ConfigKind.SUITE,
    "judges": ConfigKind.JUDGES,
    "judge": ConfigKind.JUDGES,
    "tools": ConfigKind.TOOLS,
    "training": ConfigKind.TRAINING,
    "train": ConfigKind.TRAINING,
    "promotion": ConfigKind.PROMOTION,
    "promote": ConfigKind.PROMOTION,
}

# Content sniffing, for files not named after their kind. Each kind is
# identified by a key that only it has.
_CONTENT_MARKERS: tuple[tuple[str, ConfigKind], ...] = (
    ("evaluators", ConfigKind.SUITE),
    ("source", ConfigKind.PROJECT),
    ("judges", ConfigKind.JUDGES),
    # `tools` is checked after `judges` and `source`: only the registry has it
    # at the top level, but a future file that merely mentions tools should not
    # win the sniff before the kinds that are identified by their own key.
    ("tools", ConfigKind.TOOLS),
    ("dataset_id", ConfigKind.TRAINING),
    ("slices", ConfigKind.PROMOTION),
)


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with one file, located precisely enough to go and fix."""

    file: str
    message: str
    path: str = ""
    position: Position | None = None
    hint: str | None = None
    snippet: str = ""

    @property
    def line(self) -> int | None:
        return self.position.line if self.position else None


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    kind: ConfigKind
    model: Any
    source: YamlSource


def detect_kind(path: Path, data: Any) -> ConfigKind | None:
    """Identify a config by filename, then by a key only its kind has.

    Filename first because it is what the user meant; content sniffing second so
    `voice-agent.yaml` still works. Returning None is a real answer - guessing
    wrong produces a page of errors against the wrong schema, which is far worse
    than saying "tell me what this is".
    """
    stem = path.stem.lower()
    if (kind := _FILENAME_HINTS.get(stem)) is not None:
        return kind
    for token, kind in _FILENAME_HINTS.items():
        if token in stem:
            return kind

    if isinstance(data, dict):
        for marker, kind in _CONTENT_MARKERS:
            if marker in data:
                return kind
    return None


def load_config(
    path: Path,
    *,
    kind: ConfigKind | None = None,
) -> tuple[LoadedConfig | None, list[Problem]]:
    """Parse and validate one file. Returns the model, or every problem found."""
    file = str(path)

    if not path.exists():
        return None, [Problem(file=file, message="file not found")]

    try:
        source = YamlSource.from_path(file)
        data = source.data()
    except yaml.YAMLError as exc:
        return None, [_from_yaml_error(file, exc)]

    if data is None:
        return None, [Problem(file=file, message="file is empty")]

    resolved = kind or detect_kind(path, data)
    if resolved is None:
        return None, [
            Problem(
                file=file,
                message="cannot tell which kind of config this is",
                hint=(
                    "name the file after its kind (project.yaml, eval-suite.yaml, "
                    "judges.yaml, training.yaml, promotion.yaml) or pass --as"
                ),
            )
        ]

    if resolved is ConfigKind.JUDGES:
        return _validate_judges(file, source, data)

    schema = SCHEMAS[resolved]
    try:
        model = schema.model_validate(data)
    except ValidationError as exc:
        return None, _from_validation_error(file, source, exc, schema)

    return LoadedConfig(kind=resolved, model=model, source=source), []


def validate_paths(
    paths: list[Path],
    *,
    kind: ConfigKind | None = None,
) -> tuple[list[LoadedConfig], list[Problem]]:
    """Validate several files, then the rules that only hold between them.

    Cross-file checks are the reason this takes a list. `require_distinct_
    providers` compares a base model in project.yaml against judges in
    judges.yaml; neither file can catch it alone.
    """
    loaded: list[LoadedConfig] = []
    problems: list[Problem] = []

    for path in paths:
        config, file_problems = load_config(path, kind=kind)
        problems.extend(file_problems)
        if config is not None:
            loaded.append(config)

    problems.extend(_cross_file_problems(loaded))
    return loaded, problems


def _cross_file_problems(loaded: list[LoadedConfig]) -> list[Problem]:
    project = next((c for c in loaded if c.kind is ConfigKind.PROJECT), None)
    judges = next((c for c in loaded if c.kind is ConfigKind.JUDGES), None)
    if project is None or judges is None:
        return []

    violations = check_integrity(project.model, judges.model)
    return [
        Problem(
            file=str(project.source.filename or "project.yaml"),
            message=violation,
            path="integrity",
            position=project.source.locate(("integrity",)),
        )
        for violation in violations
    ]


def _validate_judges(
    file: str,
    source: YamlSource,
    data: Any,
) -> tuple[LoadedConfig | None, list[Problem]]:
    """judges.yaml is `judges: {name: config}`, so each entry validates on its own."""
    if not isinstance(data, dict) or "judges" not in data:
        return None, [
            Problem(
                file=file,
                message="judges.yaml must have a top-level 'judges' mapping",
                position=source.locate(()),
            )
        ]

    entries = data["judges"]
    if not isinstance(entries, dict) or not entries:
        return None, [
            Problem(
                file=file,
                message="'judges' must be a non-empty mapping of name to configuration",
                path="judges",
                position=source.locate(("judges",)),
            )
        ]

    judges: dict[str, JudgeConfig] = {}
    problems: list[Problem] = []
    for name, config in entries.items():
        try:
            judges[str(name)] = JudgeConfig.model_validate(config)
        except ValidationError as exc:
            problems.extend(
                _from_validation_error(file, source, exc, JudgeConfig, prefix=("judges", str(name)))
            )

    if problems:
        return None, problems
    return LoadedConfig(kind=ConfigKind.JUDGES, model=judges, source=source), []


def _from_yaml_error(file: str, exc: yaml.YAMLError) -> Problem:
    """A syntax error, which is different from a schema error: nothing parsed."""
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    position = Position(line=mark.line + 1, column=mark.column + 1, exact=True) if mark else None
    detail = getattr(exc, "problem", None) or str(exc).splitlines()[0]
    return Problem(file=file, message=f"YAML syntax error: {detail}", position=position)


def _from_validation_error(
    file: str,
    source: YamlSource,
    exc: ValidationError,
    schema: type[BaseModel],
    *,
    prefix: tuple[Any, ...] = (),
) -> list[Problem]:
    problems: list[Problem] = []
    for error in exc.errors():
        loc = prefix + tuple(error["loc"])
        position = source.locate(loc)
        # Prefer the segments that exist in the file: a path containing a
        # discriminator tag looks like YAML but cannot be found in it.
        display = position.matched if position and position.matched else loc
        problems.append(
            Problem(
                file=file,
                message=_humanize(error, loc),
                path=_render_path(display),
                position=position,
                hint=_hint_for(error, schema, prefix),
                snippet=source.line_text(position.line) if position else "",
            )
        )
    return problems


def _render_path(loc: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for segment in loc:
        if isinstance(segment, int):
            parts.append(f"[{segment}]")
        else:
            parts.append(f".{segment}" if parts else str(segment))
    return "".join(parts)


def _humanize(error: ErrorDetails, loc: tuple[Any, ...]) -> str:
    """Pydantic's wording, adjusted where it is unhelpful out of context.

    A missing key names itself in the message, because it has no line of its own
    - the displayed path can only point at the enclosing block, so "required key
    is missing" there would leave the reader guessing which one.
    """
    kind = error["type"]
    name = str(loc[-1]) if loc else ""
    if kind == "extra_forbidden":
        return "unknown key"
    if kind == "missing":
        return f"required key '{name}' is missing" if name else "required key is missing"
    return str(error["msg"])


def _hint_for(
    error: ErrorDetails,
    schema: type[BaseModel],
    prefix: tuple[Any, ...],
) -> str | None:
    """For an unknown key, suggest the field it was probably meant to be.

    This is the whole point of forbidding unknown keys rather than ignoring
    them: `stratergy` becomes a question with an answer instead of a setting
    that silently did nothing.
    """
    if error["type"] != "extra_forbidden":
        return None

    loc = tuple(error["loc"])
    if not loc:
        return None
    bad_key = str(loc[-1])

    model = _model_at(schema, loc[:-1])
    if model is None:
        return None

    candidates = list(model.model_fields)
    close = difflib.get_close_matches(bad_key, candidates, n=1, cutoff=0.6)
    if close:
        return f"did you mean '{close[0]}'?"
    if candidates:
        return f"valid keys here: {', '.join(sorted(candidates))}"
    return None


def _model_at(schema: type[BaseModel], loc: tuple[Any, ...]) -> type[BaseModel] | None:
    """Follow a location path through annotations to the model that owns it.

    Only handles what the config shapes actually use - nested models, lists,
    dicts, and unions of models. Returns None rather than guessing, in which
    case the caller simply offers no suggestion.
    """
    current: Any = _strip_annotated(schema)
    for segment in loc:
        if isinstance(current, type) and issubclass(current, BaseModel):
            field = current.model_fields.get(str(segment))
            if field is None:
                # Not a field of this model, so the path cannot be followed and
                # any suggestion would name the wrong model's keys. A tag
                # segment never lands here - tags appear against a union, which
                # `_member_for_tag` resolves below.
                return None
            current = _strip_annotated(field.annotation)
            continue

        # Tag resolution first. Against a union, `_unwrap` can only guess - it
        # would return the first member - whereas the tag says exactly which
        # member the error came from.
        tagged = _member_for_tag(current, segment)
        if tagged is not None:
            current = tagged
            continue

        stepped = _unwrap(current, segment)
        if stepped is None:
            return None
        current = _strip_annotated(stepped)

    if isinstance(current, type) and issubclass(current, BaseModel):
        return current
    return _first_model(current)


def _strip_annotated(annotation: Any) -> Any:
    """Unwrap `Annotated[X, ...]` to X. Discriminated unions are declared that way."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _member_for_tag(annotation: Any, segment: Any) -> type[BaseModel] | None:
    """Find the union member a discriminator tag names.

    `("evaluators", 0, "llm_question", "questoin")` - the third segment is the
    tag identifying which member failed. Resolving it is what lets the hint name
    the right set of valid keys instead of giving up.
    """
    if not isinstance(segment, str):
        return None
    for arg in get_args(_strip_annotated(annotation)):
        candidate = _strip_annotated(arg)
        if not (isinstance(candidate, type) and issubclass(candidate, BaseModel)):
            continue
        field = candidate.model_fields.get("type")
        if field is not None and segment in get_args(field.annotation):
            return candidate
    return None


def _unwrap(annotation: Any, segment: Any) -> Any:
    annotation = _strip_annotated(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Both spellings: `Optional[X]` gives typing.Union, `X | None` gives
    # types.UnionType. They are distinct objects, and only checking one silently
    # drops every hint under an optional field.
    if origin is Union or origin is types.UnionType:
        for arg in args:
            if arg is type(None):
                continue
            if isinstance(segment, int) or get_origin(arg) is not None:
                unwrapped = _unwrap(arg, segment)
                if unwrapped is not None:
                    return unwrapped
            elif isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
        return None

    if origin in (list, tuple) and isinstance(segment, int):
        return args[0] if args else None
    if origin is dict:
        return args[1] if len(args) > 1 else None
    return None


def _first_model(annotation: Any) -> type[BaseModel] | None:
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None
