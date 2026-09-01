"""Config loading, kind detection, and error reporting.

The acceptance criterion for P0 is here: a typo'd key is rejected, pointing at
the right line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalloop.config import ConfigKind, Problem, detect_kind, load_config, validate_paths

GOOD_SUITE = """suite: support-bot-v1
evaluators:
  - id: tool_call_match
    type: json_match
    actual: output.tool_calls
    expected: ground_truth.tool_calls
  - id: policy_followed
    type: llm_question
    question: Did the agent follow the refund policy?
"""

GOOD_PROJECT = """name: support-bot
source:
  type: jsonl
  path: traces.jsonl
mapping:
  trace_id: id
  input.user_request: user_transcript
models:
  base:
    provider: huggingface
    model: Qwen/Qwen2.5-7B-Instruct
"""

GOOD_JUDGES = """judges:
  default:
    provider: anthropic
    model: claude-sonnet-5
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- kind detection ---


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("project.yaml", ConfigKind.PROJECT),
        ("eval-suite.yaml", ConfigKind.SUITE),
        ("eval_suite.yml", ConfigKind.SUITE),
        ("judges.yaml", ConfigKind.JUDGES),
        ("training.yaml", ConfigKind.TRAINING),
        ("promotion.yaml", ConfigKind.PROMOTION),
        ("support-bot-project.yaml", ConfigKind.PROJECT),
    ],
)
def test_kind_from_filename(filename: str, expected: ConfigKind) -> None:
    assert detect_kind(Path(filename), {}) is expected


def test_kind_from_content_when_the_filename_says_nothing() -> None:
    """So `voice-agent.yaml` still works instead of demanding a rename."""
    assert detect_kind(Path("voice-agent.yaml"), {"evaluators": []}) is ConfigKind.SUITE
    assert detect_kind(Path("anything.yaml"), {"source": {}}) is ConfigKind.PROJECT


def test_unidentifiable_returns_none_rather_than_guessing(tmp_path: Path) -> None:
    """Guessing wrong produces a page of errors against the wrong schema, which
    is far worse than asking."""
    assert detect_kind(Path("data.yaml"), {"hello": 1}) is None

    path = _write(tmp_path, "data.yaml", "hello: 1\n")
    config, problems = load_config(path)
    assert config is None
    assert "cannot tell which kind" in problems[0].message
    assert problems[0].hint is not None and "--as" in problems[0].hint


def test_explicit_kind_overrides_detection(tmp_path: Path) -> None:
    path = _write(tmp_path, "whatever.yaml", GOOD_SUITE)
    config, problems = load_config(path, kind=ConfigKind.SUITE)
    assert problems == []
    assert config is not None and config.kind is ConfigKind.SUITE


# --- happy paths ---


@pytest.mark.parametrize(
    ("name", "text", "kind"),
    [
        ("eval-suite.yaml", GOOD_SUITE, ConfigKind.SUITE),
        ("project.yaml", GOOD_PROJECT, ConfigKind.PROJECT),
        ("judges.yaml", GOOD_JUDGES, ConfigKind.JUDGES),
    ],
)
def test_valid_files_produce_no_problems(
    tmp_path: Path, name: str, text: str, kind: ConfigKind
) -> None:
    config, problems = load_config(_write(tmp_path, name, text))
    assert problems == []
    assert config is not None and config.kind is kind


def test_judges_file_yields_a_name_to_config_mapping(tmp_path: Path) -> None:
    config, _ = load_config(_write(tmp_path, "judges.yaml", GOOD_JUDGES))
    assert config is not None
    assert set(config.model) == {"default"}
    assert config.model["default"].provider == "anthropic"


# --- the P0 acceptance criterion ---


def test_typod_key_is_rejected_at_the_right_line(tmp_path: Path) -> None:
    """`expcted` would otherwise be silently dropped, the check would compare
    against nothing, and the suite would pass everything while looking healthy."""
    text = GOOD_SUITE.replace("    expected:", "    expcted:")
    _, problems = load_config(_write(tmp_path, "eval-suite.yaml", text))

    assert len(problems) == 1
    problem = problems[0]
    assert problem.message == "unknown key"
    assert problem.line == 6
    assert problem.position is not None and problem.position.exact
    assert problem.hint == "did you mean 'expected'?"
    assert "expcted" in problem.snippet


def test_discriminated_union_reports_one_error_per_mistake(tmp_path: Path) -> None:
    """An undiscriminated union makes Pydantic try every member and report all
    their failures, so one typo yielded six errors before `type` became a
    discriminator."""
    text = GOOD_SUITE.replace("    question: Did", "    questoin: Did")
    _, problems = load_config(_write(tmp_path, "eval-suite.yaml", text))

    unknown = [p for p in problems if p.message == "unknown key"]
    assert len(unknown) == 1
    assert unknown[0].hint == "did you mean 'question'?"


def test_missing_key_names_itself_in_the_message(tmp_path: Path) -> None:
    """It has no line of its own, so the path can only point at the enclosing
    block - the message has to say which key."""
    text = GOOD_SUITE.replace("    question: Did the agent follow the refund policy?\n", "")
    _, problems = load_config(_write(tmp_path, "eval-suite.yaml", text))

    assert any("required key 'question' is missing" in p.message for p in problems)


def test_every_problem_is_reported_not_just_the_first(tmp_path: Path) -> None:
    """Six mistakes should be six lines of output, not six edit-run cycles."""
    text = GOOD_SUITE.replace("    expected:", "    expcted:").replace(
        "    question: Did", "    questoin: Did"
    )
    _, problems = load_config(_write(tmp_path, "eval-suite.yaml", text))
    assert len({p.line for p in problems}) >= 2


def test_unknown_evaluator_type_lists_the_valid_ones(tmp_path: Path) -> None:
    text = GOOD_SUITE.replace("type: json_match", "type: fuzzy_match")
    _, problems = load_config(_write(tmp_path, "eval-suite.yaml", text))
    assert any("fuzzy_match" in p.message and "json_match" in p.message for p in problems)


# --- file-level failures ---


def test_missing_file(tmp_path: Path) -> None:
    _, problems = load_config(tmp_path / "absent.yaml")
    assert problems[0].message == "file not found"


def test_empty_file(tmp_path: Path) -> None:
    _, problems = load_config(_write(tmp_path, "project.yaml", ""))
    assert problems[0].message == "file is empty"


def test_yaml_syntax_error_reports_a_line(tmp_path: Path) -> None:
    """A syntax error is different from a schema error: nothing parsed at all,
    so there is no path to report - only a position."""
    path = _write(tmp_path, "eval-suite.yaml", "suite: x\nevaluators:\n  - id: a\n   type: b\n")
    _, problems = load_config(path)
    assert "YAML syntax error" in problems[0].message
    assert problems[0].line == 4


def test_judges_file_without_the_judges_key(tmp_path: Path) -> None:
    _, problems = load_config(_write(tmp_path, "judges.yaml", "default:\n  provider: anthropic\n"))
    assert "top-level 'judges' mapping" in problems[0].message


def test_empty_judges_mapping(tmp_path: Path) -> None:
    _, problems = load_config(_write(tmp_path, "judges.yaml", "judges: {}\n"))
    assert "non-empty mapping" in problems[0].message


def test_bad_judge_entry_is_located_within_its_name(tmp_path: Path) -> None:
    text = GOOD_JUDGES.replace("    model:", "    modle:")
    _, problems = load_config(_write(tmp_path, "judges.yaml", text))
    assert any(p.hint == "did you mean 'model'?" for p in problems)


# --- cross-file ---


def test_shared_provider_between_base_and_judge_is_caught(tmp_path: Path) -> None:
    """Neither file can catch this alone, which is why validation takes a list."""
    project = _write(tmp_path, "project.yaml", GOOD_PROJECT.replace("huggingface", "anthropic"))
    judges = _write(tmp_path, "judges.yaml", GOOD_JUDGES)

    _, problems = validate_paths([project, judges])
    assert len(problems) == 1
    assert "require_distinct_providers" in problems[0].message
    assert problems[0].file.endswith("project.yaml")


def test_distinct_providers_pass(tmp_path: Path) -> None:
    loaded, problems = validate_paths(
        [
            _write(tmp_path, "project.yaml", GOOD_PROJECT),
            _write(tmp_path, "judges.yaml", GOOD_JUDGES),
            _write(tmp_path, "eval-suite.yaml", GOOD_SUITE),
        ]
    )
    assert problems == []
    assert len(loaded) == 3


def test_cross_file_check_is_skipped_when_a_file_is_absent(tmp_path: Path) -> None:
    """Validating only project.yaml must not fabricate a judge to compare with."""
    _, problems = validate_paths([_write(tmp_path, "project.yaml", GOOD_PROJECT)])
    assert problems == []


def test_problem_line_is_none_without_a_position() -> None:
    assert Problem(file="x.yaml", message="file not found").line is None
