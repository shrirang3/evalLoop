"""Mapping a Pydantic error path back to a YAML line."""

from __future__ import annotations

from evalloop.config import YamlSource

DOC = """suite: support-bot-v1

evaluators:
  - id: tool_call_match
    type: json_match
    actual: output.tool_calls
    expected: ground_truth.tool_calls
    options:
      ignore_order: true

  - id: policy_followed
    type: llm_question
    question: Did the agent follow policy?
"""


def _src() -> YamlSource:
    return YamlSource(DOC, filename="eval-suite.yaml")


def test_locates_a_top_level_key() -> None:
    position = _src().locate(("suite",))
    assert position is not None
    assert (position.line, position.exact) == (1, True)


def test_locates_a_key_inside_a_list_item() -> None:
    position = _src().locate(("evaluators", 0, "expected"))
    assert position is not None
    assert position.line == 7
    assert position.exact


def test_marks_the_key_not_the_value() -> None:
    """The caret has to land under the offending name, not under whatever it was
    set to - the name is what the reader has to change."""
    position = _src().locate(("evaluators", 0, "type"))
    assert position is not None
    line = _src().line_text(position.line)
    assert line[position.column - 1 :].startswith("type:")


def test_negative_list_index() -> None:
    position = _src().locate(("evaluators", -1, "question"))
    assert position is not None
    assert position.line == 13


def test_absent_key_falls_back_to_its_enclosing_block() -> None:
    """A missing key has no line. The enclosing block is the honest answer, and
    `exact=False` says so rather than presenting a guess as precise."""
    position = _src().locate(("evaluators", 0, "nonexistent"))
    assert position is not None
    assert position.line == 4
    assert not position.exact


def test_synthetic_union_tag_is_skipped() -> None:
    """Pydantic reports a discriminated-union error as
    ("evaluators", 0, "json_match", "expected") where the tag has no YAML
    counterpart. Without skipping it, every union error would point at the top
    of its block instead of the offending key."""
    position = _src().locate(("evaluators", 0, "json_match", "expected"))
    assert position is not None
    assert position.line == 7
    assert position.exact
    assert position.matched == ("evaluators", 0, "expected")


def test_a_trailing_miss_is_not_treated_as_synthetic() -> None:
    """Skipping must not swallow a genuinely absent key: nothing follows it, so
    there is nothing to confirm the skip was right."""
    position = _src().locate(("evaluators", 1, "llm_question", "actual"))
    assert position is not None
    assert not position.exact


def test_out_of_range_index_is_inexact() -> None:
    position = _src().locate(("evaluators", 99, "id"))
    assert position is not None
    assert not position.exact


def test_line_text_past_the_end_is_empty() -> None:
    assert _src().line_text(9999) == ""


def test_empty_document_locates_nothing() -> None:
    assert YamlSource("").locate(("anything",)) is None
