"""The Python entry point.

One constructor, three stages, no database. The stages stay separate because
they fail for different reasons and cost different amounts: judging spends judge
calls, compiling spends nothing, fine-tuning spends a GPU.

What is tested here is the coercion - every argument accepts the shape you
already have - and that the result is the same object the CLI path produces,
because two ways in must not mean two engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalloop import EvalLoop, registry_check, text_matches_tools, tool_call_outcome
from evalloop.contracts import JudgeConfig, Trace

REGISTRY: dict[str, Any] = {
    "nodes": {"refunds": {"tools": ["issue_refund", "lookup_order"]}},
    "tools": {
        "issue_refund": {
            "description": "Refund an order. Irreversible.",
            "arguments": {"order_id": {"type": "string", "required": True}},
            "side_effecting": True,
        },
        "lookup_order": {
            "description": "Read-only order details.",
            "arguments": {"order_id": {"type": "string", "required": True}},
            "side_effecting": False,
        },
    },
}

TRACE: dict[str, Any] = {
    "trace_id": "t1",
    "input": {"user_request": "refund please"},
    "output": {
        "text": "Refunded.",
        "tool_calls": [{"name": "issue_refund", "arguments": {"order_id": "O1"}}],
    },
}


def _loop(**overrides: Any) -> EvalLoop:
    kwargs: dict[str, Any] = {
        "judge": "mock:stub-1",
        "tools": REGISTRY,
        "traces": [TRACE],
    }
    kwargs.update(overrides)
    return EvalLoop(**kwargs)


# --- it runs at all ---


def test_judging_needs_no_database_and_no_yaml() -> None:
    report = _loop().judge()
    assert report.tools.traces == 1
    assert report.results


def test_the_default_checks_are_the_four_that_need_no_ground_truth() -> None:
    ids = {r.evaluator_id for r in _loop().judge().results}
    assert ids == {
        "tool_registry_check",
        "tool_call_outcome",
        "tool_selection",
        "text_matches_tools",
    }


def test_checks_can_be_chosen_explicitly() -> None:
    report = _loop(checks=[registry_check(), tool_call_outcome()]).judge()
    assert {r.evaluator_id for r in report.results} == {
        "tool_registry_check",
        "tool_call_outcome",
    }


def test_a_judged_check_without_a_registry_still_builds() -> None:
    """`text_matches_tools` reads a registry when there is one and works without."""
    report = EvalLoop(judge="mock:stub-1", traces=[TRACE], checks=[text_matches_tools()]).judge()
    assert report.results


def test_limit_caps_the_traces_evaluated() -> None:
    report = _loop(traces=[TRACE, {**TRACE, "trace_id": "t2"}]).judge(limit=1)
    assert report.tools.traces == 1


# --- coercion: every argument takes the shape you already have ---


def test_judge_accepts_provider_colon_model() -> None:
    assert _loop().judges["default"].model == "stub-1"


def test_judge_accepts_a_full_config() -> None:
    loop = _loop(judge=JudgeConfig(provider="mock", model="stub-2", max_tokens=64))
    assert loop.judges["default"].max_tokens == 64


def test_judge_accepts_a_named_map_for_several_judges() -> None:
    loop = _loop(
        judge={
            "default": JudgeConfig(provider="mock", model="cheap"),
            "strong": JudgeConfig(provider="mock", model="expensive"),
        },
    )
    assert set(loop.judges) == {"default", "strong"}


def test_a_judge_string_without_a_model_is_refused() -> None:
    with pytest.raises(ValueError, match="provider:model"):
        _loop(judge="anthropic")


def test_an_unknown_provider_is_named() -> None:
    with pytest.raises(ValueError, match="unknown judge provider"):
        _loop(judge="acme:model-1")


def test_tools_accept_a_path(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "tools.yaml"
    path.write_text(yaml.safe_dump(REGISTRY), encoding="utf-8")
    assert _loop(tools=path).registry is not None


def test_traces_accept_a_jsonl_path(tmp_path: Path) -> None:
    import json

    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(TRACE) + "\n", encoding="utf-8")
    assert len(_loop(traces=path).traces) == 1


def test_traces_accept_trace_objects() -> None:
    assert len(_loop(traces=[Trace.model_validate(TRACE)]).traces) == 1


def test_raw_rows_are_mapped_when_a_mapping_is_given() -> None:
    """The customer's own column names, same as project.yaml."""
    loop = _loop(
        traces=[{"id": "x1", "question": "refund please", "reply": "Refunded."}],
        mapping={"trace_id": "id", "input.user_request": "question", "output.text": "reply"},
    )
    assert loop.traces[0].trace_id == "x1"
    assert loop.traces[0].input.user_request == "refund please"


def test_a_malformed_trace_fails_at_construction_not_mid_run() -> None:
    """Validated rather than trusted: a typo'd key is an error here instead of
    a mystery `not applicable` three checks later."""
    with pytest.raises(Exception, match=r"trace_id|validation"):
        _loop(traces=[{"input": {}, "output": {}}])


def test_a_bad_check_configuration_is_raised_at_construction() -> None:
    with pytest.raises(ValueError, match="tool registry"):
        EvalLoop(judge="mock:stub-1", traces=[TRACE], checks=[registry_check()])


# --- the report ---


def test_failures_excludes_rows_that_never_ran() -> None:
    """A check with nothing to compare against is neither a pass nor a failure,
    and counting it as one is how a report starts lying."""
    report = _loop().judge()
    assert all(r.is_failure for r in report.failures())


def test_failures_can_be_scoped_to_one_check() -> None:
    report = _loop().judge()
    assert all(r.evaluator_id == "tool_selection" for r in report.failures("tool_selection"))


def test_markdown_is_returned_and_optionally_written(tmp_path: Path) -> None:
    report = _loop().judge()
    destination = tmp_path / "nested" / "report.md"
    text = report.to_markdown(destination)
    assert text.startswith("# Tool calls")
    assert destination.read_text(encoding="utf-8") == text


def test_cost_is_reported() -> None:
    assert _loop().judge().cost_usd > 0


# --- the dataset ---


def test_a_failure_reaches_the_compiler(tmp_path: Path) -> None:
    """Stage 2, from the report stage 1 produced.

    The mock judge picks the alphabetically first tool - `issue_refund` - so a
    trace that called `lookup_order` disagrees and becomes a candidate pair.
    The mock also proposes empty arguments, which the registry rejects, so what
    this shows is the gate firing rather than a bad row being emitted.
    """
    trace = {
        **TRACE,
        "output": {
            "text": "Looking it up.",
            "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "O1"}}],
        },
    }
    loop = _loop(traces=[trace])
    dataset = loop.dataset(loop.judge(), tmp_path / "feedback.jsonl")

    assert dataset.size == 0
    assert dataset.dropped["proposal_failed_argument_check"] == 1
    assert (tmp_path / "feedback.jsonl").exists()


def test_the_manifest_explains_an_empty_dataset() -> None:
    """An empty dataset is never silent: the reasons are the useful output."""
    loop = _loop()
    manifest = loop.dataset(loop.judge()).manifest()
    assert manifest["rows"] == 0
    assert manifest["dropped_total"] > 0


# --- the stages are separate ---


def test_compiling_twice_costs_no_extra_judge_calls() -> None:
    """Stage 2 takes the report rather than re-judging, so eligibility rules can
    be changed without paying for the judge again."""
    loop = _loop()
    report = loop.judge()
    spent = report.cost_usd

    loop.dataset(report)
    loop.dataset(report, sealed_trace_ids=frozenset({"t1"}))
    assert report.cost_usd == spent


def test_each_stage_reports_its_own_wall_clock() -> None:
    loop = _loop()
    report = loop.judge()
    assert report.elapsed_s >= 0.0
    assert loop.dataset(report).elapsed_s >= 0.0


def test_fine_tuning_is_refused_by_name_not_left_to_fail_inside_trl() -> None:
    loop = _loop()
    dataset = loop.dataset(loop.judge())
    with pytest.raises(NotImplementedError, match="P5"):
        loop.finetune(dataset)
