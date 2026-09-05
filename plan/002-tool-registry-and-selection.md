# 002 — Tool registry and select-mode tool judging

**Status:** accepted, 2026-09-05
**Supersedes:** parts of [`000-build-plan.md`](000-build-plan.md) — P2 deliverable 1 (tool-call matching is no longer the GT-dependent primary check) and P4 deliverable 2 (one target source added). Extends [`001-trusted-judge-architecture.md`](001-trusted-judge-architecture.md) §2 by giving T0 a tool-correctness check that needs no labels.
**Out of scope, deliberately:** business-metric attribution. A question earns its place by catching a wrong tool call, not by being wired to a KPI.

---

## Why this document exists

`001` removed ground truth as a precondition. Tool correctness was the one check that kept requiring it anyway, and nothing in the code says so out loud.

Run the shipped support-bot suite against a production trace with no `expected_tool_calls` column and this is what comes back:

| Evaluator | Result |
|---|---|
| `tool_call_match` | `not_applicable` — "no ground truth at ground_truth.tool_calls" |
| `tool_name_match` | `not_applicable` |
| `policy_followed` | `passed = None` — "measured but not calibrated" |

An agent that refunded a 45-day-old order against a 30-day policy produces three non-verdicts. Verified against the current code: a hallucinated tool name with no GT returns `passed=None`.

`examples/support-bot/traces.jsonl` hides this. Every row carries `expected_tool_calls`, `expected_reply`, and `human_policy_verdict` — a fixture built to exercise every mapping path, reading as a claim about what production data looks like. No product emits those columns. A customer arrives with transcripts and tool calls, and the highest-value check in the system silently abstains on every row.

The fix is not more labels. It is that **the answer space was never given to the system**. A team shipping a tool-calling agent already has their tool definitions written down — that is the schema they pass to the model on every request. Asking for it is asking for an export, not for annotation.

---

## 1. The registry

`tools.yaml`, referenced from `project.yaml`. Node-scoped, because a tool being permitted *somewhere* in the graph is not the same as being permitted *here*.

```yaml
nodes:
  triage:
    tools: [lookup_order, route_to_node]
  refunds:
    tools: [issue_refund, lookup_refund_status, open_warranty_claim, cancel_order]
  warranty:
    tools: [open_warranty_claim, lookup_order]

tools:
  issue_refund:
    description: Refund an order to the original payment method. Irreversible.
    arguments:
      order_id: {type: string, required: true}
      amount:   {type: number, required: true}
    side_effecting: true
    preconditions: ["order.age_days <= 30"]

  open_warranty_claim:
    description: Open a replacement claim for a damaged or faulty item.
    arguments:
      order_id: {type: string, required: true}
      reason:   {type: string, enum: [damaged, late_request, faulty]}
    side_effecting: true

  lookup_refund_status:
    description: Read-only status of an existing refund.
    arguments:
      order_id: {type: string, required: true}
    side_effecting: false
```

`side_effecting` is not decoration. Duplicate-call detection and order-sensitivity only matter for calls that change state; flagging a repeated `lookup_order` is noise.

The registry is hashed into the suite version. Editing a tool description changes what the judge was shown, therefore changes the measurement, therefore must change the hash.

### 1.1 Node attribution

A trace records which node produced each call. Flat single-node agents leave it unset and every check still runs against the union of tools — node scoping degrades to global membership rather than failing.

---

## 2. Three checks, not one

Collapsing tool correctness into a single verdict makes "called the wrong tool" and "called the right tool with wrong arguments" indistinguishable in the report. They are different failures with different fixes.

| id | type | Judge? | GT? | Catches |
|---|---|---|---|---|
| `tool_registry_check` | deterministic | no | no | tool not in registry (hallucination); tool not allowed at this node; arguments invalid against the tool's schema; duplicate side-effecting call; unresolved ID |
| `tool_selection` | llm_question, select-mode | yes | no | wrong tool chosen for this request |
| `text_matches_tools` | llm_question | yes | no | reply asserts an outcome the calls do not support |

`tool_registry_check` takes no `expected` path. It validates a shape, exactly as `json_schema` does today, so it slots into the existing `EvaluatorSpec` with `expected: None` and no contract change.

That matters beyond convenience: it is a **deterministic signal that requires no ground truth**. Rule 10 (every promotion gate contains at least one deterministic condition) and `001` §3.2.1 (the gate must contain one signal the training process could not optimise against) were both, in practice, dependent on a GT column nobody has. They now hold on every customer's first run.

---

## 3. Select, don't grade

The suite today shows the judge what the model did and asks whether it was right:

```
Tools the agent called: [ToolCall(name='issue_refund', arguments={'order_id': 'ORD-8891', 'amount': 79.99}, call_id=None)]
Did the agent follow the refund policy?
```

Two defects. The Python repr is one (§6). The larger one is that this is the anchoring failure `001` §6.2 forbids for human labellers — showing the verdict before asking for it inflates agreement. A judge shown a decision rationalises it.

**Select-mode inverts this.** The judge sees the request and the catalogue, never `output.tool_calls` and never `output.text`, and picks. Code compares afterward.

```
Available tools at node `refunds`:

  issue_refund(order_id: string, amount: number)
      Refund an order to the original payment method. Irreversible.
  open_warranty_claim(order_id: string, reason: "damaged"|"late_request"|"faulty")
      Open a replacement claim for a damaged or faulty item.
  lookup_refund_status(order_id: string)
      Read-only status of an existing refund.
  cancel_order(order_id: string)
      Cancel an order that has not yet shipped.

Policy: Refunds are permitted within 30 days of purchase.

Customer said:
  "I ordered a blender 45 days ago and it arrived broken. I want my money back."

Which tool should be called? If none should be, answer "none".
```

Response schema, enum-constrained to the node's allowlist:

```json
{"type": "object",
 "properties": {
   "best":       {"enum": ["issue_refund", "lookup_refund_status", "open_warranty_claim", "cancel_order", "none"]},
   "acceptable": {"type": "array", "items": {"enum": ["issue_refund", "lookup_refund_status", "open_warranty_claim", "cancel_order", "none"]}},
   "reason":     {"type": "string"}},
 "required": ["best", "acceptable", "reason"]}
```

Three consequences, each load-bearing:

**Blind.** The judge cannot rationalise a decision it was not shown.

**Closed answer space.** This is classification over 5–15 labels, not free generation. Parse failures collapse, self-consistency becomes measurable per trace, and a cheap model becomes viable — which matters when the check runs over every production trace rather than a sample.

**`none` is first-class.** Calling nothing is frequently correct: sb-0801 refuses an out-of-window refund with no tool at all. Omit `none` from the enum and the judge is forced to invent a call, manufacturing a disagreement out of correct behaviour.

### 3.1 Verdict is set membership

`passed = actual_tool ∈ acceptable`, not `actual_tool == best`.

Several tools are often defensible — looking up the order before opening a claim is not wrong. Judge error is asymmetric (`001` §5.3): a false **fail** mints a training pair teaching the model to stop doing something correct. `acceptable` is the cheapest available suppressor of exactly that error, and it costs one extra field in a schema the judge is already filling in.

### 3.2 Ambiguity is not failure

Sample the judge *n* times. If it selects different tools across samples, the **case** is ambiguous — the model has not been shown to be wrong.

```
self_consistency < threshold  →  not_applicable, not False
```

This reuses the `self_consistency` probe already specced in `001` §3 (P3a) as a per-trace confidence signal rather than only an aggregate judge-health number. A judge that cannot decide has caught nothing, and recording that as a model failure is how a wrong-tool report loses its credibility on first read.

---

## 4. What comes out

Per trace, `normalized_prediction` is the judge's `best`, compared against the call that actually happened. Aggregated over a snapshot, with no ground truth anywhere in the inputs:

```
Wrong tool selections — support-bot, snapshot 2026-09-03, n=4,182, 0 labels

  called                  judge says             n    self-consist
  issue_refund         →  open_warranty_claim   34        0.94
  issue_refund         →  none                  12        0.91
  (none)               →  lookup_refund_status   9        0.88
  cancel_order,refund  →  cancel_order           6        0.96
  ───────────────────────────────────────────────────────────────
  ambiguous (judge unstable across samples)     23
```

This table is the product. Inputs: `tools.yaml` and production traces. That is the whole ask.

---

## 5. Contract changes

| Change | Where | Note |
|---|---|---|
| `ToolSpec`, `NodeSpec`, `ToolRegistry`; loader; hashed into the suite version | new `contracts/tools.py` | |
| `ToolCall.result` / `.error` | `contracts/trace.py` | `tool_call_exec` is promised at T0 (`001` §2) and the model cannot currently hold whether a call succeeded |
| Node attribution per call or per trace | `contracts/trace.py` | optional; unset degrades to global membership |
| `tools_available` wired to the registry, or removed | `contracts/trace.py:114` | declared in P0, referenced by exactly one line in the repo — its own declaration |
| `tool_registry_check` evaluator | new `evaluate/deterministic/registry_check.py` | `expected: None`, like `json_schema` |
| Select-mode: catalogue injection, enum schema generation, n-sampling | `evaluate/llm/question.py` | |
| Render `inputs` values through `canonical_json` when not scalar | `evaluate/llm/question.py` | §6 |
| `judge_tool_selection` target source | P4 feedback compiler | §8 |

---

## 6. Two defects this closes

**The judge reads Python reprs.** `render_prompt` resolves `output.tool_calls` to `list[ToolCall]` and Jinja stringifies it, so the judge is shown `call_id=None` and pydantic constructor syntax. Non-scalar `inputs` values must render through `canonical_json`.

**Prompt content is outside the evaluator version hash.** `version_payload()` covers the question text and the `inputs` *paths*, not what those paths resolve to. Adding a field to `ToolCall` changes every rendered judge prompt. The cache key includes the prompt, so answers are correctly re-fetched — but `evaluator_version` is unchanged, so runs from before and after compare as though nothing moved. The registry makes this worse (a description edit changes every prompt), which is why the registry hash folds into the suite version.

---

## 7. What this does not cover

Stated here rather than discovered at iteration four, in the manner of `001` §3.3.

**Argument values.** The registry checks argument *validity*, never *correctness*:

```
issue_refund(order_id="ORD-8891", amount=79.99)   # schema-valid, node-allowed, correct
issue_refund(order_id="ORD-8891", amount=89.99)   # schema-valid, node-allowed, wrong
```

Neither registry nor judge knows the real order total. Three routes, in preference order: a `preconditions` rule reading a trace metadata field; executable verification against a sandbox (`executably_verified_correction`); or ground truth on a scenario pack. Selection is judgeable from a catalogue; arithmetic is not.

**Call ordering.** `ignore_order: true` erases sequence by design, which is right for commutative pairs and wrong for causal ones. Order-sensitivity is a property of a tool pair and has no home in the registry yet.

**Routing.** A wrong node that then performs its own job correctly passes every check here. Multi-node routing evaluation needs the graph, not just the per-node allowlist.

---

## 8. Where ground truth still lives

Removing the *dependency* is not removing the *field*. `GroundTruth` stays exactly as it is, with two jobs and no third:

1. **Calibration labels** — the `001` §6 loop, ~150 labels, which is what makes κ and FAIL-class precision computable. `001` §5.3 gates training data on `judgecard_fail_precision_above`; delete the labels and that condition is uncomputable.
2. **Scenario-pack targets** — cases the team authored, where they genuinely know the answer.

`tool_call_match` is demoted accordingly: no longer the default suite's primary tool check, retained for scenario packs where an `expected` path actually exists. Every evaluator already returns `not_applicable` on missing GT, so nothing in that machinery changes.

**New P4 target source.** `000` P4.2 lists six, `001` §5.1 adds `judge_preference_pair`. Add `judge_tool_selection`: the judge's `best` on a trace where the actual call fell outside `acceptable`. Same admission gate as any judge-derived source — `judge_health` must pass — plus one more, because the answer space is closed and the confidence signal is free:

```yaml
require:
  - judge_health.position_flip_rate_below: 0.15
  - judge_health.self_consistency_above: 0.80
  - trace.tool_selection_self_consistency_above: 0.90   # per-trace, not aggregate
```

Rows carry `signal_provenance: judge` and the registry hash, so a dataset built against an old catalogue is identifiable after the fact.

---

## 9. Phase placement and acceptance

Ships inside P2 (evaluation engine); the registry contract lands in P0.

**Acceptance:** point EvalLoop at a JSONL of production traces containing **no ground-truth columns at all**, plus `tools.yaml`, and `evalloop evaluate` produces the §4 wrong-tool table — with `tool_registry_check` returning real pass/fail verdicts, not `not_applicable`, on every row.

The example dataset splits to prove it: `examples/support-bot/traces.jsonl` loses every `expected_*` and `human_*` column and becomes what a customer actually pastes; the authored cases move to a scenario pack, where their expectations belong.

---

## Rules added

15. Tool correctness never requires ground truth. A registry is config, not annotation.
16. A judge asked to assess a tool call is never shown the tool call.
17. A judge that disagrees with itself across samples abstains; it does not fail the trace.
18. The tool registry is hashed into the suite version — a description edit is a new measurement.

Each is enforced by a test, not by convention.
