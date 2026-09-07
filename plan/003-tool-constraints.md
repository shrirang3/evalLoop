# 003 — Tool constraints: description, not expression engine

**Status:** accepted, 2026-09-07
**Supersedes:** [`002-tool-registry-and-selection.md`](002-tool-registry-and-selection.md) §7, the "argument values" residual, which lists a precondition engine as a preferred route. It is not one. The `preconditions` field stays in `ToolSpec` as documentation; the engine is dropped from the plan.

---

## Why this document exists

`002` §7 says argument-value correctness has three routes, "in preference order": a `preconditions` rule reading trace metadata, executable verification, or ground truth on a scenario pack. The first was wrong, and wrong in a way that would have cost a parser, a binding config, a new trace field, and a permanent second copy of the customer's policy.

The argument against it is one line: **a constraint the model must obey is already in the tool description**, because the model cannot comply with a rule it was never told.

```yaml
issue_refund:
  description: >-
    Refund an order to the original payment method. Irreversible.
    Only valid for orders placed within the last 30 days.
```

That prose is already in the registry, already reaches the judge — descriptions are rendered into `tool_selection`'s catalogue and `text_matches_tools`' call list — and is therefore already being applied. A precondition restating it adds **determinism, not information**, and determinism turned out to cost more than it was worth.

---

## 1. The three things that killed it

### 1.1 The missing piece was the input, not the parser

A judge reading "within the last 30 days" still has to know how old *this* order is. Where that failed, it failed for lack of a fact, not for lack of an expression evaluator. The fix is a mapped field and an `inputs:` entry on the question — config, not machinery.

### 1.2 A restated rule is a second copy that drifts

The product extends its refund window to 45 days. `tools.yaml` still says 30. The report now fills with failures that are not failures, in the **false-fail** direction — the one that mints training pairs teaching the model to stop doing something correct (`001` §5.3). A rule mirrored from code into config is a drift surface with no owner.

### 1.3 Every such fact is as-of-the-call, and the trace cannot say when that was

Order placed day 0, refunded day 20 — correct. Evaluate on day 80 and `now() - order_date` is 80 days: a failure invented by the clock. Worse, it is unstable, so the same trace flips verdict between runs and reproducibility is gone.

Doing it properly needs `Trace.occurred_at`, derivation frozen at ingest rather than computed at evaluation, and a rule that facts are never joined live from the source database — a live join returns today's state for a call made last quarter. That is three contract-level changes to make one policy rule deterministic.

**For a voice-agent product, none of it earns its surface.** The 30-day rule stays where it already is.

---

## 2. Prefer the product's own verdict

Where the product enforces a rule, the truth is already in the trace — or would be, if the mapping captured it:

```json
{"name": "issue_refund", "arguments": {"order_id": "ORD-8891", "amount": 79.99},
 "error": "POLICY_VIOLATION: order is 45 days old, refund window is 30"}
```

`ToolCall.error` beats any restatement of the policy: it is the product's own answer, it cannot drift from the product's behaviour, and it needs no expression language. It also catches a class nothing sees today — a structurally perfect call that failed at runtime while the agent said "done".

| Product behaviour | Build |
|---|---|
| Enforces the rule and records the failure | `tool_call_exec` — read `ToolCall.error` |
| Enforces but records nothing | ingest mapping to capture it, then the above |
| Does not enforce | the tool description, read by the judge |

The blocked attempt is still a model failure worth recording: the agent decided to refund a 45-day order and the API saved it. That shows up as a failed call, so `.error` gives it to you without a precondition.

---

## 3. Only feed the judge what the agent had

A related trap, recorded because it is cheap to fall into once inputs start being added to prompts.

If a fact was never in the agent's context, handing it to the judge makes the judge better informed than the agent could possibly have been. Every such trace then reads as a model failure when the defect is upstream.

| Agent had the fact | Called `issue_refund` on a 45-day order | Finding |
|---|---|---|
| yes | ✗ | model failure — ignored a policy it could see |
| no | ✗ | **product failure** — the agent was never given the fact |

The second row is the more valuable finding and the system can currently produce neither. Prefer inputs drawn from `input.system_prompt` and `input.messages` — the agent's own recorded context, frozen in the trace by construction — over anything joined in from outside.

---

## 4. What is dropped, and what survives

**Dropped from the plan:** the expression engine, the name-to-path binding config, `Trace.occurred_at`, and live joins against the source database. All four existed only to serve preconditions.

**Kept:** `ToolSpec.preconditions` as a field. It documents intent, it is hashed into the registry, and `tool_registry_check` continues to refuse `check_preconditions: true` rather than appear to enforce what it does not. `002` §7's other two routes — executable verification and authored scenarios — stand.

**Revised order for tool correctness:**

1. `tool_call_exec` — read `ToolCall.result` / `.error`. Objective, small, catches the enforced cases.
2. Constraint inputs — map the facts the descriptions already reference into the judge prompts, subject to §3.
3. P3a judge health — the three shipped checks lean on a judge nobody has measured.
4. Preconditions — only if a specific rule appears that the product does not enforce, whose data the trace carries, and whose verdict must be gate-admissible. Not a general engine.

---

## 5. Voice note

`text_matches_tools` is the most valuable of the three tool checks in a voice product, and the most fragile.

Most valuable because there is no scroll-back. A chat customer can re-read "I've processed your refund" and notice nothing arrived; a caller hears it once and finds out in three days. Transcript-versus-tool divergence is a worse failure by voice than by text.

Most fragile because the transcript is ASR output, not what the agent emitted. "I've refunded your order" and "I've refused your order" are one phoneme apart, and flagging a mis-transcription as a contradiction is a false failure pointing at a model that behaved correctly. Mitigation when the first voice dataset lands: instruct the judge to answer `consistent: true` where the mismatch is plausibly a transcription artifact rather than a different action. One clause, no machinery.

---

## Rules added

19. A constraint stated in a tool description is not restated as a rule.
20. Prefer the product's own verdict over a restatement of its policy.
21. A judge is given only what the agent had. A fact the agent never saw is a product finding, not a model failure.

Each is enforced by a test, not by convention.
