# EvalLoop

**An evaluation and improvement control plane for AI products.**

Point it at production traces and your agent's tool definitions. Get back the tool calls that were
wrong — a tool that does not exist, one not permitted where it fired, a defensible tool with an
indefensible reply, a call the API rejected while the agent said it went through — **and a training
dataset compiled from those failures.**

**Ground truth is not a precondition.** Most teams have traces and no labels. Tool correctness comes
from a registry you already wrote; judge questions report their own provenance. Nothing is blocked
for lack of a dataset you were never going to have.

---

## Quickstart

```bash
pip install evalloop
```

```python
from evalloop import EvalLoop

loop = EvalLoop(
    judge="anthropic:claude-sonnet-5",
    tools="tools.yaml",              # your agent's tool definitions, exported
    traces="traces.jsonl",           # your production traces
    policy="Refunds are permitted within 30 days of purchase.",
)

report  = loop.judge()                       # 1. run the checks — costs judge calls
dataset = loop.dataset(report)               # 2. compile the failures — costs nothing
loop.finetune(dataset)                       # 3. P5 — refuses by name until then

report.print()                               # the tables below
report.to_markdown("report.md")              # ...as a file
dataset.to_jsonl()                           # DPO pairs, ready for a trainer
```

Three stages, three calls, on purpose: they fail for different reasons, cost
different amounts, and get re-run at different cadences. Compiling takes the
report rather than re-judging, so eligibility rules can change without paying
for the judge twice. The CLI splits the same way — `evaluate`, `feedback build`,
`train`.

No database, no migration, no config files beyond the two you point at — and
`tools=` takes a plain dict if you'd rather not have even those. What comes back:

```
Wrong tool selections
  called                     judge says            n    example
  issue_refund               open_warranty_claim   4    sb-0418
  none                       lookup_refund_status  4    sb-0433
  cancel_order,issue_refund  cancel_order          1    sb-0421
  agreed 1 · not applicable 0 · invalid answers 0

Registry violations
  code                      tools             n    example
  unregistered_tool         refund_order_now  1    sb-0417
  duplicate_side_effecting  issue_refund      1    sb-0102
  clean 8 · no tool calls 4

Calls the tool rejected
  tool          error                                                  n    example
  issue_refund  POLICY_VIOLATION: order is 45 days old, window is 30   1    sb-0417
  succeeded 1 · no outcome recorded 12
```

**No labels anywhere in that.**

### Everything the constructor takes

```python
EvalLoop(
    judge="anthropic:claude-sonnet-5",     # or a JudgeConfig, or {"default": ..., "strong": ...}
    traces="traces.jsonl",                 # or a list of dicts, or Trace objects
    tools="tools.yaml",                    # or a dict, or a ToolRegistry, or omitted
    mapping={"trace_id": "id", ...},       # only if traces aren't already in trace shape
    policy="Refunds within 30 days.",      # rules the judge should apply
    checks=[                               # defaults to all four
        registry_check(),
        tool_call_outcome(),
        tool_selection(policy=...),
        text_matches_tools(),
        llm_question("Was the tone empathetic?", id="tone"),
    ],
)
```

`report.results` is every row, `report.failures()` only the ones where a check
ran and said no, `report.cost_usd` is what it spent.

### The report is not the end of it

A `tool_selection` failure already contains the correct call — the judge chose
it, blind, before seeing what the agent did. So a preference pair needs no
labels and no human:

```json
{"prompt":   [{"role": "user", "content": "blender 45 days ago, arrived broken"}],
 "chosen":   {"tool_calls": [{"name": "open_warranty_claim",
                              "arguments": {"order_id": "ORD-8891", "reason": "damaged"}}]},
 "rejected": {"tool_calls": [{"name": "issue_refund",
                              "arguments": {"order_id": "ORD-8891", "amount": 79.99}}]},
 "target_source": "judge_tool_selection", "signal_provenance": "judge",
 "judge_version": "sha256:ec6f…", "judge_health": "unmeasured"}
```

**The judge's proposal is validated before it is trained on.** A call the judge
parameterises wrongly would teach the model the judge's mistake, so every
proposal goes through the same registry argument check that scores production
calls — a deterministic check gating judge-derived data:

```
valid proposal         EMITTED
bad enum value         dropped: proposal_failed_argument_check
missing required arg   dropped: proposal_failed_argument_check
hallucinated arg       dropped: proposal_failed_argument_check
```

**Nothing invents a target.** Of the four checks only `tool_selection` produces
one for free — knowing `refund_order_now` does not exist says nothing about what
should have been called, and an API rejection says the call failed, not what
would have worked. Those are dropped and counted, and the count is the useful
output: it says how much of your failure set is unusable and which traces are
worth a human's time first.

```
1 preference pair(s) from run run-8814…  →  feedback.jsonl
fingerprint 7dffa21247b6126f…

Dropped
  passed                            9
  proposal_failed_argument_check    3
  not_applicable                    1
```

Same thing from the CLI: `evalloop feedback build <run_id> --tools tools.yaml --out feedback.jsonl`.
`dataset.to_jsonl()` is the shape TRL's `DPOTrainer` takes, so it can be handed to a trainer today —
`loop.finetune()` exists to name what is missing (the trainer and the candidate registry, P5) rather
than let it fail somewhere inside TRL.

### Or the CLI, when you want provenance

The Python path runs nothing past your process. When results need to be
comparable six months later — versioned snapshots, hashed judges, rows in
Postgres — the same engine takes YAML:

```bash
make install && make up
evalloop validate examples/support-bot/*.yaml
evalloop ingest   examples/support-bot/project.yaml
evalloop evaluate examples/support-bot/eval-suite.yaml --split train
evalloop report   tools --out report.md
evalloop feedback build --tools examples/support-bot/tools.yaml
```

Everything above runs today; see [Status](#status) for what does not.

## Architecture

Your database is a source, never a destination. Every stage narrows: thousands of raw rows, hundreds
of results, a handful of verdicts, one decision.

```
  your DB · JSONL
      │  read-only, SELECT-checked
      ▼
 ┌──────────┐        ┌───────────────┐
 │  ingest  │───────►│   snapshot    │  Parquet, content-hashed, immutable
 │ map      │        └───────┬───────┘
 │ redact   │                │  PII gone before anything leaves the process
 └──────────┘                ▼
                     ┌───────────────┐     ┌────────────────┐     ┌──────────┐
  project.yaml ─────►│   evaluate    │────►│  judge client  │────►│ provider │
  tools.yaml ───────►│ registry      │     │ schema-forced  │     └──────────┘
  eval-suite.yaml ──►│ selection     │◄────│ + cache        │
  judges.yaml ──────►│ llm questions │     └────────────────┘
                     └───────┬───────┘        keyed by judge version
                             ▼
 ┌───────────────────────── Postgres ─────────────────────────┐
 │ runs · results · judge configs · llm cache · snapshots     │
 └─────────────────────────────┬──────────────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ judgecard  feedback │  P3–P6, not built
                    │ train      gate     │
                    └─────────────────────┘
```

**The invariant:** the judge never sees the decision it is grading, and every gate contains at least
one check the judge cannot move. Cache keys include the judge version, so editing a rubric can never
be answered by the old rubric's reply.

**Layout:**

```
evalloop/contracts/   frozen data contracts — trace, suite, tools, result
evalloop/ingest/      connectors, column mapping, redaction
evalloop/judge/       provider clients, schema-forced output, cache
evalloop/store/       Postgres metastore, Parquet traces, artifact store
evalloop/evaluate/    deterministic checks · judge questions · tool selection · consistency
evalloop/report/      rollups over stored results
evalloop/feedback/    failures → DPO pairs, with provenance and a drop histogram
evalloop/api.py       the Python entry point — no database
evalloop/cli/         validate · ingest · evaluate · report · feedback
```

`judgecard/`, `train/` and `promote/` exist as empty packages — the interfaces are reserved, the
phases are not built.

## The files

| File | Declares | Example |
|---|---|---|
| `project.yaml` | source, column mapping, splits, redaction, integrity rules | [↗](examples/support-bot/project.yaml) |
| `tools.yaml` | tools the agent may call, per node | [↗](examples/support-bot/tools.yaml) |
| `eval-suite.yaml` | the checks that run against a snapshot | [↗](examples/support-bot/eval-suite.yaml) |
| `judges.yaml` | judge models, one or many | [↗](examples/support-bot/judges.yaml) |
| `promotion.yaml` · `training.yaml` | gate conditions, LoRA config (P5–P6) | [↗](examples/support-bot/) |

A trace is your data, renamed — no migration, no `ground_truth` key:

```json
{ "trace_id": "call-123",
  "input":  { "user_request": "Cancel my order" },
  "output": { "text": "Certainly, I have cancelled it.",
              "tool_calls": [{ "name": "cancel_order", "arguments": { "order_id": "ORD-42" } }] },
  "metadata": { "language": "en", "customer_tier": "premium" } }
```

```yaml
mapping:
  trace_id:           id
  input.user_request: user_transcript
  output.tool_calls:  tool_calls_json
```

## Tool correctness, without labels

`tools.yaml` is the definitions your agent already hands the model on every request, exported —
config, not annotation ([`plan/002`](plan/002-tool-registry-and-selection.md)).

```yaml
nodes:
  refunds: { tools: [issue_refund, open_warranty_claim, lookup_order] }
tools:
  issue_refund:
    description: Refund an order to the original payment method. Irreversible.
    arguments: { order_id: {type: string, required: true}, amount: {type: number, required: true} }
    side_effecting: true
```

| Check | Asks | Catches |
|---|---|---|
| `tool_registry_check` | is this call legal? | tool that does not exist · not permitted at this node · arguments off-schema · side-effecting call repeated |
| `tool_selection` | which tool *should* have been called? | wrong choice among legal tools — the judge picks from the catalogue **without seeing the call**, and every tool called must be in its `acceptable` set |
| `text_matches_tools` | does the reply describe what the calls did? | right tool, wrong words — the call opened a warranty claim and the reply says "I've processed your full refund" |
| `tool_call_outcome` | did the call actually succeed? | legal call, right tool, matching reply — and the tool returned `POLICY_VIOLATION` while the agent said it went through. Reads the outcome your product recorded; executes nothing |

`tool_registry_check` and `tool_call_outcome` are objective — no model in either — which is what a
promotion gate needs at its floor. `tool_selection` computes a target where ground truth would have stored one, so its rows
carry the judge hash and support relative claims only. `text_matches_tools` needs no target at all —
a reply should always match its calls, so consistency *is* the answer and a mismatch is a verdict.

**Ground truth stays optional**, with two jobs and neither of them tool correctness: `policy_followed`
and friends are *labels* feeding the judgecard; `expected_tool_calls` and `expected_response` are
*targets* feeding the feedback compiler, and belong on cases you authored. A check with no ground
truth reports `not applicable`, never a failure.

## What it costs you

| Tier | You provide | You get |
|---|---|---|
| **T0** Deterministic | nothing | tool legality, schema validity, hallucinated IDs, cost, p95 latency |
| **T1** Judge health | nothing | position / verbosity / paraphrase bias, self-consistency, invalid-output rate |
| **T2** Regression | nothing | candidate vs baseline, per-slice regressions, relative gate conditions |
| **T3** Calibration | ~150 labels (≈90 min) | κ against a *measured* human ceiling, confusion matrix, FAIL-class precision |
| **T4** Training | T1 pass | SFT/DPO compilation, LoRA fine-tune, promotion decision |

**Relative claims are free. Absolute claims cost labels.** Without T3, EvalLoop will say a candidate
beat its baseline. It will refuse to say the model is 87% good.

## Why

- **A judge nobody checked is not a measurement.** A judge that flips when you swap A and B makes
  every number downstream noise. Check the instrument before reporting the reading — zero labels.
- **Training on a judge is fine; grading with the same judge is not.** The candidate is optimised to
  please the grader, then graded by it, and passes by construction. Three defences, no third model:
  a deterministic floor in every gate, held-out questions training never sees, and an automatic
  reject when judge scores climb while deterministic pass rate falls.
- **Honesty is provenance, not prohibition.** Refusing to emit a row for lack of ground truth just
  means no rows. Every row instead carries `target_source`, `signal_provenance`, `judge_version`,
  and the judge's measured health at build time.
- **A candidate cannot exceed its judge.** Fine-tuning against a stronger judge is distillation, not
  alchemy. Promotion is a record, not a deploy.

## Guarantees

Each is enforced by a test, not by convention.

1. Every snapshot is versioned; every judge config and evaluator is hashed onto every result.
2. LLM calls are cached, keyed by judge version — a rubric edit can never reuse an old answer.
3. Connectors are read-only. PII redaction runs before any external judge call.
4. A judge failing its health checks cannot mint training data.
5. Base model provider ≠ judge provider — judges favour their own family's outputs.
6. Every gate contains a deterministic condition; held-out questions never reach training data.
7. Training data never enters the sealed test set. A candidate is never deployed automatically.
8. Tool correctness never requires ground truth; a judge assessing a call is never shown the call.
9. Cost and token usage are first-class metrics.

## Status

**P0 complete** — ingest → evaluate runs end to end with queryable, fully-attributed results.

| | |
|---|---|
| ✅ P0.1–P0.8 | contracts, metastore, `validate`, JSONL ingest, deterministic + judge evaluators, cache, CI |
| ✅ plan/002–003 | tool registry, four tool checks, `report tools`, `feedback build` |
| ⬜ P1 · P2.5 | real connectors, redaction, splits, latent ground-truth harvesting |
| ⬜ P3a → P6 | `judge-health`, judgecard, SFT compilation, LoRA training, promotion gate |

`judge-health`, `judgecard`, `label`, `train`, `compare` and `bundle` appear in the design docs and
do not exist yet. The Quickstart above is the whole of what runs today.

**Voice:** traces carry audio as a URI. Tool and transcript layers are evaluated now; acoustic
evaluation is P8 — a text judge cannot hear tone, and fine-tuning a text model cannot change pitch.

---

**Code layout and extension points:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
**Design decisions:** [`plan/`](plan/README.md) — [000](plan/000-build-plan.md) build plan,
[001](plan/001-trusted-judge-architecture.md) trusted judge, [002](plan/002-tool-registry-and-selection.md) tool registry

**Stack:** Python 3.11+ · Pydantic v2 · Postgres + SQLAlchemy 2 + Alembic · Parquet · Typer + Rich ·
httpx · TRL/peft/transformers (optional `[train]`) — **License:** Apache-2.0
