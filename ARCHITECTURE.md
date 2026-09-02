# Architecture

How the codebase is put together. [`README.md`](README.md) covers what EvalLoop is for;
[`plan/`](plan/) records why decisions were made. This file is for someone about to change the code.

---

## Dependency direction

Strictly one way. Nothing lower imports anything higher.

```
cli/                      Typer commands. Thin - argument parsing and rendering only.
  ↓
config/                   YAML loading and validation. Reads contracts/, nothing else.
  ↓
promote/  train/  feedback/  judgecard/     Phase modules. Independent of each other.
  ↓
evaluate/  ingest/  judge/                  Execution: run checks, read data, call providers.
  ↓
store/                                      Postgres + the artifact store.
  ↓
contracts/                                  Pydantic models and Protocols. Imports nothing internal.
```

`contracts/` is the floor. It has no dependency on storage, HTTP, or the CLI, which is why it can be
imported by tests, by a customer's own evaluator, and by every phase without pulling in a database
driver.

The one deliberate exception is `contracts/protocols.py`, which imports `contracts/trace.py` and
`contracts/result.py`. That is within the layer, not across it.

## Module map

| Module | Owns | State |
|---|---|---|
| `contracts/` | Trace, EvalResult, config schemas, `Evaluator`/`Judge` protocols | ✅ P0.2–P0.3 |
| `store/` | SQLAlchemy models, Alembic migrations, content-addressed artifacts | ✅ P0.4 |
| `ingest/` | Connectors (jsonl done; csv/postgres/pyiter P1), mapping, redaction, splits | 🟡 P0.6 |
| `evaluate/` | Registry, runner, deterministic matchers, LLM question and rubric evaluators | ⬜ P0.7 / P2 |
| `judge/` | HTTP client, provider adapters, response parsing, version hashing | ⬜ P0.7 / P2 |
| `judgecard/` | Judge health probes, agreement metrics, bias probes, reports | ⬜ P3 |
| `feedback/` | Policy evaluation, target resolution, SFT/DPO compilers, leakage checks | ⬜ P4 |
| `train/` | `TrainerBackend` interface, TRL LoRA backend, candidate inference | ⬜ P5 |
| `promote/` | Gate DSL evaluation, baseline comparison, experiment bundle | ⬜ P6 |
| `config/` | YAML loading, kind detection, Pydantic errors mapped to source lines | ✅ P0.5 |
| `cli/` | Command surface | 🟡 `validate` only |

Every directory exists with an `__init__.py` so the layout is visible from day one and imports do not
move as phases land.

## Data flow

Four representations, each with a different lifetime.

```
source rows          the customer's schema. Read-only, never written to.
    │  ingest/mapping.py
    ▼
Trace                contracts/trace.py. Normalized, immutable, content-hashed.
    │  evaluate/runner.py
    ▼
EvalResult           contracts/result.py. One row per (trace × evaluator).
    │  feedback/
    ▼
training rows        SFT or DPO JSONL, each stamped with signal provenance.
```

## Where state lives

### Immutability

`snapshot` and `feedback_dataset` are guarded by a PL/pgSQL trigger that raises on UPDATE and DELETE
(migration `0001`). Both are cited as evidence: a judgecard only means what it said if nothing could
have edited the snapshot underneath it, and a training run can only name which rows it saw if its
dataset row cannot be rewritten. An application-level check is bypassed by the next person with `psql`
open, so the guard is in the database.

Consequence worth knowing: test rows written to those tables cannot be cleaned up afterwards. The
`pg_session` fixture wraps each test in a transaction and rolls back; `make reset-db` rebuilds the
schema when a stray row does land.

### Server defaults

Every NOT NULL column with a default declares `server_default` as well as `default`. A Python-side
default never reaches Postgres, so `psql`, raw SQL, and any non-SQLAlchemy client would hit a NOT NULL
violation on a column that looks defaulted in the model.

| Kind | Home | Why |
|---|---|---|
| Runs, results, manifests, cache | Postgres | needs querying and slicing: "every Hindi refund failure last Tuesday" |
| Traces and results in bulk | Parquet on the artifact store | wide and repetitive; Postgres holds pointers and aggregates |

Trace bodies are a JSON string in a Parquet `payload` column rather than a Parquet schema. `ground_truth`
and `metadata` are free-form and differ per customer, so a real schema would have to be inferred per
snapshot — making two snapshots of the same product incompatible — or flattened, which loses the shape.
`trace_id`, `split`, and `content_hash` stay genuinely columnar for filtering.

`ingested_at` and `content_hash` are excluded from the payload. The first because every trace in a
snapshot shares `snapshot.created_at`, and keeping a per-trace copy made identical data serialize to
different bytes — which defeated content addressing entirely. The second because it is derived:
recomputing it on read turns the round trip into an integrity check against the stored column.
| Audio, images, model adapters | Artifact store, content-addressed | large binaries; identical content is stored once |
| Secrets | Environment variables only | a key in a config file reaches git, the metastore, and the bundle |

## The three hashes

Reproducibility rests on these. Each pins a different thing, and all three are stored on the rows they
describe.

| Hash | Defined in | Covers | Changes when |
|---|---|---|---|
| `Trace.content_hash` | `contracts/trace.py` | trace id, input, output, ground truth, metadata | the trace's content changes |
| `judge_version_hash()` | `contracts/judgeconf.py` | provider, model, sampling params, prompt, question, response schema, parser version | anything that could change the judge's answer |
| `EvalSuite.suite_hash()` | `contracts/suite.py` | every evaluator's identity and configuration | what is measured changes |

Two deliberate exclusions, both to avoid fragmenting a cache or breaking idempotency for no gain:

- `content_hash` excludes `source_id` and `ingested_at`, so re-ingesting the same row tomorrow produces
  the same hash and snapshot idempotency works.
- `judge_version_hash` excludes `base_url`, `timeout_s`, and `max_retries`. Pointing at a different
  replica of the same model cannot change what the model says.

## Invariants, and where they are enforced

The README lists fourteen non-negotiable rules. Each is enforced in code, not by convention.

| Rule | Enforced by |
|---|---|
| Ground truth is optional | `GroundTruth` defaults to empty; `is_empty` is the common case |
| A stored `None` is not an absent key | `MISSING` sentinel in `contracts/paths.py` |
| Unknown config keys are errors | `extra="forbid"` on every contract model |
| Traces and results are immutable | `frozen=True` on every contract model |
| Artifacts are references, never bytes | `Artifact` validator rejects `data:` URIs |
| An errored check is not a model failure | `EvalResult.is_failure` requires `error is None` |
| Base provider ≠ judge provider | `check_integrity()` in `contracts/project.py` |
| Held-out questions never reach training | `holdout` flag on both evaluator specs; asserted in P4 |
| Deterministic checks cannot be judge-influenced | `EvalContext.judge` is `None` for them |
| Unpriced models cost `None`, not `0.0` | `TokenUsage.cost_usd` default; `eval_result.cost_usd` nullable |
| Snapshots and datasets are immutable | UPDATE/DELETE trigger, migration `0001` |
| A trace cannot be in two splits | `uq_split_assignment_trace` |
| Results record their evaluator version | `eval_result.evaluator_version` NOT NULL |
| Re-ingesting identical data is a no-op | unique `snapshot.source_fingerprint` + `upsert_snapshot` |

## Extension points

Each is a Protocol or a registry entry, so adding one touches no existing code.

- **A new evaluator** — anything with `id`, `version_hash()`, and `evaluate(trace, ctx)` satisfies the
  `Evaluator` protocol. A customer's own Python check qualifies without importing from EvalLoop.
- **A new judge provider** — implement `Judge.ask(prompt, schema) -> JudgeResponse`. It must never
  raise: a malformed answer returns `parsed=None`, a transport failure sets `error`. Both are data the
  judgecard needs, and neither should end a run of ten thousand traces.
- **A new source connector** — yield dicts; `ingest/mapping.py` handles the rest.
- **A new trainer backend** — implement `TrainerBackend` (`prepare` / `validate` / `launch` / `poll` /
  `artifact`). Remote compute slots in behind the same interface.

## Path syntax

One resolver, `contracts/paths.py`, backs every place a user names a field in YAML — mapping targets,
evaluator `actual`/`expected`, judge `inputs`, feedback target sources. So `output.artifacts[0].uri`
means exactly the same thing everywhere.

```
a.b.c        dict keys
a[0].b       list index, negative allowed
a.0.b        list index, dotted form
```

Absent data returns `MISSING`. A malformed path raises. That asymmetry is deliberate: missing data is
normal and must not crash a run, while a typo'd path is a config bug and must be loud.

## Testing

| Layer | Marker | Needs |
|---|---|---|
| Unit | none | nothing — `make test`. Schema built from metadata on in-memory SQLite |
| Integration | `@pytest.mark.integration` | the compose stack — `make itest` |
| GPU | `@pytest.mark.gpu` | a CUDA device |

`make check` runs lint, `mypy --strict`, and the unit suite — the same three commands CI runs.

Triggers, check constraints, and JSONB are Postgres-only, so anything testing them is marked
`integration`. Running those against SQLite would produce a green suite that proved nothing, which is
why `pg_engine` skips loudly rather than falling back.

Two testing commitments from the plan:

- **Golden-file tests** for every config → normalized artifact, so a formatting change cannot silently
  alter a hash.
- **Broken-judge fixtures** (`tests/fixtures/broken_judges/`) — always-yes, coin-flip, verbosity-loving,
  position-biased, schema-violating, always-timeout. Each must be caught by the metric designed to catch
  it. This is the acceptance test for P3 and the reason to trust the judgecard at all.

## Conventions

- Python 3.11+, `from __future__ import annotations` in every module.
- `mypy --strict` passes with no `ignore` comments outside a documented Pydantic decorator quirk.
- Ruff, line length 100.
- Comments explain *why*, never *what*. A comment restating the code is deleted.
- `plan/` is append-only: supersede with a new numbered document, never rewrite an old one.
