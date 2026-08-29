# EvalLoop — Build Plan (P0 → P6)

## Context

Teams shipping LLM/voice products have production traces and some ground truth, but no path from "this trace was wrong" to "a better model is in production." Today that path is hand-rolled per team: ad-hoc eval scripts, an LLM judge nobody calibrated, a training JSONL assembled by hand, and a promotion decision made on vibes.

EvalLoop is a **configurable evaluation and improvement control plane**: point it at existing production traces (no schema change to the source DB), define deterministic checks and arbitrary LLM-judge questions in YAML, measure whether the judge itself is trustworthy, compile *verified* failures into SFT/DPO datasets, fine-tune a candidate, and promote only after sealed re-evaluation.

Outcome of P0: one command chain (`ingest → evaluate → judgecard → feedback build → train → compare`) produces a defensible, reproducible promote/reject decision on a real customer dataset.

**Decisions taken (this session):**
- Metastore: **Postgres from day one** (also a P0 *source* connector — distinct database, read-only)
- Judge I/O: **own thin OpenAI-compatible client**, providers behind one interface
- Training: **local/single-GPU TRL LoRA via subprocess launch**; remote backends deferred
- Voice: **artifact refs + tool/transcript layers only**; audio-level judging deferred to P4+

---

## 0. Stack & repo layout

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | TRL/peft/transformers ecosystem |
| Packaging | `uv` + single package `evalloop`, optional extras `[postgres]`, `[train]`, `[report]` | keeps `pip install evalloop` light; training deps are heavy |
| Contracts | Pydantic v2 | JSON Schema export for free, strict validation, fast |
| Metastore | Postgres 15 + SQLAlchemy 2.0 + Alembic | run/result/manifest queries, slicing, multi-user later |
| Bulk data | Parquet on a content-addressed artifact store (local dir or S3) | traces + results are wide and repetitive; Postgres holds pointers + aggregates |
| CLI | Typer + Rich | subcommands, good tables |
| HTTP | httpx (sync + async pool) | judge calls, retries, timeouts |
| Templating | Jinja2 | HTML judgecard / promotion report |
| Sandbox (custom Python evaluators) | subprocess + `resource` rlimits + timeout, no network by default | "restricted worker with a time limit" from spec |

```
evalloop/
  contracts/     trace.py  result.py  judgeconf.py  dataset.py  suite.py  project.py
  store/         models.py  repo.py  migrations/  artifacts.py
  ingest/        connectors/{postgres,jsonl,csv,pyiter}.py  mapping.py  redact.py  split.py
  evaluate/      registry.py  runner.py  cache.py  budget.py
                 deterministic/{exact,json_match,jsonschema,regex,numeric,set_cmp,tool_exec}.py
                 llm/{question,rubric}.py   pyeval.py
  judge/         client.py  providers/{openai_compat,anthropic}.py  schema.py  parser.py  version.py
  judgecard/     metrics.py  bias.py  perturb.py  report.py  templates/
  feedback/      policy.py  filter.py  sft.py  dpo.py  leakage.py  manifest.py
  train/         backends/{base,trl_lora}.py  launcher.py  registry.py  infer.py
  promote/       gate.py  compare.py  bundle.py
  cli/           main.py  (ingest evaluate judgecard feedback train compare promote)
tests/           unit/  integration/  fixtures/{traces,broken_judges,suites}/
examples/        voice-agent/  support-bot/
docker/          docker-compose.yml  (postgres for metastore + a seeded fake "product DB")
```

---

## P0 — Contracts, storage, config validation

**Goal:** every downstream module codes against frozen schemas. Nothing else can start safely until these are stable.

### Deliverables

1. **Trace schema** (`contracts/trace.py`) — exactly the spec shape:
   `trace_id`, `input{messages[], user_request, system_prompt?, tools_available?}`,
   `output{text, tool_calls[], artifacts[]}`, `ground_truth{...free-form dict + typed known keys}`,
   `metadata{...arbitrary}`, plus system fields: `source_id`, `ingested_at`, `content_hash`.
   - `ground_truth` is `dict[str, Any]` with a `has(path)` helper — users invent their own GT keys.
   - `artifacts[]`: `{type: audio|image|file, uri, duration_ms?, mime?}`. **URI only, never bytes.**

2. **Evaluator interface**:
   ```python
   class Evaluator(Protocol):
       id: str
       def version_hash(self) -> str: ...          # config-derived, stable
       def evaluate(self, trace: Trace, ctx: EvalContext) -> EvalResult: ...
   ```

3. **Judge interface**:
   ```python
   class Judge(Protocol):
       def ask(self, prompt: RenderedPrompt, schema: dict) -> JudgeResponse:  # raw + parsed + usage
   ```

4. **Result schema** (`contracts/result.py`) — the spec's normalized result plus
   `run_id`, `judge_config_hash`, `cache_hit`, `error`, `invalid_output` flag, `tokens_in/out`.

5. **Metastore schema** (Alembic migration 0001):

   | table | key columns |
   |---|---|
   | `project` | id, name, config_yaml, created_at |
   | `snapshot` | id, project_id, source_fingerprint, row_count, created_at, **immutable** |
   | `trace` | snapshot_id, trace_id, split (`train`/`dev`/`test`), parquet_path, content_hash |
   | `split_assignment` | snapshot_id, trace_id, split, split_key, split_strategy — **unique (snapshot_id, trace_id)** |
   | `eval_run` | id, snapshot_id, suite_hash, split, started_at, status, cost_usd, tokens |
   | `eval_result` | run_id, trace_id, evaluator_id, evaluator_version, score, passed, normalized_prediction, ground_truth, latency_ms, cost_usd, raw_output(jsonb), explanation |
   | `judge_config` | hash (PK), provider, model, params(jsonb), system_prompt, rubric, response_schema |
   | `llm_cache` | key (PK = sha256(judge_hash + rendered_prompt + schema)), response(jsonb), usage(jsonb), created_at |
   | `judgecard` | run_id, evaluator_id, metrics(jsonb), created_at |
   | `feedback_dataset` | id, run_id, strategy, manifest(jsonb), path, row_count, **immutable** |
   | `train_run` | id, dataset_id, backend, config(jsonb), status, artifact_uri, base_model |
   | `model_registry` | id, name, kind(`baseline`/`candidate`), train_run_id?, endpoint_conf(jsonb) |
   | `comparison` | id, baseline_model, candidate_model, run_ids, gate_result(jsonb), decision |

   Index: `eval_result(run_id, evaluator_id)`, `eval_result(run_id, trace_id)`, `llm_cache(created_at)`.

6. **Config loading + validation**: `project.yaml`, `eval-suite.yaml`, `judges.yaml`, `training.yaml`, `promotion.yaml`. One `evalloop validate <file>` command with line-accurate errors (Pydantic error → YAML path). **Unknown keys are errors, not warnings.**

7. **Artifact store** (`store/artifacts.py`): `put(bytes|path) -> content_hash uri`, `get(uri)`. Local dir backend in P0, S3 backend interface stubbed.

8. **JSONL connector** + **exact-match evaluator** + **one LLM question** so contracts are exercised end-to-end.

**Acceptance:** `evalloop ingest examples/support-bot/project.yaml && evalloop evaluate examples/support-bot/eval-suite.yaml` runs a JSONL dataset through one exact matcher and one custom LLM question, writes rows to Postgres, and `evalloop validate` rejects a suite with a typo'd key pointing at the right line.

---

## P1 — Real data ingestion

**Goal:** ingest a customer's production traces without touching their database.

### Deliverables

1. **Read-only Postgres connector**
   - Connection forced to `default_transaction_read_only=on` + a session `SET TRANSACTION READ ONLY`; refuse to run if the URL user has write grants is *not* checked (can't be), but **statement allowlist**: reject any query whose parsed first token isn't `SELECT`/`WITH`.
   - Server-side cursor, batched fetch, `:start_date`-style bound params (never string interpolation).

2. **CSV + Python-iterator connectors** (`module:callable` yielding dicts).

3. **Mapping engine** (`ingest/mapping.py`) — the spec's YAML:
   - Path syntax on both sides: `output.artifacts[0].uri`, `ground_truth.tool_calls`.
   - Transforms: `json_parse`, `json_stringify`, `split(sep)`, `lower`, `datetime`, `default: <v>`, `template: "{a} {b}"`, `python: mod:fn`.
   - Missing-source policy per field: `error` | `null` | `skip_row` (default `error` for `trace_id`, `null` otherwise).
   - `evalloop ingest --dry-run --limit 5` prints source row → mapped trace side by side. **This is the debugging tool people will live in — make it good.**

4. **Redaction hooks** (`ingest/redact.py`)
   - Rule types: `regex`, `named_entity` (optional spaCy extra), `field_drop`, `hash`, `python`.
   - Two application points: **at ingest** (persisted redacted) and **pre-external-call** (belt and braces, in the judge client).
   - `redaction_report`: counts per rule; a rule with 0 hits across the snapshot is warned about (probably broken regex).

5. **Split creation** (`ingest/split.py`)
   - Strategies: `by_field` (customer/conversation/tool_family/scenario), `by_time` (test = newest window), `hash_of_field`, `stratified_by_field`, `random` (allowed but prints a leakage warning).
   - Near-duplicate guard: MinHash/simhash over `input.user_request` within a snapshot; identical or >0.9-similar requests are forced into the same split.
   - Emits `split_report`: sizes, per-slice label balance, collisions prevented.
   - **Invariant enforced in DB**: unique constraint on `(snapshot_id, trace_id)` in `split_assignment`; the sealed test set is additionally marked `sealed=true` and any read of it outside the promotion gate requires `--i-know-what-im-doing`.

6. **Snapshot immutability**: a snapshot row is written once with a `source_fingerprint` (hash of connector config + query + row count + sorted content hashes). Re-ingesting identical data returns the existing snapshot instead of duplicating.

**Acceptance:** ingest a realistic tool-call + voice dataset from a Postgres instance seeded from `docker/`, with audio URIs preserved as references, without any DDL/DML against the source. `evalloop snapshot show <id>` prints split sizes and the redaction report.

---

## P2 — Evaluation engine

**Goal:** deterministic + LLM evaluation producing one normalized result stream, with cost control.

### Deliverables

1. **Deterministic evaluators**
   | type | notes |
   |---|---|
   | `exact_match` | with `normalize: {lower, strip, collapse_whitespace}` |
   | `json_match` | `ignore_order`, `normalize_numbers`, `allow_extra_arguments`, `ignore_paths[]`, `coerce_types` |
   | `json_schema` | validate `actual` against a JSON Schema |
   | `regex` | `pattern`, `must_match` / `must_not_match`, capture-group extraction |
   | `numeric_tolerance` | `abs_tol` / `rel_tol` |
   | `set_comparison` | precision/recall/F1 over a list field, with a key function |
   | `tool_call_exec` | run the tool in a sandbox against a fixture/mock; compare side-effect or return |
   | `python` | `callable: my_evals.order:validate_cancellation` |

   **Tool-call matching gets first-class care** — it's the highest-value deterministic check. Normalizer handles: argument ordering, numeric-vs-string IDs, null vs missing, extra optional args, and multi-call sequences (ordered or as a set).

2. **LLM question evaluator** (`llm/question.py`) — spec section 3B: `inputs` path map → prompt render → structured response → `matcher` extracts `normalized_prediction` → compare against `ground_truth.path`.
   Matchers: `exact_label`, `label_map`, `numeric_tolerance`, `boolean`, `set_overlap`, `python`.

3. **Rubric evaluator** (`llm/rubric.py`) — spec 3C: N sub-questions in **one** call (cheaper, and consistent context), each emitting its own `EvalResult` with `evaluator_id = "<id>.<name>"` so Judgecard reports per question. Output types: `boolean`, `integer{min,max}`, `enum`, `number`, `string`.

4. **Judge client** (`judge/`)
   - `openai_compat` provider: `/chat/completions`, JSON-schema-constrained output where the endpoint supports it, else strict prompt + parse-and-repair (one repair retry, then mark `invalid_output`).
   - `anthropic` provider behind the same interface (tool-use forced schema).
   - Retries with jittered backoff on 429/5xx only; **timeouts are not retried silently** — they count toward the invalid rate.
   - **Judge version hash** = sha256 of canonical JSON of `{provider, model, temperature, top_p, max_tokens, system_prompt, question(s), response_schema, parser_version}`. Stored in `judge_config`; every result references it. Change one rubric sentence → new hash → new version. Enforced by test.

5. **Cache** — key = `sha256(judge_hash ‖ rendered_prompt ‖ schema)`. Postgres-backed, `--no-cache` and `--refresh-cache` flags. Cache hits cost $0 and are flagged in the result. Cache is *keyed by judge version*, so recalibrating a prompt never silently reuses old answers.

6. **Budget + concurrency**
   - `budget.max_cost_usd`, `max_tokens`, `max_traces` per run; runner aborts cleanly at the limit and marks the run `partial` (results already written stay valid).
   - Async worker pool with per-provider rate limits; deterministic evaluators run in a thread/process pool.
   - Pricing table per model in config; unknown model → cost recorded as `null`, not zero, and the run warns.

7. **Python evaluator sandbox** — subprocess, `RLIMIT_CPU`/`RLIMIT_AS`, wall-clock timeout, no network via a stub `socket` guard, `PYTHONPATH` restricted to the project's eval package. Failure → result with `error` set, never crashes the run.

8. **Resume**: a run is checkpointed per trace; `evalloop evaluate --resume <run_id>` skips completed (trace, evaluator) pairs.

**Acceptance:** `evalloop evaluate eval-suite.yaml --split dev` produces a per-question report with agreement, invalid rate, per-evaluator cost and latency, and stays under a configured USD budget.

---

## P3 — Judgecard

**Goal:** answer "can I trust this judge, on this question?" — per question, not per judge.

### Deliverables

1. **Per-question metrics** (`judgecard/metrics.py`)
   - Agreement with ground truth; confusion matrix; per-label precision/recall/F1/support
   - Cohen's κ (and Krippendorff's α when >2 raters exist)
   - Bootstrap 95% CI on agreement (BCa, 2000 resamples), plus n
   - Invalid/missing output rate, parse-failure rate, timeout rate
   - Cost and p50/p95 latency
   - **Explicit small-n guard**: below a configurable `min_gt_samples` (default 30), metrics render but are stamped `LOW CONFIDENCE (n=…)` and are ineligible to gate feedback.

2. **Bias probes** (`judgecard/bias.py`, `perturb.py`) — run only when `--probes` is on (they cost extra calls):
   - **Position bias**: for pairwise/comparative questions, swap A/B, measure flip rate
   - **Verbosity bias**: pad the response with irrelevant-but-fluent filler, measure score delta
   - **Formatting sensitivity**: reformat (markdown ↔ plain, bullet ↔ prose), measure score delta
   - **Perturbation sensitivity**: semantics-preserving paraphrase of the *input*, measure prediction flip rate
   - Each reported as a % with its own CI. These are *judge* properties, sampled on a subset (default 100 traces).

3. **The no-ground-truth rule** — enforced in code, not just documented. Every question in the card carries:
   ```
   Measured: Yes
   Calibrated against GT: No
   Eligible as training signal: No     (unless feedback.allow_uncalibrated: true, which logs a loud warning
                                        and stamps the dataset manifest)
   ```
   `feedback build` reads this flag from the judgecard table; it cannot be bypassed by editing YAML alone.

4. **Reports**: JSON (canonical, diffable) + HTML (Jinja2, self-contained, charts inline as SVG) + a Rich terminal table matching the spec's layout.

5. **Broken-judge fixtures** (`tests/fixtures/broken_judges/`): always-"yes" judge, coin-flip judge, verbosity-loving judge, position-biased judge, schema-violating judge, always-timeout judge. **Test asserts each is flagged by the right metric.** This is the acceptance test for the whole phase.

**Acceptance:** `evalloop judgecard <run_id>` produces the per-question card; the broken-judge suite is caught — coin-flip shows κ≈0, verbosity-lover shows a >15% verbosity delta, schema-violator shows a high invalid rate.

---

## P4 — Feedback compiler

**Goal:** turn *verified* failures into training rows — and refuse to fabricate a target when none exists.

### Deliverables

1. **Feedback policy** per evaluator (spec section 8), evaluated as a hard precondition:
   ```yaml
   feedback:
     eligible: true
     strategy: dpo            # sft | dpo | none
     chosen:  { source: ground_truth.expected_response }
     rejected:{ source: output.text }
     require:
       - ground_truth_available
       - judgecard_kappa_above: 0.6
       - judgecard_min_samples: 30
       - not_in_sealed_test
   ```

2. **Target-source resolution** — a row is emitted **only** if a target comes from one of the spec's six allowed sources, and the source is recorded per row:
   `ground_truth_response` | `ground_truth_tool_call` | `human_correction` | `approved_exemplar` | `executably_verified_correction` | `trusted_teacher_correction`.
   A failing score with no target source → **dropped**, counted in the manifest's `dropped_no_target` bucket with a per-evaluator breakdown. This number being large is the honest, useful signal: "you need more ground truth."

3. **Compilers**
   - `sft.py`: messages + target (text and/or tool_calls), chat-template-agnostic — stores structured, renders at train time.
   - `dpo.py`: prompt / chosen / rejected, with a guard that chosen ≠ rejected after normalization, and a length-balance report (DPO on systematically longer `chosen` teaches length, not quality).

4. **Leakage checks** (`leakage.py`) — run at build, fail the build not warn:
   - No trace_id from `test` split
   - No near-duplicate (simhash) of any `test` input
   - No generated/augmented row without `provenance: generated` (and those are train-pool-only by construction)

5. **Dataset manifest** (immutable, hashed) — snapshot id, run id, suite hash, judge hashes used, evaluator versions, filter predicates, per-evaluator row counts, target-source histogram, dropped-reason histogram, split fingerprints, builder version, timestamp. `evalloop feedback show <dataset_id>` prints it; two builds from the same inputs produce byte-identical output.

6. **Human-correction ingress** (small but essential): `evalloop feedback export-review <run_id>` → CSV/JSONL of failures needing targets; `evalloop feedback import-review <file>` → attaches corrections as `human_correction` targets. Without this, most teams' `dropped_no_target` never shrinks.

**Acceptance:** one evaluation run produces a reproducible SFT and DPO JSONL with a manifest; a deliberately leaky config is rejected; a judge below the κ threshold produces zero rows from its evaluator.

---

## P5 — Fine-tuning

**Goal:** one working trainer behind an interface that other backends can slot into.

### Deliverables

1. **`TrainerBackend` interface**
   ```python
   prepare(dataset) -> PreparedDataset      # render chat template, tokenize-check, length stats
   validate(prepared) -> ValidationReport   # truncation rate, malformed rows, class balance
   launch(config) -> TrainHandle            # subprocess/detached, returns run id
   poll(handle) -> Status                   # + streamed loss/eval curves to the metastore
   artifact(handle) -> ModelArtifact        # adapter dir + config + hashes
   ```

2. **`trl_lora` backend** — `transformers` + `peft` + `trl` in the `[train]` extra. Launcher writes a generated `train.py` + config into the run dir and shells out (so a crashed trainer never takes the CLI with it, and the exact script is archived as part of the run). Supports LoRA and QLoRA (bitsandbytes), gradient checkpointing, `bf16`, configurable `rank/alpha/dropout/target_modules`, epochs, LR, warmup, packing off by default.
   - **CPU smoke path**: a tiny model (e.g. a 100M-param stand-in) + 20 rows so CI proves the whole pipeline without a GPU.

3. **Validation before launch** — refuse to start on: >5% truncated examples, any malformed chat structure, dataset row count below a floor, `chosen == rejected` rows, or a dataset whose manifest fails its own hash check.

4. **Training run manifest**: dataset id + hash, base model + revision, all hyperparameters, library versions, git sha of the eval package, GPU/driver, seed, final + best metrics, wall time, adapter hash.

5. **Candidate registration + inference** (`train/infer.py`)
   - Register the adapter in `model_registry` with an `endpoint_conf` describing how to call it: local `transformers` generate, or an OpenAI-compatible endpoint (vLLM serving the adapter).
   - `evalloop infer <model> --split test` regenerates outputs for a split → writes a **new snapshot of candidate traces** carrying the original `ground_truth` and `metadata` forward, so the *same* eval suite runs against it unmodified. This is the key trick that makes candidate evaluation free of special-casing.

**Acceptance:** train one LoRA adapter from a P4 dataset and evaluate it against the sealed test set with the same suite that produced the baseline numbers.

---

## P6 — Promotion gate

**Goal:** one command, one defensible decision, one reproducible bundle.

### Deliverables

1. **Gate DSL** (spec section 12): `all` / `any` blocks, conditions of the form
   `candidate >= baseline + delta`, `candidate >= absolute`, `candidate <= absolute`, `candidate >= baseline * factor`.
   Metrics resolve to any evaluator id (including rubric sub-questions), plus built-ins: `invalid_output_rate`, `cost_per_trace`, `p95_latency_ms`.

2. **Slice regression rules**: `slices: [{field: metadata.language, no_regression: true, min_n: 25, tolerance: 0.02}]`.
   Slices below `min_n` are reported as `INSUFFICIENT DATA` and **do not silently pass** — the gate's default for them is configurable (`ignore` / `fail`), defaulting to a visible warning.

3. **Statistical honesty**: every delta reported with a bootstrap CI and a paired significance test (McNemar for binary, paired bootstrap otherwise). A gate can require `significant: true`. Multiple-comparison note printed when >10 slices are checked.

4. **Sealed-set enforcement**: the promotion gate is the **only** code path allowed to read `split='test'` rows; it records the read in an audit table with run id and timestamp. Repeated reads of the sealed set by the same project are counted and warned about ("you have queried the sealed set 7 times — it is becoming a dev set").

5. **Output**: the spec's table (baseline → candidate, PASS/FAIL per row), a `REJECT`/`PROMOTE` verdict, and a **reproducible experiment bundle** — a directory/tarball with every config, every hash, both run ids, the judgecards, the dataset manifest, the training manifest, and the gate result. Re-running from the bundle reproduces the decision.

6. **Promotion is a record, not a deploy.** `evalloop promote <comparison_id>` marks the model `promoted` in the registry and emits a webhook/exit code. **Nothing is deployed automatically** — non-negotiable rule #9.

**Acceptance:** one command produces the promote/reject report; a candidate that improves overall but regresses on Hindi conversations is rejected, and the bundle reproduces that verdict on a clean machine.

---

## Post-P0 roadmap (P7+, not in the first release)

| Phase | Content |
|---|---|
| P7 | Connector breadth: Snowflake, BigQuery, ClickHouse, Langfuse/Phoenix, webhook ingestion |
| P8 | Audio-level evaluation: multimodal/audio judge provider, speech-signal features (WPM, pause distribution, pitch variance, clipping), pronunciation scoring. **Explicit doc: text fine-tuning cannot fix pitch/cadence — those route to TTS config or TTS training.** |
| P9 | Trainer backends: OpenAI-compatible fine-tuning API, Axolotl, Unsloth, Modal/RunPod, Kubernetes job, custom webhook |
| P10 | Similar-case generation (spec §9): programmatic tool-call construction with LLM paraphrase of the request only; stricter validation for subjective/tone cases; **train-pool-only enforcement already built in P4** |
| P11 | Web UI: run browser, trace inspector, judgecard viewer, human-correction review queue |
| P12 | Multi-tenancy, RBAC, hosted SaaS, scheduled/continuous eval, streaming eval |
| P13 | RL/GRPO, active learning for which traces to label next |

---

## Cross-cutting engineering rules (enforced by tests, not convention)

| Rule | Enforcement |
|---|---|
| Every dataset snapshot versioned | `snapshot`/`feedback_dataset` rows immutable; re-ingest is idempotent by fingerprint |
| Every judge configuration hashed | `judge_config.hash` PK; test asserts a one-word prompt edit changes the hash |
| Every result stores evaluator version | NOT NULL column; runner refuses to write without it |
| LLM calls cached | cache keyed by judge hash + prompt + schema; cache-hit rate reported per run |
| Connectors read-only | SELECT-only allowlist + read-only transaction; test asserts an INSERT query is rejected |
| PII redaction before external judge calls | redaction runs inside the judge client, after prompt render, before the HTTP call; test asserts a seeded PII string never appears in the outbound body |
| Judge feedback without GT not trusted | `feedback build` reads the judgecard eligibility flag from the DB |
| Training data never in sealed test | unique split assignment + simhash near-dup check + build-time hard failure |
| No automatic deployment | `promote` writes a registry row and exits; no deploy code exists in the package |
| Cost/tokens first-class | columns on `eval_run` and `eval_result`; budget aborts; every report shows USD |

**Testing strategy:** golden-file tests for every config → normalized artifact; property tests for the mapping engine and JSON matcher; the broken-judge fixture suite for P3; a full `docker-compose` integration test running ingest → promote on a seeded fake product DB in CI (CPU-only, tiny model, mocked judge).

---

## Critical files to create first (P0 order)

1. `evalloop/contracts/trace.py`, `result.py`, `judgeconf.py` — freeze these before anything else
2. `evalloop/store/models.py` + `migrations/0001_initial.py`
3. `evalloop/contracts/suite.py` + `evalloop/cli/main.py::validate`
4. `evalloop/ingest/connectors/jsonl.py` + `ingest/mapping.py`
5. `evalloop/evaluate/registry.py`, `runner.py`, `deterministic/exact.py`
6. `evalloop/judge/client.py` + `providers/openai_compat.py` + `judge/version.py`
7. `docker/docker-compose.yml` + `examples/support-bot/` fixtures

---

## Verification (end-to-end, per phase)

```bash
docker compose -f docker/docker-compose.yml up -d      # metastore + seeded fake product DB
uv run alembic upgrade head

evalloop validate examples/voice-agent/*.yaml           # P0
evalloop ingest   examples/voice-agent/project.yaml --dry-run --limit 5   # P1 mapping check
evalloop ingest   examples/voice-agent/project.yaml
evalloop snapshot show <snapshot_id>                    # splits + redaction report

evalloop evaluate examples/voice-agent/eval-suite.yaml --split dev --budget-usd 2   # P2
evalloop judgecard <run_id> --probes --html out/card.html                            # P3
pytest tests/integration/test_broken_judges.py                                       # P3 acceptance

evalloop feedback build <run_id> --strategy dpo         # P4
evalloop feedback show <dataset_id>                     # manifest + dropped-reason histogram

evalloop train examples/voice-agent/training.yaml       # P5 (CPU smoke: --smoke)
evalloop infer  candidate-v3 --split test               # regenerate candidate outputs
evalloop evaluate examples/voice-agent/eval-suite.yaml --split test --model candidate-v3

evalloop compare baseline candidate-v3 --gate examples/voice-agent/promotion.yaml    # P6
evalloop bundle  <comparison_id> --out bundles/         # reproducible decision
```

Each phase's acceptance test lives in `tests/integration/test_phase_<n>.py` and runs in CI against the compose stack.
