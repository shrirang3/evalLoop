# Plan

Design documents for EvalLoop. Numbered, append-only — supersede rather than rewrite, so decisions keep
their history.

| Doc | Covers |
|---|---|
| [`000-build-plan.md`](000-build-plan.md) | Full phased build plan: stack, repo layout, P0 (contracts) → P6 (promotion gate), P7+ roadmap, cross-cutting engineering rules, per-phase verification commands |

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Metastore | Postgres from day one | Run/result/manifest queries and slicing need real SQL; multi-user later without a rewrite. Distinct from the *source* Postgres connector, which is read-only. |
| Judge I/O | Own thin OpenAI-compatible client | One httpx client covers OpenAI, vLLM, Together, Groq, LiteLLM proxy. Full control over the cache key, retry policy, and cost ledger — all three are product surface, not plumbing. |
| Training compute | Local/single GPU, subprocess launch | Runs anywhere with a GPU, and CPU-only with a tiny model for CI. Remote backends (Modal, RunPod, k8s) slot in behind `TrainerBackend` later. |
| Voice scope | Artifact refs + tool/transcript layers | P0 stores audio URIs and evaluates tool behaviour deterministically plus transcript behaviour by judge. Audio-level evaluation is P8 with the interface reserved now. |

## Phase index

| Phase | Deliverable | Acceptance |
|---|---|---|
| P0 | Contracts, Postgres metastore, config validation | One JSONL dataset runs through exact matching and a custom LLM question |
| P1 | Read-only connectors, mapping, redaction, splits | Ingest a realistic tool-call + voice dataset without changing its source database |
| P2 | Deterministic + LLM evaluators, judge client, cache, budget | Per-question evaluation report with costs, under a USD budget |
| P3 | Judgecard: agreement, κ, CIs, bias probes | Deliberately broken judges are correctly flagged |
| P4 | Feedback compiler: SFT/DPO, eligibility, leakage checks | Reproducible training JSONL from one evaluation run |
| P5 | TRL LoRA trainer, candidate registration and inference | Train one adapter, evaluate it against the sealed test set |
| P6 | Promotion gate, slice regressions, experiment bundle | One command produces a defensible promotion decision |
