# Plan

Design documents for EvalLoop. Numbered, append-only — supersede rather than rewrite, so decisions keep
their history.

| Doc | Covers |
|---|---|
| [`000-build-plan.md`](000-build-plan.md) | Full phased build plan: stack, repo layout, P0 (contracts) → P6 (promotion gate), P7+ roadmap, cross-cutting engineering rules, per-phase verification commands |
| [`001-trusted-judge-architecture.md`](001-trusted-judge-architecture.md) | Supersedes parts of 000 (P3, P4, P6). Ground truth is no longer a precondition: trusted-judge default path, capability tiers, two-model topology, provenance-not-prohibition, human calibration loop, dataset choices |

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Metastore | Postgres from day one | Run/result/manifest queries and slicing need real SQL; multi-user later without a rewrite. Distinct from the *source* Postgres connector, which is read-only. |
| Judge I/O | Own thin OpenAI-compatible client | One httpx client covers OpenAI, vLLM, Together, Groq, LiteLLM proxy. Full control over the cache key, retry policy, and cost ledger — all three are product surface, not plumbing. |
| Training compute | Local/single GPU, subprocess launch | Runs anywhere with a GPU, and CPU-only with a tiny model for CI. Remote backends (Modal, RunPod, k8s) slot in behind `TrainerBackend` later. |
| Voice scope | Artifact refs + tool/transcript layers | P0 stores audio URIs and evaluates tool behaviour deterministically plus transcript behaviour by judge. Audio-level evaluation is P8 with the interface reserved now. |
| Ground truth | **Not a precondition** ([`001`](001-trusted-judge-architecture.md) §1–2) | Most teams have traces and no GT. GT-required gating means the pipeline runs and emits zero rows. Judge-derived signal becomes the default path; GT calibration becomes an upgrade tier that unlocks absolute claims. |
| Model topology | **Two models, base provider ≠ judge provider** ([`001`](001-trusted-judge-architecture.md) §3) | Distinct providers kill self-preference bias at near-zero cost. A third gate judge was considered and deferred — circularity is broken more cheaply by a deterministic gate floor, held-out judge questions, and a divergence block. |
| Judge trust | **Provenance, not prohibition** ([`001`](001-trusted-judge-architecture.md) §5.2) | Blocking uncalibrated judges blocks the whole market. Instead every training row is stamped with `signal_provenance`, `judge_version`, and `judge_health`, so nothing is silently unaccounted for and any dataset can be audited after the fact. |
| Human involvement | ~150 labels, two pools, blind ([`001`](001-trusted-judge-architecture.md) §6) | ~90 min of one domain expert makes the judgecard real. Anchor pool gives unbiased κ, targeted pool gives FAIL-class precision — the metric that actually protects training data. |

## Phase index

| Phase | Deliverable | Acceptance |
|---|---|---|
| P0 | Contracts, Postgres metastore, config validation | One JSONL dataset runs through exact matching and a custom LLM question |
| P1 | Read-only connectors, mapping, redaction, splits | Ingest a realistic tool-call + voice dataset without changing its source database |
| P2 | Deterministic + LLM evaluators, judge client, cache, budget | Per-question evaluation report with costs, under a USD budget |
| P2.5 | Latent GT harvesting: handoff mining, outcome joins, active-learning sampler | Targets recovered from a product that reports having no ground truth |
| P3a | Judge health (GT-free): bias probes, self-consistency, invalid rate | `evalloop judge-health` flags a broken judge with zero labels |
| P3b | Judge calibration: human labelling loop, agreement, κ, CIs, human ceiling | 150 labels produce an interpretable card; stale labels are auto-flagged on judge version change |
| P4 | Feedback compiler: SFT/DPO, provenance stamping, eligibility, leakage checks | Reproducible training JSONL where every row records its signal source and judge health |
| P5 | TRL LoRA trainer, candidate registration and inference | Train one adapter, evaluate it against the sealed test set |
| P6 | Promotion gate, slice regressions, experiment bundle | One command produces a defensible promotion decision |
