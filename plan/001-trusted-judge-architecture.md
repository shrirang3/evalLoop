# 001 — Trusted-judge architecture

**Status:** accepted, 2026-08-30
**Supersedes:** parts of [`000-build-plan.md`](000-build-plan.md) — P3 (judgecard), P4 (feedback compiler eligibility), P6 (promotion gate). Everything else in 000 stands.

---

## Why this document exists

`000` assumes the customer has ground truth. Read literally, EvalLoop without GT produces:

- κ undefined → judgecard `LOW CONFIDENCE` → ineligible to gate feedback (`000` P3.3)
- no target source → every failure lands in `dropped_no_target` (`000` P4.2)
- feedback build emits **zero rows**

The pipeline runs end to end and delivers nothing. That is not a positioning problem, it is a product problem: most teams shipping LLM products have production traces and *no* ground truth, and cannot cheaply acquire it.

This document changes the default path from *ground-truth-required* to *trusted-judge*, and moves the integrity machinery from **gate** to **provenance** — nothing is blocked, everything is labelled with where its signal came from and how trustworthy that source was at the time.

---

## 1. The two claims

The original design conflates two claims with very different requirements.

| Claim | Requires | Who needs it |
|---|---|---|
| "The model is good" (absolute) | ground truth | few |
| "The model got better / did not regress" (relative) | nothing but a consistent judge | everyone |

Judge bias is largely a **constant**, and constants cancel inside a comparison. The same rubric bias applied to both baseline and candidate subtracts out of the delta. It does *not* subtract out of an absolute score.

**Rule:** without GT calibration, EvalLoop reports **relative** claims only. Absolute scores are rendered but stamped `UNCALIBRATED — relative use only`, and cannot be used as an `absolute` condition in the promotion gate DSL (`000` P6.1).

## 2. Capability tiers

Each tier is independently shippable and independently useful. A customer entering at T0 gets value on day one with zero labelling.

| Tier | Needs from customer | Delivers |
|---|---|---|
| **T0 — Deterministic** | nothing | schema validity, tool-call correctness, hallucinated-ID checks, policy-rule checks, `tool_call_exec`, cost, p95 latency |
| **T1 — Judge health** | nothing | position / verbosity / formatting / perturbation bias, self-consistency, invalid-output rate, judge-vs-judge agreement. *"Your judge flips on 30% of paraphrases — your eval numbers are noise."* |
| **T2 — Regression detection** | nothing | pairwise candidate-vs-baseline, per-slice regression, promotion gate on relative conditions |
| **T3 — Judge calibration** | ~150 human labels (§6) | κ vs human ceiling, confusion matrix, per-class precision/recall, absolute claims unlocked |
| **T4 — Training** | T1 pass (+ T3 for absolute gating) | SFT/DPO compilation, LoRA fine-tune, promotion |

T0–T2 require **zero ground truth**. This is the wedge: `evalloop judge-health` runs against a customer's traces with no labels, no setup, and returns a finding they cannot get anywhere else.

---

## 3. Model topology — two models, not three

Three roles exist conceptually; two models fill them.

| Role | Model |
|---|---|
| **Base** — served / fine-tuned (`train_run.base_model`) | open-weights, local (TRL LoRA, `000` P5.2) |
| **Judge** — mints preference pairs *and* scores the gate | API provider |

### 3.1 Constraint: base provider ≠ judge provider

Kills **self-preference bias** — judges measurably favour outputs from their own family. Given the `000` stack (open-weights base + API judge) this is nearly free; assert it rather than hope for it.

### 3.2 Circular measurement, and why a third model is not the fix

If the judge that mints training pairs also scores the promotion gate, the candidate is optimised to please the grader and then graded by it. It passes by construction. The failure is **silent and inverted**: judge scores rise while real quality falls.

A separate gate model was considered and **rejected for the first release** — cost and config burden out of proportion to the benefit, given cheaper fixes below. The requirement is not a third model. It is: **the gate must contain at least one signal the training process could not optimise against.** Three mechanisms, no extra model:

1. **Deterministic gate floor** — ≥1 T0 check must be present in every gate. A judge cannot game a JSON parser. Reward-hacked models characteristically degrade on objective checks while judge scores climb.
2. **Held-out judge questions** — a subset of rubric questions is marked `holdout: true`, never compiled into training data, used only at the gate. Same judge model, but the candidate was never optimised against those specific rubrics.
3. **Divergence block** — if judge score rises while deterministic pass rate falls, `REJECT` regardless of the judge. One rule, catches the exact failure mode.

**Revisit a distinct gate judge when:** iteration ≥3 on the same product (drift compounds), or the product is pure free-text with no meaningful T0 checks.

### 3.3 The distillation ceiling — documented, not hidden

Training a base model against judge preferences approaches the judge; it does not exceed it. This is the product (compress an expensive model's quality into a cheap servable one), not a defect — but it must be stated plainly in the README so nobody discovers it at iteration four.

> A candidate cannot exceed its judge on judged dimensions.

### 3.4 Config

```yaml
models:
  base:
    provider: huggingface
    model: Qwen/Qwen2.5-7B-Instruct

judge:
  provider: anthropic
  model: claude-sonnet-5

integrity:
  require_distinct_providers: [base, judge]      # §3.1
  gate:
    holdout_questions: [policy_followed, escalation_correct]   # §3.2.2 — never compiled to training
    deterministic_required: true                 # §3.2.1
    block_on_divergence: true                    # §3.2.3
```

Enforced at `evalloop validate` as an **error**, not a runtime warning — it is a comparison between `judge_config.provider` and `train_run.base_model` (`000` P0.5).

---

## 4. P3 split — judge health (GT-free) vs judge calibration (GT)

`000` P3 is one phase requiring ground truth throughout. Split it.

### P3a — Judge health (`judgecard/health.py`), zero labels

Everything already specced in `000` P3.2 needs no ground truth, plus additions:

- Position bias — swap A/B on pairwise questions, flip rate
- Verbosity bias — pad with fluent filler, score delta
- Formatting sensitivity — markdown ↔ prose, score delta
- Perturbation sensitivity — semantics-preserving input paraphrase, prediction flip rate
- **Self-consistency** *(new)* — same input sampled n times, prediction variance
- **Judge-vs-judge agreement** *(new, optional)* — second provider on a subset, agreement %
- Invalid / parse-failure / timeout rate, cost, p50/p95 latency

Ships as its own command: `evalloop judge-health <suite> --traces <snapshot>`. No labels, no run history, no GT. **This is the T1 wedge and it should be the first thing a prospect runs.**

### P3b — Judge calibration (`judgecard/metrics.py`), human labels

`000` P3.1 unchanged (agreement, κ, confusion matrix, per-class P/R/F1, BCa bootstrap CI, small-n guard) — but the labels now come from §6 rather than being assumed present in `ground_truth`.

### P3c — Card status line

Replaces `000` P3.3. Every question in the card carries provenance, not a prohibition:

```
Measured:                    Yes
Judge health:                PASS  (pos 4%, verb 6%, self-consist 0.91)
Calibrated against GT:       No
Claim scope:                 Relative only
Eligible as training signal: Yes (judge-derived, provenance stamped)
```

---

## 5. P4 changes — provenance, not prohibition

### 5.1 New target source: `judge_preference_pair`

The six sources in `000` P4.2 stand. A seventh is added for the trusted-judge path:

`ground_truth_response` | `ground_truth_tool_call` | `human_correction` | `approved_exemplar` | `executably_verified_correction` | `trusted_teacher_correction` | **`judge_preference_pair`**

Judge-derived pairs may be constructed from: best-of-n sampling ranked by judge, two model versions compared pairwise, or deterministic-pass vs deterministic-fail responses to the same prompt (this last one is *objective*, not judged).

**Admission gate — enforced in code:**

```yaml
require:
  - judge_health.position_flip_rate_below: 0.15
  - judge_health.self_consistency_above: 0.80
```

A judge that flips on a coin toss cannot mint training data. Zero labels needed to enforce this.

### 5.2 `allow_uncalibrated` flips from gate to label

`000` P3.3 makes uncalibrated judges opt-in with a loud warning and ineligible by default. Reverse it: **uncalibrated is the default path.** Do not block it — stamp it. Every emitted row carries:

```json
{
  "prompt": [...],
  "chosen": "...",
  "rejected": "...",
  "target_source": "judge_preference_pair",
  "signal_provenance": "judge",
  "judge_version": "sha256:a91f...",
  "judge_health": {
    "verbosity_delta": 0.06,
    "position_flip_rate": 0.04,
    "self_consistency": 0.91
  }
}
```

Nobody is blocked; every row is auditable after the fact and carries its own trust score. Honesty becomes provenance, not prohibition — the stronger form of the original thesis.

### 5.3 Gate on FAIL-class precision, not only κ

Judge error cost is asymmetric for training data:

| Judge error | Consequence |
|---|---|
| False **fail** (said bad, was fine) | mints a pair teaching the model to stop doing something correct. **Actively harmful.** |
| False **pass** (said fine, was bad) | missed opportunity. Harmless. |

κ is a blunt aggregate that can look acceptable while FAIL-class precision is 0.5. `000` P3.1 already computes per-label precision/recall/support — promote it to a gate condition. Updated feedback policy:

```yaml
feedback:
  eligible: true
  strategy: dpo
  chosen:  { source: judge_preference }        # or ground_truth.expected_response
  rejected:{ source: output.text }
  require:
    - judge_health_pass                        # §5.1 — always required
    - not_in_sealed_test
    # the following apply only when human labels exist (T3):
    - judgecard_kappa_above: 0.6
    - judgecard_fail_precision_above: 0.8      # NEW — protects training data
    - judgecard_min_samples: 30
```

### 5.4 Length balancing becomes mandatory for judge-derived pairs

`000` P4.3 specs a length-balance *report*. When `signal_provenance: judge` and `judge_health.verbosity_delta > 0.10`, length balancing is **enforced**, not reported — otherwise DPO on judge-chosen pairs teaches length, which is the documented verbosity bias arriving as a training objective.

---

## 6. Human calibration loop (new, P3b)

Small, bounded, and the input to a consumer `000` already designed. Total ask: **~150 labels + 30 double-labelled ≈ 90 minutes of one domain expert.**

### 6.1 Two pools, reported separately

Random sampling on a 90%-pass product yields ~5 failures — nothing learned about the class that mints training data. Stratified sampling fixes that but biases the agreement estimate. Solution: two pools, never merged into one number.

| Pool | Size | Selection | Used for |
|---|---|---|---|
| **Anchor** | 100 | pure random | unbiased agreement, κ, CI |
| **Targeted** | 50 | 50/50 judge-pass / judge-fail, plus judge-vs-deterministic disagreements, plus paraphrase-flippers from `perturb.py` | FAIL-class precision, finding failures |

Judge-vs-deterministic disagreement is free to compute and near-guaranteed to be interesting: the deterministic check passed but the judge said fail, or the reverse.

### 6.2 Blind labelling — mandatory

The human answers the same question, with the same rubric text, **without seeing the judge's verdict**. Showing it produces anchoring and inflates κ. This is a UI decision, so it costs nothing to get right.

### 6.3 Human ceiling

Double-label 30 of the anchor pool with a second person → **human-human agreement**.

Without it κ is uninterpretable. If two domain experts agree only 72% on "was the tone empathetic", a judge κ of 0.6 is near ceiling and *excellent*; measured against an imaginary 1.0 it looks like failure and invites weeks of rubric tuning against noise. Highest-leverage 30 labels in the process.

### 6.4 Staleness tied to judge version

Labels are valid for one `judge_config.hash` (`000` P2.4). On a rubric edit the hash changes and the card auto-marks:

```
STALE — labels collected against judge v3, current judge is v4
```

Re-label triggers: judge version change, silent provider model update, bias-probe drift, monthly, or every N new traces.

### 6.5 Tooling — CLI + JSONL, no web app

```bash
evalloop label export <run_id> --pool anchor   --n 100 --out anchor.jsonl
evalloop label export <run_id> --pool targeted --n 50  --out targeted.jsonl
evalloop label import anchor.jsonl targeted.jsonl     # recomputes the judgecard
evalloop label status <run_id>                        # coverage, staleness, human ceiling
```

Human fills a `human_label` field in any editor or spreadsheet. A review UI is P11.

### 6.6 Reading the result

| κ (relative to human ceiling) | Verdict |
|---|---|
| < 0.2 | judge broken — fix the rubric, do not ship |
| 0.2 – 0.4 | relative claims only, no absolute scores |
| 0.4 – 0.6 | usable for gating, not for minting training data |
| > 0.6 **and** FAIL precision > 0.8 | trusted for training signal |

---

## 7. P2.5 — Latent ground-truth harvesting (new phase)

"No ground truth" almost always means "no GT *dataset*", not "no GT *possible*". Production systems leak targets constantly, and none of `000` harvests them.

| Source | Yields | Maps to target source |
|---|---|---|
| **Human handoff** — agent fails, a human takes over; the human's action is the correct action | response + tool call | `human_correction` |
| **Executable verification** — tool ran, side effect confirmed | tool call | `executably_verified_correction` |
| **Business outcome** — order completed, ticket closed without reopen, placement made | weak binary label | judgecard label |
| **Retry / rephrase** — user repeated themselves | implicit failure marker | judgecard label |
| **Thumbs up/down** | sparse preference | `judge_preference_pair` seed |

Deliverables: outcome-join mapping in `project.yaml` (same path/transform syntax as `ingest/mapping.py`), a handoff-detection helper, and an **active-learning sampler** that ranks unlabelled traces by judge uncertainty so §6's 150 labels are the 150 most informative ones rather than 150 random ones.

Sequenced after P2, before P3b — harvested targets make the human loop smaller.

---

## 8. Datasets

Two distinct needs, previously conflated.

### 8.1 Demo / example — `examples/support-bot/`

**Choose a tool-calling dataset, not plain chat.** T0 deterministic gates (§3.2.1) are the unhackable signal that makes the whole trusted-judge design safe, and they need objectively checkable outputs. Plain chat provides nothing the judge cannot game.

| Candidate | Licence | Notes |
|---|---|---|
| **[Schema-Guided Dialogue (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)** — *chosen* | **CC BY-SA 4.0** (verified 2026-08-30) | large, many domains, GT API calls + slots; `metadata.domain` gives free slices for `000` P6.2. Share-alike — see §8.4 |
| MultiWOZ 2.2 | **unverified** | smaller, multi-turn, well known; lighter fixture. Check licence before use |
| Glaive function-calling v2 / BFCL | **unverified** | simplest mapping, but single-turn and synthetic — weaker "production trace" story. Check licence before use |

Failures are manufactured: run a weak base model over the inputs, capture as `output`, keep the dataset's own call as an *optional* gold. Under trusted-judge the gold is not required, but having it lets the demo show both T2 and T3 side by side.

### 8.2 Judge validation — `tests/fixtures/`

These carry real human preference labels and exist to prove §3 and P3a are not decorative.

| Dataset | Licence | Use |
|---|---|---|
| **[LLMBar](https://github.com/princeton-nlp/LLMBar)** — *chosen* | **MIT** (verified 2026-08-30) | adversarially built so naive judges fail; the best single fixture for the bias probes. Permissive — **commit the rows**, so the P3a acceptance test runs in CI with no download |
| **[MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)** — *chosen* | **CC BY 4.0** (verified 2026-08-30) | 3.3k expert pairwise votes over 6 models; validates `judgecard/metrics.py` κ against real human votes. Attribution to LMSYS required |
| Chatbot Arena conversations | **unverified** | real prompts + human votes at scale; good for position-bias probes |
| RewardBench | **unverified** | chosen/rejected pairs; validates the §5.1 admission gate |

**LLMBar caveat:** the *Natural* set is assembled from upstream human-preference datasets whose individual instances may carry their own terms. The *Adversarial* set is Princeton-authored and is the half that matters for bias probes — prefer it for the committed fixture.

Adding LLMBar upgrades the `000` P3.5 acceptance test from "we catch judges we built to be broken" to "we catch judges on data designed to break judges."

### 8.3 CI fixture

Keep the hand-written ~50-trace synthetic set in `tests/fixtures/traces/` — deterministic, no downloads, no licence risk. Must cover four kinds: fail-with-target, pass, fail-without-target (must land in `dropped_no_target`), and judge-trap traces.

### 8.4 Licensing — verified 2026-08-30

Licences attach to the **data**, not to the tool. None of the below affects EvalLoop's own Apache-2.0 code licence, and a customer running EvalLoop over their own traces is unaffected entirely.

| Dataset | Licence | Commit rows into this repo? |
|---|---|---|
| SGD | CC BY-SA 4.0 | **No** — fetch script only |
| LLMBar | MIT | **Yes** |
| MT-Bench human judgments | CC BY 4.0 | Yes, with attribution |
| MultiWOZ 2.2, Glaive/BFCL, Chatbot Arena, RewardBench | unverified | Verify before use |

**The share-alike question.** SGD is CC BY-**SA** 4.0, not plain BY. A compiled SFT/DPO dataset derived from SGD prompts is a derivative of the dataset, so *distributing* one obliges you to release it under CC BY-SA 4.0 as well. This does not reach the code, and it does not reach a customer who never distributes their training data — but it does mean EvalLoop must not ship a prebuilt example dataset built from SGD.

**Standing rule:** for anything not permissively licensed, commit a `fetch.py` + mapping YAML and let the user materialise the rows locally. This sidesteps redistribution entirely and keeps the repo small. LLMBar is the exception — MIT, so its rows are committed directly and CI needs no network.

---

## 9. Positioning correction

`README.md` currently reads *"Bring your production traces and ground truth"* and *"it refuses to let you skip the parts that make that claim defensible."* Under this document the first excludes most of the market at the opening sentence and the second becomes false. Replacement thesis:

> Judge-derived signal is the default, because ground truth usually does not exist. Every row records where its signal came from and how trustworthy that judge was at the time. Deterministic checks and held-out questions sit outside the training loop, so improvement is measured by something the training signal could not influence.

Sequence to lead with: **we check your judge before we check your model → we catch regressions between versions → we find the ground truth already sitting in your product → then we train.**

`dropped_no_target` stops being an apology and becomes the upsell: *"680 of your 900 failures have no recoverable target. Here are the 200 worth labelling first."*

---

## 10. Phase deltas against `000`

| Phase | Change |
|---|---|
| P2.5 | **NEW** — latent GT harvesting + active-learning sampler (§7) |
| P3 | **SPLIT** — P3a judge health (GT-free, ships standalone as `evalloop judge-health`), P3b calibration + human labelling loop (§4, §6) |
| P4 | `judge_preference_pair` source; provenance stamping; `allow_uncalibrated` becomes default; FAIL-precision gate; mandatory length balancing for judge-derived pairs (§5) |
| P5 | unchanged |
| P6 | `integrity` block enforced at validate; holdout questions; deterministic gate floor; divergence block; absolute conditions refused on uncalibrated questions (§1, §3) |
| P11 | review UI now also serves the §6 labelling loop |
| P13 | active learning promoted out of P13 into P2.5 |

## 11. Cross-cutting rule changes

| `000` rule | Replacement |
|---|---|
| "Judge feedback without GT is never trusted automatically" | **"Judge feedback without GT is never *unlabelled*."** Every row carries `signal_provenance`, `judge_version`, and `judge_health` at build time; judge-derived rows require a judge-health pass. Test asserts an unstamped row cannot be written. |
| — | **NEW** — "Base provider ≠ judge provider." Test asserts `validate` rejects a config where they match. |
| — | **NEW** — "Every gate contains ≥1 deterministic condition." Test asserts a judge-only gate is rejected. |
| — | **NEW** — "Holdout questions never reach training data." Test asserts a holdout question's results are absent from every compiled dataset. |

## 12. Open questions

1. Judge-vs-judge agreement (P3a) needs a second provider — optional extra, or required for the health card to be considered complete?
2. Best-of-n preference-pair generation requires sampling from the base model at build time. Does that live in `feedback/` or in `train/infer.py`?
3. Weak business-outcome labels (§7) are noisy. Admit them into the judgecard at a discount, or keep them advisory only?
