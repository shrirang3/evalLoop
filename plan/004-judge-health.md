# 004 — Judge health (P3a)

**Status:** accepted, 2026-09-07
**Extends:** [`001-trusted-judge-architecture.md`](001-trusted-judge-architecture.md) §4 P3a, which specifies the probes generically. This makes them specific to the three judged checks that now exist, adds the sampling mechanics `001` left open, and puts a cost ceiling on the whole thing.
**Completes:** [`002-tool-registry-and-selection.md`](002-tool-registry-and-selection.md) §3.2 — ambiguity abstention was specified there and refused in code for want of a per-sample cache key. That key lands here.

---

## Why now

Four tool checks ship. Two of them ask a judge nobody has measured:

| Check | Judge? | Measured? |
|---|---|---|
| `tool_registry_check` | no | n/a — objective |
| `tool_call_outcome` | no | n/a — objective |
| `tool_selection` | **yes** | **no** |
| `text_matches_tools` | **yes** | **no** |

The wrong-tool table is the product, and half of it rests on an instrument with no calibration of any kind — not even the free kind. A judge that picks whichever tool appears first in the catalogue would produce a table that looks exactly like a real finding.

Zero labels are required for any of this. That is the point: `evalloop judge-health` is the first thing a prospect runs, before ingest of any labelled data, before a judgecard, before anything is promised (`001` §2, T1).

---

## 1. The probes are check-shaped

`001` §4 lists probes written for a generic `llm_question`. Two of the three judged checks are not that shape, and the generic version of each probe is either meaningless or measures the wrong thing.

| Probe | `llm_question` | `tool_selection` | `text_matches_tools` |
|---|---|---|---|
| **Position** | swap A/B in a pairwise question | **shuffle the catalogue order** | — (no ordered options) |
| **Paraphrase** | reword the input | reword the customer request | reword the reply, keeping the claim |
| **Verbosity** | pad with fluent filler | pad the request | **pad the reply** — the direct false-fail risk |
| **Formatting** | markdown ↔ prose | — | markdown ↔ prose in the reply |
| **Self-consistency** | sample *n* | sample *n* | sample *n* |

**Catalogue shuffle is the headline probe for this product.** `tool_selection` hands the judge an enum whose order comes from `sorted()` over a YAML file. If the pick moves when the order moves, every row of the wrong-tool table is an artifact of alphabetisation. Nothing else in the system would reveal that, and it costs one extra call per sampled trace to find out.

**Verbosity on the reply is the headline for `text_matches_tools`.** The judge is asked whether a reply matches its calls; padding the reply with apology and warmth must not move that answer, because a longer reply is not a different action. A flip here is the false-fail direction (`001` §5.3) arriving in the only check that reads prose.

---

## 2. Perturbations are frozen, never regenerated

Paraphrase and padding need text that does not exist in the trace. Generating it means a model call, which means the probe would produce different inputs on every run — and a bias measurement that moves between runs measures nothing.

**Rule:** a probe's perturbed inputs are generated once, stored in the snapshot alongside the trace, and reused. Regenerating is a new probe version, not a new run of the same probe.

```
perturbation
  trace_id · probe · variant_index
  text            the generated variant
  generator_hash  model + prompt that produced it
```

Two consequences worth stating. Generation is itself a judge-family model call, so it is budgeted and cached like any other. And a perturbation set is part of the measurement's identity: comparing a flip rate against one built from different paraphrases is comparing two different numbers.

Position and formatting probes are deterministic transforms — shuffle with a fixed seed, markdown ↔ prose by rule — and need no generation or storage.

---

## 3. Sampling, and the cache key that blocks it

`ToolSelectionEvaluator` currently refuses `samples > 1`:

> Repeated sampling needs a cache key that varies per sample, which arrives with the P3a self-consistency probe; asking a temperature-0 judge the same question n times and calling the agreement 'consistency' would be measuring the cache.

Both halves of that are the design here.

**3.1 The cache key gains a sample index.** `cache_key(judge_version, prompt, schema)` becomes `cache_key(judge_version, prompt, schema, sample_index)`. Sample 0 keeps the existing key, so every cached answer in every existing metastore stays valid and no run pays twice for work already done.

**3.2 `samples > 1` requires `temperature > 0`.** Asserted at suite construction, not warned about at runtime. A temperature-0 judge asked the same question five times returns the same answer five times, and reporting that as `self_consistency: 1.00` is a confident lie about the least reliable part of the system.

**3.3 Consistency is per trace, not only aggregate.** `001` §3 treats self-consistency as one number on a card. It is also a per-trace confidence signal, and that is what `002` §3.2 needs.

---

## 4. Abstention — completing `002` §3.2

Rule 17 already stands: *a judge that disagrees with itself across samples abstains; it does not fail the trace.* It has been unenforceable. With §3 it becomes:

```yaml
- id: tool_selection
  samples: 5
  abstain_below: 0.6      # 3 of 5 must agree
```

Below the threshold the result is `not_applicable` with the vote recorded, not `False`. The wrong-tool table gains the row `002` §4 always showed and could never produce:

```
  ambiguous (judge unstable across samples)     23
```

Default is `samples: 1`, which means no abstention and no extra cost — the behaviour that ships today. Turning it on is a deliberate spend.

---

## 5. Cost, which `001` does not address

Probes multiply judge calls, and the multiplier is the product of three numbers:

```
calls = traces × judged_checks × (1 + probes + samples - 1)
```

The shipped example at full sampling: 14 × 2 × (1 + 4 + 4) = 252 calls for 14 traces. On a real snapshot of 40,000 that is not a run anybody starts twice.

**So probes run on a sample, never the snapshot.** `--sample 200` by default, drawn with a fixed seed so the number is reproducible, with the sample size printed on the card because a flip rate over 30 traces and one over 2,000 are different claims. The existing `--budget-usd` ceiling applies and a run that hits it reports `partial` rather than silently truncating.

Ordering matters too: the cheap deterministic probes (position, formatting) run first, because a judge that fails catalogue-shuffle at 40% needs no paraphrase budget spent on it.

---

## 6. What the card says

Per judged evaluator, not per judge — the same model can be steady on one question and useless on the next (`001` §3, "can I trust this judge, on **this** question?").

```
tool_selection                          judge claude-sonnet-5 · n=200 sampled

  catalogue position flip     4%   PASS   (<15%)
  paraphrase flip            11%   PASS   (<15%)
  self-consistency          0.91   PASS   (>0.80)
  invalid answers            0.5%  PASS   (<2%)
  p95 latency               1.9s
  cost                      $0.41

  VERDICT: PASS — eligible to mint training data (001 §5.1)
```

The verdict gates **training data**, per `001` §5.1, and nothing else. It does not gate the report: provenance, not prohibition (`001` §5.2). A failing judge's wrong-tool table still renders, stamped with the health verdict that says how much to believe it.

---

## 7. Explicitly not in P3a

κ, the confusion matrix, per-class precision, and FAIL-class precision all need human labels and are P3b (`001` §4, §6). Judge health answers *"is this judge stable?"*. It cannot answer *"is this judge right?"* — a judge that is confidently and consistently wrong passes every probe here.

Worth saying plainly in the card's own output, because a green health card is the easiest thing in this system to over-read.

---

## 8. Deliverables

| | |
|---|---|
| `judgecard/health.py` | probe runner, per-check probe registry, thresholds |
| `judgecard/perturb.py` | deterministic transforms; generated variants frozen to the snapshot |
| `judge/client.py` | sample index in the cache key; temperature assertion |
| `contracts/suite.py` | `samples`, `abstain_below` on the two judged tool checks |
| `evaluate/llm/selection.py` | lift the `samples > 1` refusal; per-trace vote and abstention |
| `store/models.py` | `perturbation` table, `judge_health` rows keyed by (run, evaluator) |
| `cli/judge_health.py` | `evalloop judge-health <suite> --traces <snapshot> --sample 200` |
| `report/tool_calls.py` | ambiguous row; health verdict in the footer |

**Acceptance:** point `evalloop judge-health` at a snapshot with no labels and no run history, and it reports per-check flip rates and self-consistency, with a deliberately broken judge (a provider that answers by list position) failing the catalogue-shuffle probe.

---

## Rules added

22. A probe never regenerates its perturbation. Generated inputs are frozen into the snapshot; regenerating is a new probe version.
23. `samples > 1` requires `temperature > 0`, asserted at construction. Otherwise the measurement is of the cache.
24. Probes run on a sample with a recorded size. A flip rate without its denominator is not a number.
25. Judge health gates training data only. It never gates a report — the report carries the verdict instead.

Each is enforced by a test, not by convention.