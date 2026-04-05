# Latest Experiment Protocol

This document records the current canonical reporting protocol for the latest
LEAF experiments.

It is intentionally focused on the experiment contract rather than on the full
benchmark runner implementation. The current public `LEAF` repository packages
the method itself; some older internal benchmark scripts remain tied to the
historical `TraceNav` workspace and are not copied here verbatim.

## Canonical Method Variant

Use the current mature LEAF method as the default experiment target.

At a high level, this means:

- layered evolutionary abstraction
- fidelity-preserved raw details at the leaf layer
- incremental ingest
- hierarchical retrieval across root, session, entity, object-version, and raw
  evidence levels

## Canonical Metrics

The current default report should include:

- `answer_em`
- `answer_f1`
- `bleu1`
- `avg_elapsed_ms`
- `avg_answer_input_tokens_est`
- `judge_score`
- `judge_mean`
- `judge_std`

The intended computation is:

- `answer_em`: exact match after lightweight answer normalization
- `answer_f1`: token-overlap F1 after the same normalization
- `bleu1`: unigram BLEU-style score for answer quality
- `avg_elapsed_ms`: average end-to-end per-question latency for retrieval plus
  answer generation
- `avg_answer_input_tokens_est`: average estimated prompt size fed to the answer
  model
- `judge_score`: aggregate score for each sample after combining the judge runs
- `judge_mean` / `judge_std`: mean and standard deviation across the
  multi-run judge aggregates

The current canonical report should not treat retrieval evidence overlap as a
primary headline metric. In other words:

- do not rely on `evidence_hit_rate`
- do not rely on `evidence_recall`

This is partly a methodological choice and partly a practical one: those fields
can become misleading when different systems expose retrieval evidence in very
different formats.

## LLM-as-Judge

The canonical setting is:

- `LLM-as-judge = enabled`
- `judge_runs = 5`

The expected behavior is:

- keep per-sample judge scores
- aggregate them into report-level `judge_mean` and `judge_std`
- report the number of completed judge runs when relevant

In practice, a good report should retain:

- per-row `judge_scores`
- final per-row `judge_score`
- summary-level `judge_run_scores`
- summary-level `judge_mean`
- summary-level `judge_std`

## Comparison Rules

When LEAF is compared against another method:

- keep the answer model fixed
- keep the memory model fixed when possible
- keep the embedding model fixed when possible
- state clearly whether the experiment uses full sessions or a truncated subset
- state clearly whether the benchmark setting is non-adversarial or adversarial

The comparison goal is to isolate the memory system itself. That means:

- avoid changing the answer model across methods
- avoid changing the embedding model across methods unless unavoidable
- avoid mixing retrieval-policy changes with answer-template changes without
  explicit reporting

## Open-Domain Note

Open-domain evaluation remains sensitive to protocol details. To keep the
comparison defensible:

- use the same conversation scope across methods
- avoid method-specific answer templates
- avoid mixing partial-session and full-session settings in one summary table
- log malformed outputs and repair/retry behavior explicitly

## Public-Repo Scope

The purpose of this repository is to package LEAF itself cleanly for external
use. If a future public release needs a fully runnable benchmark suite, that
suite should be ported into this repository explicitly instead of copying
historical internal experiment scripts wholesale.
