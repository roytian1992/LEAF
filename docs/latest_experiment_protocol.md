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

## Metric Definitions

### Answer Normalization

Before computing text-overlap metrics, normalize both gold and predicted
answers:

1. lowercase
2. strip leading and trailing whitespace
3. remove punctuation
4. remove articles: `a`, `an`, `the`
5. split on whitespace and rejoin

In pseudocode:

```text
normalize(answer):
  s = lowercase(trim(answer))
  s = remove_punctuation(s)
  tokens = split_whitespace(s)
  tokens = [t for t in tokens if t not in {"a", "an", "the"}]
  return join_with_space(tokens)
```

### Answer EM

`answer_em` is exact match after normalization.

Formula:

```text
answer_em = 1, if normalize(gold) == normalize(pred)
answer_em = 0, otherwise
```

### Answer F1

`answer_f1` is token-overlap F1 after the same normalization.

Let:

- `G` be the normalized gold tokens
- `P` be the normalized predicted tokens
- `overlap` be the multiset token overlap count

Then:

```text
precision = overlap / |P|
recall    = overlap / |G|
f1        = 2 * precision * recall / (precision + recall)
```

If one side is empty and the other is not, the score is `0`. If both are
empty, the score is `1`.

### BLEU-1

`bleu1` is a unigram BLEU-style score computed on normalized tokens.

Let:

- `overlap_1` be the clipped unigram overlap
- `|P|` be the predicted token count
- `|G|` be the gold token count

Then:

```text
precision_1 = overlap_1 / |P|
brevity_penalty = 1,                         if |P| > |G|
brevity_penalty = exp(1 - |G| / |P|),       otherwise
bleu1 = precision_1 * brevity_penalty
```

### Average Latency

`avg_elapsed_ms` is the mean end-to-end latency per question:

```text
avg_elapsed_ms = mean(elapsed_ms_i)
```

where each `elapsed_ms_i` covers the full retrieval-plus-answer pipeline for a
single evaluated QA item.

### Average Answer Input Tokens

`avg_answer_input_tokens_est` is the mean estimated answer-side prompt size.

The estimate is lightweight rather than tokenizer-exact. At a high level:

```text
estimated_tokens(message_text) ~= ceil(char_count / 4)
```

and the final message estimate adds fixed chat-format overhead on top of the
per-message text estimate.

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

### Judge Aggregation

For each QA item, run the judge `5` times and retain the per-run scores.

Then:

```text
judge_score(row) = aggregate(row.judge_scores)
```

At the report level, compute the mean score for each judge run across all
judged rows:

```text
run_mean_j = mean(score_{i,j})
```

Finally:

```text
judge_mean = mean(run_mean_j)
judge_std  = std(run_mean_j)
```

where `j` ranges over the judge runs and `i` ranges over evaluated rows with a
valid judge score.

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
