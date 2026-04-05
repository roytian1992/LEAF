# LoCoMo Reproduction Notes

This document records the evaluation assumptions we use when adapting LEAF to
LoCoMo-style long-conversation question answering. The goal is not to freeze a
single model choice, but to make the memory-building and retrieval procedure
explicit enough that the experiment can be reproduced fairly.

## Scope

The alignment target is the non-adversarial LoCoMo setting with full sessions.
In other words:

- use the non-adversarial split
- ingest all available sessions for a conversation when running the full
  conversation setting
- avoid truncating the conversation to only the first few sessions unless the
  experiment explicitly states that restriction

LEAF can also be run incrementally in an online setting, but that should be
reported separately from the full-conversation setup.

## Separation of Roles

LEAF separates three roles that are often conflated in memory-system
evaluations:

- answer model: produces the final answer
- memory model: extracts and reconciles memory during ingest
- embedding model: supports retrieval over the stored hierarchy

This separation matters because changes in answer quality should not be
misattributed to the memory-construction model, and vice versa.

## Ingest Procedure

For each conversation, LEAF ingests raw turns and preserves the original text
at the leaf level.

Each turn is processed as follows:

1. Normalize the turn into a `(session_id, speaker, text, timestamp, metadata)`
   record.
2. Write an event record anchored to the raw turn.
3. Extract memory atoms from that turn.
4. Reconcile state-like atoms into evolving memory objects and versions.
5. Refresh the session, entity, and root snapshots.

The important design choice is that raw textual evidence is not discarded after
abstraction. The hierarchy is additive:

- leaf raw spans preserve source fidelity
- events keep turn-level structure
- object versions track evolving state
- session/entity/root snapshots provide compressed navigation layers

## Retrieval Procedure

At question time, LEAF does not rely on a single summary. Instead, it retrieves
across multiple levels of the hierarchy.

The retrieval path is:

1. Embed the question.
2. Score root, entity, and session snapshots using embedding similarity and
   lexical overlap.
3. Retrieve supporting memory-object versions and turn-level evidence.
4. Return a bundle containing:
   - traversal path
   - selected pages
   - selected atoms
   - raw supporting spans

This makes the system robust to questions that require either:

- abstract summary-level navigation
- precise recovery of source-level details

## Open-Domain Questions

Open-domain questions are the easiest place to accidentally under-report memory
system quality because they are especially sensitive to evaluation setup.

When aligning LEAF on open-domain questions, the important choices are:

- do not truncate the conversation if the comparison target uses full sessions
- keep retrieval evidence grounded in the preserved raw spans
- avoid introducing question-type-specific answer templates that would turn the
  evaluation into prompt engineering rather than memory evaluation
- report parser failures or malformed outputs separately from semantic failures

In LEAF, open-domain support comes primarily from two design choices:

- entity/session/root snapshots let the system navigate large histories quickly
- leaf raw spans preserve the details needed when the answer cannot be safely
  reconstructed from abstraction alone

## Fair-Comparison Guidance

When comparing LEAF against other memory systems:

- keep the answer model fixed across methods
- keep the embedding model fixed across methods when possible
- do not silently change from partial sessions to full sessions
- do not mix adversarial and non-adversarial settings in one table
- distinguish memory-build cost from answer-time latency

This last point is particularly important because some systems front-load work
into ingest while others defer more work to retrieval.

## Reporting

At minimum, report:

- answer F1
- BLEU-1 if used by the benchmark setup
- retrieval/evidence metrics when available
- average answer input tokens
- answer-time latency
- whether the run used full sessions or a truncated subset
- whether malformed generations were repaired, retried, or skipped

## Practical Note

The implementation in this repository is intended to preserve the current LEAF
method faithfully while making the packaging clean enough for external use. If
you change retrieval heuristics, question rewriting, or evaluation-time answer
formatting, those changes should be logged explicitly as part of the reported
setup.
