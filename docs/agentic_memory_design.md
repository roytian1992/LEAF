# Agentic Tree Memory Design

## Goal

Build a self-evolving memory system on top of the existing LEAF dynamic-tree store.

The online memory path must stay stable and low-latency. Background evolution creates versioned memory views that can be evaluated, promoted, or rolled back without rewriting the immutable event / atom / object evidence layer.

## Architecture

```text
Immutable memory core
  leaf_events
  leaf_atoms
  leaf_objects
  leaf_object_versions
  leaf_evidence_links
  leaf_snapshots

Versioned agentic memory view
  leaf_memory_views
  leaf_topic_nodes
  leaf_topic_assignments
  leaf_topic_edges
  leaf_retrieval_policies
  leaf_search_traces
  leaf_selfqa_tasks
  leaf_evolution_runs
```

Online usage reads only the active view:

```text
question
  -> query schema
  -> active topic routing
  -> entity / session / time / bridge retrieval
  -> evidence role selection
  -> answer view
```

Background evolution writes a candidate view:

```text
trigger
  -> collect new atoms and search failures
  -> build candidate topic tree
  -> split / merge / rename / promote / demote topics
  -> update bridges and per-topic retrieval policies
  -> generate evidence-grounded self-QA
  -> evaluate retrieval-first metrics
  -> promote only if gates pass
```

## Current MVP

The first implementation is intentionally conservative:

- create a versioned seed topic tree
- assign existing atoms to seed topics with deterministic keyword matching for benchmark backfills
- incrementally assign newly ingested atoms to the current active topic view
- accumulate online evolution pending state on the active view
- trigger online topic growth every configured number of new turns or atoms
- fork the active tree into a promoted grown view when stable new concepts appear
- expose the active tree outline for inspection
- inject active topic hints into later atom extraction prompts and atom metadata
- keep topic assignment out of baseline retrieval when no active view exists
- optionally promote the view as active
- optionally record search traces against the active view
- generate evidence-grounded self-QA tasks from stored atoms
- validate generated self-QA tasks with a memory-side LLM before freezing them
- evaluate retrieval against frozen gold event / atom evidence paths
- write promotion-gate results back into memory view metrics
- optionally record promotion-gate runs in `leaf_evolution_runs`
- route queries to active-view topics as a shadow signal during evaluation
- compare shadow-routed topics with gold and retrieved atom topic assignments
- do not alter default retrieval behavior

Initial seed topics:

- `personal_profile`
- `relationships`
- `work_education`
- `health_emotion`
- `hobbies_media`
- `travel_places`
- `plans_tasks`
- `preferences_opinions`
- `events_timeline`
- `misc`

## Online Incremental Design

The production design treats seed topics as a global prior, not as a corpus-specific topic model induced from a complete conversation.

Online hot path:

```text
new turn / document chunk
  -> event / atom / object ingest
  -> active topic hints guide atom extraction, if an active view exists
  -> if active memory view exists:
       assign only newly written atoms to that active view
     else:
       leave retrieval unchanged
  -> accumulate pending turns / atoms for background evolution
  -> query uses the active view as a read-only routing / expansion signal
```

Background cold path:

```text
growth trigger or schedule
  -> inspect recent atoms, repeated new concepts, low-confidence / misc assignments
  -> inspect search traces and self-QA shadow misses
  -> write candidate memory view
  -> run frozen shadow eval and churn gates
  -> promote candidate only if gates pass
```

For experiments, batch backfill is allowed because benchmark corpora are static. That backfill should be described as an offline approximation of the incremental state after all atoms have arrived, not as the online learning mechanism itself.

Implemented online MVP:

- `LEAFService.append_turns(...)` skips online evolution for `migration` mode, preserving the historical batch-import path.
- For normal appends, the active memory view stores `online_evolution_pending` with pending turn count, atom count, and a bounded list of recent atom ids.
- Default trigger thresholds are 50 new turns or 40 new atoms.
- `grow_topic_tree_from_recent_atoms(...)` copies the current active tree into a new view, grows level-2 child topics from stable repeated recent concepts, reassigns evidence atoms to the new topics, and promotes the view.
- The current implementation uses a conservative lexical clusterer. It is a production-shaped scaffold, not the final self-QA-gated evolution policy.
- The next step is to run the same self-QA shadow gate before promotion, so automatic growth is evaluated before becoming active.

## Promotion Gate

A candidate view should become active only when a frozen evaluation snapshot shows:

- path recall does not regress
- multi-hop role coverage improves
- unsupported answer rate does not increase
- search latency increase stays within the configured budget
- retrieved token count increase stays within the configured budget

Current gate implementation:

- reads a frozen retrieval eval report
- checks minimum task count, event recall, event path hit rate, atom recall, atom path hit rate
- optionally checks absolute latency and baseline regressions
- writes `promotion_gate` metrics into the memory view
- optionally promotes the view and records an evolution run

## First Experiment Track

1. Bootstrap seed topic views on existing reusable DBs.
2. Generate a frozen self-QA snapshot with gold evidence paths.
3. Run retrieval-only eval against current dynamic-tree search.
4. Add topic routing as a shadow retrieval signal.
5. Compare evidence recall, path recall, topic route accuracy, latency, and token budget.

Current topic-routing result on the first `gvd_emily` validated smoke:

- keyword shadow router: mean topic recall 0.0, topic path hit rate 0.0
- LLM shadow router: mean topic recall 0.5, topic path hit rate 0.5
- retrieved evidence topic coverage: mean retrieval topic recall 1.0

Interpretation: topic routing is not ready for hard filtering. It should first be used as a soft expansion or reranking signal, and topic assignment noise should feed the next background evolution step.

## Evolution A/B Protocol

Use the same frozen self-QA file for both conditions:

- no evolution: evaluate the base active view with `--topic-view-id <base_view_id>`
- with evolution: create a candidate view from shadow misses, then evaluate it with `--topic-view-id <candidate_view_id>`
- compare the two eval reports with `scripts/compare_memory_view_eval.py`
- run the promotion gate against the candidate report and baseline report

For the current MVP, default retrieval does not consume the topic view. Therefore event / atom recall should stay unchanged unless we later enable topic-aware retrieval. The meaningful early signals are:

- topic route recall
- topic path hit rate
- retrieved evidence topic coverage
- latency and task-count parity
- assignment churn, especially how many atoms moved to new topics

First evolved candidate on `gvd_emily`:

- candidate view: `view_fcb7db2f310faa3bc4`
- added topics: `food_cooking`, `social_chitchat`, `reading_books`
- reassigned atoms: 97
- mean topic recall improved from 0.5 to 0.8333
- topic path hit rate improved from 0.5 to 0.8333
- event / atom recall stayed at 1.0

This candidate passed the report-based gate but was not promoted. The reassignment count is high for a 6-task smoke, so it should be treated as validated evidence for the evolution mechanism, not as a production-ready taxonomy update.

The first branch is `agentic-memory-evolution` in:

```text
/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory
```
