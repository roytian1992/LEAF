# LEAF Dynamic-Tree Ingest/Search Reality Check - 2026-04-16

## Why this note exists

This note pins down what the current `dynamic-tree` code actually does from ingest to search, and distinguishes:

- structures that are now part of the normal ingest path
- structures that only exist if we explicitly backfill old DBs
- structures that search code currently consumes
- structures that are generated but not actually used by search

The repository now exposes this distinction as two formal ingest workflows:

- `online`: normal incremental append path
- `migration`: append plus optional corpus-wide post-ingest rebuild / backfill


## 1. What normal ingest does now

Entry point:

- `src/leaf/service.py` -> `LEAFService.append_turns()`
- `src/leaf/indexer.py` -> `LEAFIndexer.append_turns()`

Current normal ingest flow:

1. normalize input turns and assign `session_id` / `turn_index`
2. prepare events in parallel
3. merge local turns into extraction chunks before atom extraction
4. extract atoms / state candidates / evidence links
5. write events, atoms, objects, versions, evidence links
6. refresh derived snapshots for all touched sessions and touched subjects
7. refresh the root snapshot

Important point:

- derived snapshots are no longer only an ad hoc backfill step
- `append_turns()` now directly calls:
  - `_refresh_session_snapshot()`
  - `_refresh_entity_snapshot()`
  - `_refresh_root_snapshot()`


## 2. What ingest now materializes by default

### Session side

`_refresh_session_snapshot()` currently rebuilds:

- `session`
- `session_page`
- `session_block`

These are in the normal ingest path now, not just a later repair step.

### Entity side

`_refresh_entity_snapshot()` currently rebuilds by default:

- `entity`
- `entity_slot`

Important nuance:

- `entity_aspect` and `entity_facet` still exist in code, but are now disabled in default ingest
- they are only meant to appear via explicit experimental backfill / opt-in refresh

### Corpus side

`_refresh_root_snapshot()` rebuilds:

- `root`


## 3. What still needs explicit backfill

Backfill entry points exist in `src/leaf/service.py` and `src/leaf/indexer.py`:

- `backfill_derived_snapshots()`
- `backfill_entity_facets()`
- `backfill_entity_bridges()`

Meaning:

- for a fresh ingest, `session/session_page/session_block/entity/entity_slot/root` are produced automatically
- `entity_facet` snapshots only appear if we explicitly run facet backfill
- `bridge` snapshots are still outside the default ingest path and only appear if we run explicit backfill


## 4. What search actually reads today

`src/leaf/search.py` currently loads and scores:

- `root`
- `entity`
- `entity_slot`
- `entity_facet`
- `bridge`
- `session`
- `session_page`
- `session_block`

Notably:

- `entity_aspect` is **not** part of the search snapshot load path right now
- `entity_facet` is supported by search, but only matters if a DB was explicitly backfilled to contain it

So the current reality is:

- `entity_aspect` remains implemented in ingest code
- cached in `service.py`
- but not consumed in `retrieve_leaf_memory()`
- and is now disabled in default ingest

In other words, `entity_aspect` is currently ingest overhead / storage overhead, not an active retrieval mechanism.


## 5. What the current full LoCoMo run actually used

The current full QA report:

- `reports/locomo10_dynamic_tree_mergeatom_yaketag_official_currentsearch_fullqa_6x8_parallel8_20260416.json`

used this DB:

- `data/locomo10_dynamic_tree_mergeatom_yaketag_official_20260415.sqlite3`

Snapshot-kind counts in that DB:

- `entity`: `137`
- `entity_slot`: `260`
- `entity_aspect`: `3370`
- `session`: `272`
- `session_page`: `848`
- `session_block`: `2266`
- `root`: `10`
- `entity_facet`: `0`
- `bridge`: `0`

So that full run could actually use:

- `entity`
- `entity_slot`
- `session`
- `session_page`
- `session_block`
- `root`

But it could **not** use:

- `entity_facet` because the DB had none
- `bridge` because the DB had none

And although the DB contains `entity_aspect`, search currently does not consume it.


## 6. What seems method-defining vs still experimental

### Reasonably safe to lock in as current main pipeline

- merged-turn atom extraction before state building
- `session` + `session_page` + `session_block`
- `entity`
- `entity_slot`
- `root`

These are both:

- in the current normal ingest path
- actually usable by the current search path

### Not part of the locked default method path

- `entity_facet`
- `bridge`
- `entity_aspect`

Why:

- `entity_facet`: search can use it, but the main DB / full run above did not contain or validate it
- `bridge`: only exists via explicit backfill, so it is still experimental by workflow definition
- `entity_aspect`: not used by search, so it is not part of effective retrieval yet


## 7. Recommendation for locking down ingest

If the goal is to freeze a clean and honest ingest definition now, the safest stance is:

Keep in formal ingest:

- `session`
- `session_page`
- `session_block`
- `entity`
- `entity_slot`
- `root`

Treat as experimental until revalidated end-to-end:

- `entity_facet`
- `bridge`

Keep disabled until there is a reason to wire it into search:

- `entity_aspect`

Because the current codebase would otherwise sit in an awkward middle state where `entity_aspect` is generated but not actually used.


## 8. Practical implication

If we want to claim a new ingest pipeline as the method, we should do one of these before the next “official” comparison:

1. re-ingest a fresh DB with the frozen ingest path and run full QA on that DB
2. or explicitly trim ingest so only the structures that search really uses are generated

Until then, the fairest reading of current results is:

- the latest full run validates the current search logic mainly on top of `entity + entity_slot + session_page/session_block`
- it does **not** validate `entity_facet`
- it does **not** validate `bridge`
- it also does not justify keeping `entity_aspect` as a method component yet
