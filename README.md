# LEAF

LEAF stands for **Layered Evolutionary Abstraction with Fidelity-preserved Memory**.

This repository packages our long-horizon conversational memory system as a standalone project. The design has two defining properties:

- **Layered evolutionary abstraction**: memory is organized into a hierarchy of evolving snapshots rather than a single flat summary.
- **Fidelity-preserved leaves**: the bottom layer retains raw textual detail, so retrieval can always fall back to source-level evidence instead of relying on lossy compression alone.

## What LEAF Provides

LEAF is intended for long conversational histories where a memory system must both:

- evolve over time as new turns arrive
- preserve fine-grained details that may become important later

The current package includes:

- incremental memory ingestion over conversation turns
- root / session / entity snapshots
- entity timelines
- hierarchical retrieval over snapshots, events, and object versions
- a lightweight CLI for local use
- a small Python API for integration into other projects

## Repository Layout

```text
LEAF/
├── examples/
│   ├── config.yaml
│   └── demo_conversation.json
├── src/
│   └── leaf/
│       ├── cli.py
│       ├── clients.py
│       ├── config.py
│       ├── extract.py
│       ├── grounding.py
│       ├── indexer.py
│       ├── records.py
│       ├── service.py
│       ├── store.py
│       ├── normalize.py
│       ├── schemas.py
│       └── search.py
└── pyproject.toml
```

## Installation

```bash
cd LEAF
pip install -e .
```

## Configuration

Edit `examples/config.yaml` to point to your OpenAI-compatible service endpoints:

- an answer model
- an embedding model
- an optional separate memory model

The memory model is used during ingestion for atom extraction and reconciliation. If `additional_llm` is omitted, LEAF falls back to `llm`.

LEAF now exposes two ingest workflows through the same API:

- `online`: incremental append, intended for normal live usage
- `migration`: append plus corpus-wide post-processing, intended for offline historical loads, rebuilds, or snapshot backfills

The optional `ingest` section in `examples/config.yaml` controls the default:

```yaml
ingest:
  mode: online
  migration_refresh_derived: true
  migration_build_entity_facets: false
  migration_build_entity_bridges: false
  migration_bridge_mode: hybrid
```

## Quick Start

### 1. Ingest a conversation JSON

```bash
leaf \
  --config examples/config.yaml \
  --db data/leaf.sqlite3 \
  ingest-json \
  --corpus-id demo \
  --title "Demo Conversation" \
  --input examples/demo_conversation.json
```

To force the offline migration-style path for a bulk historical load:

```bash
leaf \
  --config examples/config.yaml \
  --db data/leaf.sqlite3 \
  ingest-json \
  --corpus-id demo \
  --title "Demo Conversation" \
  --input examples/demo_conversation.json \
  --ingest-mode migration
```

Accepted JSON formats include:

- a raw list of turns
- an object with `turns`
- an object with `conversation`
- an object with `messages`

Each turn should ideally contain:

- `session_id`
- `speaker`
- `text`
- optional `timestamp`

### 2. Retrieve memory for a question

```bash
leaf \
  --config examples/config.yaml \
  --db data/leaf.sqlite3 \
  search \
  --corpus-id demo \
  --text "What fields would Caroline be likely to pursue in her education?" \
  --raw-span-limit 8
```

The search result returns:

- `traversal_path`
- `pages`
- `atoms`
- `raw_spans`
- retrieval timing

### 3. Inspect the hierarchy

```bash
leaf --config examples/config.yaml --db data/leaf.sqlite3 get-root --corpus-id demo

leaf --config examples/config.yaml --db data/leaf.sqlite3 get-session --corpus-id demo --session-id session-2

leaf --config examples/config.yaml --db data/leaf.sqlite3 get-entity --corpus-id demo --entity caroline

leaf --config examples/config.yaml --db data/leaf.sqlite3 get-timeline --corpus-id demo --entity caroline
```

To run migration-only backfill steps on an already ingested corpus:

```bash
leaf \
  --config examples/config.yaml \
  --db data/leaf.sqlite3 \
  migrate-corpus \
  --corpus-id demo \
  --build-entity-bridges
```

### 4. Use the Python API

```python
from leaf import LEAFService

service = LEAFService(config_path="examples/config.yaml", db_path="data/leaf.sqlite3")
try:
    service.append_json(
        corpus_id="demo",
        title="Demo Conversation",
        path="examples/demo_conversation.json",
    )
    result = service.search(
        corpus_id="demo",
        question="What is Caroline planning to study?",
        raw_span_limit=6,
    )
    print(result["raw_spans"])
finally:
    service.close()
```

If you want the offline migration-style path through the Python API:

```python
service.append_json(
    corpus_id="demo",
    title="Demo Conversation",
    path="examples/demo_conversation.json",
    ingest_mode="migration",
)
```

### 5. Run an Incremental Chat-Loop Demo

The repository also includes a minimal script showing how LEAF is updated turn by turn and then queried:

```bash
python examples/chat_loop.py \
  --config examples/config.yaml \
  --db data/chat_loop.sqlite3 \
  --corpus-id demo-chat
```

This is the closest example to an online memory workflow:

- append new conversational turns incrementally
- maintain the evolving memory state in SQLite
- retrieve grounded memory for a downstream question

For offline historical imports or benchmark rebuilds, prefer `migration` mode instead of this chat-loop style path.

## Design Notes

LEAF does not reduce the entire history into a single summary. Instead, it maintains:

- event-level records grounded in raw turns
- object/version state for evolving memory facts
- session snapshots
- entity snapshots
- a root snapshot spanning the whole corpus

This structure is meant to support both:

- efficient high-level navigation
- recovery of raw textual evidence when detail matters

## Reproduction Notes

For LoCoMo-specific setup details, evaluation assumptions, and open-domain alignment notes, see [docs/locomo_reproduction.md](docs/locomo_reproduction.md).

For the current canonical experiment reporting protocol, including the latest
metric set and the `LLM-as-judge x 5` convention, see
[docs/latest_experiment_protocol.md](docs/latest_experiment_protocol.md).

## Development

Run the smoke test suite with:

```bash
python -m unittest tests.test_smoke
```

## Status

This repository is a cleaned packaging of the current mature version of our method. It is designed to be understandable and publishable as a standalone GitHub project, while staying close to the code path used in our latest experiments.
