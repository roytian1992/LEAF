from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import bootstrap_seed_memory_view
from leaf.store import SQLiteMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a versioned agentic memory seed topic view for an existing LEAF corpus.",
    )
    parser.add_argument("--db", required=True, help="Path to an existing LEAF SQLite database.")
    parser.add_argument("--corpus-id", required=True, help="Corpus id to initialize.")
    parser.add_argument("--name", default="seed-topic-tree-v0", help="Memory view name.")
    parser.add_argument("--parent-view-id", default=None, help="Optional parent memory view id.")
    parser.add_argument("--activate", action="store_true", help="Promote the new view as active for this corpus.")
    parser.add_argument("--no-assign-atoms", action="store_true", help="Only create topics; skip atom assignment.")
    parser.add_argument("--assignment-limit", type=int, default=None, help="Optional atom assignment limit for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SQLiteMemoryStore(args.db)
    try:
        result = bootstrap_seed_memory_view(
            store,
            corpus_id=args.corpus_id,
            name=args.name,
            parent_view_id=args.parent_view_id,
            activate=bool(args.activate),
            assign_atoms=not bool(args.no_assign_atoms),
            assignment_limit=args.assignment_limit,
            metadata={"created_by": "scripts/init_agentic_memory_view.py"},
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
