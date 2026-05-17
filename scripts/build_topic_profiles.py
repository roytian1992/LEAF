from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from leaf.clients import EmbeddingClient
from leaf.config import load_config
from leaf.store import SQLiteMemoryStore
from leaf.topic_profile import build_topic_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build language-agnostic topic profiles for LEAF memory views.")
    parser.add_argument("--config", default="", help="Optional LEAF config. Required only when --embed-missing is set.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", default="", help="Build profiles for active view of this corpus.")
    parser.add_argument("--view-id", default="", help="Build profiles for one explicit memory view.")
    parser.add_argument("--all-active", action="store_true", help="Build profiles for all active memory views.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embed-missing", action="store_true", help="Use configured embedding model for topics without event centroids.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-exemplars", type=int, default=12)
    parser.add_argument("--max-terms", type=int, default=24)
    parser.add_argument("--max-entities", type=int, default=16)
    parser.add_argument(
        "--max-document-frequency-ratio",
        type=float,
        default=0.5,
        help="Drop profile terms that appear in more than this ratio of topics in the same view.",
    )
    return parser.parse_args()


def _active_view_ids(store: SQLiteMemoryStore) -> list[str]:
    rows = store.conn.execute(
        """
        select view_id
        from leaf_memory_views
        where active = 1
        order by corpus_id, view_id
        """
    ).fetchall()
    return [str(row["view_id"]) for row in rows]


def resolve_view_ids(store: SQLiteMemoryStore, args: argparse.Namespace) -> list[str]:
    if args.view_id:
        return [str(args.view_id)]
    if args.corpus_id:
        view = store.get_active_memory_view(str(args.corpus_id))
        if view is None:
            raise RuntimeError(f"No active memory view for corpus: {args.corpus_id}")
        return [str(view["view_id"])]
    if args.all_active:
        return _active_view_ids(store)
    raise RuntimeError("Provide --view-id, --corpus-id, or --all-active.")


def main() -> None:
    args = parse_args()
    embedding = None
    if args.embed_missing:
        if not args.config:
            raise RuntimeError("--config is required with --embed-missing")
        config = load_config(args.config)
        if not config.embedding.base_url:
            raise RuntimeError("Embedding model is not configured.")
        embedding = EmbeddingClient(config.embedding)

    store = SQLiteMemoryStore(args.db)
    try:
        view_ids = resolve_view_ids(store, args)
        results: list[dict[str, Any]] = []
        for view_id in view_ids:
            results.append(
                build_topic_profiles(
                    store,
                    view_id=view_id,
                    embedding=embedding,
                    max_exemplars=args.max_exemplars,
                    max_terms=args.max_terms,
                    max_entities=args.max_entities,
                    max_document_frequency_ratio=args.max_document_frequency_ratio,
                    write=not args.dry_run,
                )
            )
        payload = {
            "db": args.db,
            "view_ids": view_ids,
            "dry_run": bool(args.dry_run),
            "embed_missing": bool(args.embed_missing),
            "result_count": len(results),
            "results": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        store.close()


if __name__ == "__main__":
    main()
