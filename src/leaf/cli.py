from __future__ import annotations

import argparse
import json

from .service import LEAFService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LEAF CLI")
    parser.add_argument("--config", default="examples/config.yaml")
    parser.add_argument("--db", default="data/leaf.sqlite3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-json", help="Ingest a conversation JSON file")
    ingest.add_argument("--corpus-id", required=True)
    ingest.add_argument("--title", required=True)
    ingest.add_argument("--input", required=True)

    search = subparsers.add_parser("search", help="Search memory using LEAF retrieval")
    search.add_argument("--corpus-id", required=True)
    search.add_argument("--text", required=True)
    search.add_argument("--raw-span-limit", type=int, default=8)

    root = subparsers.add_parser("get-root", help="Get the root snapshot")
    root.add_argument("--corpus-id", required=True)

    session = subparsers.add_parser("get-session", help="Get a session snapshot")
    session.add_argument("--corpus-id", required=True)
    session.add_argument("--session-id", required=True)

    entity = subparsers.add_parser("get-entity", help="Get an entity snapshot")
    entity.add_argument("--corpus-id", required=True)
    entity.add_argument("--entity", required=True)

    timeline = subparsers.add_parser("get-timeline", help="Get an entity timeline")
    timeline.add_argument("--corpus-id", required=True)
    timeline.add_argument("--entity", required=True)

    corpora = subparsers.add_parser("list-corpora", help="List corpora")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = LEAFService(config_path=args.config, db_path=args.db)
    try:
        if args.command == "ingest-json":
            result = service.append_json(corpus_id=args.corpus_id, title=args.title, path=args.input)
        elif args.command == "search":
            result = service.search(corpus_id=args.corpus_id, question=args.text, raw_span_limit=args.raw_span_limit)
        elif args.command == "get-root":
            result = service.get_root_snapshot(corpus_id=args.corpus_id)
        elif args.command == "get-session":
            result = service.get_session_snapshot(corpus_id=args.corpus_id, session_id=args.session_id)
        elif args.command == "get-entity":
            result = service.get_entity_snapshot(corpus_id=args.corpus_id, entity=args.entity)
        elif args.command == "get-timeline":
            result = service.get_entity_timeline(corpus_id=args.corpus_id, entity=args.entity)
        elif args.command == "list-corpora":
            result = service.list_corpora()
        else:
            raise ValueError(f"Unsupported command: {args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
