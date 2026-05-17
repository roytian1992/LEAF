from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf import LEAFService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill atom-to-topic assignments for the active memory view of an existing corpus.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = LEAFService(config_path=args.config, db_path=args.db)
    try:
        result = service.backfill_active_memory_view_assignments(
            args.corpus_id,
            limit=args.limit if args.limit > 0 else None,
        )
    finally:
        service.close()
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
