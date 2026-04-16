from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.service import LEAFService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill derived snapshots (session_page, entity_slot, refreshed session/entity/root) from existing LEAF data."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", action="append", default=[])
    parser.add_argument("--no-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = LEAFService(config_path=args.config, db_path=args.db)
    try:
        corpus_ids = [str(item).strip() for item in args.corpus_id if str(item).strip()]
        if not corpus_ids:
            corpus_ids = service.list_corpora()
        rows: list[dict[str, Any]] = []
        for corpus_id in corpus_ids:
            result = service.backfill_derived_snapshots(
                corpus_id=corpus_id,
                refresh=not args.no_refresh,
            )
            rows.append(
                {
                    "corpus_id": corpus_id,
                    "result": result,
                }
            )
        print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
