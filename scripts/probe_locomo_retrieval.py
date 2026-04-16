from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from leaf.service import LEAFService  # noqa: E402
import eval_locomo as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a retrieval-only LoCoMo probe and summarize missed_in_retrieval."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-limit", type=int, default=6)
    parser.add_argument("--raw-span-limit", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = base.load_locomo_samples(args.input)
    service = LEAFService(config_path=args.config, db_path=args.db)
    try:
        rows: list[dict[str, Any]] = []
        by_sample: dict[str, dict[str, Any]] = {}
        search_times: list[float] = []
        for sample in samples:
            sample_id, _turns = base.locomo_sample_to_turns(sample)
            qas = base.locomo_sample_to_qas(sample)
            corpus_id = f"locomo_{base.sanitize_sample_id(sample_id)}"
            sample_rows: list[dict[str, Any]] = []
            for qa in qas:
                started_at = time.perf_counter()
                retrieval = service.search(
                    corpus_id=corpus_id,
                    question=str(qa["question"]),
                    snapshot_limit=args.snapshot_limit,
                    raw_span_limit=args.raw_span_limit,
                )
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                search_times.append(elapsed_ms)
                retrieved_dia_ids: list[str] = []
                for span in retrieval.get("raw_spans") or []:
                    dia_id = str(((span.get("metadata") or {}).get("dia_id") or "")).strip()
                    if dia_id:
                        retrieved_dia_ids.append(dia_id)
                gold_evidence = [str(item).strip() for item in (qa.get("evidence") or []) if str(item).strip()]
                missed = bool(gold_evidence) and not set(gold_evidence).intersection(retrieved_dia_ids)
                row = {
                    "sample_id": sample_id,
                    "corpus_id": corpus_id,
                    "question_index": int(qa["question_index"]),
                    "category_name": str(qa["category_name"]),
                    "question": str(qa["question"]),
                    "gold_evidence": gold_evidence,
                    "retrieved_dia_ids": retrieved_dia_ids,
                    "missed_in_retrieval": missed,
                    "search_elapsed_ms": round(elapsed_ms, 2),
                }
                rows.append(row)
                sample_rows.append(row)
            sample_missed = sum(1 for row in sample_rows if row["missed_in_retrieval"])
            by_sample[sample_id] = {
                "question_count": len(sample_rows),
                "missed_in_retrieval": sample_missed,
                "avg_search_elapsed_ms": round(mean(row["search_elapsed_ms"] for row in sample_rows), 2)
                if sample_rows
                else None,
                "median_search_elapsed_ms": round(median(row["search_elapsed_ms"] for row in sample_rows), 2)
                if sample_rows
                else None,
            }
        payload = {
            "input": str(Path(args.input).resolve()),
            "db": str(Path(args.db).resolve()),
            "snapshot_limit": int(args.snapshot_limit),
            "raw_span_limit": int(args.raw_span_limit),
            "summary": {
                "sample_count": len(samples),
                "question_count": len(rows),
                "missed_in_retrieval": sum(1 for row in rows if row["missed_in_retrieval"]),
                "avg_search_elapsed_ms": round(mean(search_times), 2) if search_times else None,
                "median_search_elapsed_ms": round(median(search_times), 2) if search_times else None,
            },
            "by_sample": by_sample,
            "rows": rows,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(output_path.resolve()))
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
