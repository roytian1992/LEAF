from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
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
    parser = argparse.ArgumentParser(description="Ingest LoCoMo samples into a fresh LEAF SQLite DB.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--ingest-prepare-workers", type=int, default=0)
    parser.add_argument("--ingest-mode", choices=["online", "migration"], default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def add_numeric_maps(rows: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            total[str(key)] = int(total.get(str(key), 0)) + int(value)
    return dict(sorted(total.items()))


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.ingest_prepare_workers > 0:
        os.environ["LEAF_INGEST_PREPARE_WORKERS"] = str(max(1, args.ingest_prepare_workers))

    service = LEAFService(config_path=args.config, db_path=args.db)
    try:
        if not args.ingest_mode:
            args.ingest_mode = str(service.config.ingest.mode)
        samples = base.load_locomo_samples(args.input)
        if args.sample_limit > 0:
            samples = samples[: args.sample_limit]

        started_at = time.perf_counter()
        ingest_rows: list[dict[str, Any]] = []
        ingest_metric_rows: list[dict[str, int]] = []
        snapshot_rows: list[dict[str, int]] = []
        state_action_rows: list[dict[str, int]] = []
        llm_call_rows: list[dict[str, int]] = []

        for sample in samples:
            sample_id, turns = base.locomo_sample_to_turns(sample)
            corpus_id = f"locomo_{base.sanitize_sample_id(sample_id)}"
            print(
                f"[locomo-ingest] sample={sample_id} corpus={corpus_id} turns={len(turns)} ingest_start",
                flush=True,
            )
            ingest_started = time.perf_counter()
            ingest_result = base.maybe_ingest_sample(
                service,
                corpus_id=corpus_id,
                title=f"LoCoMo {sample_id}",
                turns=turns,
                refresh=args.refresh,
                ingest_mode=args.ingest_mode,
            )
            ingest_elapsed_ms = (time.perf_counter() - ingest_started) * 1000.0
            session_ids = base.ordered_unique([str(turn.get("session_id") or "").strip() for turn in turns])
            row = {
                "sample_id": sample_id,
                "corpus_id": corpus_id,
                "turn_count": len(turns),
                "session_count": len([item for item in session_ids if item]),
                "ingested": bool(ingest_result["ingested"]),
                "reused": bool(ingest_result["reused"]),
                "ingest_elapsed_ms": round(ingest_elapsed_ms, 2),
                "ingest_metrics": ingest_result.get("result") or {},
            }
            ingest_rows.append(row)
            metrics = dict(row["ingest_metrics"] or {})
            if metrics:
                ingest_metric_rows.append(
                    {
                        "events_written": int(metrics.get("events_written") or 0),
                        "atoms_written": int(metrics.get("atoms_written") or 0),
                        "objects_written": int(metrics.get("objects_written") or 0),
                        "state_candidates": int(metrics.get("state_candidates") or 0),
                        "evidence_links_written": int(metrics.get("evidence_links_written") or 0),
                        "input_text_chars": int(metrics.get("input_text_chars") or 0),
                        "input_text_tokens_est": int(metrics.get("input_text_tokens_est") or 0),
                        "snapshot_upserts": int(metrics.get("snapshot_upserts") or 0),
                    }
                )
                snapshot_rows.append(dict(metrics.get("snapshot_upserts_by_kind") or {}))
                state_action_rows.append(dict(metrics.get("state_action_counts") or {}))
                llm_call_rows.append(dict(metrics.get("memory_llm_calls_est") or {}))
            print(
                f"[locomo-ingest] sample={sample_id} ingest_done reused={ingest_result['reused']} "
                f"elapsed_ms={round(ingest_elapsed_ms, 2)}",
                flush=True,
            )

        payload = {
            "input": str(Path(args.input).resolve()),
            "db": str(Path(args.db).resolve()),
            "config": str(Path(args.config).resolve()),
            "ingest_mode": str(args.ingest_mode),
            "sample_limit": int(args.sample_limit),
            "ingest_prepare_workers": int(args.ingest_prepare_workers),
            "refresh": bool(args.refresh),
            "completed": True,
            "summary": {
                "sample_count": len(samples),
                "ingest_reused_count": sum(1 for row in ingest_rows if row["reused"]),
                "ingest_new_count": sum(1 for row in ingest_rows if row["ingested"]),
                "ingest_elapsed_ms_total": round((time.perf_counter() - started_at) * 1000.0, 2),
                "ingest_avg_elapsed_ms": round(
                    sum(float(row["ingest_elapsed_ms"]) for row in ingest_rows) / len(ingest_rows),
                    2,
                )
                if ingest_rows
                else None,
                "ingest_turn_count_total": sum(int(row["turn_count"]) for row in ingest_rows),
                "ingest_session_count_total": sum(int(row["session_count"]) for row in ingest_rows),
                **add_numeric_maps(ingest_metric_rows),
                "ingest_snapshot_upserts_by_kind_total": add_numeric_maps(snapshot_rows),
                "ingest_state_action_counts_total": add_numeric_maps(state_action_rows),
                "ingest_memory_llm_calls_est_total": add_numeric_maps(llm_call_rows),
            },
            "ingest": ingest_rows,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(output_path.resolve()))
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
