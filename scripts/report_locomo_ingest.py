from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.service import LEAFService  # noqa: E402


def load_locomo_samples(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LoCoMo file should be a list of samples.")
    return [item for item in payload if isinstance(item, dict)]


def sanitize_sample_id(sample_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(sample_id or "").strip())
    return cleaned.strip("_").lower() or "locomo_sample"


def locomo_sample_to_turns(sample: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    sample_id = str(sample.get("sample_id") or "locomo-sample")
    conversation = sample.get("conversation") or {}
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo sample conversation must be a dict.")
    session_keys = sorted(
        [key for key in conversation.keys() if re.fullmatch(r"session_\d+", str(key)) and isinstance(conversation.get(key), list)],
        key=lambda value: int(str(value).split("_")[1]),
    )
    turns: list[dict[str, Any]] = []
    for session_key in session_keys:
        session_turns = conversation.get(session_key) or []
        timestamp = conversation.get(f"{session_key}_date_time")
        if not isinstance(session_turns, list):
            continue
        for turn in session_turns:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            turns.append(
                {
                    "session_id": session_key,
                    "speaker": str(turn.get("speaker") or "unknown"),
                    "text": text,
                    "timestamp": str(timestamp) if timestamp else None,
                    "dia_id": str(turn.get("dia_id") or "").strip() or None,
                    "img_url": turn.get("img_url"),
                    "blip_caption": turn.get("blip_caption"),
                }
            )
    return sample_id, turns


def add_numeric_maps(rows: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            total[str(key)] = int(total.get(str(key), 0)) + int(value)
    return dict(sorted(total.items()))


def ingest_one(
    *,
    config_path: str,
    db_dir: str,
    ingest_mode: str,
    sample: dict[str, Any],
    refresh: bool,
    prepare_cache_dir: str | None,
    refresh_prepare_cache: bool,
) -> dict[str, Any]:
    sample_id, turns = locomo_sample_to_turns(sample)
    corpus_id = f"locomo_{sanitize_sample_id(sample_id)}"
    db_path = Path(db_dir) / f"{corpus_id}.sqlite3"
    if refresh and db_path.exists():
        db_path.unlink()
    service = LEAFService(config_path=config_path, db_path=db_path)
    prepare_cache_path = Path(prepare_cache_dir) / f"{corpus_id}.prepared.json" if prepare_cache_dir else None
    prepare_cache_hit = False
    prepare_cache_read_ms: float | None = None
    prepare_cache_write_ms: float | None = None
    prepare_elapsed_ms: float | None = None
    try:
        started_at = time.perf_counter()
        if ingest_mode == "migration" and prepare_cache_path is not None:
            if prepare_cache_path.exists() and not refresh_prepare_cache:
                cache_read_started_at = time.perf_counter()
                cache_payload = service.load_prepared_turns_cache(prepare_cache_path)
                prepared_turns = list(cache_payload.get("prepared_turns") or [])
                prepare_cache_read_ms = round((time.perf_counter() - cache_read_started_at) * 1000.0, 2)
                prepare_cache_hit = True
            else:
                prepare_started_at = time.perf_counter()
                prepared_turns = service.prepare_turns(
                    corpus_id=corpus_id,
                    turns=turns,
                    ingest_mode=ingest_mode,
                )
                prepare_elapsed_ms = round((time.perf_counter() - prepare_started_at) * 1000.0, 2)
                cache_write_started_at = time.perf_counter()
                service.save_prepared_turns_cache(
                    prepare_cache_path,
                    corpus_id=corpus_id,
                    title=f"LoCoMo {sample_id}",
                    prepared_turns=prepared_turns,
                    ingest_mode=ingest_mode,
                )
                prepare_cache_write_ms = round((time.perf_counter() - cache_write_started_at) * 1000.0, 2)
            result = service.append_prepared_turns(
                corpus_id=corpus_id,
                title=f"LoCoMo {sample_id}",
                prepared_turns=prepared_turns,
                ingest_mode=ingest_mode,
            )
            stage_timings_ms = dict(result.get("stage_timings_ms") or {})
            if prepare_elapsed_ms is not None:
                stage_timings_ms["prepare_total_ms"] = prepare_elapsed_ms
            if prepare_cache_read_ms is not None:
                stage_timings_ms["prepare_cache_read_ms"] = prepare_cache_read_ms
            if prepare_cache_write_ms is not None:
                stage_timings_ms["prepare_cache_write_ms"] = prepare_cache_write_ms
            result["stage_timings_ms"] = dict(sorted(stage_timings_ms.items()))
            result["ingest_elapsed_ms"] = round((time.perf_counter() - started_at) * 1000.0, 2)
            result["prepared_turn_count"] = len(prepared_turns)
        else:
            result = service.append_turns(
                corpus_id=corpus_id,
                title=f"LoCoMo {sample_id}",
                turns=turns,
                ingest_mode=ingest_mode,
            )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    finally:
        service.close()
    row = {
        "sample_id": sample_id,
        "corpus_id": corpus_id,
        "db_path": str(db_path),
        "turn_count": len(turns),
        "session_count": len({str(turn.get("session_id") or "") for turn in turns if str(turn.get("session_id") or "")}),
        "ingest_wall_ms": elapsed_ms,
        "prepare_cache_path": str(prepare_cache_path) if prepare_cache_path is not None else None,
        "prepare_cache_hit": prepare_cache_hit,
    }
    row.update(result)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full LoCoMo ingest and report stats.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--ingest-mode", choices=["online", "migration"], default="migration")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ingest-prepare-workers", type=int, default=0)
    parser.add_argument("--migration-prepare-workers", type=int, default=0)
    parser.add_argument("--embedding-workers", type=int, default=0)
    parser.add_argument("--embedding-batch-size", type=int, default=0)
    parser.add_argument("--prepare-cache-dir")
    parser.add_argument("--refresh-prepare-cache", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ingest_prepare_workers > 0:
        os.environ["LEAF_INGEST_PREPARE_WORKERS"] = str(max(1, args.ingest_prepare_workers))
    if args.migration_prepare_workers > 0:
        os.environ["LEAF_MIGRATION_PREPARE_WORKERS"] = str(max(1, args.migration_prepare_workers))
    if args.embedding_workers > 0:
        os.environ["LEAF_EMBED_WORKERS"] = str(max(1, args.embedding_workers))
    if args.embedding_batch_size > 0:
        os.environ["LEAF_EMBED_BATCH_SIZE"] = str(max(1, args.embedding_batch_size))
    samples = load_locomo_samples(args.input)
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    worker_count = min(max(1, args.workers), len(samples) or 1)
    if worker_count == 1:
        for sample in samples:
            row = ingest_one(
                config_path=args.config,
                db_dir=str(db_dir),
                ingest_mode=args.ingest_mode,
                sample=sample,
                refresh=args.refresh,
                prepare_cache_dir=args.prepare_cache_dir,
                refresh_prepare_cache=args.refresh_prepare_cache,
            )
            rows.append(row)
            print(
                f"[locomo-ingest] sample={row['sample_id']} mode={args.ingest_mode} turns={row['turn_count']} wall_ms={row['ingest_wall_ms']}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    ingest_one,
                    config_path=args.config,
                    db_dir=str(db_dir),
                    ingest_mode=args.ingest_mode,
                    sample=sample,
                    refresh=args.refresh,
                    prepare_cache_dir=args.prepare_cache_dir,
                    refresh_prepare_cache=args.refresh_prepare_cache,
                )
                for sample in samples
            ]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"[locomo-ingest] sample={row['sample_id']} mode={args.ingest_mode} turns={row['turn_count']} wall_ms={row['ingest_wall_ms']}",
                    flush=True,
                )
        rows.sort(key=lambda item: str(item["sample_id"]))

    ingest_elapsed_values = [float(row["ingest_elapsed_ms"]) for row in rows if row.get("ingest_elapsed_ms") is not None]
    wall_elapsed_values = [float(row["ingest_wall_ms"]) for row in rows if row.get("ingest_wall_ms") is not None]
    summary = {
        "sample_count": len(rows),
        "ingest_mode": args.ingest_mode,
        "input": str(args.input),
        "config": str(args.config),
        "db_dir": str(db_dir),
        "workers": worker_count,
        "ingest_prepare_workers": int(os.environ.get("LEAF_INGEST_PREPARE_WORKERS", "4") or "4"),
        "migration_prepare_workers": int(os.environ.get("LEAF_MIGRATION_PREPARE_WORKERS", "16") or "16"),
        "embedding_workers": int(os.environ.get("LEAF_EMBED_WORKERS", "8") or "8"),
        "embedding_batch_size": int(os.environ.get("LEAF_EMBED_BATCH_SIZE", "32") or "32"),
        "elapsed_wall_total_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
        "ingest_elapsed_ms_total": round(sum(ingest_elapsed_values), 2) if ingest_elapsed_values else None,
        "ingest_elapsed_ms_avg": round(sum(ingest_elapsed_values) / len(ingest_elapsed_values), 2) if ingest_elapsed_values else None,
        "ingest_wall_ms_total": round(sum(wall_elapsed_values), 2) if wall_elapsed_values else None,
        "ingest_wall_ms_avg": round(sum(wall_elapsed_values) / len(wall_elapsed_values), 2) if wall_elapsed_values else None,
        "prepare_cache_dir": str(args.prepare_cache_dir) if args.prepare_cache_dir else None,
        "prepare_cache_hits": sum(1 for row in rows if row.get("prepare_cache_hit")),
        "turn_count_total": sum(int(row.get("turn_count") or 0) for row in rows),
        "session_count_total": sum(int(row.get("session_count") or 0) for row in rows),
        "events_written_total": sum(int(row.get("events_written") or 0) for row in rows),
        "atoms_written_total": sum(int(row.get("atoms_written") or 0) for row in rows),
        "objects_written_total": sum(int(row.get("objects_written") or 0) for row in rows),
        "state_candidates_total": sum(int(row.get("state_candidates") or 0) for row in rows),
        "evidence_links_written_total": sum(int(row.get("evidence_links_written") or 0) for row in rows),
        "state_action_counts_total": add_numeric_maps(
            [dict(row.get("state_action_counts") or {}) for row in rows if isinstance(row.get("state_action_counts"), dict)]
        ),
        "state_cache_metrics_total": add_numeric_maps(
            [
                {key: value for key, value in dict(row.get("state_cache_metrics") or {}).items() if isinstance(value, int)}
                for row in rows
                if isinstance(row.get("state_cache_metrics"), dict)
            ]
        ),
    }
    payload = {
        "summary": summary,
        "samples": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
