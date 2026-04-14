from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from leaf.service import LEAFService


def sanitize_persona_name(name: str) -> str:
    cleaned = "_".join(str(name).strip().split())
    return cleaned or "unknown_persona"


def load_memory_bank(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("GVD memory bank must be a JSON object keyed by persona name.")
    return payload


def persona_to_turns(persona_name: str, persona_payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = persona_payload.get("history") or {}
    if not isinstance(history, dict):
        raise ValueError(f"Persona {persona_name} has invalid history format.")
    turns: list[dict[str, Any]] = []
    turn_index = 0
    for session_id in sorted(history.keys()):
        session_rows = history.get(session_id) or []
        if not isinstance(session_rows, list):
            continue
        for pair_index, row in enumerate(session_rows):
            if not isinstance(row, dict):
                continue
            user_text = str(row.get("query") or "").strip()
            assistant_text = str(row.get("response") or "").strip()
            if user_text:
                turns.append(
                    {
                        "session_id": session_id,
                        "speaker": persona_name.strip() or "user",
                        "text": user_text,
                        "timestamp": session_id,
                        "turn_index": turn_index,
                        "pair_index": pair_index,
                        "role": "user",
                    }
                )
                turn_index += 1
            if assistant_text:
                turns.append(
                    {
                        "session_id": session_id,
                        "speaker": "AI Companion",
                        "text": assistant_text,
                        "timestamp": session_id,
                        "turn_index": turn_index,
                        "pair_index": pair_index,
                        "role": "assistant",
                    }
                )
                turn_index += 1
    return turns


def add_numeric_maps(rows: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            total[str(key)] = int(total.get(str(key), 0)) + int(value)
    return dict(sorted(total.items()))


def maybe_ingest_persona(
    service: LEAFService,
    corpus_id: str,
    title: str,
    turns: list[dict[str, Any]],
    *,
    refresh: bool,
) -> dict[str, Any]:
    existing = set(service.list_corpora())
    if corpus_id in existing and not refresh:
        return {"ingested": False, "reused": True, "turn_count": len(turns)}
    if corpus_id in existing and refresh:
        raise RuntimeError(
            f"Corpus {corpus_id} already exists in the SQLite store. "
            "Use a fresh DB path for refresh runs because LEAF does not yet support corpus deletion."
        )
    result = service.append_turns(corpus_id=corpus_id, title=title, turns=turns)
    return {"ingested": True, "reused": False, "turn_count": len(turns), "result": result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ingest-only reporting for GVD / MemoryBank personas.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--memory-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--personas", nargs="+", default=[])
    parser.add_argument("--persona-limit", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = LEAFService(config_path=args.config, db_path=args.db)
    started_at = time.perf_counter()
    try:
        bank = load_memory_bank(args.memory_bank)
        persona_names = sorted(bank.keys(), key=lambda item: str(item).strip().lower())
        if args.personas:
            allowed = {str(item).strip() for item in args.personas if str(item).strip()}
            persona_names = [name for name in persona_names if str(name).strip() in allowed]
        if args.persona_limit > 0:
            persona_names = persona_names[: args.persona_limit]

        ingest_rows: list[dict[str, Any]] = []
        for persona_name in persona_names:
            normalized_name = str(persona_name).strip()
            corpus_id = f"gvd_{sanitize_persona_name(normalized_name).lower()}"
            turns = persona_to_turns(normalized_name, bank[persona_name])
            ingest_started = time.perf_counter()
            ingest_result = maybe_ingest_persona(
                service,
                corpus_id=corpus_id,
                title=f"GVD Persona {normalized_name}",
                turns=turns,
                refresh=args.refresh,
            )
            ingest_rows.append(
                {
                    "persona": normalized_name,
                    "corpus_id": corpus_id,
                    "turn_count": len(turns),
                    "ingested": ingest_result["ingested"],
                    "reused": ingest_result["reused"],
                    "ingest_elapsed_ms": round((time.perf_counter() - ingest_started) * 1000.0, 2),
                    "ingest_metrics": ingest_result.get("result"),
                    "corpus_stats": service.store.get_corpus_stats(corpus_id),
                }
            )

        ingest_metric_rows = [
            dict(row["ingest_metrics"])
            for row in ingest_rows
            if isinstance(row.get("ingest_metrics"), dict)
        ]
        ingest_elapsed_values = [float(row["ingest_elapsed_ms"]) for row in ingest_rows if row.get("ingest_elapsed_ms") is not None]

        db_path = Path(args.db)
        summary = {
            "persona_count": len(persona_names),
            "ingest_new_count": sum(1 for row in ingest_rows if row["ingested"]),
            "ingest_reused_count": sum(1 for row in ingest_rows if row["reused"]),
            "wall_clock_elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
            "ingest_elapsed_ms_total": round(sum(ingest_elapsed_values), 2) if ingest_elapsed_values else None,
            "ingest_avg_elapsed_ms": round(sum(ingest_elapsed_values) / len(ingest_elapsed_values), 2) if ingest_elapsed_values else None,
            "turn_count_total": sum(int(row.get("turn_count") or 0) for row in ingest_rows),
            "events_written_total": sum(int(row.get("events_written") or 0) for row in ingest_metric_rows),
            "atoms_written_total": sum(int(row.get("atoms_written") or 0) for row in ingest_metric_rows),
            "objects_written_total": sum(int(row.get("objects_written") or 0) for row in ingest_metric_rows),
            "state_candidates_total": sum(int(row.get("state_candidates") or 0) for row in ingest_metric_rows),
            "evidence_links_written_total": sum(int(row.get("evidence_links_written") or 0) for row in ingest_metric_rows),
            "input_text_chars_total": sum(int(row.get("input_text_chars") or 0) for row in ingest_metric_rows),
            "input_text_tokens_est_total": sum(int(row.get("input_text_tokens_est") or 0) for row in ingest_metric_rows),
            "snapshot_upserts_total": sum(int(row.get("snapshot_upserts_total") or 0) for row in ingest_metric_rows),
            "memory_llm_calls_est_total": add_numeric_maps(
                [
                    dict(row.get("memory_llm_calls_est") or {})
                    for row in ingest_metric_rows
                    if isinstance(row.get("memory_llm_calls_est"), dict)
                ]
            ),
            "state_action_counts_total": add_numeric_maps(
                [
                    dict(row.get("state_action_counts") or {})
                    for row in ingest_metric_rows
                    if isinstance(row.get("state_action_counts"), dict)
                ]
            ),
            "snapshot_upserts_by_kind_total": add_numeric_maps(
                [
                    dict(row.get("snapshot_upserts_by_kind") or {})
                    for row in ingest_metric_rows
                    if isinstance(row.get("snapshot_upserts_by_kind"), dict)
                ]
            ),
            "db_file_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        }

        payload = {
            "config": str(args.config),
            "memory_bank": str(args.memory_bank),
            "db": str(args.db),
            "personas": list(args.personas),
            "persona_limit": args.persona_limit,
            "refresh": args.refresh,
            "summary": summary,
            "ingest": ingest_rows,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
