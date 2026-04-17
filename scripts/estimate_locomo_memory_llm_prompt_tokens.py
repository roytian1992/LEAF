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
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from leaf.service import LEAFService  # noqa: E402
import eval_locomo as base  # noqa: E402

ATOM_SYSTEM_PROMPT = (
    "Extract up to 5 memory atoms from a single interaction span. "
    "Return JSON with key 'atoms'. Each atom must contain type, content, entities, confidence."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate LEAF LoCoMo memory-side prompt tokens from an existing ingest/eval report."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--reconcile-prompt-tokens-per-call", type=int, default=91)
    return parser.parse_args()


def _load_ingested_sample_ids(report_payload: dict[str, Any]) -> list[str]:
    ingest_rows = report_payload.get("ingest") or []
    sample_ids: list[str] = []
    for row in ingest_rows:
        if not isinstance(row, dict):
            continue
        if row.get("ingested") is False and row.get("reused") is True:
            continue
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id:
            sample_ids.append(sample_id)
    return sample_ids


def _reconstruct_atom_prompt_tokens(
    *,
    service: LEAFService,
    samples_by_id: dict[str, list[dict[str, Any]]],
    sample_ids: list[str],
) -> tuple[int, int]:
    atom_prompt_tokens_total = 0
    atom_call_total = 0
    for sample_id in sample_ids:
        turns = samples_by_id.get(sample_id) or []
        corpus_id = f"locomo_{base.sanitize_sample_id(sample_id)}"
        prepared_inputs: list[dict[str, Any]] = []
        per_session_index: dict[str, int] = {}
        for turn in turns:
            session_id = str(turn.get("session_id") or "session-1")
            speaker = str(turn.get("speaker") or turn.get("role") or "unknown")
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            turn_index = per_session_index.get(session_id, 0)
            per_session_index[session_id] = turn_index + 1
            timestamp = str(turn.get("timestamp")) if turn.get("timestamp") else None
            metadata = {
                key: value
                for key, value in turn.items()
                if key not in {"session_id", "speaker", "role", "text", "content", "timestamp"}
            }
            prepared_inputs.append(
                {
                    "corpus_id": corpus_id,
                    "session_id": session_id,
                    "speaker": speaker,
                    "text": text,
                    "turn_index": turn_index,
                    "timestamp": timestamp,
                    "metadata": metadata,
                }
            )
        prepared_turns = service.indexer._prepare_turns_parallel(prepared_inputs, max_workers=1)
        grouped_by_session: dict[str, list[dict[str, Any]]] = {}
        session_order: list[str] = []
        for prepared_turn in prepared_turns:
            session_id = str(prepared_turn["event"].session_id)
            if session_id not in grouped_by_session:
                grouped_by_session[session_id] = []
                session_order.append(session_id)
            grouped_by_session[session_id].append(prepared_turn)
        for session_id in session_order:
            previous_chunk: list[dict[str, Any]] | None = None
            for chunk in service.indexer._build_extraction_chunks(grouped_by_session[session_id]):
                span = service.indexer._build_chunk_extraction_span(chunk, previous_chunk)
                messages = [
                    {"role": "system", "content": ATOM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"speaker={span.speaker}\ntext={span.text}"},
                ]
                atom_prompt_tokens_total += base.estimate_message_tokens(messages)
                atom_call_total += 1
                previous_chunk = chunk
    return atom_call_total, atom_prompt_tokens_total


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    sample_ids = _load_ingested_sample_ids(report_payload)
    samples = base.load_locomo_samples(args.input)
    samples_by_id: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        sample_id, turns = base.locomo_sample_to_turns(sample)
        samples_by_id[sample_id] = turns

    service = LEAFService(config_path=args.config, db_path="/tmp/leaf_locomo_prompt_token_estimate.sqlite3")
    service.indexer.embedding_client = None
    try:
        atom_call_total, atom_prompt_tokens_total = _reconstruct_atom_prompt_tokens(
            service=service,
            samples_by_id=samples_by_id,
            sample_ids=sample_ids,
        )
    finally:
        service.close()

    summary = dict(report_payload.get("summary") or {})
    reconcile_calls = int(
        (summary.get("ingest_memory_llm_calls_est_total") or {}).get("reconciliation")
        or (summary.get("memory_llm_calls_est_total") or {}).get("reconciliation")
        or 0
    )
    reconcile_prompt_tokens_est = reconcile_calls * int(args.reconcile_prompt_tokens_per_call)
    payload = {
        "report": str(report_path.resolve()),
        "input": str(Path(args.input).resolve()),
        "sample_ids": sample_ids,
        "atom_extraction_calls_reconstructed": atom_call_total,
        "atom_prompt_tokens_est": atom_prompt_tokens_total,
        "reconciliation_calls_from_report": reconcile_calls,
        "reconcile_prompt_tokens_per_call_est": int(args.reconcile_prompt_tokens_per_call),
        "reconcile_prompt_tokens_est": reconcile_prompt_tokens_est,
        "memory_llm_prompt_tokens_est_total": atom_prompt_tokens_total + reconcile_prompt_tokens_est,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
