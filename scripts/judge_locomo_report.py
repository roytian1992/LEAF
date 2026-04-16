from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import eval_locomo as base  # noqa: E402
from leaf.clients import ChatClient  # noqa: E402
from leaf.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add LLM-as-judge results to an existing LoCoMo report without rerunning QA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--judge-style", choices=["legacy_binary", "partial_credit"], default="legacy_binary")
    parser.add_argument("--judge-runs", type=int, default=5)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--judge-max-workers", type=int, default=4)
    parser.add_argument("--row-limit", type=int, default=0, help="Optional dry-run limit for the first N rows.")
    parser.add_argument("--resume-progress", action="store_true")
    return parser.parse_args()


def load_judge_client(config_path: str | Path) -> tuple[ChatClient, dict[str, Any]]:
    config = load_config(config_path)
    if not config.llm.base_url:
        raise RuntimeError("Judge requested but config.llm.base_url is empty.")
    client = ChatClient(config.llm)
    meta = {
        "judge_model_name": str(config.llm.model_name),
        "judge_model_base_url": str(config.llm.base_url),
    }
    return client, meta


def recompute_payload_summary(payload: dict[str, Any], *, results: list[dict[str, Any]], judge_runs: int) -> dict[str, Any]:
    old_summary = dict(payload.get("summary") or {})
    sample_count = int(old_summary.get("sample_count") or len({str(row.get("sample_id") or "") for row in results}))
    ingest_rows = list(payload.get("ingest") or [])
    summary = base.build_summary(
        samples=[{} for _ in range(max(0, sample_count))],
        ingest_rows=ingest_rows,
        results=results,
        judge_with_llm=True,
        judge_runs=judge_runs,
    )
    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_report)
    output_path = Path(args.output_report)
    progress_path = output_path.with_name(f"{output_path.stem}.judge_progress.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume_progress or not progress_path.exists():
        progress_path.write_text("", encoding="utf-8")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    results = list(payload.get("results") or [])
    if not results:
        raise RuntimeError(f"No results found in report: {input_path}")
    if args.row_limit > 0:
        results = results[: args.row_limit]

    judge_client, judge_meta = load_judge_client(args.config)

    def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
        return (
            str(row.get("sample_id") or ""),
            str(row.get("qa_id") or ""),
            int(row.get("question_index") or 0),
        )

    resumed_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.resume_progress and progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if int(row.get("judge_valid_runs") or 0) >= max(1, args.judge_runs):
                    resumed_rows[row_key(row)] = row

    def judge_single(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        judgments = [
            base.judge_answer(
                judge_client,
                question=str(row["question"]),
                gold_answer=str(row["gold_answer"]),
                predicted_answer=str(row["predicted_answer"]),
                retries=args.judge_retries,
                judge_style=args.judge_style,
            )
            for _ in range(max(1, args.judge_runs))
        ]
        updated = dict(row)
        updated.update(base.aggregate_row_judgments(judgments))
        return index, updated

    judged_rows: list[dict[str, Any] | None] = [None] * len(results)
    pending_tasks: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(results):
        resumed = resumed_rows.get(row_key(row))
        if resumed is not None:
            judged_rows[index] = resumed
        else:
            pending_tasks.append((index, row))

    if resumed_rows:
        print(f"[locomo-judge] resumed {len(resumed_rows)}/{len(results)} rows from {progress_path}", flush=True)

    worker_count = min(max(1, args.judge_max_workers), max(1, len(pending_tasks) or 1))
    completed_count = len(resumed_rows)
    if worker_count <= 1 or len(pending_tasks) <= 1:
        for index, row in pending_tasks:
            _, updated = judge_single(index, row)
            judged_rows[index] = updated
            base.append_jsonl(progress_path, updated)
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(results):
                print(f"[locomo-judge] judged {completed_count}/{len(results)} rows", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(judge_single, index, row) for index, row in pending_tasks]
            for future in as_completed(futures):
                index, updated = future.result()
                judged_rows[index] = updated
                base.append_jsonl(progress_path, updated)
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(results):
                    print(f"[locomo-judge] judged {completed_count}/{len(results)} rows", flush=True)

    finalized_rows = [row for row in judged_rows if row is not None]
    updated_payload = dict(payload)
    updated_payload["source_report"] = str(input_path)
    updated_payload["judge_with_llm"] = True
    updated_payload["judge_style"] = str(args.judge_style)
    updated_payload["judge_runs"] = int(args.judge_runs)
    updated_payload["judge_max_workers"] = int(worker_count)
    updated_payload["judge_retries"] = int(args.judge_retries)
    updated_payload["judge_progress_path"] = str(progress_path)
    updated_payload.update(judge_meta)
    updated_payload["results"] = finalized_rows
    updated_payload["summary"] = recompute_payload_summary(
        updated_payload,
        results=finalized_rows,
        judge_runs=int(args.judge_runs),
    )
    base.write_json_atomic(output_path, updated_payload)
    print(json.dumps({"output": str(output_path), "summary": updated_payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
