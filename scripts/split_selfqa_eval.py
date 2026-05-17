from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a self-QA JSONL and matching search-eval report by task_id.")
    parser.add_argument("--selfqa", required=True)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--train-selfqa", required=True)
    parser.add_argument("--heldout-selfqa", required=True)
    parser.add_argument("--train-eval-report", required=True)
    parser.add_argument("--heldout-eval-report", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--heldout-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout-min-count", type=int, default=1)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_eval_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_eval_report(path: str | Path, source: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(source.get("summary") or {})
    summary.update(
        {
            "task_count": len(rows),
            "split_source_eval_report": str(source.get("summary", {}).get("selfqa") or ""),
            "split_row_count": len(rows),
        }
    )
    output.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def row_task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or "").strip()


def main() -> None:
    args = parse_args()
    selfqa_rows = load_jsonl(args.selfqa)
    eval_report = load_eval_report(args.eval_report)
    eval_rows = [row for row in (eval_report.get("rows") or []) if isinstance(row, dict)]
    selfqa_by_task_id = {row_task_id(row): row for row in selfqa_rows if row_task_id(row)}
    eval_by_task_id = {row_task_id(row): row for row in eval_rows if row_task_id(row)}
    common_task_ids = sorted(set(selfqa_by_task_id).intersection(eval_by_task_id))
    if not common_task_ids:
        raise SystemExit("No overlapping task_id values between self-QA and eval report.")
    rng = random.Random(int(args.seed))
    shuffled = list(common_task_ids)
    rng.shuffle(shuffled)
    heldout_count = round(len(shuffled) * max(0.0, min(1.0, float(args.heldout_ratio))))
    heldout_count = max(int(args.heldout_min_count), int(heldout_count))
    if len(shuffled) > 1:
        heldout_count = min(len(shuffled) - 1, heldout_count)
    heldout_ids = set(shuffled[:heldout_count])
    train_ids = set(shuffled[heldout_count:])
    train_selfqa = [row for row in selfqa_rows if row_task_id(row) in train_ids]
    heldout_selfqa = [row for row in selfqa_rows if row_task_id(row) in heldout_ids]
    train_eval_rows = [row for row in eval_rows if row_task_id(row) in train_ids]
    heldout_eval_rows = [row for row in eval_rows if row_task_id(row) in heldout_ids]
    write_jsonl(args.train_selfqa, train_selfqa)
    write_jsonl(args.heldout_selfqa, heldout_selfqa)
    write_eval_report(args.train_eval_report, eval_report, train_eval_rows)
    write_eval_report(args.heldout_eval_report, eval_report, heldout_eval_rows)
    summary = {
        "selfqa": str(args.selfqa),
        "eval_report": str(args.eval_report),
        "seed": int(args.seed),
        "heldout_ratio": float(args.heldout_ratio),
        "input_selfqa_count": len(selfqa_rows),
        "input_eval_row_count": len(eval_rows),
        "common_task_count": len(common_task_ids),
        "train_task_count": len(train_ids),
        "heldout_task_count": len(heldout_ids),
        "train_task_ids": sorted(train_ids),
        "heldout_task_ids": sorted(heldout_ids),
        "train_selfqa": str(args.train_selfqa),
        "heldout_selfqa": str(args.heldout_selfqa),
        "train_eval_report": str(args.train_eval_report),
        "heldout_eval_report": str(args.heldout_eval_report),
    }
    output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
