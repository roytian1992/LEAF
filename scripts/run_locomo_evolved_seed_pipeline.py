from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.store import SQLiteMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoCoMo per-corpus evolved seed-topic pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--min-accepted", type=int, default=6)
    parser.add_argument("--prefix", default="reports/agentic_memory")
    return parser.parse_args()


def corpus_id_for_sample(sample_id: str) -> str:
    return f"locomo_{str(sample_id).replace('-', '_')}"


def run(cmd: list[str], *, retries: int = 3) -> None:
    print("[run]", " ".join(cmd), flush=True)
    for attempt in range(retries + 1):
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.returncode == 0:
            return
        retryable = "HTTP 503" in result.stdout or "system_memory_overloaded" in result.stdout
        if not retryable or attempt >= retries:
            raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
        sleep_seconds = 20 * (attempt + 1)
        print(f"[run] retryable failure; retrying in {sleep_seconds}s ({attempt + 1}/{retries})", flush=True)
        time.sleep(sleep_seconds)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    python = sys.executable
    prefix = Path(args.prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []

    store = SQLiteMemoryStore(args.db)
    try:
        for sample_id in args.samples:
            corpus_id = corpus_id_for_sample(sample_id)
            print(f"[evolve] sample={sample_id} corpus={corpus_id} start", flush=True)
            base_view = store.get_active_memory_view(corpus_id)
            if base_view is None:
                row = {"sample_id": sample_id, "corpus_id": corpus_id, "status": "no_active_view"}
                rows.append(row)
                output_path.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            base_view_id = str(base_view["view_id"])
            stem = f"locomo_{sample_id}"
            selfqa = prefix / f"{stem}_selfqa_evolved_seed_20260516.jsonl"
            selfqa_summary = prefix / f"{stem}_selfqa_evolved_seed_summary_20260516.json"
            seed_eval = prefix / f"{stem}_seed_shadow_eval_20260516.json"
            candidate = prefix / f"{stem}_evolved_topic_view_20260516.json"
            evolved_eval = prefix / f"{stem}_evolved_shadow_eval_20260516.json"
            gate = prefix / f"{stem}_evolved_gate_20260516.json"

            run(
                [
                    python,
                    "scripts/build_selfqa_from_memory.py",
                    "--config",
                    args.config,
                    "--db",
                    args.db,
                    "--corpus-id",
                    corpus_id,
                    "--active-view-id",
                    base_view_id,
                    "--output",
                    str(selfqa),
                    "--summary-output",
                    str(selfqa_summary),
                    "--limit",
                    str(args.limit),
                    "--candidate-limit",
                    str(args.candidate_limit),
                    "--validate",
                    "--min-validation-score",
                    "0.75",
                ]
            )
            selfqa_payload = load_json(selfqa_summary)
            accepted_count = int(selfqa_payload.get("accepted_count") or 0)
            if accepted_count < args.min_accepted:
                row = {
                    "sample_id": sample_id,
                    "corpus_id": corpus_id,
                    "base_view_id": base_view_id,
                    "status": "insufficient_selfqa",
                    "accepted_count": accepted_count,
                }
                rows.append(row)
                output_path.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            run(
                [
                    python,
                    "scripts/eval_memory_search.py",
                    "--config",
                    args.config,
                    "--db",
                    args.db,
                    "--corpus-id",
                    corpus_id,
                    "--selfqa",
                    str(selfqa),
                    "--output",
                    str(seed_eval),
                    "--snapshot-limit",
                    "6",
                    "--raw-span-limit",
                    "8",
                    "--trace-memory",
                    "--topic-routing-shadow",
                    "--topic-router",
                    "keyword",
                    "--topic-route-top-k",
                    "3",
                    "--topic-view-id",
                    base_view_id,
                ]
            )
            run(
                [
                    python,
                    "scripts/evolve_topic_view_from_shadow.py",
                    "--config",
                    args.config,
                    "--db",
                    args.db,
                    "--corpus-id",
                    corpus_id,
                    "--base-view-id",
                    base_view_id,
                    "--selfqa",
                    str(selfqa),
                    "--shadow-eval-report",
                    str(seed_eval),
                    "--output",
                    str(candidate),
                    "--name",
                    f"locomo-{sample_id}-evolved-topic-shadow-v1",
                    "--strategy",
                    "llm",
                    "--max-new-topics",
                    "4",
                    "--max-miss-rows",
                    "12",
                    "--max-auto-reassignments",
                    "40",
                    "--min-auto-reassign-keyword-matches",
                    "2",
                    "--max-reassigned-atom-ratio",
                    "0.15",
                    "--record-run",
                ]
            )
            candidate_payload = load_json(candidate)
            candidate_view_id = str(candidate_payload["candidate"]["view_id"])
            run(
                [
                    python,
                    "scripts/eval_memory_search.py",
                    "--config",
                    args.config,
                    "--db",
                    args.db,
                    "--corpus-id",
                    corpus_id,
                    "--selfqa",
                    str(selfqa),
                    "--output",
                    str(evolved_eval),
                    "--snapshot-limit",
                    "6",
                    "--raw-span-limit",
                    "8",
                    "--trace-memory",
                    "--topic-routing-shadow",
                    "--topic-router",
                    "keyword",
                    "--topic-route-top-k",
                    "3",
                    "--topic-view-id",
                    candidate_view_id,
                ]
            )
            run(
                [
                    python,
                    "scripts/evaluate_memory_view_gate.py",
                    "--db",
                    args.db,
                    "--corpus-id",
                    corpus_id,
                    "--view-id",
                    candidate_view_id,
                    "--eval-report",
                    str(evolved_eval),
                    "--baseline-report",
                    str(seed_eval),
                    "--output",
                    str(gate),
                    "--min-task-count",
                    str(args.min_accepted),
                    "--min-mean-event-recall",
                    "0.45",
                    "--min-event-path-hit-rate",
                    "0.30",
                    "--min-mean-atom-recall",
                    "0.45",
                    "--min-atom-path-hit-rate",
                    "0.30",
                    "--min-mean-topic-recall",
                    "0.20",
                    "--min-topic-path-hit-rate",
                    "0.15",
                    "--max-event-recall-regression",
                    "0.05",
                    "--max-atom-recall-regression",
                    "0.05",
                    "--min-topic-recall-improvement",
                    "0.05",
                    "--max-latency-ratio",
                    "2.00",
                    "--max-reassigned-atom-ratio",
                    "0.15",
                    "--promote",
                    "--record-run",
                ]
            )
            gate_payload = load_json(gate)
            row = {
                "sample_id": sample_id,
                "corpus_id": corpus_id,
                "base_view_id": base_view_id,
                "candidate_view_id": candidate_view_id,
                "status": (
                    "promoted"
                    if gate_payload.get("promoted")
                    else ("passed_not_promoted" if gate_payload.get("gate", {}).get("passed") else "gate_failed")
                ),
                "accepted_count": accepted_count,
                "seed_summary": load_json(seed_eval).get("summary"),
                "evolved_summary": load_json(evolved_eval).get("summary"),
                "gate": gate_payload.get("gate"),
                "promoted": bool(gate_payload.get("promoted")),
                "assignment_churn": gate_payload.get("assignment_churn"),
            }
            rows.append(row)
            output_path.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[evolve] sample={sample_id} status={row['status']}", flush=True)
    finally:
        store.close()

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "count": len(rows),
        "promoted_count": sum(1 for row in rows if row.get("promoted")),
        "statuses": [{"status": status, "count": count} for status, count in sorted(status_counts.items())],
    }
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
