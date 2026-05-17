from __future__ import annotations

import argparse
import json
import sqlite3
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
    parser = argparse.ArgumentParser(
        description="Run per-GVD-corpus topic-tree evolution over an existing GVD LEAF database.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-ids", nargs="+", default=[])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--date-suffix", default="20260518")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--candidate-limit", type=int, default=90)
    parser.add_argument("--min-accepted", type=int, default=6)
    parser.add_argument("--topic-router", choices=["keyword", "profile_hybrid", "llm"], default="llm")
    parser.add_argument("--topic-route-top-k", type=int, default=3)
    parser.add_argument("--max-new-topics", type=int, default=4)
    parser.add_argument("--max-miss-rows", type=int, default=12)
    parser.add_argument("--max-auto-reassignments", type=int, default=40)
    parser.add_argument("--max-reassigned-atom-ratio", type=float, default=0.15)
    parser.add_argument(
        "--auto-reassignment-mode",
        choices=["keyword", "profile"],
        default="profile",
    )
    parser.add_argument("--profile-auto-min-score", type=float, default=3.0)
    parser.add_argument("--profile-auto-min-term-overlap", type=int, default=2)
    parser.add_argument("--profile-auto-min-embedding-score", type=float, default=0.0)
    parser.add_argument(
        "--profile-auto-text-mode",
        choices=["content_entities", "content_only"],
        default="content_only",
    )
    parser.add_argument("--profile-auto-min-score-margin", type=float, default=1.0)
    parser.add_argument("--profile-auto-min-score-ratio", type=float, default=0.0)
    parser.add_argument("--profile-auto-max-profile-terms", type=int, default=24)
    parser.add_argument(
        "--min-topic-path-hit-improvement",
        type=float,
        default=0.0,
        help="0 disables the topic-path improvement gate; event/atom regression gates still apply.",
    )
    parser.add_argument(
        "--include-existing-evolved",
        action="store_true",
        help="Also evolve corpora whose active view is already an evolved_topic_tree.",
    )
    return parser.parse_args()


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
        retryable = (
            "HTTP 503" in result.stdout
            or "system_memory_overloaded" in result.stdout
            or "timeout" in result.stdout.lower()
            or "timed out" in result.stdout.lower()
        )
        if not retryable or attempt >= retries:
            raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
        sleep_seconds = 20 * (attempt + 1)
        print(f"[run] retryable failure; retrying in {sleep_seconds}s ({attempt + 1}/{retries})", flush=True)
        time.sleep(sleep_seconds)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def discover_gvd_corpus_ids(db_path: str) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "select distinct corpus_id from leaf_events where corpus_id like 'gvd_%' order by corpus_id"
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        con.close()


def active_view(db_path: str, corpus_id: str) -> dict[str, Any] | None:
    store = SQLiteMemoryStore(db_path)
    try:
        view = store.get_active_memory_view(corpus_id)
        return dict(view) if view is not None else None
    finally:
        store.close()


def active_evolved_count(db_path: str, corpus_ids: list[str]) -> int:
    count = 0
    for corpus_id in corpus_ids:
        view = active_view(db_path, corpus_id)
        metadata = dict((view or {}).get("metadata") or {})
        if metadata.get("view_type") == "evolved_topic_tree":
            count += 1
    return count


def view_type(view: dict[str, Any] | None) -> str:
    return str(dict((view or {}).get("metadata") or {}).get("view_type") or "")


def selfqa_accepted_count(selfqa: Path, summary: Path) -> int:
    if summary.exists():
        payload = load_json(summary)
        value = payload.get("accepted_count")
        if value is not None:
            return int(value)
    if not selfqa.exists():
        return 0
    return sum(1 for line in selfqa.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    args = parse_args()
    python = sys.executable
    prefix = Path(args.prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    corpus_ids = list(args.corpus_ids) if args.corpus_ids else discover_gvd_corpus_ids(args.db)
    rows: list[dict[str, Any]] = []
    for corpus_id in corpus_ids:
        print(f"[gvd-evolve] corpus={corpus_id} start", flush=True)
        base_view = active_view(args.db, corpus_id)
        if base_view is None:
            row = {"corpus_id": corpus_id, "status": "no_active_view"}
            rows.append(row)
            append_jsonl(output_path, row)
            continue
        base_view_id = str(base_view["view_id"])
        base_view_type = view_type(base_view)
        if base_view_type == "evolved_topic_tree" and not args.include_existing_evolved:
            row = {
                "corpus_id": corpus_id,
                "base_view_id": base_view_id,
                "base_view_name": base_view.get("name"),
                "base_view_type": base_view_type,
                "status": "existing_evolved_active",
                "promoted": True,
            }
            rows.append(row)
            append_jsonl(output_path, row)
            print(f"[gvd-evolve] corpus={corpus_id} status=existing_evolved_active", flush=True)
            continue

        stem = corpus_id
        selfqa = prefix / f"{stem}_selfqa_evolved_topic_{args.date_suffix}.jsonl"
        selfqa_summary = prefix / f"{stem}_selfqa_evolved_topic_summary_{args.date_suffix}.json"
        seed_eval = prefix / f"{stem}_seed_{args.topic_router}_shadow_eval_{args.date_suffix}.json"
        candidate = prefix / f"{stem}_evolved_topic_view_{args.date_suffix}.json"
        evolved_eval = prefix / f"{stem}_evolved_{args.topic_router}_shadow_eval_{args.date_suffix}.json"
        comparison = prefix / f"{stem}_seed_vs_evolved_compare_{args.date_suffix}.json"
        gate = prefix / f"{stem}_evolved_gate_{args.date_suffix}.json"

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
                "--task-types",
                "single_fact",
                "multi_hop",
                "temporal",
                "--candidate-sampling",
                "stratified",
                "--validate",
                "--min-validation-score",
                "0.75",
                "--write-db",
                "--stream-output",
            ]
        )
        accepted_count = selfqa_accepted_count(selfqa, selfqa_summary)
        if accepted_count < args.min_accepted:
            row = {
                "corpus_id": corpus_id,
                "base_view_id": base_view_id,
                "base_view_type": base_view_type,
                "status": "insufficient_selfqa",
                "accepted_count": accepted_count,
            }
            rows.append(row)
            append_jsonl(output_path, row)
            print(f"[gvd-evolve] corpus={corpus_id} status=insufficient_selfqa accepted={accepted_count}", flush=True)
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
                args.topic_router,
                "--topic-route-top-k",
                str(args.topic_route_top_k),
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
                f"{corpus_id}-evolved-topic-shadow-v1",
                "--strategy",
                "llm",
                "--max-new-topics",
                str(args.max_new_topics),
                "--max-miss-rows",
                str(args.max_miss_rows),
                "--max-auto-reassignments",
                str(args.max_auto_reassignments),
                "--min-auto-reassign-keyword-matches",
                "2",
                "--auto-reassignment-mode",
                args.auto_reassignment_mode,
                "--profile-auto-min-score",
                str(args.profile_auto_min_score),
                "--profile-auto-min-term-overlap",
                str(args.profile_auto_min_term_overlap),
                "--profile-auto-min-embedding-score",
                str(args.profile_auto_min_embedding_score),
                "--profile-auto-text-mode",
                args.profile_auto_text_mode,
                "--profile-auto-min-score-margin",
                str(args.profile_auto_min_score_margin),
                "--profile-auto-min-score-ratio",
                str(args.profile_auto_min_score_ratio),
                "--profile-auto-max-profile-terms",
                str(args.profile_auto_max_profile_terms),
                "--max-reassigned-atom-ratio",
                str(args.max_reassigned_atom_ratio),
                "--preserve-base-assignments",
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
                args.topic_router,
                "--topic-route-top-k",
                str(args.topic_route_top_k),
                "--topic-view-id",
                candidate_view_id,
            ]
        )
        run(
            [
                python,
                "scripts/compare_memory_view_eval.py",
                "--baseline-report",
                str(seed_eval),
                "--candidate-report",
                str(evolved_eval),
                "--output",
                str(comparison),
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
                "--min-topic-path-hit-improvement",
                str(args.min_topic_path_hit_improvement),
                "--max-latency-ratio",
                "2.00",
                "--max-reassigned-atom-ratio",
                str(args.max_reassigned_atom_ratio),
                "--promote",
                "--record-run",
            ]
        )
        gate_payload = load_json(gate)
        row = {
            "corpus_id": corpus_id,
            "base_view_id": base_view_id,
            "base_view_name": base_view.get("name"),
            "base_view_type": base_view_type,
            "candidate_view_id": candidate_view_id,
            "candidate_view_name": candidate_payload.get("candidate", {}).get("name"),
            "status": (
                "promoted"
                if gate_payload.get("promoted")
                else ("passed_not_promoted" if gate_payload.get("gate", {}).get("passed") else "gate_failed")
            ),
            "promoted": bool(gate_payload.get("promoted")),
            "accepted_count": accepted_count,
            "selfqa": str(selfqa),
            "selfqa_summary": str(selfqa_summary),
            "seed_eval": str(seed_eval),
            "evolved_eval": str(evolved_eval),
            "comparison": str(comparison),
            "gate_report": str(gate),
            "proposal": candidate_payload.get("proposal"),
            "candidate": candidate_payload.get("candidate"),
            "seed_summary": load_json(seed_eval).get("summary"),
            "evolved_summary": load_json(evolved_eval).get("summary"),
            "gate": gate_payload.get("gate"),
            "assignment_churn": gate_payload.get("assignment_churn"),
        }
        rows.append(row)
        append_jsonl(output_path, row)
        print(f"[gvd-evolve] corpus={corpus_id} status={row['status']}", flush=True)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "count": len(rows),
        "corpus_ids": corpus_ids,
        "promoted_count": sum(1 for row in rows if row.get("status") == "promoted"),
        "existing_evolved_active_count": sum(1 for row in rows if row.get("status") == "existing_evolved_active"),
        "active_evolved_after_count": active_evolved_count(args.db, corpus_ids),
        "statuses": [{"status": status, "count": count} for status, count in sorted(status_counts.items())],
        "db": args.db,
        "prefix": str(prefix),
        "output": str(output_path),
    }
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
