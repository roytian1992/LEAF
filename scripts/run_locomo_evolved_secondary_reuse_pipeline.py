from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.store import SQLiteMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LoCoMo evolved-topic retrofit using existing self-QA/proposals, "
            "preserving seed assignments as primary labels and writing evolved labels as secondary labels."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--source-prefix", required=True, help="Directory containing prior self-QA and proposal outputs.")
    parser.add_argument("--prefix", required=True, help="Output directory for this secondary-label run.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--date-suffix", default="20260516")
    parser.add_argument("--min-accepted", type=int, default=6)
    parser.add_argument("--max-auto-reassignments", type=int, default=40)
    parser.add_argument("--min-auto-reassign-keyword-matches", type=int, default=2)
    parser.add_argument("--auto-reassignment-mode", choices=["keyword", "profile"], default="keyword")
    parser.add_argument("--profile-auto-min-score", type=float, default=3.0)
    parser.add_argument("--profile-auto-min-term-overlap", type=int, default=2)
    parser.add_argument("--profile-auto-min-embedding-score", type=float, default=0.0)
    parser.add_argument(
        "--profile-auto-text-mode",
        choices=["content_entities", "content_only"],
        default="content_entities",
    )
    parser.add_argument("--profile-auto-min-score-margin", type=float, default=0.0)
    parser.add_argument("--profile-auto-min-score-ratio", type=float, default=0.0)
    parser.add_argument("--profile-auto-max-profile-terms", type=int, default=0)
    parser.add_argument("--max-reassigned-atom-ratio", type=float, default=0.15)
    parser.add_argument("--min-topic-path-hit-improvement", type=float, default=0.05)
    return parser.parse_args()


def corpus_id_for_sample(sample_id: str) -> str:
    return f"locomo_{str(sample_id).replace('-', '_')}"


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def active_view_id(db_path: str, corpus_id: str) -> str | None:
    store = SQLiteMemoryStore(db_path)
    try:
        view = store.get_active_memory_view(corpus_id)
        return str(view["view_id"]) if view is not None else None
    finally:
        store.close()


def selfqa_accepted_count(selfqa: Path, summary: Path) -> int:
    if summary.exists():
        payload = load_json(summary)
        value = payload.get("accepted_count")
        if value is not None:
            return int(value)
    return sum(1 for line in selfqa.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    args = parse_args()
    python = sys.executable
    source_prefix = Path(args.source_prefix)
    prefix = Path(args.prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for sample_id in args.samples:
        corpus_id = corpus_id_for_sample(sample_id)
        stem = f"locomo_{sample_id}"
        print(f"[secondary-evolve] sample={sample_id} corpus={corpus_id} start", flush=True)

        base_view_id = active_view_id(args.db, corpus_id)
        if base_view_id is None:
            row = {"sample_id": sample_id, "corpus_id": corpus_id, "status": "no_active_view"}
            rows.append(row)
            append_jsonl(output_path, row)
            continue

        selfqa = source_prefix / f"{stem}_selfqa_evolved_seed_{args.date_suffix}.jsonl"
        selfqa_summary = source_prefix / f"{stem}_selfqa_evolved_seed_summary_{args.date_suffix}.json"
        proposal = source_prefix / f"{stem}_evolved_topic_view_{args.date_suffix}.json"
        seed_eval = prefix / f"{stem}_seed_shadow_eval_secondary_{args.date_suffix}.json"
        candidate = prefix / f"{stem}_evolved_topic_view_secondary_{args.date_suffix}.json"
        evolved_eval = prefix / f"{stem}_evolved_shadow_eval_secondary_{args.date_suffix}.json"
        gate = prefix / f"{stem}_evolved_gate_secondary_{args.date_suffix}.json"

        missing = [str(path) for path in (selfqa, proposal) if not path.exists()]
        if missing:
            row = {
                "sample_id": sample_id,
                "corpus_id": corpus_id,
                "base_view_id": base_view_id,
                "status": "missing_reuse_artifact",
                "missing": missing,
            }
            rows.append(row)
            append_jsonl(output_path, row)
            continue

        accepted_count = selfqa_accepted_count(selfqa, selfqa_summary)
        if accepted_count < args.min_accepted:
            row = {
                "sample_id": sample_id,
                "corpus_id": corpus_id,
                "base_view_id": base_view_id,
                "status": "insufficient_selfqa",
                "accepted_count": accepted_count,
            }
            rows.append(row)
            append_jsonl(output_path, row)
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
                "--proposal-json",
                str(proposal),
                "--output",
                str(candidate),
                "--name",
                f"locomo-{sample_id}-evolved-topic-secondary-v1",
                "--strategy",
                "llm",
                "--max-new-topics",
                "4",
                "--max-miss-rows",
                "12",
                "--max-auto-reassignments",
                str(args.max_auto_reassignments),
                "--min-auto-reassign-keyword-matches",
                str(args.min_auto_reassign_keyword_matches),
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
            "assignment_mode": candidate_payload.get("candidate", {}).get("assignment_mode"),
            "assignment_churn": gate_payload.get("assignment_churn"),
        }
        rows.append(row)
        append_jsonl(output_path, row)
        print(f"[secondary-evolve] sample={sample_id} status={row['status']}", flush=True)

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
