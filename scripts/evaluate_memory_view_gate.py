from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import stable_hash, utc_now_iso
from leaf.store import SQLiteMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a memory view promotion gate from a frozen retrieval eval report.",
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--baseline-report", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-task-count", type=int, default=1)
    parser.add_argument("--min-mean-event-recall", type=float, default=0.95)
    parser.add_argument("--min-event-path-hit-rate", type=float, default=0.95)
    parser.add_argument("--min-mean-atom-recall", type=float, default=0.95)
    parser.add_argument("--min-atom-path-hit-rate", type=float, default=0.95)
    parser.add_argument("--min-mean-topic-recall", type=float, default=0.0, help="0 disables the topic recall gate.")
    parser.add_argument("--min-topic-path-hit-rate", type=float, default=0.0, help="0 disables the topic path gate.")
    parser.add_argument("--max-avg-elapsed-ms", type=float, default=0.0, help="0 disables the absolute latency gate.")
    parser.add_argument("--max-event-recall-regression", type=float, default=0.0)
    parser.add_argument("--max-atom-recall-regression", type=float, default=0.0)
    parser.add_argument("--min-topic-recall-improvement", type=float, default=0.0)
    parser.add_argument(
        "--min-topic-path-hit-improvement",
        type=float,
        default=0.0,
        help="Minimum topic_path_hit_rate improvement over baseline. 0 disables the topic path improvement gate.",
    )
    parser.add_argument("--max-latency-ratio", type=float, default=0.0, help="0 disables the baseline latency gate.")
    parser.add_argument("--max-reassigned-atom-count", type=int, default=0, help="0 disables assignment churn count gate.")
    parser.add_argument("--max-reassigned-atom-ratio", type=float, default=0.0, help="0 disables assignment churn ratio gate.")
    parser.add_argument("--promote", action="store_true", help="Promote the view if all gate checks pass.")
    parser.add_argument("--record-run", action="store_true", help="Record this gate as a leaf_evolution_runs row.")
    return parser.parse_args()


def load_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summary" in payload and isinstance(payload["summary"], dict):
        return dict(payload["summary"])
    return dict(payload)


def number(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(summary.get(key, default))
    except (TypeError, ValueError):
        return default


def check_min(name: str, value: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "mode": "min",
        "value": value,
        "threshold": threshold,
        "passed": value >= threshold,
    }


def check_max(name: str, value: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "mode": "max",
        "value": value,
        "threshold": threshold,
        "passed": value <= threshold,
    }


def check_required(name: str, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "mode": "required",
        "value": bool(passed),
        "threshold": True,
        "passed": bool(passed),
    }


def normalize_slug(value: str) -> str:
    lowered = str(value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered or "topic"


def topic_slug(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    value = (
        metadata.get("topic_slug")
        or metadata.get("evolved_slug")
        or metadata.get("seed_slug")
        or node.get("name")
        or node.get("topic_id")
    )
    return normalize_slug(str(value))


def build_assignment_churn(store: SQLiteMemoryStore, view: dict[str, Any]) -> dict[str, Any] | None:
    parent_view_id = str(view.get("parent_view_id") or "").strip()
    if not parent_view_id:
        return None
    parent_view = store.get_memory_view(parent_view_id)
    if parent_view is None:
        return None
    parent_topics = {
        str(node["topic_id"]): topic_slug(node)
        for node in store.list_topic_nodes(parent_view_id)
    }
    candidate_topics = {
        str(node["topic_id"]): topic_slug(node)
        for node in store.list_topic_nodes(str(view["view_id"]))
    }
    parent_assignments = {
        str(assignment["item_id"]): parent_topics.get(str(assignment["topic_id"]), str(assignment["topic_id"]))
        for assignment in store.list_topic_assignments(parent_view_id, item_kind="atom")
    }
    candidate_assignments = {
        str(assignment["item_id"]): candidate_topics.get(str(assignment["topic_id"]), str(assignment["topic_id"]))
        for assignment in store.list_topic_assignments(str(view["view_id"]), item_kind="atom")
    }
    candidate_secondary_assignments: dict[str, list[str]] = {}
    for assignment in store.list_topic_assignments(str(view["view_id"]), item_kind="atom_secondary"):
        atom_id = str(assignment["item_id"])
        candidate_secondary_assignments.setdefault(atom_id, []).append(
            candidate_topics.get(str(assignment["topic_id"]), str(assignment["topic_id"]))
        )
    comparable_atom_ids = sorted(set(parent_assignments) | set(candidate_assignments))
    reassigned_atom_ids = [
        atom_id
        for atom_id in comparable_atom_ids
        if parent_assignments.get(atom_id) != candidate_assignments.get(atom_id)
    ]
    secondary_atom_ids = sorted(candidate_secondary_assignments)
    parent_assignment_count = len(parent_assignments)
    ratio = len(reassigned_atom_ids) / parent_assignment_count if parent_assignment_count > 0 else 0.0
    secondary_ratio = len(secondary_atom_ids) / parent_assignment_count if parent_assignment_count > 0 else 0.0
    return {
        "parent_view_id": parent_view_id,
        "candidate_view_id": str(view["view_id"]),
        "parent_assignment_count": parent_assignment_count,
        "candidate_assignment_count": len(candidate_assignments),
        "candidate_secondary_assignment_count": sum(len(items) for items in candidate_secondary_assignments.values()),
        "candidate_secondary_atom_count": len(secondary_atom_ids),
        "candidate_secondary_atom_ratio": secondary_ratio,
        "comparable_atom_count": len(comparable_atom_ids),
        "reassigned_atom_count": len(reassigned_atom_ids),
        "reassigned_atom_ratio": ratio,
        "reassigned_atom_ids_sample": reassigned_atom_ids[:50],
        "secondary_atom_ids_sample": secondary_atom_ids[:50],
    }


def build_gate(
    args: argparse.Namespace,
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
    assignment_churn: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        check_min("task_count", number(summary, "task_count"), float(args.min_task_count)),
        check_min("mean_event_recall", number(summary, "mean_event_recall"), args.min_mean_event_recall),
        check_min("event_path_hit_rate", number(summary, "event_path_hit_rate"), args.min_event_path_hit_rate),
        check_min("mean_atom_recall", number(summary, "mean_atom_recall"), args.min_mean_atom_recall),
        check_min("atom_path_hit_rate", number(summary, "atom_path_hit_rate"), args.min_atom_path_hit_rate),
    ]
    avg_elapsed_ms = number(summary, "avg_elapsed_ms")
    if args.max_avg_elapsed_ms > 0:
        checks.append(check_max("avg_elapsed_ms", avg_elapsed_ms, args.max_avg_elapsed_ms))
    if args.min_mean_topic_recall > 0:
        checks.append(check_min("mean_topic_recall", number(summary, "mean_topic_recall"), args.min_mean_topic_recall))
    if args.min_topic_path_hit_rate > 0:
        checks.append(check_min("topic_path_hit_rate", number(summary, "topic_path_hit_rate"), args.min_topic_path_hit_rate))
    if baseline is not None:
        baseline_event = number(baseline, "mean_event_recall")
        baseline_atom = number(baseline, "mean_atom_recall")
        baseline_topic = number(baseline, "mean_topic_recall")
        checks.append(
            check_min(
                "mean_event_recall_vs_baseline",
                number(summary, "mean_event_recall"),
                baseline_event - args.max_event_recall_regression,
            )
        )
        checks.append(
            check_min(
                "mean_atom_recall_vs_baseline",
                number(summary, "mean_atom_recall"),
                baseline_atom - args.max_atom_recall_regression,
            )
        )
        if args.min_topic_recall_improvement > 0:
            checks.append(
                check_min(
                    "mean_topic_recall_vs_baseline",
                    number(summary, "mean_topic_recall"),
                    baseline_topic + args.min_topic_recall_improvement,
                )
            )
        if args.min_topic_path_hit_improvement > 0:
            baseline_topic_path_hit = number(baseline, "topic_path_hit_rate")
            checks.append(
                check_min(
                    "topic_path_hit_rate_vs_baseline",
                    number(summary, "topic_path_hit_rate"),
                    baseline_topic_path_hit + args.min_topic_path_hit_improvement,
                )
            )
        if args.max_latency_ratio > 0:
            baseline_latency = number(baseline, "avg_elapsed_ms")
            checks.append(check_max("avg_elapsed_ms_vs_baseline", avg_elapsed_ms, baseline_latency * args.max_latency_ratio))
    if args.max_reassigned_atom_count > 0 or args.max_reassigned_atom_ratio > 0:
        checks.append(check_required("assignment_churn_available", assignment_churn is not None))
    if assignment_churn is not None:
        reassigned_count = number(assignment_churn, "reassigned_atom_count")
        reassigned_ratio = number(assignment_churn, "reassigned_atom_ratio")
        if args.max_reassigned_atom_count > 0:
            checks.append(check_max("reassigned_atom_count", reassigned_count, float(args.max_reassigned_atom_count)))
        if args.max_reassigned_atom_ratio > 0:
            checks.append(check_max("reassigned_atom_ratio", reassigned_ratio, args.max_reassigned_atom_ratio))
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "thresholds": {
            "min_task_count": args.min_task_count,
            "min_mean_event_recall": args.min_mean_event_recall,
            "min_event_path_hit_rate": args.min_event_path_hit_rate,
            "min_mean_atom_recall": args.min_mean_atom_recall,
            "min_atom_path_hit_rate": args.min_atom_path_hit_rate,
            "min_mean_topic_recall": args.min_mean_topic_recall,
            "min_topic_path_hit_rate": args.min_topic_path_hit_rate,
            "max_avg_elapsed_ms": args.max_avg_elapsed_ms,
            "max_event_recall_regression": args.max_event_recall_regression,
            "max_atom_recall_regression": args.max_atom_recall_regression,
            "min_topic_recall_improvement": args.min_topic_recall_improvement,
            "min_topic_path_hit_improvement": args.min_topic_path_hit_improvement,
            "max_latency_ratio": args.max_latency_ratio,
            "max_reassigned_atom_count": args.max_reassigned_atom_count,
            "max_reassigned_atom_ratio": args.max_reassigned_atom_ratio,
        },
    }


def main() -> None:
    args = parse_args()
    summary = load_summary(args.eval_report)
    baseline = load_summary(args.baseline_report) if args.baseline_report else None
    now = utc_now_iso()
    store = SQLiteMemoryStore(args.db)
    assignment_churn: dict[str, Any] | None = None
    view: dict[str, Any] | None = None
    try:
        view = store.get_memory_view(args.view_id)
        if view is None:
            raise RuntimeError(f"Unknown memory view: {args.view_id}")
        assignment_churn = build_assignment_churn(store, view)
        gate = build_gate(args, summary, baseline, assignment_churn)
    except Exception:
        store.close()
        raise
    report = {
        "corpus_id": args.corpus_id,
        "view_id": args.view_id,
        "eval_report": str(args.eval_report),
        "baseline_report": str(args.baseline_report) if args.baseline_report else None,
        "gate": gate,
        "assignment_churn": assignment_churn,
        "eval_summary": summary,
        "baseline_summary": baseline,
        "created_at": now,
        "promote_requested": bool(args.promote),
        "promoted": False,
    }

    try:
        is_active = bool(view.get("active"))
        status = None
        if gate["passed"]:
            status = None if is_active else "validated"
            if args.promote:
                store.promote_memory_view(args.view_id, promoted_at=now)
                report["promoted"] = True
                status = None
        elif not is_active:
            status = "gate_failed"
        store.update_memory_view_metrics(
            args.view_id,
            metrics={
                "promotion_gate": gate,
                "last_eval_report": str(args.eval_report),
                "last_eval_summary": summary,
                "last_gate_at": now,
            },
            status=status,
        )
        if args.record_run:
            run_id = f"evo_{stable_hash(args.corpus_id, args.view_id, str(args.eval_report), now, length=24)}"
            store.add_evolution_run(
                run_id=run_id,
                corpus_id=args.corpus_id,
                base_view_id=str(view.get("parent_view_id") or "") or None,
                candidate_view_id=args.view_id,
                trigger={
                    "kind": "promotion_gate",
                    "eval_report": str(args.eval_report),
                    "baseline_report": str(args.baseline_report) if args.baseline_report else None,
                },
                status="promoted" if report["promoted"] else ("passed" if gate["passed"] else "failed"),
                result=report,
                created_at=now,
                completed_at=now,
            )
        store.commit()
    finally:
        store.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
