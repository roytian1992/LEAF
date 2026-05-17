from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Gate a criteria-driven topic patch using held-out self-QA eval reports.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-heldout-task-count", type=int, default=3)
    parser.add_argument("--min-route-hit-improvement", type=float, default=0.0)
    parser.add_argument("--min-event-recall-improvement", type=float, default=0.0)
    parser.add_argument("--min-atom-recall-improvement", type=float, default=0.0)
    parser.add_argument("--max-event-recall-regression", type=float, default=0.0)
    parser.add_argument("--max-atom-recall-regression", type=float, default=0.0)
    parser.add_argument("--max-candidate-atom-ratio", type=float, default=1.25)
    parser.add_argument("--max-route-count-ratio", type=float, default=1.25)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--record-run", action="store_true")
    return parser.parse_args()


def load_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("summary"), dict):
        raise ValueError(f"Report has no summary: {path}")
    return payload


def number(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(summary.get(key, default))
    except (TypeError, ValueError):
        return default


def check(name: str, mode: str, value: float | bool, threshold: float | bool) -> dict[str, Any]:
    if mode == "min":
        passed = float(value) >= float(threshold)
    elif mode == "max":
        passed = float(value) <= float(threshold)
    elif mode == "required":
        passed = bool(value) is bool(threshold)
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    return {
        "name": name,
        "mode": mode,
        "value": value,
        "threshold": threshold,
        "passed": passed,
    }


def ratio(candidate: float, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    return candidate / baseline


def main() -> None:
    args = parse_args()
    candidate = load_report(args.candidate_report)
    baseline = load_report(args.baseline_report)
    candidate_summary = dict(candidate["summary"])
    baseline_summary = dict(baseline["summary"])
    candidate_task_count = int(number(candidate_summary, "task_count", 0.0))
    baseline_task_count = int(number(baseline_summary, "task_count", 0.0))
    candidate_route_hit = number(candidate_summary, "criteria_expected_topic_route_hit_rate", 0.0)
    baseline_route_hit = number(baseline_summary, "criteria_expected_topic_route_hit_rate", 0.0)
    candidate_event_recall = number(candidate_summary, "mean_event_recall", 0.0)
    baseline_event_recall = number(baseline_summary, "mean_event_recall", 0.0)
    candidate_atom_recall = number(candidate_summary, "mean_atom_recall", 0.0)
    baseline_atom_recall = number(baseline_summary, "mean_atom_recall", 0.0)
    candidate_atoms = number(candidate_summary, "avg_topic_soft_candidate_atom_count", 0.0)
    baseline_atoms = number(baseline_summary, "avg_topic_soft_candidate_atom_count", 0.0)
    candidate_routes = sum(
        int(value)
        for value in (candidate_summary.get("routed_topic_counts") or {}).values()
        if isinstance(value, int | float)
    )
    baseline_routes = sum(
        int(value)
        for value in (baseline_summary.get("routed_topic_counts") or {}).values()
        if isinstance(value, int | float)
    )
    candidate_atom_ratio = ratio(candidate_atoms, baseline_atoms)
    route_count_ratio = ratio(float(candidate_routes), float(baseline_routes))
    checks: list[dict[str, Any]] = [
        check("candidate_task_count", "min", candidate_task_count, int(args.min_heldout_task_count)),
        check("baseline_task_count_match", "required", baseline_task_count == candidate_task_count, True),
        check(
            "criteria_expected_topic_route_hit_improvement",
            "min",
            candidate_route_hit - baseline_route_hit,
            float(args.min_route_hit_improvement),
        ),
        check(
            "mean_event_recall_improvement",
            "min",
            candidate_event_recall - baseline_event_recall,
            float(args.min_event_recall_improvement),
        ),
        check(
            "mean_atom_recall_improvement",
            "min",
            candidate_atom_recall - baseline_atom_recall,
            float(args.min_atom_recall_improvement),
        ),
        check(
            "mean_event_recall_regression",
            "max",
            max(0.0, baseline_event_recall - candidate_event_recall),
            float(args.max_event_recall_regression),
        ),
        check(
            "mean_atom_recall_regression",
            "max",
            max(0.0, baseline_atom_recall - candidate_atom_recall),
            float(args.max_atom_recall_regression),
        ),
    ]
    if candidate_atom_ratio is not None and float(args.max_candidate_atom_ratio) > 0.0:
        checks.append(
            check(
                "topic_soft_candidate_atom_ratio",
                "max",
                round(candidate_atom_ratio, 6),
                float(args.max_candidate_atom_ratio),
            )
        )
    if route_count_ratio is not None and float(args.max_route_count_ratio) > 0.0:
        checks.append(
            check("route_count_ratio", "max", round(route_count_ratio, 6), float(args.max_route_count_ratio))
        )
    passed = all(item["passed"] for item in checks)
    now = utc_now_iso()
    result = {
        "corpus_id": args.corpus_id,
        "view_id": args.view_id,
        "candidate_report": str(args.candidate_report),
        "baseline_report": str(args.baseline_report),
        "passed": passed,
        "promoted": False,
        "checks": checks,
        "metrics": {
            "candidate_task_count": candidate_task_count,
            "baseline_task_count": baseline_task_count,
            "candidate_route_hit": candidate_route_hit,
            "baseline_route_hit": baseline_route_hit,
            "route_hit_delta": round(candidate_route_hit - baseline_route_hit, 6),
            "candidate_event_recall": candidate_event_recall,
            "baseline_event_recall": baseline_event_recall,
            "event_recall_delta": round(candidate_event_recall - baseline_event_recall, 6),
            "candidate_atom_recall": candidate_atom_recall,
            "baseline_atom_recall": baseline_atom_recall,
            "atom_recall_delta": round(candidate_atom_recall - baseline_atom_recall, 6),
            "candidate_avg_topic_soft_candidate_atom_count": candidate_atoms,
            "baseline_avg_topic_soft_candidate_atom_count": baseline_atoms,
            "candidate_atom_ratio": candidate_atom_ratio,
            "candidate_route_count": candidate_routes,
            "baseline_route_count": baseline_routes,
            "route_count_ratio": route_count_ratio,
        },
    }
    store = SQLiteMemoryStore(args.db)
    try:
        view = store.get_memory_view(args.view_id)
        if view is None:
            raise SystemExit(f"Unknown candidate view: {args.view_id}")
        if str(view.get("corpus_id") or "") != args.corpus_id:
            raise SystemExit(f"View {args.view_id} does not belong to corpus_id={args.corpus_id!r}.")
        if args.promote and passed:
            store.promote_memory_view(args.view_id, promoted_at=now)
            result["promoted"] = True
        if args.record_run:
            store.add_evolution_run(
                run_id=f"run_{stable_hash(args.corpus_id, args.view_id, 'selfqa_topic_patch_gate', now, length=20)}",
                corpus_id=args.corpus_id,
                base_view_id=str(view.get("parent_view_id") or ""),
                candidate_view_id=args.view_id,
                trigger={
                    "kind": "selfqa_topic_patch_gate",
                    "candidate_report": str(args.candidate_report),
                    "baseline_report": str(args.baseline_report),
                    "promote_requested": bool(args.promote),
                },
                status="passed" if passed else "failed",
                result=result,
                created_at=now,
                completed_at=utc_now_iso(),
            )
        store.commit()
    finally:
        store.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
