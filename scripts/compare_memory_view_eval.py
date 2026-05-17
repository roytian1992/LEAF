from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "task_count",
    "mean_event_recall",
    "event_path_hit_rate",
    "mean_atom_recall",
    "atom_path_hit_rate",
    "mean_topic_recall",
    "topic_path_hit_rate",
    "mean_retrieval_topic_recall",
    "retrieval_topic_path_hit_rate",
    "avg_elapsed_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and evolved memory-view eval reports.")
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload.get("summary"), dict):
        return dict(payload["summary"])
    return dict(payload)


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_delta(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key in METRIC_KEYS:
        base_value = base.get(key)
        candidate_value = candidate.get(key)
        base_number = to_float(base_value)
        candidate_number = to_float(candidate_value)
        delta = None if base_number is None or candidate_number is None else round(candidate_number - base_number, 6)
        rows[key] = {
            "baseline": base_value,
            "candidate": candidate_value,
            "delta": delta,
        }
    return rows


def count_delta(base_counts: dict[str, Any], candidate_counts: dict[str, Any]) -> dict[str, dict[str, int]]:
    keys = sorted(set(base_counts) | set(candidate_counts))
    return {
        key: {
            "baseline": int(base_counts.get(key) or 0),
            "candidate": int(candidate_counts.get(key) or 0),
            "delta": int(candidate_counts.get(key) or 0) - int(base_counts.get(key) or 0),
        }
        for key in keys
    }


def main() -> None:
    args = parse_args()
    baseline = load_summary(args.baseline_report)
    candidate = load_summary(args.candidate_report)
    comparison = {
        "baseline_report": str(args.baseline_report),
        "candidate_report": str(args.candidate_report),
        "baseline_topic_view_id": baseline.get("topic_view_id"),
        "candidate_topic_view_id": candidate.get("topic_view_id"),
        "metric_deltas": metric_delta(baseline, candidate),
        "gold_topic_counts_delta": count_delta(
            dict(baseline.get("gold_topic_counts") or {}),
            dict(candidate.get("gold_topic_counts") or {}),
        ),
        "routed_topic_counts_delta": count_delta(
            dict(baseline.get("routed_topic_counts") or {}),
            dict(candidate.get("routed_topic_counts") or {}),
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(comparison["metric_deltas"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
