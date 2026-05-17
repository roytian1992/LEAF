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

from leaf.agentic_memory import utc_now_iso
from leaf.store import SQLiteMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn a retrieval-time topic-soft usage policy from frozen eval diagnostics.",
    )
    parser.add_argument("--input", required=True, help="LoCoMo QA result JSON or self-QA retrieval eval JSON.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--db", default="", help="Optional LEAF SQLite DB for writing policy to a memory view.")
    parser.add_argument("--view-id", default="", help="Memory view to update when --db is provided.")
    parser.add_argument(
        "--mode",
        choices=["qa_delta", "selfqa_retrieval"],
        default="qa_delta",
        help="qa_delta expects rows with baseline/variant metrics; selfqa_retrieval uses topic-routing shadow rows.",
    )
    parser.add_argument("--baseline", default="", help="Baseline LoCoMo QA JSON for --mode qa_delta.")
    parser.add_argument("--variant", default="", help="Variant LoCoMo QA JSON for --mode qa_delta. Defaults to --input.")
    parser.add_argument("--min-topic-count", type=int, default=5)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--max-negative-mean-delta", type=float, default=-0.001)
    parser.add_argument("--min-topic-recall", type=float, default=0.5)
    parser.add_argument("--policy-name", default="learned_topic_soft_policy_v0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("sample_id") or ""), int(row.get("question_index") or 0)


def normalize_slug(value: str) -> str:
    lowered = str(value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered


def active_topic_slugs(row: dict[str, Any]) -> list[str]:
    return [
        normalize_slug(str(item))
        for item in (row.get("topic_soft_active_topic_slugs") or [])
        if normalize_slug(str(item))
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def learn_from_qa_delta(
    *,
    baseline_path: Path,
    variant_path: Path,
    min_topic_count: int,
    min_mean_delta: float,
    max_negative_mean_delta: float,
) -> dict[str, Any]:
    baseline_payload = load_json(baseline_path)
    variant_payload = load_json(variant_path)
    baseline_rows = {row_key(row): row for row in baseline_payload.get("results") or []}
    variant_rows = {row_key(row): row for row in variant_payload.get("results") or []}
    if set(baseline_rows) != set(variant_rows):
        raise RuntimeError("Baseline and variant result keys do not match.")

    by_slug: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(variant_rows, key=lambda item: (item[0], item[1])):
        baseline = baseline_rows[key]
        variant = variant_rows[key]
        slugs = active_topic_slugs(variant)
        if not slugs:
            continue
        delta_f1 = float(variant.get("answer_f1") or 0.0) - float(baseline.get("answer_f1") or 0.0)
        delta_bleu1 = float(variant.get("bleu1") or 0.0) - float(baseline.get("bleu1") or 0.0)
        for slug in slugs:
            by_slug.setdefault(slug, []).append(
                {
                    "delta_f1": delta_f1,
                    "delta_bleu1": delta_bleu1,
                    "category": variant.get("category_name"),
                    "sample_id": variant.get("sample_id"),
                    "question_index": variant.get("question_index"),
                }
            )

    topic_stats: dict[str, dict[str, Any]] = {}
    allow_slugs: list[str] = []
    deny_slugs: list[str] = []
    for slug, rows in sorted(by_slug.items()):
        deltas = [float(row["delta_f1"]) for row in rows]
        bleu_deltas = [float(row["delta_bleu1"]) for row in rows]
        stat = {
            "count": len(rows),
            "mean_delta_f1": round(mean(deltas), 6),
            "mean_delta_bleu1": round(mean(bleu_deltas), 6),
            "positive_count": sum(1 for value in deltas if value > 0.0),
            "negative_count": sum(1 for value in deltas if value < 0.0),
            "zero_count": sum(1 for value in deltas if value == 0.0),
        }
        topic_stats[slug] = stat
        if len(rows) < min_topic_count:
            continue
        if stat["mean_delta_f1"] >= min_mean_delta:
            allow_slugs.append(slug)
        elif stat["mean_delta_f1"] <= max_negative_mean_delta:
            deny_slugs.append(slug)

    return {
        "source_mode": "qa_delta",
        "baseline": str(baseline_path),
        "variant": str(variant_path),
        "topic_stats": topic_stats,
        "allow_topic_slugs": sorted(allow_slugs),
        "deny_topic_slugs": sorted(deny_slugs),
        "thresholds": {
            "min_topic_count": min_topic_count,
            "min_mean_delta": min_mean_delta,
            "max_negative_mean_delta": max_negative_mean_delta,
        },
    }


def learn_from_selfqa_retrieval(
    *,
    report_path: Path,
    min_topic_count: int,
    min_topic_recall: float,
) -> dict[str, Any]:
    payload = load_json(report_path)
    by_slug: dict[str, list[float]] = {}
    for row in payload.get("rows") or []:
        shadow = row.get("topic_routing_shadow") or {}
        if not isinstance(shadow, dict):
            continue
        recall = shadow.get("topic_recall")
        if recall is None:
            continue
        for slug in shadow.get("routed_topic_slugs") or []:
            normalized = normalize_slug(str(slug))
            if normalized:
                by_slug.setdefault(normalized, []).append(float(recall))

    topic_stats: dict[str, dict[str, Any]] = {}
    allow_slugs: list[str] = []
    deny_slugs: list[str] = []
    for slug, recalls in sorted(by_slug.items()):
        stat = {
            "count": len(recalls),
            "mean_topic_recall": round(mean(recalls), 6),
        }
        topic_stats[slug] = stat
        if len(recalls) < min_topic_count:
            continue
        if stat["mean_topic_recall"] >= min_topic_recall:
            allow_slugs.append(slug)
        else:
            deny_slugs.append(slug)

    return {
        "source_mode": "selfqa_retrieval",
        "report": str(report_path),
        "topic_stats": topic_stats,
        "allow_topic_slugs": sorted(allow_slugs),
        "deny_topic_slugs": sorted(deny_slugs),
        "thresholds": {
            "min_topic_count": min_topic_count,
            "min_topic_recall": min_topic_recall,
        },
    }


def write_policy_to_view(db_path: Path, view_id: str, policy: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    store = SQLiteMemoryStore(db_path)
    try:
        view = store.get_memory_view(view_id)
        if view is None:
            raise RuntimeError(f"Unknown memory view: {view_id}")
        metadata = dict(view.get("metadata") or {})
        metadata["topic_soft_policy"] = {
            "name": policy["name"],
            "created_at": policy["created_at"],
            "source_mode": policy["source_mode"],
            "allow_topic_slugs": list(policy.get("allow_topic_slugs") or []),
            "deny_topic_slugs": list(policy.get("deny_topic_slugs") or []),
            "thresholds": dict(policy.get("thresholds") or {}),
        }
        if not dry_run:
            store.update_memory_view_metadata(view_id, metadata=metadata)
            store.commit()
        return {
            "db": str(db_path),
            "view_id": view_id,
            "dry_run": dry_run,
            "written": not dry_run,
            "metadata_topic_soft_policy": metadata["topic_soft_policy"],
        }
    finally:
        store.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if args.mode == "qa_delta":
        baseline = Path(args.baseline)
        variant = Path(args.variant or args.input)
        if not args.baseline:
            raise RuntimeError("--baseline is required for --mode qa_delta")
        learned = learn_from_qa_delta(
            baseline_path=baseline,
            variant_path=variant,
            min_topic_count=max(1, int(args.min_topic_count)),
            min_mean_delta=float(args.min_mean_delta),
            max_negative_mean_delta=float(args.max_negative_mean_delta),
        )
    else:
        learned = learn_from_selfqa_retrieval(
            report_path=input_path,
            min_topic_count=max(1, int(args.min_topic_count)),
            min_topic_recall=float(args.min_topic_recall),
        )

    policy = {
        "name": str(args.policy_name),
        "created_at": utc_now_iso(),
        **learned,
    }
    if args.db and args.view_id:
        policy["view_write"] = write_policy_to_view(
            Path(args.db),
            args.view_id,
            policy,
            dry_run=bool(args.dry_run),
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
