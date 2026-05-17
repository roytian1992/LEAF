from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import utc_now_iso
from leaf.memory_overlay import default_retrieval_policy_overlay, overlay_query_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a runtime-feature-only memory overlay retrieval policy from LoCoMo/self-QA failure logs. "
            "This writes a proposal JSON; it does not mutate the memory database."
        )
    )
    parser.add_argument("--failure-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-id", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def diagnostic_group(row: dict[str, Any]) -> str:
    label = str(row.get("diagnostic_question_type") or row.get("diagnostic_task_type") or "").strip().lower()
    if "temporal" in label:
        return "runtime_temporal_feature"
    question = str(row.get("question") or "")
    features = overlay_query_features(question)
    if features.get("temporal"):
        return "runtime_temporal_feature"
    if features.get("relationship"):
        return "runtime_relationship_feature"
    if features.get("profile"):
        return "runtime_profile_feature"
    if features.get("plan"):
        return "runtime_plan_feature"
    if features.get("media_hobby"):
        return "runtime_media_hobby_feature"
    if features.get("place_travel"):
        return "runtime_place_travel_feature"
    if features.get("activity"):
        return "runtime_activity_feature"
    return "runtime_general_feature"


def topic_soft_stats(row: dict[str, Any]) -> dict[str, Any]:
    topic_soft = row.get("topic_soft") if isinstance(row.get("topic_soft"), dict) else {}
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    return {
        "event_count": int(topic_soft.get("event_count") or 0),
        "candidate_atom_count": int(topic_soft.get("candidate_atom_count") or diagnostics.get("topic_candidate_atom_count") or 0),
        "policy_reason": str(topic_soft.get("policy_reason") or diagnostics.get("topic_policy_reason") or ""),
        "f1_delta_vs_baseline": as_float(diagnostics.get("f1_delta_vs_baseline"), 0.0),
        "answer_f1": as_float(row.get("answer_f1"), 0.0),
        "baseline_f1": as_float((row.get("baseline") or {}).get("answer_f1") if isinstance(row.get("baseline"), dict) else None, 0.0),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    modes = Counter()
    source_types = Counter()
    corpora = Counter()
    for row in rows:
        groups.setdefault(diagnostic_group(row), []).append(row)
        modes.update(string_list(row.get("failure_modes")))
        source_types[str(row.get("source_type") or "")] += 1
        corpora[str(row.get("corpus_id") or "")] += 1

    group_summary: dict[str, dict[str, Any]] = {}
    for group, group_rows in sorted(groups.items()):
        stats = [topic_soft_stats(row) for row in group_rows]
        negative_deltas = [item["f1_delta_vs_baseline"] for item in stats if item["f1_delta_vs_baseline"] < 0]
        group_summary[group] = {
            "failure_count": len(group_rows),
            "avg_topic_event_count": round(sum(item["event_count"] for item in stats) / len(stats), 4),
            "avg_candidate_atom_count": round(sum(item["candidate_atom_count"] for item in stats) / len(stats), 4),
            "policy_reason_counts": dict(Counter(item["policy_reason"] for item in stats)),
            "regression_count": len(negative_deltas),
            "avg_negative_delta": round(sum(negative_deltas) / len(negative_deltas), 4) if negative_deltas else 0.0,
        }
    return {
        "failure_count": len(rows),
        "by_source_type": dict(source_types.most_common()),
        "by_corpus_id": dict(corpora.most_common()),
        "by_failure_mode": dict(modes.most_common()),
        "by_runtime_feature_group": group_summary,
    }


def derive_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = default_retrieval_policy_overlay()
    summary = summarize(rows)
    failure_modes = Counter()
    for row in rows:
        failure_modes.update(string_list(row.get("failure_modes")))

    policy = json.loads(json.dumps(base, ensure_ascii=False))
    policy["version"] = "retrieval_policy_overlay_v2_failure_adapted"
    policy["source"] = {
        "kind": "failure_log_selfqa_locomo",
        "generated_at": utc_now_iso(),
    }
    policy["runtime_feature_routes"] = {
        "temporal": {
            "prefer_temporal_overlay": True,
            "suppress_topic_soft_when_temporal_feature": True,
            "force_raw_timestamp": True,
        },
        "entity_profile": {
            "prefer_entity_profile_overlay": True,
            "fallback_when_topic_candidate_atoms_zero": True,
            "require_query_entity_match": True,
            "min_event_score": 0.42,
        },
        "topic_soft": {
            "suppress_when_candidate_atoms_above": 20,
            "require_selected_content_overlap": 2,
        },
    }

    if failure_modes.get("topic_noise", 0) + failure_modes.get("single_fact_topic_interference", 0) >= 3:
        policy["topic_soft"]["max_candidate_atoms"] = 16
        policy["topic_soft"]["min_selected_overlap"] = 2
        policy["context_assembly"] = {
            "primary_evidence_first": True,
            "overlay_as_supplemental": True,
            "deduplicate_raw_spans": True,
            "group_by_entity_when_profile_feature": True,
            "sort_by_timestamp_when_temporal_feature": True,
        }
    else:
        policy["context_assembly"] = {
            "primary_evidence_first": True,
            "overlay_as_supplemental": True,
        }

    if failure_modes.get("topic_missing_or_unused", 0) + failure_modes.get("selfqa_event_miss", 0) >= 2:
        policy["entity_profile"]["max_events"] = 4
        policy["entity_profile"]["require_query_entity_match"] = True
        policy["entity_profile"]["min_event_score"] = 0.42
        policy["fallback"]["baseline_on_unknown"] = True
        policy["fallback"]["suppress_when_primary_strong"] = True
        policy["fallback"]["primary_strong_min_raw_spans"] = 4
        policy["fallback"]["primary_strong_min_content_overlap"] = 2
        policy["fallback"]["primary_strong_min_top3_overlap_sum"] = 4

    temporal_failures = sum(
        int(item.get("failure_count") or 0)
        for key, item in (summary.get("by_runtime_feature_group") or {}).items()
        if key == "runtime_temporal_feature"
    )
    if temporal_failures:
        policy["temporal"]["max_events"] = 5
        policy["temporal"]["include_neighbor_events"] = True
        policy["topic_soft"]["skip_temporal"] = True

    return policy


def compact_examples(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (-as_float(row.get("severity")), str(row.get("failure_id") or "")))
    examples: list[dict[str, Any]] = []
    for row in rows[:limit]:
        stats = topic_soft_stats(row)
        examples.append(
            {
                "failure_id": row.get("failure_id"),
                "source_type": row.get("source_type"),
                "corpus_id": row.get("corpus_id"),
                "runtime_feature_group": diagnostic_group(row),
                "failure_modes": string_list(row.get("failure_modes")),
                "question": re.sub(r"\s+", " ", str(row.get("question") or "")).strip()[:240],
                "topic_soft": stats,
            }
        )
    return examples


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.failure_log)
    corpus_filter = {str(item).strip() for item in args.corpus_id if str(item).strip()}
    if corpus_filter:
        rows = [row for row in rows if str(row.get("corpus_id") or "") in corpus_filter]
    if args.max_rows and int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    policy = derive_policy(rows)
    payload = {
        "proposal_version": "memory_overlay_policy_proposal_v1",
        "created_at": utc_now_iso(),
        "source_failure_log": str(args.failure_log),
        "corpus_filter": sorted(corpus_filter),
        "summary": summarize(rows),
        "policy_patch": policy,
        "examples": compact_examples(rows),
        "guard": {
            "selfqa_required": True,
            "locomo_guard": "Run a small held-out corpus guard before promotion.",
            "reject_if": "Overall F1/BLEU1 below matched active-view control or temporal category regresses materially.",
        },
        "notes": [
            "The proposal uses runtime-visible query features and failure modes only.",
            "Diagnostic question_type labels are used only to summarize failures, not as runtime routing keys.",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "failure_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
