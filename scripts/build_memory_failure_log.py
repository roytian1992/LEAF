from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
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
        description=(
            "Build a compact failure log for memory evolution proposals from LoCoMo QA "
            "reports and/or self-QA search reports."
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--locomo-report", action="append", default=[])
    parser.add_argument("--baseline-locomo-report", action="append", default=[])
    parser.add_argument("--selfqa-report", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--min-f1", type=float, default=0.5)
    parser.add_argument("--regression-margin", type=float, default=0.08)
    parser.add_argument("--latency-outlier-ms", type=float, default=15000.0)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--max-context-lines", type=int, default=8)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_from_report(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    rows = payload.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return round(float(statistics.mean(values)), 4) if values else 0.0


def sanitize_sample_id(sample_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(sample_id or "").strip())
    return cleaned.strip("_") or "sample"


def corpus_from_row(row: dict[str, Any]) -> str:
    corpus_id = str(row.get("corpus_id") or "").strip()
    if corpus_id:
        return corpus_id
    sample_id = str(row.get("sample_id") or "").strip()
    return f"locomo_{sanitize_sample_id(sample_id)}" if sample_id else ""


def normalized_question(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("question") or "").strip().split()).lower()


def baseline_keys(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    sample_id = str(row.get("sample_id") or "").strip()
    corpus_id = corpus_from_row(row)
    question = normalized_question(row)
    qa_id = str(row.get("qa_id") or row.get("question_index") or "").strip()
    return [
        ("sample_question", sample_id, question),
        ("corpus_question", corpus_id, question),
        ("qa_id", f"{sample_id}:{qa_id}", question),
        ("question", "", question),
    ]


def build_baseline_index(paths: list[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        for row in rows_from_report(payload):
            for key in baseline_keys(row):
                index.setdefault(key, row)
    return index


def find_baseline(row: dict[str, Any], index: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any] | None:
    for key in baseline_keys(row):
        baseline = index.get(key)
        if baseline is not None:
            return baseline
    return None


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def compact_context_lines(row: dict[str, Any], max_lines: int) -> list[str]:
    lines = row.get("answer_context_lines")
    if not isinstance(lines, list):
        return []
    compact: list[str] = []
    for line in lines:
        text = str(line).strip()
        if text:
            compact.append(text[:500])
        if len(compact) >= max(0, int(max_lines)):
            break
    return compact


def classify_locomo_failure(
    row: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    min_f1: float,
    regression_margin: float,
    latency_outlier_ms: float,
) -> tuple[list[str], float, dict[str, Any]]:
    f1 = as_float(row.get("answer_f1"))
    baseline_f1 = as_float(baseline.get("answer_f1")) if baseline else None
    delta = None if baseline_f1 is None else round(f1 - baseline_f1, 4)
    category = str(row.get("category_name") or row.get("category") or "").strip().lower()
    topic_event_count = as_int(row.get("topic_soft_event_count"))
    topic_candidate_count = as_int(row.get("topic_soft_candidate_atom_count"))
    active_topic_slugs = string_list(row.get("topic_soft_active_topic_slugs"))
    policy_reason = str(row.get("topic_soft_policy_reason") or "")
    search_ms = as_float(row.get("search_elapsed_ms"))

    modes: list[str] = []
    if f1 < min_f1:
        modes.append("answer_wrong")
    if delta is not None and delta <= -abs(regression_margin):
        modes.append("regressed_vs_baseline")
    if topic_event_count > 0 and delta is not None and delta <= -abs(regression_margin):
        modes.append("topic_noise")
    if category == "single_hop" and topic_event_count > 0 and delta is not None and delta < 0.0:
        modes.append("single_fact_topic_interference")
    if category == "temporal" and f1 < min_f1:
        modes.append("temporal_conflict_or_missing_evidence")
    if category in {"multi_hop", "open_domain"} and not active_topic_slugs and f1 < min_f1:
        modes.append("topic_missing_or_unused")
    if category == "open_domain" and f1 < min_f1:
        modes.append("open_domain_gap")
    if topic_candidate_count > 48 and topic_event_count == 0 and f1 < min_f1:
        modes.append("topic_route_noisy_pool")
    if "suppressed" in policy_reason and category != "temporal" and f1 < min_f1:
        modes.append("topic_policy_over_suppression")
    if latency_outlier_ms > 0.0 and search_ms >= latency_outlier_ms:
        modes.append("latency_outlier")
    modes = list(dict.fromkeys(modes))

    severity = max(0.0, 1.0 - f1)
    if delta is not None and delta < 0.0:
        severity += min(0.5, abs(delta))
    if "topic_noise" in modes:
        severity += 0.15
    if "topic_missing_or_unused" in modes:
        severity += 0.1
    if "latency_outlier" in modes:
        severity += 0.05

    diagnostics = {
        "baseline_f1": baseline_f1,
        "f1_delta_vs_baseline": delta,
        "topic_event_count": topic_event_count,
        "topic_candidate_atom_count": topic_candidate_count,
        "topic_policy_reason": policy_reason,
        "search_elapsed_ms": search_ms,
    }
    return modes, round(severity, 4), diagnostics


def topic_slug_from_node(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    return str(
        metadata.get("topic_slug")
        or metadata.get("evolved_slug")
        or metadata.get("seed_slug")
        or node.get("name")
        or node.get("topic_id")
        or ""
    ).strip()


def topic_role_from_node(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    if metadata.get("evolved_from") or metadata.get("evolved_slug"):
        return "evolved"
    if metadata.get("seed_slug") or metadata.get("seed_role"):
        return "seed"
    return str(metadata.get("topic_role") or "unknown")


def build_topic_context(store: SQLiteMemoryStore, corpus_id: str) -> dict[str, Any]:
    if not corpus_id:
        return {}
    view = store.get_active_memory_view(corpus_id)
    if view is None:
        return {}
    nodes = store.list_topic_nodes(str(view["view_id"]))
    assignment_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()
    for assignment in store.list_topic_assignments(str(view["view_id"])):
        topic_id = str(assignment.get("topic_id") or "")
        if not topic_id:
            continue
        if str(assignment.get("item_kind") or "") == "atom_secondary":
            secondary_counts[topic_id] += 1
        else:
            assignment_counts[topic_id] += 1
    by_slug: dict[str, dict[str, Any]] = {}
    for node in nodes:
        slug = topic_slug_from_node(node)
        metadata = dict(node.get("metadata") or {})
        by_slug[slug] = {
            "topic_id": node.get("topic_id"),
            "slug": slug,
            "name": node.get("name"),
            "parent_id": node.get("parent_id"),
            "level": node.get("level"),
            "role": topic_role_from_node(node),
            "keywords": string_list(node.get("keywords"))[:16],
            "route_keywords": string_list(metadata.get("route_keywords"))[:16],
            "profile_terms": string_list(metadata.get("profile_terms"))[:16],
            "answer_exposure": metadata.get("answer_exposure"),
            "route_exposure": metadata.get("route_exposure"),
            "assignment_count": assignment_counts[str(node.get("topic_id") or "")],
            "secondary_assignment_count": secondary_counts[str(node.get("topic_id") or "")],
        }
    return {
        "active_view_id": view.get("view_id"),
        "active_view_name": view.get("name"),
        "active_view_status": view.get("status"),
        "topic_by_slug": by_slug,
    }


def referenced_topic_slugs_from_locomo(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "topic_soft_active_topic_slugs",
        "topic_soft_topic_slugs",
        "topic_soft_suppressed_topic_slugs",
    ):
        values.extend(string_list(row.get(key)))
    return sorted(set(values))


def locomo_failure_rows(
    *,
    path: str,
    payload: dict[str, Any],
    baseline_index: dict[tuple[str, str, str], dict[str, Any]],
    topic_contexts: dict[str, dict[str, Any]],
    min_f1: float,
    regression_margin: float,
    latency_outlier_ms: float,
    max_context_lines: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows_from_report(payload), start=1):
        baseline = find_baseline(row, baseline_index) if baseline_index else None
        modes, severity, diagnostics = classify_locomo_failure(
            row,
            baseline,
            min_f1=min_f1,
            regression_margin=regression_margin,
            latency_outlier_ms=latency_outlier_ms,
        )
        if not modes:
            continue
        corpus_id = corpus_from_row(row)
        sample_id = str(row.get("sample_id") or "").strip()
        question = str(row.get("question") or "").strip()
        active_slugs = referenced_topic_slugs_from_locomo(row)
        topic_context = topic_contexts.get(corpus_id) or {}
        topic_by_slug = dict(topic_context.get("topic_by_slug") or {})
        active_topic_refs = [topic_by_slug[slug] for slug in active_slugs if slug in topic_by_slug]
        failure_id = f"fail_{stable_hash('locomo', path, sample_id, row.get('qa_id'), question, ','.join(modes), length=20)}"
        rows.append(
            {
                "failure_id": failure_id,
                "source_type": "locomo_qa",
                "source_report": str(path),
                "row_index": row_index,
                "corpus_id": corpus_id,
                "sample_id": sample_id,
                "qa_id": row.get("qa_id"),
                "question_index": row.get("question_index"),
                "diagnostic_question_type": row.get("category_name") or row.get("category"),
                "question": question,
                "gold_answer": row.get("gold_answer"),
                "predicted_answer": row.get("predicted_answer"),
                "answer_f1": as_float(row.get("answer_f1")),
                "bleu1": as_float(row.get("bleu1")),
                "failure_modes": modes,
                "severity": severity,
                "diagnostics": diagnostics,
                "topic_soft": {
                    "router": row.get("topic_soft_router"),
                    "policy": row.get("topic_soft_policy"),
                    "policy_reason": row.get("topic_soft_policy_reason"),
                    "active_topic_slugs": string_list(row.get("topic_soft_active_topic_slugs")),
                    "candidate_topic_slugs": string_list(row.get("topic_soft_topic_slugs")),
                    "suppressed_topic_slugs": string_list(row.get("topic_soft_suppressed_topic_slugs")),
                    "event_count": as_int(row.get("topic_soft_event_count")),
                    "candidate_atom_count": as_int(row.get("topic_soft_candidate_atom_count")),
                    "raw_candidate_atom_count": as_int(row.get("topic_soft_raw_candidate_atom_count")),
                    "filtered_atom_count": as_int(row.get("topic_soft_filtered_atom_count")),
                },
                "retrieval": {
                    "mode": row.get("retrieval_mode"),
                    "retrieved_dia_ids": string_list(row.get("retrieved_dia_ids")),
                    "raw_span_count": as_int(row.get("raw_span_count")),
                    "atom_count": as_int(row.get("atom_count")),
                    "search_elapsed_ms": as_float(row.get("search_elapsed_ms")),
                },
                "baseline": (
                    None
                    if baseline is None
                    else {
                        "source_report": "baseline_locomoreport_index",
                        "answer_f1": as_float(baseline.get("answer_f1")),
                        "bleu1": as_float(baseline.get("bleu1")),
                        "predicted_answer": baseline.get("predicted_answer"),
                        "retrieval_mode": baseline.get("retrieval_mode"),
                        "search_elapsed_ms": as_float(baseline.get("search_elapsed_ms")),
                    }
                ),
                "topic_context": {
                    "active_view_id": topic_context.get("active_view_id"),
                    "active_view_name": topic_context.get("active_view_name"),
                    "active_topic_refs": active_topic_refs,
                },
                "answer_context_lines_sample": compact_context_lines(row, max_context_lines),
            }
        )
    return rows


def selfqa_failure_modes(row: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    if row.get("event_path_hit") is False or as_float(row.get("event_recall"), 1.0) < 1.0:
        modes.append("selfqa_event_miss")
    if row.get("atom_path_hit") is False or as_float(row.get("atom_recall"), 1.0) < 1.0:
        modes.append("selfqa_atom_miss")
    criteria_metrics = row.get("criteria_path_metrics")
    if isinstance(criteria_metrics, dict):
        for key in (
            "evidence_required_event_hit",
            "evidence_required_atom_hit",
            "must_retrieve_any_event_hit",
            "must_retrieve_all_event_hit",
            "must_retrieve_any_atom_hit",
            "must_retrieve_all_atom_hit",
        ):
            if criteria_metrics.get(key) is False:
                modes.append(f"criteria_{key}_failed")
    if row.get("answer_criteria_static_pass") is False:
        modes.append("criteria_answer_static_failed")
    topic_shadow = row.get("topic_routing_shadow")
    if isinstance(topic_shadow, dict):
        if topic_shadow.get("criteria_expected_topic_route_hit") is False:
            modes.append("topic_route_miss")
        if topic_shadow.get("criteria_expected_topic_retrieval_hit") is False:
            modes.append("topic_retrieval_miss")
        if topic_shadow.get("topic_path_hit") is False:
            modes.append("gold_topic_path_miss")
    if as_int(row.get("topic_soft_candidate_atom_count")) > 48 and as_float(row.get("event_recall"), 1.0) < 1.0:
        modes.append("topic_noise_large_candidate_pool")
    return list(dict.fromkeys(modes))


def selfqa_failure_rows(
    *,
    path: str,
    payload: dict[str, Any],
    topic_contexts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = dict(payload.get("summary") or {})
    default_corpus_id = str(summary.get("corpus_id") or "").strip()
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows_from_report(payload), start=1):
        modes = selfqa_failure_modes(row)
        if not modes:
            continue
        corpus_id = str(row.get("corpus_id") or default_corpus_id or "").strip()
        topic_shadow = row.get("topic_routing_shadow") if isinstance(row.get("topic_routing_shadow"), dict) else {}
        slugs = sorted(
            set(
                string_list(topic_shadow.get("routed_topic_slugs"))
                + string_list(topic_shadow.get("gold_topic_slugs"))
                + string_list(topic_shadow.get("retrieved_topic_slugs"))
                + string_list(topic_shadow.get("criteria_expected_topic_slugs"))
            )
        )
        topic_context = topic_contexts.get(corpus_id) or {}
        topic_by_slug = dict(topic_context.get("topic_by_slug") or {})
        topic_refs = [topic_by_slug[slug] for slug in slugs if slug in topic_by_slug]
        event_recall = row.get("event_recall")
        atom_recall = row.get("atom_recall")
        severity = 0.0
        if event_recall is not None:
            severity += 1.0 - as_float(event_recall)
        if atom_recall is not None:
            severity += 1.0 - as_float(atom_recall)
        if "topic_route_miss" in modes:
            severity += 0.25
        if "topic_retrieval_miss" in modes:
            severity += 0.25
        failure_id = f"fail_{stable_hash('selfqa', path, row.get('task_id'), row.get('question'), ','.join(modes), length=20)}"
        rows.append(
            {
                "failure_id": failure_id,
                "source_type": "selfqa_search",
                "source_report": str(path),
                "row_index": row_index,
                "corpus_id": corpus_id,
                "task_id": row.get("task_id"),
                "task_index": row.get("task_index"),
                "diagnostic_task_type": row.get("task_type"),
                "tags": string_list(row.get("tags")),
                "question": row.get("question"),
                "gold_answer": row.get("answer"),
                "failure_modes": modes,
                "severity": round(severity, 4),
                "retrieval": {
                    "event_recall": row.get("event_recall"),
                    "atom_recall": row.get("atom_recall"),
                    "returned_atom_recall": row.get("returned_atom_recall"),
                    "event_path_hit": row.get("event_path_hit"),
                    "atom_path_hit": row.get("atom_path_hit"),
                    "retrieved_event_ids": string_list(row.get("retrieved_event_ids")),
                    "retrieved_atom_ids": string_list(row.get("retrieved_atom_ids")),
                    "gold_event_ids": string_list(row.get("gold_event_ids")),
                    "gold_atom_ids": string_list(row.get("gold_atom_ids")),
                },
                "criteria": {
                    "path_metrics": row.get("criteria_path_metrics"),
                    "answer_criteria_static_pass": row.get("answer_criteria_static_pass"),
                },
                "topic_routing_shadow": topic_shadow,
                "topic_soft": {
                    "event_count": as_int(row.get("topic_soft_event_count")),
                    "candidate_atom_count": as_int(row.get("topic_soft_candidate_atom_count")),
                    "raw_candidate_atom_count": as_int(row.get("topic_soft_raw_candidate_atom_count")),
                    "filtered_atom_count": as_int(row.get("topic_soft_filtered_atom_count")),
                },
                "topic_context": {
                    "active_view_id": topic_context.get("active_view_id"),
                    "active_view_name": topic_context.get("active_view_name"),
                    "active_topic_refs": topic_refs,
                },
            }
        )
    return rows


def corpus_ids_from_reports(paths: list[str], baseline_paths: list[str]) -> set[str]:
    corpus_ids: set[str] = set()
    for path in list(paths) + list(baseline_paths):
        if not Path(path).exists():
            continue
        payload = load_json(path)
        summary_corpus = str((payload.get("summary") or {}).get("corpus_id") or "").strip()
        if summary_corpus:
            corpus_ids.add(summary_corpus)
        for row in rows_from_report(payload):
            corpus_id = corpus_from_row(row)
            if corpus_id:
                corpus_ids.add(corpus_id)
    return corpus_ids


def summarize_failures(rows: list[dict[str, Any]], *, inputs: dict[str, Any]) -> dict[str, Any]:
    by_mode: Counter[str] = Counter()
    by_corpus: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_question_type: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    severities: list[float] = []
    for row in rows:
        by_source[str(row.get("source_type") or "")] += 1
        by_corpus[str(row.get("corpus_id") or "")] += 1
        diagnostic_type = str(row.get("diagnostic_question_type") or row.get("diagnostic_task_type") or "")
        if diagnostic_type:
            by_question_type[diagnostic_type] += 1
        for mode in string_list(row.get("failure_modes")):
            by_mode[mode] += 1
        severities.append(as_float(row.get("severity")))
        topic_soft = row.get("topic_soft") if isinstance(row.get("topic_soft"), dict) else {}
        for slug in string_list(topic_soft.get("active_topic_slugs")) + string_list(topic_soft.get("candidate_topic_slugs")):
            topic_counts[slug] += 1
        topic_shadow = row.get("topic_routing_shadow") if isinstance(row.get("topic_routing_shadow"), dict) else {}
        for slug in string_list(topic_shadow.get("routed_topic_slugs")) + string_list(topic_shadow.get("gold_topic_slugs")):
            topic_counts[slug] += 1
    return {
        "created_at": utc_now_iso(),
        "inputs": inputs,
        "failure_count": len(rows),
        "by_source_type": dict(sorted(by_source.items())),
        "by_corpus_id": dict(sorted(by_corpus.items())),
        "by_failure_mode": dict(by_mode.most_common()),
        "by_diagnostic_type": dict(by_question_type.most_common()),
        "top_topic_slugs": dict(topic_counts.most_common(30)),
        "severity": {
            "mean": mean(severities),
            "max": round(max(severities), 4) if severities else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    report_paths = list(args.locomo_report or [])
    selfqa_paths = list(args.selfqa_report or [])
    baseline_paths = list(args.baseline_locomo_report or [])
    corpus_ids = corpus_ids_from_reports(report_paths + selfqa_paths, baseline_paths)
    store = SQLiteMemoryStore(args.db)
    try:
        topic_contexts = {corpus_id: build_topic_context(store, corpus_id) for corpus_id in sorted(corpus_ids)}
    finally:
        store.close()

    baseline_index = build_baseline_index(baseline_paths)
    failures: list[dict[str, Any]] = []
    for path in report_paths:
        payload = load_json(path)
        failures.extend(
            locomo_failure_rows(
                path=path,
                payload=payload,
                baseline_index=baseline_index,
                topic_contexts=topic_contexts,
                min_f1=float(args.min_f1),
                regression_margin=float(args.regression_margin),
                latency_outlier_ms=float(args.latency_outlier_ms),
                max_context_lines=int(args.max_context_lines),
            )
        )
    for path in selfqa_paths:
        payload = load_json(path)
        failures.extend(selfqa_failure_rows(path=path, payload=payload, topic_contexts=topic_contexts))

    failures.sort(key=lambda row: (-as_float(row.get("severity")), str(row.get("failure_id") or "")))
    if args.max_items > 0:
        failures = failures[: int(args.max_items)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize_failures(
        failures,
        inputs={
            "db": str(args.db),
            "locomo_reports": report_paths,
            "baseline_locomoreports": baseline_paths,
            "selfqa_reports": selfqa_paths,
            "min_f1": args.min_f1,
            "regression_margin": args.regression_margin,
            "latency_outlier_ms": args.latency_outlier_ms,
            "max_items": args.max_items,
            "output": str(output),
        },
    )
    if args.summary_output:
        summary_output = Path(args.summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
