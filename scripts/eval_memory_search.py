from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf import LEAFService
from leaf.agentic_memory import route_query_to_topics
from leaf.clients import ChatClient, extract_json_object
from leaf.topic_soft import (
    build_topic_context as build_topic_soft_context,
    merge_topic_soft_evidence,
    topic_soft_expand_events,
)

CRITERIA_VERSION = "criteria_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LEAF memory search against a frozen self-QA JSONL file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--selfqa", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-limit", type=int, default=6)
    parser.add_argument("--raw-span-limit", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--retrieval-mode", choices=["baseline", "topic_soft"], default="baseline")
    parser.add_argument("--topic-routing-shadow", action="store_true")
    parser.add_argument("--topic-router", choices=["keyword", "profile_hybrid", "llm"], default="keyword")
    parser.add_argument("--topic-route-top-k", type=int, default=3)
    parser.add_argument("--topic-view-id", default="", help="Defaults to the active memory view.")
    parser.add_argument("--topic-soft-event-limit", type=int, default=4)
    parser.add_argument("--topic-soft-per-topic-atom-limit", type=int, default=16)
    parser.add_argument(
        "--topic-soft-use-stemmed-content-tokens",
        action="store_true",
        help="Use English Snowball stemming in topic-soft content-token overlap and atom scoring.",
    )
    return parser.parse_args()


def load_selfqa(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def extract_gold_ids(task: dict[str, Any]) -> tuple[set[str], set[str]]:
    event_ids: set[str] = set()
    atom_ids: set[str] = set()
    for item in task.get("gold_evidence_path") or []:
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id") or "").strip()
        atom_id = str(item.get("atom_id") or "").strip()
        if event_id:
            event_ids.add(event_id)
        if atom_id:
            atom_ids.add(atom_id)
    return event_ids, atom_ids


def task_criteria_v1(task: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(task.get("metadata") or {})
    criteria = metadata.get(CRITERIA_VERSION) or task.get(CRITERIA_VERSION) or {}
    return dict(criteria) if isinstance(criteria, dict) else {}


def criteria_id_set(criteria: dict[str, Any], section: str, key: str) -> set[str]:
    payload = dict(criteria.get(section) or {})
    return {str(value).strip() for value in (payload.get(key) or []) if str(value).strip()}


def normalized_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def answer_criteria_static_pass(task: dict[str, Any], criteria: dict[str, Any]) -> bool | None:
    answer_criteria = dict(criteria.get("answer_criteria") or {})
    must_contain = [str(item).strip() for item in (answer_criteria.get("must_contain") or []) if str(item).strip()]
    wrong_if_contains = [
        str(item).strip() for item in (answer_criteria.get("wrong_if_contains") or []) if str(item).strip()
    ]
    if not must_contain and not wrong_if_contains:
        return None
    answer_text = normalized_text(str(task.get("answer") or ""))
    for phrase in must_contain:
        if normalized_text(phrase) not in answer_text:
            return False
    for phrase in wrong_if_contains:
        if normalized_text(phrase) and normalized_text(phrase) in answer_text:
            return False
    return True


def criteria_path_metrics(
    criteria: dict[str, Any],
    *,
    retrieved_event_ids: set[str],
    retrieved_atom_ids: set[str],
) -> dict[str, Any]:
    evidence_required_event_ids = criteria_id_set(criteria, "evidence_criteria", "required_event_ids")
    evidence_required_atom_ids = criteria_id_set(criteria, "evidence_criteria", "required_atom_ids")
    must_any_event_ids = criteria_id_set(criteria, "retrieval_criteria", "must_retrieve_any_event_ids")
    must_all_event_ids = criteria_id_set(criteria, "retrieval_criteria", "must_retrieve_all_event_ids")
    must_any_atom_ids = criteria_id_set(criteria, "retrieval_criteria", "must_retrieve_any_atom_ids")
    must_all_atom_ids = criteria_id_set(criteria, "retrieval_criteria", "must_retrieve_all_atom_ids")
    return {
        "criteria_version": str(criteria.get("version") or ""),
        "evidence_required_event_ids": sorted(evidence_required_event_ids),
        "evidence_required_atom_ids": sorted(evidence_required_atom_ids),
        "evidence_required_event_hit": bool(
            evidence_required_event_ids and evidence_required_event_ids.issubset(retrieved_event_ids)
        ),
        "evidence_required_atom_hit": bool(
            evidence_required_atom_ids and evidence_required_atom_ids.issubset(retrieved_atom_ids)
        ),
        "must_retrieve_any_event_hit": (
            None if not must_any_event_ids else bool(must_any_event_ids.intersection(retrieved_event_ids))
        ),
        "must_retrieve_all_event_hit": (
            None if not must_all_event_ids else bool(must_all_event_ids.issubset(retrieved_event_ids))
        ),
        "must_retrieve_any_atom_hit": (
            None if not must_any_atom_ids else bool(must_any_atom_ids.intersection(retrieved_atom_ids))
        ),
        "must_retrieve_all_atom_hit": (
            None if not must_all_atom_ids else bool(must_all_atom_ids.issubset(retrieved_atom_ids))
        ),
    }


def extract_retrieved_ids(result: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    event_ids = {str(value).strip() for value in (result.get("selected_event_ids") or []) if str(value).strip()}
    atom_ids: set[str] = set()
    span_ids: set[str] = set()
    for atom in result.get("atoms") or []:
        atom_id = str(atom.get("atom_id") or "").strip()
        event_id = str(atom.get("event_id") or "").strip()
        if atom_id:
            atom_ids.add(atom_id)
        if event_id:
            event_ids.add(event_id)
    for span in result.get("raw_spans") or []:
        span_id = str(span.get("span_id") or "").strip()
        if span_id:
            span_ids.add(span_id)
    return event_ids, atom_ids, span_ids


def get_event_atom_ids(service: LEAFService, event_ids: set[str]) -> set[str]:
    atoms = service.store.get_atoms_for_events(sorted(event_ids))
    return {str(atom.atom_id).strip() for atom in atoms if str(atom.atom_id).strip()}


def build_topic_shadow_context(service: LEAFService, corpus_id: str, view_id: str | None = None) -> dict[str, Any] | None:
    active_view = service.store.get_memory_view(view_id) if view_id else service.get_active_agentic_memory_view(corpus_id)
    if active_view is None or str(active_view.get("corpus_id") or "") != corpus_id:
        return None
    view_id = str(active_view["view_id"])
    topic_by_id = {str(node["topic_id"]): node for node in service.store.list_topic_nodes(view_id)}
    assignment_by_atom: dict[str, dict[str, Any]] = {}
    assignments_by_atom: dict[str, list[dict[str, Any]]] = {}
    for assignment in service.store.list_topic_assignments(view_id):
        item_kind = str(assignment.get("item_kind") or "")
        if item_kind not in {"atom", "atom_secondary"}:
            continue
        atom_id = str(assignment["item_id"])
        assignments_by_atom.setdefault(atom_id, []).append(assignment)
        if item_kind == "atom" or atom_id not in assignment_by_atom:
            assignment_by_atom[atom_id] = assignment
    return {
        "view_id": view_id,
        "view_name": active_view.get("name"),
        "topic_by_id": topic_by_id,
        "assignment_by_atom": assignment_by_atom,
        "assignments_by_atom": assignments_by_atom,
    }


def topic_slug(topic_by_id: dict[str, dict[str, Any]], topic_id: str) -> str:
    node = topic_by_id.get(topic_id) or {}
    metadata = dict(node.get("metadata") or {})
    return str(metadata.get("topic_slug") or metadata.get("evolved_slug") or metadata.get("seed_slug") or node.get("name") or topic_id)


def _metadata_id_values(metadata: dict[str, Any], *keys: str) -> set[str]:
    values: set[str] = set()
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, list):
            values.update(str(item).strip() for item in raw if str(item).strip())
        elif isinstance(raw, str) and raw.strip():
            values.add(raw.strip())
    return values


def resolve_topic_ids(topic_context: dict[str, Any], topic_ids: set[str]) -> set[str]:
    if not topic_ids:
        return set()
    topic_by_id: dict[str, dict[str, Any]] = topic_context["topic_by_id"]
    resolved = {topic_id for topic_id in topic_ids if topic_id in topic_by_id}
    unresolved = set(topic_ids) - resolved
    if not unresolved:
        return resolved
    for current_topic_id, node in topic_by_id.items():
        metadata = dict(node.get("metadata") or {})
        aliases = _metadata_id_values(
            metadata,
            "base_topic_id",
            "source_topic_id",
            "criteria_patch_clone_of_topic_id",
            "criteria_patch_source_topic_id",
        )
        if aliases.intersection(unresolved):
            resolved.add(str(current_topic_id))
    return resolved


def topic_ids_for_atoms(topic_context: dict[str, Any], atom_ids: set[str]) -> set[str]:
    assignment_by_atom: dict[str, dict[str, Any]] = topic_context["assignment_by_atom"]
    assignments_by_atom: dict[str, list[dict[str, Any]]] = topic_context.get("assignments_by_atom") or {}
    topic_ids: set[str] = set()
    for atom_id in atom_ids:
        assignments = assignments_by_atom.get(atom_id) or []
        if not assignments:
            assignment = assignment_by_atom.get(atom_id)
            assignments = [assignment] if assignment is not None else []
        if not assignments:
            continue
        for assignment in assignments:
            topic_id = str(assignment.get("topic_id") or "").strip()
            if topic_id:
                topic_ids.add(topic_id)
    return topic_ids


def topic_path_hit_for_atoms(
    topic_context: dict[str, Any],
    atom_ids: set[str],
    routed_topic_ids: set[str],
) -> bool:
    assignments_by_atom: dict[str, list[dict[str, Any]]] = topic_context.get("assignments_by_atom") or {}
    assignment_by_atom: dict[str, dict[str, Any]] = topic_context["assignment_by_atom"]
    if not atom_ids:
        return False
    for atom_id in atom_ids:
        assignments = assignments_by_atom.get(atom_id) or []
        if not assignments:
            assignment = assignment_by_atom.get(atom_id)
            assignments = [assignment] if assignment is not None else []
        atom_topic_ids = {
            str(assignment.get("topic_id") or "").strip()
            for assignment in assignments
            if assignment is not None and str(assignment.get("topic_id") or "").strip()
        }
        if not atom_topic_ids or atom_topic_ids.isdisjoint(routed_topic_ids):
            return False
    return True


def route_query_to_topics_llm(
    llm: ChatClient,
    topic_context: dict[str, Any],
    *,
    question: str,
    top_k: int,
) -> list[dict[str, Any]]:
    topic_by_id: dict[str, dict[str, Any]] = topic_context["topic_by_id"]
    topic_rows: list[dict[str, Any]] = []
    allowed_ids: set[str] = set()
    for topic_id, node in sorted(topic_by_id.items(), key=lambda item: (int(item[1].get("level") or 0), str(item[1].get("name") or ""))):
        metadata = dict(node.get("metadata") or {})
        slug = str(metadata.get("topic_slug") or metadata.get("evolved_slug") or metadata.get("seed_slug") or node.get("name") or topic_id)
        if metadata.get("seed_role") == "root":
            continue
        allowed_ids.add(topic_id)
        topic_rows.append(
            {
                "topic_id": topic_id,
                "slug": slug,
                "name": node.get("name"),
                "description": node.get("description"),
                "keywords": node.get("keywords") or [],
            }
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You route memory-search questions to topic nodes. "
                "Return only one valid JSON object. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Choose the most likely topic nodes for this memory question. "
                "Use the topic descriptions, not only keyword overlap. "
                "Return at most the requested top_k topics, ordered by confidence.\n\n"
                "Return this JSON shape:\n"
                "{\n"
                '  "topics": [\n'
                '    {"topic_id": "...", "confidence": 0.0, "reason": "short reason"}\n'
                "  ]\n"
                "}\n\n"
                f"top_k: {top_k}\n"
                f"question: {question}\n"
                f"topics: {json.dumps(topic_rows, ensure_ascii=False)}"
            ),
        },
    ]
    text = llm.text(messages, max_tokens=512, temperature=0.0)
    payload = extract_json_object(text)
    routes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload.get("topics") or []:
        if not isinstance(item, dict):
            continue
        topic_id = str(item.get("topic_id") or "").strip()
        if topic_id not in allowed_ids or topic_id in seen:
            continue
        node = topic_by_id[topic_id]
        metadata = dict(node.get("metadata") or {})
        slug = str(metadata.get("topic_slug") or metadata.get("evolved_slug") or metadata.get("seed_slug") or node.get("name") or topic_id)
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        routes.append(
            {
                "topic_id": topic_id,
                "slug": slug,
                "name": str(node.get("name") or ""),
                "level": int(node.get("level") or 0),
                "matched_keywords": [],
                "score": round(confidence, 4),
                "confidence": round(confidence, 3),
                "router": "llm_shadow_v0",
                "reason": str(item.get("reason") or "").strip(),
            }
        )
        seen.add(topic_id)
        if len(routes) >= top_k:
            break
    return routes


def build_topic_shadow_row(
    service: LEAFService,
    topic_context: dict[str, Any] | None,
    *,
    question: str,
    gold_atom_ids: set[str],
    retrieved_atom_ids: set[str],
    top_k: int,
    router: str,
    criteria: dict[str, Any] | None = None,
    routed_topics: list[dict[str, Any]] | None = None,
    routed_router_name: str | None = None,
) -> dict[str, Any] | None:
    if topic_context is None:
        return None
    if routed_topics is not None:
        routed = routed_topics
        router_name = routed_router_name or str(routed[0].get("router") if routed else "topic_shadow")
    else:
        router_name = "keyword_shadow_v0"
        try:
            if router == "llm":
                if service.memory_llm is None:
                    raise RuntimeError("memory_llm is not configured")
                routed = route_query_to_topics_llm(
                    service.memory_llm,
                    topic_context,
                    question=question,
                    top_k=top_k,
                )
                router_name = "llm_shadow_v0"
            else:
                routed = route_query_to_topics(
                    service.store,
                    view_id=str(topic_context["view_id"]),
                    query=question,
                    top_k=top_k,
                    query_embedding=None,
                    router=router,
                )
                router_name = "profile_hybrid_v0" if router == "profile_hybrid" else "keyword_shadow_v0"
        except Exception as exc:
            routed = route_query_to_topics(
                service.store,
                view_id=str(topic_context["view_id"]),
                query=question,
                top_k=top_k,
            )
            router_name = f"{router}_shadow_failed_keyword_fallback"
            for item in routed:
                item["router_error"] = str(exc)[:300]
    routed_topic_ids = {str(item["topic_id"]) for item in routed if str(item.get("topic_id") or "").strip()}
    gold_topic_ids = topic_ids_for_atoms(topic_context, gold_atom_ids)
    retrieved_topic_ids = topic_ids_for_atoms(topic_context, retrieved_atom_ids)
    raw_expected_topic_ids = criteria_id_set(criteria, "topic_criteria", "expected_topic_ids") if criteria else set()
    expected_topic_ids = resolve_topic_ids(topic_context, raw_expected_topic_ids)
    topic_by_id: dict[str, dict[str, Any]] = topic_context["topic_by_id"]
    topic_recall = safe_recall(gold_topic_ids, routed_topic_ids)
    retrieval_topic_recall = safe_recall(gold_topic_ids, retrieved_topic_ids)
    return {
        "view_id": topic_context["view_id"],
        "view_name": topic_context.get("view_name"),
        "router": router_name,
        "top_k": top_k,
        "routed_topics": routed,
        "routed_topic_ids": sorted(routed_topic_ids),
        "routed_topic_slugs": [str(item.get("slug") or item.get("topic_id")) for item in routed],
        "gold_topic_ids": sorted(gold_topic_ids),
        "gold_topic_slugs": sorted(topic_slug(topic_by_id, topic_id) for topic_id in gold_topic_ids),
        "retrieved_topic_ids": sorted(retrieved_topic_ids),
        "retrieved_topic_slugs": sorted(topic_slug(topic_by_id, topic_id) for topic_id in retrieved_topic_ids),
        "topic_recall": topic_recall,
        "topic_path_hit": topic_path_hit_for_atoms(topic_context, gold_atom_ids, routed_topic_ids),
        "retrieval_topic_recall": retrieval_topic_recall,
        "retrieval_topic_path_hit": topic_path_hit_for_atoms(topic_context, gold_atom_ids, retrieved_topic_ids),
        "gold_atom_assignment_rate": safe_recall(gold_atom_ids, set(topic_context["assignment_by_atom"]).intersection(gold_atom_ids)),
        "criteria_expected_topic_ids": sorted(raw_expected_topic_ids),
        "criteria_expected_topic_resolved_ids": sorted(expected_topic_ids),
        "criteria_expected_topic_unresolved_ids": sorted(raw_expected_topic_ids - resolve_topic_ids(topic_context, raw_expected_topic_ids)),
        "criteria_expected_topic_slugs": sorted(
            topic_slug(topic_by_id, topic_id) for topic_id in expected_topic_ids if topic_id in topic_by_id
        ),
        "criteria_expected_topic_route_hit": (
            None if not expected_topic_ids else bool(expected_topic_ids.intersection(routed_topic_ids))
        ),
        "criteria_expected_topic_retrieval_hit": (
            None if not expected_topic_ids else bool(expected_topic_ids.intersection(retrieved_topic_ids))
        ),
    }


def safe_recall(gold: set[str], retrieved: set[str]) -> float | None:
    if not gold:
        return None
    return len(gold.intersection(retrieved)) / len(gold)


def mean(values: list[float]) -> float:
    return round(float(statistics.mean(values)), 4) if values else 0.0


def main() -> None:
    args = parse_args()
    tasks = load_selfqa(args.selfqa, args.limit)
    service = LEAFService(config_path=args.config, db_path=args.db)
    rows: list[dict[str, Any]] = []
    try:
        topic_context = (
            build_topic_soft_context(service.store, args.corpus_id, args.topic_view_id or None)
            if args.topic_routing_shadow or args.retrieval_mode == "topic_soft"
            else None
        )
        for index, task in enumerate(tasks, start=1):
            question = str(task.get("question") or "").strip()
            criteria = task_criteria_v1(task)
            gold_event_ids, gold_atom_ids = extract_gold_ids(task)
            started = time.perf_counter()
            result = service.search(
                corpus_id=args.corpus_id,
                question=question,
                snapshot_limit=args.snapshot_limit,
                raw_span_limit=args.raw_span_limit,
                trace_memory=args.trace_memory,
            )
            baseline_retrieved_event_ids, baseline_returned_atom_ids, baseline_retrieved_span_ids = extract_retrieved_ids(result)
            baseline_retrieved_event_atom_ids = get_event_atom_ids(service, baseline_retrieved_event_ids)
            routed_topics: list[dict[str, Any]] | None = None
            routed_router_name: str | None = None
            topic_soft_payload: dict[str, Any] | None = None
            if topic_context is not None and (args.topic_routing_shadow or args.retrieval_mode == "topic_soft"):
                topic_shadow_for_routes = build_topic_shadow_row(
                    service,
                    topic_context,
                    question=question,
                    gold_atom_ids=gold_atom_ids,
                    retrieved_atom_ids=baseline_retrieved_event_atom_ids,
                    top_k=args.topic_route_top_k,
                    router=args.topic_router,
                    criteria=criteria,
                )
                routed_topics = list((topic_shadow_for_routes or {}).get("routed_topics") or [])
                routed_router_name = str((topic_shadow_for_routes or {}).get("router") or "")
            if args.retrieval_mode == "topic_soft" and topic_context is not None:
                expansion = topic_soft_expand_events(
                    service.store,
                    topic_context,
                    question=question,
                    routed_topics=routed_topics or [],
                    exclude_event_ids=baseline_retrieved_event_ids,
                    event_limit=args.topic_soft_event_limit,
                    per_topic_atom_limit=args.topic_soft_per_topic_atom_limit,
                    use_stemmed_content_tokens=bool(args.topic_soft_use_stemmed_content_tokens),
                )
                result = merge_topic_soft_evidence(result, expansion)
                topic_soft_payload = expansion
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            retrieved_event_ids, returned_atom_ids, retrieved_span_ids = extract_retrieved_ids(result)
            retrieved_event_atom_ids = get_event_atom_ids(service, retrieved_event_ids)
            retrieved_combined_atom_ids = returned_atom_ids | retrieved_event_atom_ids
            event_recall = safe_recall(gold_event_ids, retrieved_event_ids)
            atom_recall = safe_recall(gold_atom_ids, retrieved_event_atom_ids)
            returned_atom_recall = safe_recall(gold_atom_ids, returned_atom_ids)
            topic_shadow = build_topic_shadow_row(
                service,
                topic_context,
                question=question,
                gold_atom_ids=gold_atom_ids,
                retrieved_atom_ids=retrieved_event_atom_ids,
                top_k=args.topic_route_top_k,
                router=args.topic_router,
                criteria=criteria,
                routed_topics=routed_topics,
                routed_router_name=routed_router_name,
            )
            if topic_shadow is not None and topic_soft_payload is not None:
                topic_shadow["topic_soft"] = {
                    "event_ids": topic_soft_payload.get("event_ids") or [],
                    "atom_ids": topic_soft_payload.get("atom_ids") or [],
                    "candidate_atom_count": topic_soft_payload.get("candidate_atom_count"),
                }
            row = {
                "task_index": index,
                "task_id": task.get("task_id"),
                "question": question,
                "answer": task.get("answer"),
                "tags": task.get("tags") or [],
                "task_type": (task.get("metadata") or {}).get("task_type"),
                "criteria_v1": criteria,
                "criteria_path_metrics": criteria_path_metrics(
                    criteria,
                    retrieved_event_ids=retrieved_event_ids,
                    retrieved_atom_ids=retrieved_event_atom_ids,
                ) if criteria else None,
                "answer_criteria_static_pass": answer_criteria_static_pass(task, criteria) if criteria else None,
                "gold_event_ids": sorted(gold_event_ids),
                "gold_atom_ids": sorted(gold_atom_ids),
                "retrieved_event_ids": sorted(retrieved_event_ids),
                "baseline_retrieved_event_ids": sorted(baseline_retrieved_event_ids),
                "retrieved_atom_ids": sorted(retrieved_event_atom_ids),
                "baseline_retrieved_atom_ids": sorted(baseline_retrieved_event_atom_ids),
                "returned_atom_ids": sorted(returned_atom_ids),
                "retrieved_combined_atom_ids": sorted(retrieved_combined_atom_ids),
                "retrieved_span_ids": sorted(retrieved_span_ids),
                "event_recall": event_recall,
                "atom_recall": atom_recall,
                "returned_atom_recall": returned_atom_recall,
                "event_path_hit": bool(gold_event_ids and gold_event_ids.issubset(retrieved_event_ids)),
                "atom_path_hit": bool(gold_atom_ids and gold_atom_ids.issubset(retrieved_event_atom_ids)),
                "returned_atom_path_hit": bool(gold_atom_ids and gold_atom_ids.issubset(returned_atom_ids)),
                "raw_span_count": len(result.get("raw_spans") or []),
                "baseline_raw_span_count": len(result.get("raw_spans") or []) - len((topic_soft_payload or {}).get("raw_spans") or []),
                "atom_count": len(result.get("atoms") or []),
                "retrieved_event_atom_count": len(retrieved_event_atom_ids),
                "topic_soft_event_count": len((topic_soft_payload or {}).get("event_ids") or []),
                "topic_soft_candidate_atom_count": int((topic_soft_payload or {}).get("candidate_atom_count") or 0),
                "topic_soft_raw_candidate_atom_count": int((topic_soft_payload or {}).get("raw_candidate_atom_count") or 0),
                "topic_soft_filtered_atom_count": int((topic_soft_payload or {}).get("filtered_atom_count") or 0),
                "page_count": len(result.get("pages") or []),
                "elapsed_ms": elapsed_ms,
                "search_total_ms": (result.get("timing") or {}).get("search_total_ms"),
                "agentic_memory": result.get("agentic_memory"),
                "topic_routing_shadow": topic_shadow,
            }
            rows.append(row)
    finally:
        service.close()

    event_recalls = [float(row["event_recall"]) for row in rows if row["event_recall"] is not None]
    atom_recalls = [float(row["atom_recall"]) for row in rows if row["atom_recall"] is not None]
    returned_atom_recalls = [
        float(row["returned_atom_recall"]) for row in rows if row["returned_atom_recall"] is not None
    ]
    topic_shadow_rows = [row["topic_routing_shadow"] for row in rows if row.get("topic_routing_shadow")]
    topic_recalls = [
        float(row["topic_recall"]) for row in topic_shadow_rows if row.get("topic_recall") is not None
    ]
    retrieval_topic_recalls = [
        float(row["retrieval_topic_recall"])
        for row in topic_shadow_rows
        if row.get("retrieval_topic_recall") is not None
    ]
    criteria_rows = [row for row in rows if row.get("criteria_v1")]
    criteria_path_rows = [
        row["criteria_path_metrics"]
        for row in criteria_rows
        if isinstance(row.get("criteria_path_metrics"), dict)
    ]
    criteria_topic_rows = [
        row
        for row in topic_shadow_rows
        if row.get("criteria_expected_topic_route_hit") is not None
    ]
    answer_criteria_rows = [
        row for row in rows if row.get("answer_criteria_static_pass") is not None
    ]
    routed_topic_slugs = sorted({slug for row in topic_shadow_rows for slug in row.get("routed_topic_slugs", [])})
    gold_topic_slugs = sorted({slug for row in topic_shadow_rows for slug in row.get("gold_topic_slugs", [])})
    summary = {
        "corpus_id": args.corpus_id,
        "selfqa": str(args.selfqa),
        "task_count": len(rows),
        "snapshot_limit": args.snapshot_limit,
        "raw_span_limit": args.raw_span_limit,
        "retrieval_mode": args.retrieval_mode,
        "topic_routing_shadow": bool(args.topic_routing_shadow),
        "topic_router": args.topic_router,
        "topic_view_id": topic_context.get("view_id") if topic_context else None,
        "topic_route_top_k": args.topic_route_top_k,
        "topic_soft_event_limit": args.topic_soft_event_limit,
        "topic_soft_per_topic_atom_limit": args.topic_soft_per_topic_atom_limit,
        "topic_soft_use_stemmed_content_tokens": bool(args.topic_soft_use_stemmed_content_tokens),
        "mean_event_recall": mean(event_recalls),
        "mean_atom_recall": mean(atom_recalls),
        "mean_returned_atom_recall": mean(returned_atom_recalls),
        "event_path_hit_rate": mean([1.0 if row["event_path_hit"] else 0.0 for row in rows]),
        "atom_path_hit_rate": mean([1.0 if row["atom_path_hit"] else 0.0 for row in rows]),
        "returned_atom_path_hit_rate": mean([1.0 if row["returned_atom_path_hit"] else 0.0 for row in rows]),
        "criteria_task_count": len(criteria_rows),
        "criteria_evidence_required_event_hit_rate": mean(
            [1.0 if row.get("evidence_required_event_hit") else 0.0 for row in criteria_path_rows]
        ),
        "criteria_evidence_required_atom_hit_rate": mean(
            [1.0 if row.get("evidence_required_atom_hit") else 0.0 for row in criteria_path_rows]
        ),
        "criteria_must_retrieve_any_event_hit_rate": mean(
            [
                1.0 if row.get("must_retrieve_any_event_hit") else 0.0
                for row in criteria_path_rows
                if row.get("must_retrieve_any_event_hit") is not None
            ]
        ),
        "criteria_must_retrieve_all_event_hit_rate": mean(
            [
                1.0 if row.get("must_retrieve_all_event_hit") else 0.0
                for row in criteria_path_rows
                if row.get("must_retrieve_all_event_hit") is not None
            ]
        ),
        "criteria_must_retrieve_any_atom_hit_rate": mean(
            [
                1.0 if row.get("must_retrieve_any_atom_hit") else 0.0
                for row in criteria_path_rows
                if row.get("must_retrieve_any_atom_hit") is not None
            ]
        ),
        "criteria_must_retrieve_all_atom_hit_rate": mean(
            [
                1.0 if row.get("must_retrieve_all_atom_hit") else 0.0
                for row in criteria_path_rows
                if row.get("must_retrieve_all_atom_hit") is not None
            ]
        ),
        "answer_criteria_static_pass_rate": mean(
            [1.0 if row.get("answer_criteria_static_pass") else 0.0 for row in answer_criteria_rows]
        ),
        "topic_shadow_task_count": len(topic_shadow_rows),
        "mean_topic_recall": mean(topic_recalls),
        "topic_path_hit_rate": mean(
            [1.0 if row["topic_path_hit"] else 0.0 for row in topic_shadow_rows]
        ),
        "mean_retrieval_topic_recall": mean(retrieval_topic_recalls),
        "retrieval_topic_path_hit_rate": mean(
            [1.0 if row["retrieval_topic_path_hit"] else 0.0 for row in topic_shadow_rows]
        ),
        "criteria_expected_topic_task_count": len(criteria_topic_rows),
        "criteria_expected_topic_route_hit_rate": mean(
            [1.0 if row.get("criteria_expected_topic_route_hit") else 0.0 for row in criteria_topic_rows]
        ),
        "criteria_expected_topic_retrieval_hit_rate": mean(
            [1.0 if row.get("criteria_expected_topic_retrieval_hit") else 0.0 for row in criteria_topic_rows]
        ),
        "routed_topic_counts": {
            slug: sum(1 for row in topic_shadow_rows if slug in row.get("routed_topic_slugs", []))
            for slug in routed_topic_slugs
        },
        "gold_topic_counts": {
            slug: sum(1 for row in topic_shadow_rows if slug in row.get("gold_topic_slugs", []))
            for slug in gold_topic_slugs
        },
        "avg_elapsed_ms": mean([float(row["elapsed_ms"]) for row in rows]),
        "avg_topic_soft_event_count": mean([float(row["topic_soft_event_count"]) for row in rows]),
        "avg_topic_soft_candidate_atom_count": mean(
            [float(row["topic_soft_candidate_atom_count"]) for row in rows]
        ),
        "avg_topic_soft_raw_candidate_atom_count": mean(
            [float(row["topic_soft_raw_candidate_atom_count"]) for row in rows]
        ),
        "avg_topic_soft_filtered_atom_count": mean(
            [float(row["topic_soft_filtered_atom_count"]) for row in rows]
        ),
        "tag_counts": {
            tag: sum(1 for row in rows if tag in row.get("tags", []))
            for tag in sorted({tag for row in rows for tag in row.get("tags", [])})
        },
        "task_type_counts": {
            task_type: sum(1 for row in rows if row.get("task_type") == task_type)
            for task_type in sorted({str(row.get("task_type")) for row in rows if row.get("task_type")})
        },
    }
    report = {"summary": summary, "rows": rows}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
