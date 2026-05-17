from __future__ import annotations

import json
import re
from typing import Any

from .agentic_memory import route_query_to_topics
from .clients import ChatClient, cosine_similarity, extract_json_object
from .grounding import derive_temporal_grounding, match_temporal_pattern
from .memory_overlay import overlay_event_bonus, overlay_query_features
from .normalize import language_aware_content_terms, language_aware_stemmed_content_terms
from .search import event_to_raw_spans
from .store import SQLiteMemoryStore


def topic_slug(topic_by_id: dict[str, dict[str, Any]], topic_id: str) -> str:
    node = topic_by_id.get(topic_id) or {}
    metadata = dict(node.get("metadata") or {})
    return str(
        metadata.get("topic_slug")
        or metadata.get("evolved_slug")
        or metadata.get("seed_slug")
        or node.get("name")
        or topic_id
    )


def content_tokens(text: str, *, use_stemmed: bool = False) -> set[str]:
    extractor = language_aware_stemmed_content_terms if use_stemmed else language_aware_content_terms
    return set(extractor(text, mode="auto", include_cjk_subgrams=True))


def build_topic_context(store: SQLiteMemoryStore, corpus_id: str, view_id: str | None = None) -> dict[str, Any] | None:
    view = store.get_memory_view(view_id) if view_id else store.get_active_memory_view(corpus_id)
    if view is None or str(view.get("corpus_id") or "") != corpus_id:
        return None
    resolved_view_id = str(view["view_id"])
    topic_by_id = {str(node["topic_id"]): node for node in store.list_topic_nodes(resolved_view_id)}
    assignment_by_atom: dict[str, dict[str, Any]] = {}
    assignments_by_atom: dict[str, list[dict[str, Any]]] = {}
    assignments_by_topic: dict[str, list[dict[str, Any]]] = {}
    atom_ids_by_topic: dict[str, list[str]] = {}
    for assignment in store.list_topic_assignments(resolved_view_id):
        item_kind = str(assignment.get("item_kind") or "")
        if item_kind not in {"atom", "atom_secondary"}:
            continue
        atom_id = str(assignment["item_id"])
        topic_id = str(assignment["topic_id"])
        assignments_by_atom.setdefault(atom_id, []).append(assignment)
        assignments_by_topic.setdefault(topic_id, []).append(assignment)
        if item_kind == "atom" or atom_id not in assignment_by_atom:
            assignment_by_atom[atom_id] = assignment
        topic_atoms = atom_ids_by_topic.setdefault(topic_id, [])
        if atom_id not in topic_atoms:
            topic_atoms.append(atom_id)
    atom_by_id = {str(atom.atom_id): atom for atom in store.list_atoms(corpus_id)}
    event_overlay = dict(dict(view.get("metadata") or {}).get("event_overlay") or {})
    atom_overlay = dict(dict(view.get("metadata") or {}).get("atom_overlay") or {})
    entity_profile_overlay = dict(dict(view.get("metadata") or {}).get("entity_profile_overlay") or {})
    temporal_overlay = dict(dict(view.get("metadata") or {}).get("temporal_overlay") or {})
    return {
        "view_id": resolved_view_id,
        "view_name": view.get("name"),
        "view_metadata": dict(view.get("metadata") or {}),
        "view_metrics": dict(view.get("metrics") or {}),
        "topic_by_id": topic_by_id,
        "assignment_by_atom": assignment_by_atom,
        "assignments_by_atom": assignments_by_atom,
        "assignments_by_topic": assignments_by_topic,
        "atom_ids_by_topic": atom_ids_by_topic,
        "atom_by_id": atom_by_id,
        "event_overlay": event_overlay,
        "atom_overlay": atom_overlay,
        "entity_profile_overlay": entity_profile_overlay,
        "temporal_overlay": temporal_overlay,
    }


def raw_span_content(span: dict[str, Any]) -> str:
    metadata = dict(span.get("metadata") or {})
    return " ".join(
        [
            str(span.get("text") or ""),
            str(span.get("speaker") or ""),
            str(metadata.get("dia_id") or ""),
        ]
    )


def raw_span_event_id(span: dict[str, Any]) -> str:
    metadata = dict(span.get("metadata") or {})
    return (
        str(metadata.get("original_span_id") or "").strip()
        or str(span.get("span_id") or "").split("#", 1)[0].strip()
    )


def evidence_strength(
    question: str,
    evidence: dict[str, Any] | None,
    *,
    use_stemmed_content_tokens: bool = True,
) -> dict[str, Any]:
    query_terms = content_tokens(question, use_stemmed=use_stemmed_content_tokens)
    distinctive_query_terms = {
        term
        for term in query_terms
        if len(term) >= 5 and term not in {"maria", "john", "question", "answer", "thing", "things"}
    }
    raw_spans = list((evidence or {}).get("raw_spans") or [])
    overlaps: list[int] = []
    distinctive_overlaps: list[int] = []
    for span in raw_spans:
        if not isinstance(span, dict):
            continue
        span_terms = content_tokens(raw_span_content(span), use_stemmed=use_stemmed_content_tokens)
        overlaps.append(len(query_terms.intersection(span_terms)))
        distinctive_overlaps.append(len(distinctive_query_terms.intersection(span_terms)))
    overlaps.sort(reverse=True)
    distinctive_overlaps.sort(reverse=True)
    top3 = overlaps[:3]
    return {
        "raw_span_count": len(raw_spans),
        "atom_count": len((evidence or {}).get("atoms") or []),
        "selected_event_count": len((evidence or {}).get("selected_event_ids") or []),
        "max_content_overlap": max(overlaps) if overlaps else 0,
        "max_distinctive_overlap": max(distinctive_overlaps) if distinctive_overlaps else 0,
        "top3_content_overlap_sum": sum(top3),
        "query_term_count": len(query_terms),
        "distinctive_query_term_count": len(distinctive_query_terms),
    }


def evidence_is_strong(
    question: str,
    evidence: dict[str, Any] | None,
    policy: dict[str, Any],
    *,
    use_stemmed_content_tokens: bool = True,
) -> tuple[bool, dict[str, Any]]:
    fallback_policy = dict(policy.get("fallback") or {})
    strength = evidence_strength(question, evidence, use_stemmed_content_tokens=use_stemmed_content_tokens)
    min_raw_spans = max(1, int(fallback_policy.get("primary_strong_min_raw_spans") or 4))
    min_max_overlap = max(1, int(fallback_policy.get("primary_strong_min_content_overlap") or 2))
    min_top3_overlap_sum = max(min_max_overlap, int(fallback_policy.get("primary_strong_min_top3_overlap_sum") or 4))
    min_distinctive_hits = max(0, int(fallback_policy.get("primary_strong_min_distinctive_hits") or 0))
    distinctive_query_term_count = int(strength.get("distinctive_query_term_count") or 0)
    required_distinctive_hits = min(min_distinctive_hits, distinctive_query_term_count)
    distinctive_ok = (
        min_distinctive_hits <= 0
        or distinctive_query_term_count <= 0
        or int(strength.get("max_distinctive_overlap") or 0) >= required_distinctive_hits
    )
    strong = (
        int(strength.get("raw_span_count") or 0) >= min_raw_spans
        and distinctive_ok
        and (
            int(strength.get("max_content_overlap") or 0) >= min_max_overlap
            or int(strength.get("top3_content_overlap_sum") or 0) >= min_top3_overlap_sum
        )
    )
    strength["policy_thresholds"] = {
        "primary_strong_min_raw_spans": min_raw_spans,
        "primary_strong_min_content_overlap": min_max_overlap,
        "primary_strong_min_top3_overlap_sum": min_top3_overlap_sum,
        "primary_strong_min_distinctive_hits": min_distinctive_hits,
        "primary_strong_required_distinctive_hits": required_distinctive_hits,
    }
    return strong, strength


def evidence_has_relative_temporal_match(
    question: str,
    evidence: dict[str, Any] | None,
    *,
    use_stemmed_content_tokens: bool = True,
) -> bool:
    question_text = str(question or "").strip().lower()
    if not question_text.startswith("when"):
        return False
    query_terms = content_tokens(question, use_stemmed=use_stemmed_content_tokens)
    distinctive_query_terms = {
        term
        for term in query_terms
        if len(term) >= 5 and term not in {"maria", "john", "question", "answer", "thing", "things"}
    }
    spans = list((evidence or {}).get("raw_spans") or []) + list((evidence or {}).get("supporting_raw_spans") or [])
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = str(span.get("text") or "").strip()
        if not text:
            continue
        spec, _ = match_temporal_pattern(text)
        if spec is None:
            continue
        grounding = derive_temporal_grounding(text, span.get("timestamp"))
        precision = str(grounding.get("precision") or "")
        if precision not in {"date", "month", "relative", "year", "season"}:
            continue
        span_terms = content_tokens(raw_span_content(span), use_stemmed=use_stemmed_content_tokens)
        if distinctive_query_terms and not distinctive_query_terms.intersection(span_terms):
            continue
        if query_terms and len(query_terms.intersection(span_terms)) < 2:
            continue
        return True
    return False


_GENERIC_QUERY_TERMS = {
    "answer",
    "ask",
    "asked",
    "child",
    "children",
    "des",
    "did",
    "doe",
    "does",
    "doing",
    "done",
    "event",
    "family",
    "friend",
    "friends",
    "having",
    "item",
    "items",
    "john",
    "kind",
    "maria",
    "mention",
    "mentioned",
    "people",
    "person",
    "question",
    "reaction",
    "thing",
    "things",
    "type",
    "year",
}


def _distinctive_terms(terms: set[str]) -> set[str]:
    return {term for term in terms if len(term) >= 4 and term not in _GENERIC_QUERY_TERMS}


def _query_month_numbers(question: str) -> set[int]:
    month_numbers = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    text = str(question or "").lower()
    return {number for name, number in month_numbers.items() if name in text}


def _overlay_month_number(overlay: dict[str, Any]) -> int | None:
    key = str(overlay.get("temporal_key") or "")
    match = re.match(r"^\d{4}-(\d{2})", key)
    if match:
        return int(match.group(1))
    timestamp = str(overlay.get("timestamp") or "").lower()
    for index, month in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    ):
        if month in timestamp:
            return index
    return None


def _same_session_neighbors(
    *,
    temporal_overlay: dict[str, Any],
    event_ids: list[str],
    forward_window: int,
    backward_window: int,
) -> list[tuple[str, str, int]]:
    if not event_ids:
        return []
    session_event_ids = dict(temporal_overlay.get("session_event_ids") or {})
    event_positions: dict[str, tuple[list[str], int]] = {}
    for ids in session_event_ids.values():
        ordered = [str(item) for item in ids]
        for index, event_id in enumerate(ordered):
            event_positions[event_id] = (ordered, index)
    output: list[tuple[str, str, int]] = []
    seen: set[str] = set(event_ids)
    for source_event_id in event_ids:
        located = event_positions.get(str(source_event_id))
        if located is None:
            continue
        ordered, index = located
        for offset in range(1, max(0, int(forward_window)) + 1):
            next_index = index + offset
            if next_index >= len(ordered):
                break
            event_id = ordered[next_index]
            if event_id in seen:
                continue
            seen.add(event_id)
            output.append((event_id, "next_in_session", offset))
        for offset in range(1, max(0, int(backward_window)) + 1):
            prev_index = index - offset
            if prev_index < 0:
                break
            event_id = ordered[prev_index]
            if event_id in seen:
                continue
            seen.add(event_id)
            output.append((event_id, "previous_in_session", offset))
    return output


def _event_positions_by_id(temporal_overlay: dict[str, Any]) -> dict[str, tuple[list[str], int]]:
    session_event_ids = dict(temporal_overlay.get("session_event_ids") or {})
    positions: dict[str, tuple[list[str], int]] = {}
    for ids in session_event_ids.values():
        ordered = [str(item) for item in ids]
        for index, event_id in enumerate(ordered):
            positions[event_id] = (ordered, index)
    return positions


def _is_profile_attribute_query(question: str) -> bool:
    text = str(question or "").strip().lower()
    return bool(
        re.search(r"\b(attributes?|traits?|qualities|personality|describe\s+\w+)\b", text)
        and not re.search(r"\b(say|said|specific|what did|when|where|who)\b", text)
    )


def _is_inference_overlay_sensitive_query(question: str) -> bool:
    features = overlay_query_features(question)
    if not features.get("inference"):
        return False
    if features.get("geo_request") or features.get("childhood") or features.get("reaction"):
        return False
    return True


_OVERLAY_TEMPORAL_SIGNAL_RE = re.compile(
    r"\b(when|date|year|month|week|day|recently|before|after|during|first|last|later|ago|time|current|previous|next)\b"
)
_OVERLAY_WHY_HOW_RE = re.compile(r"^\s*(why|how)\b")
_OVERLAY_WHERE_RE = re.compile(r"\b(where|which country|what country)\b")
_OVERLAY_OPEN_SLOT_RE = re.compile(
    r"\bwhat\s+(?:activity|class|kind|type|group|country|place|location|new activity|sports activity|additional country)\b"
)


def _overlay_selected_sources(payload: dict[str, Any]) -> set[str]:
    sources: set[str] = set()
    for item in payload.get("selected") or []:
        if not isinstance(item, dict):
            continue
        for source in item.get("sources") or []:
            text = str(source or "").strip()
            if text:
                sources.add(text)
    return sources


def _is_overlay_open_slot_query(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not text:
        return False
    if _OVERLAY_WHY_HOW_RE.search(text) or _OVERLAY_WHERE_RE.search(text) or _OVERLAY_OPEN_SLOT_RE.search(text):
        return True
    # Keep the policy usable for Chinese memories without depending on benchmark labels.
    return bool(re.search(r"(为什么|为何|如何|哪里|哪儿|哪个国家|什么活动|什么课程|什么类型|什么地方)", text))


def apply_overlay_runtime_policy(
    payload: dict[str, Any],
    *,
    question: str,
    policy: str = "none",
) -> dict[str, Any]:
    """Gate overlay evidence with query/runtime-visible signals before answer assembly."""

    runtime_policy = str(policy or "none").strip().lower()
    payload = dict(payload or {})
    payload["runtime_policy"] = runtime_policy
    payload["runtime_policy_applied"] = False
    payload["runtime_policy_allowed"] = True
    payload["runtime_policy_reason"] = ""
    payload["runtime_policy_features"] = {}
    if runtime_policy in {"", "none"}:
        return payload
    if runtime_policy != "pre_gate_v2_no_open":
        payload["runtime_policy_allowed"] = True
        payload["runtime_policy_reason"] = "unknown_policy_passthrough"
        return payload

    selected_event_ids = [str(item) for item in (payload.get("event_ids") or []) if str(item).strip()]
    if not selected_event_ids:
        return payload

    sources = _overlay_selected_sources(payload)
    query_features = overlay_query_features(question)
    question_text = str(question or "").strip().lower()
    has_temporal_signal = bool(query_features.get("temporal")) or bool(_OVERLAY_TEMPORAL_SIGNAL_RE.search(question_text))
    has_entity_signal = "entity_profile_overlay" in sources
    has_neighbor_signal = "local_neighbor_overlay" in sources
    open_slot_query = _is_overlay_open_slot_query(question)
    has_memory_signal = has_temporal_signal or has_entity_signal or has_neighbor_signal
    allowed = has_memory_signal and not open_slot_query
    payload["runtime_policy_applied"] = True
    payload["runtime_policy_allowed"] = bool(allowed)
    payload["runtime_policy_features"] = {
        "has_temporal_signal": bool(has_temporal_signal),
        "has_entity_signal": bool(has_entity_signal),
        "has_neighbor_signal": bool(has_neighbor_signal),
        "open_slot_query": bool(open_slot_query),
        "sources": sorted(sources),
    }
    if allowed:
        payload["runtime_policy_reason"] = "accepted_memory_signal"
        return payload

    reason = "open_slot_query" if open_slot_query else "missing_temporal_entity_neighbor_signal"
    payload["runtime_policy_reason"] = reason
    payload["suppressed"] = True
    payload["suppress_reason"] = f"runtime_policy:{reason}"
    payload["suppressed_event_ids"] = list(selected_event_ids)
    payload["event_ids"] = []
    payload["raw_spans"] = []
    payload["selected"] = []
    return payload


def overlay_expand_events(
    store: SQLiteMemoryStore,
    topic_context: dict[str, Any] | None,
    *,
    question: str,
    baseline_evidence: dict[str, Any] | None = None,
    exclude_event_ids: set[str] | None = None,
    event_limit: int = 4,
    use_stemmed_content_tokens: bool = True,
) -> dict[str, Any]:
    if topic_context is None or event_limit <= 0:
        return {
            "event_ids": [],
            "raw_spans": [],
            "candidate_count": 0,
            "selected": [],
            "policy": {},
        }
    exclude_event_ids = exclude_event_ids or set()
    metadata = dict(topic_context.get("view_metadata") or {})
    policy = dict(metadata.get("retrieval_policy_overlay") or {})
    entity_policy = dict(policy.get("entity_profile") or {})
    event_lexical_policy = dict(policy.get("event_lexical") or {})
    local_neighbor_policy = dict(policy.get("local_neighbors") or {})
    temporal_policy = dict(policy.get("temporal") or {})
    fallback_policy = dict(policy.get("fallback") or {})
    primary_strong = False
    primary_strength: dict[str, Any] = {}
    primary_has_relative_temporal = False
    if baseline_evidence is not None and bool(fallback_policy.get("suppress_when_primary_strong", True)):
        primary_strong, primary_strength = evidence_is_strong(
            question,
            baseline_evidence,
            policy,
            use_stemmed_content_tokens=use_stemmed_content_tokens,
        )
        primary_has_relative_temporal = evidence_has_relative_temporal_match(
            question,
            baseline_evidence,
            use_stemmed_content_tokens=use_stemmed_content_tokens,
        )
        if primary_strong and not any(
            bool(route_policy.get("run_when_primary_strong", False))
            for route_policy in (entity_policy, event_lexical_policy, local_neighbor_policy, temporal_policy)
        ):
            return {
                "event_ids": [],
                "raw_spans": [],
                "candidate_count": 0,
                "selected": [],
                "policy": policy,
                "suppressed": True,
                "suppress_reason": "primary_evidence_strong",
                "primary_strength": primary_strength,
            }
    query_features = overlay_query_features(question)
    query_terms = set(query_features.get("terms") or [])
    if not query_terms:
        return {
            "event_ids": [],
            "raw_spans": [],
            "candidate_count": 0,
            "selected": [],
            "policy": policy,
            "primary_strength": primary_strength,
        }

    candidates: dict[str, dict[str, Any]] = {}

    def add_candidate(event_id: str, *, score: float, source: str, reason: dict[str, Any]) -> None:
        event_id = str(event_id or "").strip()
        if not event_id or event_id in exclude_event_ids:
            return
        row = candidates.setdefault(event_id, {"event_id": event_id, "score": 0.0, "sources": [], "reasons": []})
        row["score"] = max(float(row.get("score") or 0.0), float(score))
        if source not in row["sources"]:
            row["sources"].append(source)
        row["reasons"].append(reason)

    event_overlay_by_id: dict[str, dict[str, Any]] = topic_context.get("event_overlay") or {}
    suppress_temporal_relative = primary_has_relative_temporal and bool(
        fallback_policy.get("suppress_expansion_when_primary_has_relative_temporal", True)
    )
    suppress_inference = bool(fallback_policy.get("suppress_inference_queries", False)) and _is_inference_overlay_sensitive_query(question)
    suppress_profile_attribute_all = bool(fallback_policy.get("suppress_profile_attribute_queries", False)) and _is_profile_attribute_query(question)

    if bool(entity_policy.get("enabled", True)) and not suppress_temporal_relative and not suppress_inference and not suppress_profile_attribute_all and (
        not primary_strong or bool(entity_policy.get("run_when_primary_strong", False))
    ):
        entity_overlay: dict[str, dict[str, Any]] = topic_context.get("entity_profile_overlay") or {}
        max_events = max(0, int(entity_policy.get("max_events") or 3))
        min_overlap = max(0, int(entity_policy.get("min_query_term_overlap") or 1))
        require_query_entity_match = bool(entity_policy.get("require_query_entity_match", True))
        min_event_score = float(entity_policy.get("min_event_score") or 0.42)
        for entity, profile in entity_overlay.items():
            entity_key = str(entity).strip().lower()
            aliases = {
                str(alias).strip().lower()
                for alias in (profile.get("aliases") or [entity_key])
                if str(alias).strip()
            }
            alias_terms = set()
            for alias in aliases | {entity_key}:
                alias_terms.update(content_tokens(alias, use_stemmed=use_stemmed_content_tokens))
            entity_hit = bool(query_terms.intersection(alias_terms))
            if require_query_entity_match and not entity_hit:
                continue
            profile_terms = set(profile.get("terms") or [])
            profile_overlap = sorted(query_terms.intersection(profile_terms))
            if not entity_hit and len(profile_overlap) < min_overlap:
                continue
            facets = set((profile.get("facets") or {}).keys())
            facet_hits = sorted(facet for facet in facets if query_features.get(facet))
            event_candidates: list[dict[str, Any]] = []
            for event_id in list(profile.get("event_ids") or []):
                overlay = dict(event_overlay_by_id.get(str(event_id)) or {})
                event_terms = set(overlay.get("terms") or [])
                event_term_hits = sorted(query_terms.intersection(event_terms))
                event_facets = set(overlay.get("facets") or [])
                event_facet_hits = sorted(facet for facet in event_facets if query_features.get(facet))
                utility = dict(overlay.get("utility") or {})
                if not event_term_hits and not event_facet_hits and len(profile_overlap) < min_overlap:
                    continue
                event_score = (
                    (0.25 if entity_hit else 0.0)
                    + min(0.45, 0.06 * len(event_term_hits))
                    + min(0.25, 0.08 * len(event_facet_hits))
                    + min(0.15, 0.12 * float(utility.get("answerability") or 0.0))
                    + min(0.15, 0.03 * len(profile_overlap))
                )
                if event_score < min_event_score:
                    continue
                event_candidates.append(
                    {
                        "event_id": str(event_id),
                        "score": event_score,
                        "term_hits": event_term_hits,
                        "facet_hits": event_facet_hits,
                    }
                )
            event_candidates.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("event_id") or "")))
            for item in event_candidates[:max_events]:
                add_candidate(
                    str(item["event_id"]),
                    score=float(item.get("score") or 0.0),
                    source="entity_profile_overlay",
                    reason={
                        "entity": entity,
                        "entity_hit": entity_hit,
                        "profile_term_overlap": profile_overlap[:8],
                        "event_term_hits": list(item.get("term_hits") or [])[:8],
                        "facet_hits": sorted(set(facet_hits).union(item.get("facet_hits") or [])),
                    },
                )

    suppress_profile_attribute_event = bool(event_lexical_policy.get("suppress_profile_attribute_queries", True)) and _is_profile_attribute_query(question)
    if bool(event_lexical_policy.get("enabled", True)) and not suppress_temporal_relative and not suppress_inference and not suppress_profile_attribute_all and not suppress_profile_attribute_event and (
        not primary_strong or bool(event_lexical_policy.get("run_when_primary_strong", False))
    ):
        min_overlap = max(0, int(event_lexical_policy.get("min_query_term_overlap") or 2))
        min_distinctive_overlap = max(0, int(event_lexical_policy.get("min_distinctive_overlap") or 1))
        min_score = float(event_lexical_policy.get("min_score") or 0.36)
        geo_min_score = float(event_lexical_policy.get("geo_request_min_score") or min_score)
        max_events = max(0, int(event_lexical_policy.get("max_events") or event_limit))
        query_distinctive_terms = _distinctive_terms(query_terms)
        query_months = _query_month_numbers(question)
        lexical_candidates: list[dict[str, Any]] = []
        for event_id, overlay in event_overlay_by_id.items():
            event_terms = set(overlay.get("event_terms") or overlay.get("terms") or [])
            all_terms = set(overlay.get("terms") or event_terms)
            term_hits = sorted(query_terms.intersection(event_terms))
            all_term_hits = sorted(query_terms.intersection(all_terms))
            distinctive_hits = sorted(query_distinctive_terms.intersection(event_terms))
            all_distinctive_hits = sorted(query_distinctive_terms.intersection(all_terms))
            event_facets = set(overlay.get("facets") or [])
            facet_hits = sorted(facet for facet in event_facets if query_features.get(facet))
            utility = dict(overlay.get("utility") or {})
            geo_terms = [str(item).lower() for item in overlay.get("geo_terms") or []]
            geo_hit = bool(query_features.get("geo_request") and geo_terms)
            score = 0.0
            score += min(0.48, 0.08 * len(term_hits))
            score += min(0.36, 0.12 * len(distinctive_hits))
            score += min(0.18, 0.04 * len(all_term_hits))
            score += min(0.18, 0.06 * len(all_distinctive_hits))
            score += min(0.22, 0.075 * len(facet_hits))
            score += min(0.12, 0.08 * float(utility.get("answerability") or 0.0))
            if geo_hit:
                score += 0.18
            if query_features.get("childhood") and {"childhood", "kid"}.intersection(event_terms):
                score += 0.16
            if query_features.get("reaction") and len(distinctive_hits) >= 1:
                score += 0.08
            event_month = _overlay_month_number(dict(overlay))
            if query_months and event_month in query_months:
                score += 0.10
            if not all_term_hits and not facet_hits and not geo_hit:
                continue
            if len(all_term_hits) < min_overlap and not geo_hit:
                continue
            if (
                len(query_distinctive_terms) >= min_distinctive_overlap
                and len(all_distinctive_hits) < min_distinctive_overlap
                and not geo_hit
            ):
                continue
            required_score = geo_min_score if geo_hit else min_score
            if score < required_score:
                continue
            lexical_candidates.append(
                {
                    "event_id": str(event_id),
                    "score": score,
                    "term_hits": term_hits,
                    "all_term_hits": all_term_hits,
                    "distinctive_hits": distinctive_hits,
                    "all_distinctive_hits": all_distinctive_hits,
                    "facet_hits": facet_hits,
                    "geo_terms": geo_terms[:8],
                    "utility_answerability": utility.get("answerability"),
                }
            )
        lexical_candidates.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("event_id") or "")))
        selected_lexical_candidates: list[dict[str, Any]] = []
        seen_geo_terms: set[str] = set()
        if query_features.get("geo_request"):
            for item in lexical_candidates:
                geo_terms = [str(term).lower() for term in item.get("geo_terms") or []]
                if not geo_terms:
                    continue
                if seen_geo_terms.intersection(geo_terms):
                    continue
                selected_lexical_candidates.append(item)
                seen_geo_terms.update(geo_terms)
                if len(selected_lexical_candidates) >= max_events:
                    break
        for item in lexical_candidates:
            if len(selected_lexical_candidates) >= max_events:
                break
            if item in selected_lexical_candidates:
                continue
            selected_lexical_candidates.append(item)
        for item in selected_lexical_candidates[:max_events]:
            add_candidate(
                str(item["event_id"]),
                score=float(item.get("score") or 0.0),
                source="event_lexical_overlay",
                reason={
                    "term_hits": list(item.get("term_hits") or [])[:10],
                    "all_term_hits": list(item.get("all_term_hits") or [])[:10],
                    "distinctive_hits": list(item.get("distinctive_hits") or [])[:10],
                    "all_distinctive_hits": list(item.get("all_distinctive_hits") or [])[:10],
                    "facet_hits": list(item.get("facet_hits") or [])[:8],
                    "geo_terms": list(item.get("geo_terms") or [])[:8],
                    "utility_answerability": item.get("utility_answerability"),
                },
            )

    if bool(temporal_policy.get("enabled", True)) and bool(query_features.get("temporal")) and (
        not primary_strong or bool(temporal_policy.get("run_when_primary_strong", False))
    ):
        temporal_overlay: dict[str, Any] = topic_context.get("temporal_overlay") or {}
        text = str(question or "").lower()
        date_keys = set()
        month_keys = set()
        include_neighbor_events = bool(temporal_policy.get("include_neighbor_events", False))
        previous_by_event = dict(temporal_overlay.get("previous_by_event") or {})
        next_by_event = dict(temporal_overlay.get("next_by_event") or {})

        def add_temporal_event(event_id: str, *, score: float, reason: dict[str, Any]) -> None:
            add_candidate(str(event_id), score=score, source="temporal_overlay", reason=reason)
            if not include_neighbor_events:
                return
            for neighbor_key, neighbor_id in (("previous_event_id", previous_by_event.get(str(event_id))), ("next_event_id", next_by_event.get(str(event_id)))):
                if neighbor_id:
                    neighbor_reason = dict(reason)
                    neighbor_reason[neighbor_key] = str(neighbor_id)
                    add_candidate(
                        str(neighbor_id),
                        score=max(0.1, float(score) * 0.62),
                        source="temporal_overlay_neighbor",
                        reason=neighbor_reason,
                    )

        for match in re.finditer(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", text):
            date_keys.add(match.group(0))
            month_keys.add(match.group(0)[:7])
        for match in re.finditer(r"\b((?:19|20)\d{2})\b", text):
            year = match.group(1)
            for month_key in (temporal_overlay.get("by_month") or {}):
                if str(month_key).startswith(year):
                    month_keys.add(str(month_key))
        max_events = max(0, int(temporal_policy.get("max_events") or 4))
        for date_key in sorted(date_keys):
            for event_id in list((temporal_overlay.get("by_date") or {}).get(date_key) or [])[:max_events]:
                add_temporal_event(
                    str(event_id),
                    score=float(temporal_policy.get("date_match_bonus") or 1.0),
                    reason={"date_key": date_key},
                )
        if not date_keys:
            for month_key in sorted(month_keys):
                for event_id in list((temporal_overlay.get("by_month") or {}).get(month_key) or [])[:max_events]:
                    add_temporal_event(
                        str(event_id),
                        score=float(temporal_policy.get("month_match_bonus") or 0.5),
                        reason={"month_key": month_key},
                    )

    if bool(local_neighbor_policy.get("enabled", True)) and not suppress_temporal_relative and not suppress_inference and not suppress_profile_attribute_all and (
        not primary_strong or bool(local_neighbor_policy.get("run_when_primary_strong", False))
    ):
        temporal_overlay = topic_context.get("temporal_overlay") or {}
        baseline_event_ids = [
            str(event_id)
            for event_id in (baseline_evidence or {}).get("selected_event_ids") or []
            if str(event_id).strip()
        ]
        raw_spans_by_event_id: dict[str, list[dict[str, Any]]] = {}
        event_lookup: dict[str, Any] = {}
        for span in (baseline_evidence or {}).get("raw_spans") or []:
            if not isinstance(span, dict):
                continue
            event_id = raw_span_event_id(span)
            if not event_id:
                continue
            raw_spans_by_event_id.setdefault(event_id, []).append(span)
        for source_event_id in baseline_event_ids:
            event = store.get_event(source_event_id)
            if event is not None:
                event_lookup[source_event_id] = event
        lexical_seed_ids = [
            str(item.get("event_id") or "")
            for item in candidates.values()
            if "event_lexical_overlay" in set(item.get("sources") or [])
        ]
        for source_event_id in lexical_seed_ids:
            if source_event_id not in event_lookup:
                event = store.get_event(source_event_id)
                if event is not None:
                    event_lookup[source_event_id] = event
        max_events = max(0, int(local_neighbor_policy.get("max_events") or 4))
        neighbor_rows = _same_session_neighbors(
            temporal_overlay=temporal_overlay,
            event_ids=baseline_event_ids + lexical_seed_ids,
            forward_window=max(0, int(local_neighbor_policy.get("forward_window") or 0)),
            backward_window=max(0, int(local_neighbor_policy.get("backward_window") or 0)),
        )
        query_distinctive_terms = _distinctive_terms(query_terms)
        require_source_question = bool(local_neighbor_policy.get("require_source_question", True))
        allow_bridge_question = bool(local_neighbor_policy.get("allow_bridge_question", True))
        min_source_overlap = float(local_neighbor_policy.get("min_source_overlap") or 0.0)
        min_source_distinctive_overlap = max(0, int(local_neighbor_policy.get("min_source_distinctive_overlap") or 0))
        positions_by_event_id = _event_positions_by_id(temporal_overlay)
        for event_id, direction, offset in neighbor_rows[: max_events * 3]:
            if direction == "next_in_session":
                source_event_ids = [
                    source_event_id
                    for source_event_id in baseline_event_ids + lexical_seed_ids
                    if event_id in {neighbor_id for neighbor_id, _, _ in _same_session_neighbors(
                        temporal_overlay=temporal_overlay,
                        event_ids=[source_event_id],
                        forward_window=max(0, int(local_neighbor_policy.get("forward_window") or 0)),
                        backward_window=0,
                    )}
                ]
            else:
                source_event_ids = []
            if require_source_question and source_event_ids:
                source_ok = False
                for source_event_id in source_event_ids:
                    source_spans = raw_spans_by_event_id.get(source_event_id) or []
                    source_text = " ".join(str(span.get("text") or "") for span in source_spans)
                    if not source_text and source_event_id in event_lookup:
                        source_text = str(getattr(event_lookup[source_event_id], "text", "") or "")
                    if "?" not in source_text:
                        continue
                    source_terms = content_tokens(source_text, use_stemmed=use_stemmed_content_tokens)
                    source_overlap = len(query_terms.intersection(source_terms)) / max(1, len(query_terms))
                    source_distinctive_overlap = len(query_distinctive_terms.intersection(source_terms))
                    direct_source_ok = (
                        source_overlap >= min_source_overlap
                        and source_distinctive_overlap >= min_source_distinctive_overlap
                    )
                    bridge_source_ok = False
                    if allow_bridge_question and not direct_source_ok:
                        located = positions_by_event_id.get(source_event_id)
                        if located is not None:
                            ordered_ids, source_index = located
                            next_id = ordered_ids[source_index + 1] if source_index + 1 < len(ordered_ids) else ""
                            next_event = store.get_event(next_id) if next_id else None
                            next_text = str(getattr(next_event, "text", "") or "")
                            bridge_terms = content_tokens(" ".join([source_text, next_text]), use_stemmed=use_stemmed_content_tokens)
                            bridge_overlap = len(query_terms.intersection(bridge_terms)) / max(1, len(query_terms))
                            bridge_distinctive_overlap = len(query_distinctive_terms.intersection(bridge_terms))
                            bridge_source_ok = (
                                bridge_overlap >= min_source_overlap
                                and bridge_distinctive_overlap >= min_source_distinctive_overlap
                            )
                    if direct_source_ok or bridge_source_ok:
                        source_ok = True
                        break
                if not source_ok:
                    continue
            elif require_source_question:
                continue
            overlay = dict(event_overlay_by_id.get(str(event_id)) or {})
            event_terms = set(overlay.get("event_terms") or overlay.get("terms") or [])
            term_hits = sorted(query_terms.intersection(event_terms))
            distinctive_hits = sorted(query_distinctive_terms.intersection(event_terms))
            event_facets = set(overlay.get("facets") or [])
            facet_hits = sorted(facet for facet in event_facets if query_features.get(facet))
            score = float(local_neighbor_policy.get("score") or 0.58)
            score += min(0.16, 0.05 * len(term_hits))
            score += min(0.16, 0.08 * len(distinctive_hits))
            score += min(0.12, 0.05 * len(facet_hits))
            score -= max(0, int(offset) - 1) * 0.08
            if score <= 0:
                continue
            add_candidate(
                str(event_id),
                score=score,
                source="local_neighbor_overlay",
                reason={
                    "direction": direction,
                    "offset": offset,
                    "term_hits": term_hits[:10],
                    "distinctive_hits": distinctive_hits[:10],
                    "facet_hits": facet_hits[:8],
                },
            )

    ranked = sorted(candidates.values(), key=lambda item: (-float(item.get("score") or 0.0), str(item.get("event_id") or "")))
    selected = ranked[: max(0, int(event_limit))]
    raw_spans: list[dict[str, Any]] = []
    selected_event_ids: list[str] = []
    for item in selected:
        event_id = str(item.get("event_id") or "")
        event = store.get_event(event_id)
        if event is None:
            continue
        selected_event_ids.append(event_id)
        for span in event_to_raw_spans(event):
            span_metadata = dict(span.get("metadata") or {})
            span_metadata["memory_overlay"] = {
                "view_id": topic_context.get("view_id"),
                "sources": list(item.get("sources") or []),
                "score": round(float(item.get("score") or 0.0), 4),
                "reasons": list(item.get("reasons") or [])[:3],
            }
            span["metadata"] = span_metadata
            raw_spans.append(span)
    return {
        "event_ids": selected_event_ids,
        "raw_spans": raw_spans,
        "candidate_count": len(ranked),
        "selected": selected,
        "policy": policy,
        "suppressed": False,
        "suppress_reason": "",
        "primary_strength": primary_strength,
    }


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
    for topic_id, node in sorted(
        topic_by_id.items(),
        key=lambda item: (int(item[1].get("level") or 0), str(item[1].get("name") or "")),
    ):
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            continue
        allowed_ids.add(topic_id)
        topic_rows.append(
            {
                "topic_id": topic_id,
                "slug": topic_slug(topic_by_id, topic_id),
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
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        routes.append(
            {
                "topic_id": topic_id,
                "slug": topic_slug(topic_by_id, topic_id),
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


def route_topics(
    store: SQLiteMemoryStore,
    topic_context: dict[str, Any],
    *,
    question: str,
    router: str,
    top_k: int,
    llm: ChatClient | None = None,
    query_embedding: list[float] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if router == "llm":
        if llm is None:
            raise RuntimeError("memory_llm is not configured")
        return route_query_to_topics_llm(llm, topic_context, question=question, top_k=top_k), "llm_shadow_v0"
    if router == "evolved_profile_first":
        evolved_routes = route_query_to_topics(
            store,
            view_id=str(topic_context["view_id"]),
            query=question,
            top_k=top_k,
            query_embedding=query_embedding,
            router="profile_hybrid",
            topic_scope="evolved",
        )
        seed_profile_routes = route_query_to_topics(
            store,
            view_id=str(topic_context["view_id"]),
            query=question,
            top_k=top_k,
            query_embedding=query_embedding,
            router="profile_hybrid",
            topic_scope="seed",
        )
        seed_best_score = max((float(route.get("score") or 0.0) for route in seed_profile_routes), default=0.0)
        competitive_evolved_routes = [
            route
            for route in evolved_routes
            if not seed_profile_routes or float(route.get("score") or 0.0) > seed_best_score
        ]
        if competitive_evolved_routes:
            for route in competitive_evolved_routes:
                route["router"] = "evolved_profile_first_v0"
                route["router_stage"] = "evolved"
                route["evolved_confidence_gate"] = "beats_seed_profile"
                route["seed_best_score"] = round(seed_best_score, 4)
            return competitive_evolved_routes, "evolved_profile_first_v0"

        fallback_routes = route_query_to_topics(
            store,
            view_id=str(topic_context["view_id"]),
            query=question,
            top_k=top_k,
            query_embedding=None,
            router="keyword",
            topic_scope="all",
        )
        for route in fallback_routes:
            route["router"] = "evolved_profile_first_keyword_fallback_v0"
            route["router_stage"] = "keyword_fallback"
            route["evolved_confidence_gate"] = "keyword_fallback"
            route["evolved_candidate_count"] = len(evolved_routes)
            route["evolved_max_score"] = round(
                max((float(item.get("score") or 0.0) for item in evolved_routes), default=0.0),
                4,
            )
            route["seed_profile_best_score"] = round(seed_best_score, 4)
        return fallback_routes, "evolved_profile_first_keyword_fallback_v0"
    if router == "overlay_facet_hybrid":
        routes = route_query_to_topics(
            store,
            view_id=str(topic_context["view_id"]),
            query=question,
            top_k=top_k,
            query_embedding=query_embedding,
            router="overlay_facet_hybrid",
        )
        return routes, "overlay_facet_hybrid_v0"
    routes = route_query_to_topics(
        store,
        view_id=str(topic_context["view_id"]),
        query=question,
        top_k=top_k,
        query_embedding=query_embedding,
        router=router,
    )
    if router == "profile_hybrid":
        return routes, "profile_hybrid_v0"
    if router == "profile_quality":
        return routes, "profile_quality_v0"
    return routes, "keyword_shadow_v0"


def _atom_route_score(
    *,
    question_tokens: set[str],
    atom: Any,
    assignment: dict[str, Any],
    route_by_topic: dict[str, dict[str, Any]],
    atom_overlay: dict[str, Any] | None = None,
    query_features: dict[str, Any] | None = None,
    max_overlay_bonus: float = 0.0,
    use_stemmed_content_tokens: bool = False,
) -> float:
    atom_text = " ".join(
        [
            str(atom.content or ""),
            " ".join(str(entity) for entity in (atom.entities or [])),
            " ".join(str(entity) for entity in (atom.canonical_entities or [])),
        ]
    )
    atom_tokens = content_tokens(atom_text, use_stemmed=use_stemmed_content_tokens)
    overlap = len(question_tokens.intersection(atom_tokens))
    topic_id = str(assignment.get("topic_id") or "")
    route = route_by_topic.get(topic_id) or {}
    route_score = float(route.get("score") or route.get("confidence") or 0.0)
    assignment_conf = float(assignment.get("confidence") or 0.0)
    overlay_bonus = 0.0
    overlay = dict(atom_overlay or {})
    if overlay and max_overlay_bonus > 0:
        facets = set(overlay.get("facets") or [])
        utility = dict(overlay.get("utility") or {})
        feature_hits = 0
        for facet in ["temporal", "profile", "relationship", "plan", "media_hobby", "place_travel", "activity"]:
            if (query_features or {}).get(facet) and facet in facets:
                feature_hits += 1
        overlay_bonus += min(max_overlay_bonus * 0.55, 0.12 * feature_hits)
        overlay_bonus += min(max_overlay_bonus * 0.45, float(utility.get("answerability") or 0.0) * 0.18)
    return float(overlap) * 2.0 + route_score + assignment_conf + min(max_overlay_bonus, overlay_bonus)


def _event_semantic_similarity(
    *,
    query_embedding: list[float] | None,
    event: Any | None,
) -> float | None:
    if not query_embedding or event is None or not getattr(event, "embedding", None):
        return None
    return cosine_similarity(query_embedding, event.embedding or [])


def _route_is_fallback(route: dict[str, Any], topic_by_id: dict[str, dict[str, Any]]) -> bool:
    topic_id = str(route.get("topic_id") or "").strip()
    slug = str(route.get("slug") or topic_slug(topic_by_id, topic_id)).strip().lower()
    matched_keywords = [str(item).strip() for item in (route.get("matched_keywords") or []) if str(item).strip()]
    try:
        score = float(route.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return slug == "misc" or (score <= 0.0 and not matched_keywords)


def _assignment_item_kind(assignment: dict[str, Any] | None) -> str:
    return str((assignment or {}).get("item_kind") or "").strip()


def _topic_has_primary_assignment(topic_id: str, assignments_by_topic: dict[str, list[dict[str, Any]]]) -> bool:
    return any(_assignment_item_kind(assignment) == "atom" for assignment in assignments_by_topic.get(topic_id, []))


def _topic_has_secondary_assignment(topic_id: str, assignments_by_topic: dict[str, list[dict[str, Any]]]) -> bool:
    return any(_assignment_item_kind(assignment) == "atom_secondary" for assignment in assignments_by_topic.get(topic_id, []))


def _topic_is_secondary_only(topic_id: str, assignments_by_topic: dict[str, list[dict[str, Any]]]) -> bool:
    return _topic_has_secondary_assignment(topic_id, assignments_by_topic) and not _topic_has_primary_assignment(
        topic_id,
        assignments_by_topic,
    )


def _route_keyword_overlap(
    route: dict[str, Any],
    topic: dict[str, Any],
    query_content_tokens: set[str],
    *,
    use_stemmed_content_tokens: bool = False,
) -> int:
    matched_keywords = [str(item) for item in (route.get("matched_keywords") or []) if str(item).strip()]
    matched_tokens = content_tokens(" ".join(matched_keywords), use_stemmed=use_stemmed_content_tokens)
    topic_keywords = [str(item) for item in (topic.get("keywords") or []) if str(item).strip()]
    topic_tokens = content_tokens(" ".join(topic_keywords), use_stemmed=use_stemmed_content_tokens)
    route_tokens = matched_tokens or topic_tokens
    if not query_content_tokens:
        return 0
    return len(query_content_tokens.intersection(route_tokens))


def _normalize_slug_set(values: list[Any] | tuple[Any, ...] | set[Any] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [item for item in values.split(",") if item.strip()]
    return {str(value).strip().lower() for value in values if str(value).strip()}


def topic_policy_from_context(topic_context: dict[str, Any] | None) -> dict[str, Any]:
    if topic_context is None:
        return {}
    metadata = dict(topic_context.get("view_metadata") or {})
    metrics = dict(topic_context.get("view_metrics") or {})
    policy = metadata.get("topic_soft_policy") or metrics.get("topic_soft_policy") or {}
    return dict(policy) if isinstance(policy, dict) else {}


def filter_routes_by_topic_policy(
    routes: list[dict[str, Any]],
    topic_context: dict[str, Any] | None,
    *,
    deny_topic_slugs: set[str] | None = None,
    allow_topic_slugs: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = topic_policy_from_context(topic_context)
    denied = _normalize_slug_set(policy.get("deny_topic_slugs")) | _normalize_slug_set(deny_topic_slugs)
    allowed = _normalize_slug_set(policy.get("allow_topic_slugs")) | _normalize_slug_set(allow_topic_slugs)
    if not denied and not allowed:
        return routes, []
    topic_by_id = (topic_context or {}).get("topic_by_id") or {}
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for route in routes:
        topic_id = str(route.get("topic_id") or "").strip()
        slug = str(route.get("slug") or topic_slug(topic_by_id, topic_id)).strip().lower()
        if allowed and slug not in allowed:
            suppressed.append({**route, "topic_policy_reason": "not_allowed_topic_slug"})
            continue
        if denied and slug in denied:
            suppressed.append({**route, "topic_policy_reason": "denied_topic_slug"})
            continue
        kept.append(route)
    return kept, suppressed


def topic_soft_expand_events(
    store: SQLiteMemoryStore,
    topic_context: dict[str, Any] | None,
    *,
    question: str,
    routed_topics: list[dict[str, Any]],
    exclude_event_ids: set[str] | None = None,
    event_limit: int = 4,
    per_topic_atom_limit: int = 16,
    min_content_overlap: int = 0,
    allow_fallback_topic: bool = True,
    score_with_content_tokens: bool = False,
    deny_topic_slugs: set[str] | None = None,
    allow_topic_slugs: set[str] | None = None,
    secondary_policy: str = "all",
    secondary_min_content_overlap: int = 2,
    secondary_min_route_keyword_overlap: int = 1,
    query_embedding: list[float] | None = None,
    min_event_embedding_similarity: float = 0.0,
    use_stemmed_content_tokens: bool = False,
) -> dict[str, Any]:
    secondary_policy = str(secondary_policy or "all").strip()
    if secondary_policy not in {"all", "primary_only", "strict_text_v0"}:
        raise ValueError(f"Unknown topic-soft secondary policy: {secondary_policy}")
    if topic_context is None or event_limit <= 0:
        return {
            "event_ids": [],
            "atom_ids": [],
            "raw_spans": [],
            "routes": routed_topics,
            "active_routes": [],
            "suppressed_routes": [],
            "candidate_atom_count": 0,
            "raw_candidate_atom_count": 0,
            "filtered_atom_count": 0,
            "skipped_fallback_route_count": 0,
            "skipped_low_overlap_count": 0,
            "secondary_policy": secondary_policy,
            "secondary_candidate_atom_count": 0,
            "secondary_selected_atom_count": 0,
            "skipped_secondary_route_count": 0,
            "skipped_secondary_policy_count": 0,
            "skipped_secondary_low_overlap_count": 0,
            "semantic_gate_enabled": False,
            "semantic_gate_min_similarity": 0.0,
            "semantic_gate_missing_embedding_count": 0,
            "semantic_gate_skipped_event_count": 0,
            "semantic_gate_max_similarity": None,
            "use_stemmed_content_tokens": bool(use_stemmed_content_tokens),
        }
    exclude_event_ids = exclude_event_ids or set()
    atom_by_id: dict[str, Any] = topic_context["atom_by_id"]
    assignment_by_atom: dict[str, dict[str, Any]] = topic_context["assignment_by_atom"]
    assignments_by_atom: dict[str, list[dict[str, Any]]] = topic_context.get("assignments_by_atom") or {}
    assignments_by_topic: dict[str, list[dict[str, Any]]] = topic_context.get("assignments_by_topic") or {}
    atom_ids_by_topic: dict[str, list[str]] = topic_context["atom_ids_by_topic"]
    topic_by_id: dict[str, dict[str, Any]] = topic_context["topic_by_id"]
    event_overlay_by_id: dict[str, dict[str, Any]] = topic_context.get("event_overlay") or {}
    atom_overlay_by_id: dict[str, dict[str, Any]] = topic_context.get("atom_overlay") or {}
    overlay_policy = dict(dict(topic_context.get("view_metadata") or {}).get("overlay_policy") or {})
    overlay_topic_soft_enabled = bool(overlay_policy.get("topic_soft_utility_enabled"))
    overlay_max_atom_bonus = float(overlay_policy.get("max_topic_soft_atom_bonus") or 0.0)
    query_features = overlay_query_features(question) if overlay_topic_soft_enabled else {}
    active_routes: list[dict[str, Any]] = []
    policy_routes, suppressed_routes = filter_routes_by_topic_policy(
        routed_topics,
        topic_context,
        deny_topic_slugs=deny_topic_slugs,
        allow_topic_slugs=allow_topic_slugs,
    )
    query_content_tokens = content_tokens(question, use_stemmed=use_stemmed_content_tokens)
    question_tokens = query_content_tokens
    skipped_fallback_route_count = 0
    skipped_secondary_route_count = 0
    for route in policy_routes:
        topic_id = str(route.get("topic_id") or "").strip()
        if not topic_id:
            continue
        if not allow_fallback_topic and _route_is_fallback(route, topic_by_id):
            skipped_fallback_route_count += 1
            continue
        topic = topic_by_id.get(topic_id) or {}
        if _topic_is_secondary_only(topic_id, assignments_by_topic):
            if secondary_policy == "primary_only":
                skipped_secondary_route_count += 1
                suppressed_routes.append({**route, "topic_policy_reason": "secondary_primary_only"})
                continue
            if secondary_policy == "strict_text_v0":
                route_overlap = _route_keyword_overlap(
                    route,
                    topic,
                    query_content_tokens,
                    use_stemmed_content_tokens=use_stemmed_content_tokens,
                )
                if route_overlap < max(0, int(secondary_min_route_keyword_overlap)):
                    skipped_secondary_route_count += 1
                    suppressed_routes.append(
                        {
                            **route,
                            "topic_policy_reason": "secondary_low_route_keyword_overlap",
                            "secondary_route_keyword_overlap": route_overlap,
                        }
                    )
                    continue
        active_routes.append(route)
    route_by_topic = {str(route["topic_id"]): route for route in active_routes if str(route.get("topic_id") or "").strip()}

    scored_atoms: list[dict[str, Any]] = []
    raw_candidate_atom_count = 0
    skipped_low_overlap_count = 0
    secondary_candidate_atom_count = 0
    skipped_secondary_policy_count = 0
    skipped_secondary_low_overlap_count = 0
    semantic_similarity_enabled = bool(query_embedding)
    semantic_gate_enabled = semantic_similarity_enabled and float(min_event_embedding_similarity or 0.0) > 0.0
    semantic_gate_min_similarity = float(min_event_embedding_similarity or 0.0) if semantic_gate_enabled else 0.0
    semantic_gate_missing_embedding_count = 0
    semantic_gate_skipped_event_count = 0
    semantic_gate_similarity_by_event: dict[str, float | None] = {}
    for topic_id in route_by_topic:
        topic_atoms: list[dict[str, Any]] = []
        for atom_id in atom_ids_by_topic.get(topic_id, []):
            atom = atom_by_id.get(atom_id)
            assignment = next(
                (
                    item
                    for item in assignments_by_atom.get(atom_id, [])
                    if str(item.get("topic_id") or "") == topic_id
                ),
                assignment_by_atom.get(atom_id),
            )
            if atom is None or assignment is None:
                continue
            assignment_kind = _assignment_item_kind(assignment)
            is_secondary_assignment = assignment_kind == "atom_secondary"
            raw_candidate_atom_count += 1
            if is_secondary_assignment:
                secondary_candidate_atom_count += 1
                if secondary_policy == "primary_only":
                    skipped_secondary_policy_count += 1
                    continue
            required_overlap = max(0, int(min_content_overlap))
            if is_secondary_assignment and secondary_policy == "strict_text_v0":
                required_overlap = max(required_overlap, max(0, int(secondary_min_content_overlap)))
            if required_overlap > 0 and query_content_tokens:
                atom_text_for_overlap = " ".join(
                    [
                        str(atom.content or ""),
                        " ".join(str(entity) for entity in (atom.entities or [])),
                        " ".join(str(entity) for entity in (atom.canonical_entities or [])),
                    ]
                )
                overlap = len(
                    query_content_tokens.intersection(
                        content_tokens(atom_text_for_overlap, use_stemmed=use_stemmed_content_tokens)
                    )
                )
                if overlap < required_overlap:
                    if is_secondary_assignment:
                        skipped_secondary_low_overlap_count += 1
                    else:
                        skipped_low_overlap_count += 1
                    continue
            elif required_overlap > 0 and is_secondary_assignment and secondary_policy == "strict_text_v0":
                overlap = 0
                skipped_secondary_low_overlap_count += 1
                continue
            else:
                overlap = 0
            score = _atom_route_score(
                question_tokens=question_tokens,
                atom=atom,
                assignment=assignment,
                route_by_topic=route_by_topic,
                atom_overlay=atom_overlay_by_id.get(atom_id),
                query_features=query_features,
                max_overlay_bonus=overlay_max_atom_bonus,
                use_stemmed_content_tokens=use_stemmed_content_tokens,
            )
            overlay_atom = dict(atom_overlay_by_id.get(atom_id) or {})
            event_overlay_bonus = 0.0
            event_overlay_match: dict[str, Any] = {}
            if overlay_topic_soft_enabled:
                event_overlay_bonus, event_overlay_match = overlay_event_bonus(
                    question=question,
                    overlay=dict(event_overlay_by_id.get(str(atom.event_id)) or {}),
                    max_bonus=0.18,
                )
                score += event_overlay_bonus
            topic_atoms.append(
                {
                    "score": score,
                    "atom_id": atom_id,
                    "event_id": str(atom.event_id),
                    "topic_id": topic_id,
                    "content_overlap": overlap,
                    "assignment_kind": assignment_kind,
                    "secondary_assignment": is_secondary_assignment,
                    "overlay_facets": list(overlay_atom.get("facets") or []),
                    "overlay_utility": dict(overlay_atom.get("utility") or {}),
                    "overlay_event_bonus": round(float(event_overlay_bonus), 4),
                    "overlay_match": event_overlay_match,
                }
            )
        topic_atoms.sort(key=lambda item: (-float(item["score"]), str(item["atom_id"])))
        scored_atoms.extend(topic_atoms[: max(1, per_topic_atom_limit)])

    scored_atoms.sort(key=lambda item: (-float(item["score"]), str(item["atom_id"])))
    selected_event_ids: list[str] = []
    selected_atom_ids: list[str] = []
    selected_atom_metadata: list[dict[str, Any]] = []
    seen_events: set[str] = set(exclude_event_ids)
    selected_by_event: dict[str, dict[str, Any]] = {}
    for item in scored_atoms:
        score = float(item["score"])
        atom_id = str(item["atom_id"])
        event_id = str(item["event_id"])
        if event_id in seen_events:
            continue
        event = store.get_event(event_id) if semantic_similarity_enabled else None
        semantic_similarity = _event_semantic_similarity(query_embedding=query_embedding, event=event)
        if semantic_gate_enabled:
            semantic_gate_similarity_by_event[event_id] = semantic_similarity
            if semantic_similarity is None:
                semantic_gate_missing_embedding_count += 1
                continue
            if semantic_similarity < semantic_gate_min_similarity:
                semantic_gate_skipped_event_count += 1
                continue
        elif semantic_similarity_enabled:
            semantic_gate_similarity_by_event[event_id] = semantic_similarity
        if score <= 0 and len(selected_event_ids) >= 1:
            continue
        seen_events.add(event_id)
        selected_event_ids.append(event_id)
        selected_atom_ids.append(atom_id)
        selected_by_event[event_id] = {**item, "semantic_similarity": semantic_similarity}
        selected_atom_metadata.append(
            {
                "atom_id": atom_id,
                "event_id": event_id,
                "topic_id": str(item.get("topic_id") or ""),
                "score": round(score, 4),
                "content_overlap": int(item.get("content_overlap") or 0),
                "assignment_kind": str(item.get("assignment_kind") or ""),
                "secondary_assignment": bool(item.get("secondary_assignment")),
                "semantic_similarity": (
                    round(float(semantic_similarity), 4)
                    if semantic_similarity is not None
                    else None
                ),
                "overlay_facets": list(item.get("overlay_facets") or []),
                "overlay_utility": dict(item.get("overlay_utility") or {}),
                "overlay_event_bonus": round(float(item.get("overlay_event_bonus") or 0.0), 4),
                "overlay_match": dict(item.get("overlay_match") or {}),
            }
        )
        if len(selected_event_ids) >= event_limit:
            break

    raw_spans: list[dict[str, Any]] = []
    for event_id in selected_event_ids:
        event = store.get_event(event_id)
        if event is None:
            continue
        for span in event_to_raw_spans(event):
            metadata = dict(span.get("metadata") or {})
            selected_item = selected_by_event.get(event_id) or {}
            metadata["topic_soft"] = {
                "view_id": topic_context["view_id"],
                "router": next((route.get("router") for route in active_routes if route.get("topic_id")), "topic_soft"),
                "selected_atom_id": str(selected_item.get("atom_id") or ""),
                "selected_topic_id": str(selected_item.get("topic_id") or ""),
                "score": round(float(selected_item.get("score") or 0.0), 4),
                "content_overlap": int(selected_item.get("content_overlap") or 0),
                "assignment_kind": str(selected_item.get("assignment_kind") or ""),
                "secondary_assignment": bool(selected_item.get("secondary_assignment")),
                "semantic_similarity": (
                    round(float(selected_item.get("semantic_similarity")), 4)
                    if selected_item.get("semantic_similarity") is not None
                    else None
                ),
                "overlay_facets": list(selected_item.get("overlay_facets") or []),
                "overlay_utility": dict(selected_item.get("overlay_utility") or {}),
                "overlay_match": dict(selected_item.get("overlay_match") or {}),
            }
            span["metadata"] = metadata
            raw_spans.append(span)

    semantic_similarity_values = [
        float(value)
        for value in semantic_gate_similarity_by_event.values()
        if value is not None
    ]

    return {
        "event_ids": selected_event_ids,
        "atom_ids": selected_atom_ids,
        "selected_atoms": selected_atom_metadata,
        "raw_spans": raw_spans,
        "routes": routed_topics,
        "active_routes": active_routes,
        "suppressed_routes": suppressed_routes,
        "candidate_atom_count": len(scored_atoms),
        "raw_candidate_atom_count": raw_candidate_atom_count,
        "filtered_atom_count": raw_candidate_atom_count - len(scored_atoms),
        "skipped_fallback_route_count": skipped_fallback_route_count,
        "skipped_low_overlap_count": skipped_low_overlap_count,
        "secondary_policy": secondary_policy,
        "secondary_min_content_overlap": max(0, int(secondary_min_content_overlap)),
        "secondary_min_route_keyword_overlap": max(0, int(secondary_min_route_keyword_overlap)),
        "secondary_candidate_atom_count": secondary_candidate_atom_count,
        "secondary_selected_atom_count": len(
            [item for item in selected_atom_metadata if bool(item.get("secondary_assignment"))]
        ),
        "skipped_secondary_route_count": skipped_secondary_route_count,
        "skipped_secondary_policy_count": skipped_secondary_policy_count,
        "skipped_secondary_low_overlap_count": skipped_secondary_low_overlap_count,
        "semantic_gate_enabled": semantic_gate_enabled,
        "semantic_similarity_enabled": semantic_similarity_enabled,
        "semantic_gate_min_similarity": round(float(semantic_gate_min_similarity), 4),
        "semantic_gate_missing_embedding_count": semantic_gate_missing_embedding_count,
        "semantic_gate_skipped_event_count": semantic_gate_skipped_event_count,
        "semantic_gate_max_similarity": (
            round(max(semantic_similarity_values), 4)
            if semantic_similarity_values
            else None
        ),
        "use_stemmed_content_tokens": bool(use_stemmed_content_tokens),
    }


def apply_topic_soft_policy(
    expansion: dict[str, Any],
    *,
    policy: str = "always",
    min_selected_overlap: int = 2,
    max_candidate_atom_count: int = 20,
    suppress_for_temporal_query: bool = False,
    min_selected_semantic_similarity: float = 0.0,
    suppress_multi_route: bool = False,
) -> dict[str, Any]:
    payload = dict(expansion)
    selected_atoms = list(payload.get("selected_atoms") or [])
    selected_overlaps = [
        int(item.get("content_overlap") or 0)
        for item in selected_atoms
        if isinstance(item, dict)
    ]
    max_selected_overlap = max(selected_overlaps) if selected_overlaps else 0
    candidate_atom_count = int(payload.get("candidate_atom_count") or 0)
    event_ids = [str(item) for item in (payload.get("event_ids") or []) if str(item).strip()]
    atom_ids = [str(item) for item in (payload.get("atom_ids") or []) if str(item).strip()]
    selected_semantic_similarities = [
        float(item.get("semantic_similarity"))
        for item in selected_atoms
        if isinstance(item, dict) and item.get("semantic_similarity") is not None
    ]
    max_selected_semantic_similarity = (
        max(selected_semantic_similarities)
        if selected_semantic_similarities
        else None
    )
    active_route_count = len(payload.get("active_routes") or [])

    payload["policy"] = policy
    payload["policy_applied"] = bool(event_ids)
    payload["policy_reason"] = "selected" if event_ids else "no_topic_event"
    payload["policy_max_selected_content_overlap"] = max_selected_overlap
    payload["policy_max_candidate_atom_count"] = max_candidate_atom_count
    payload["policy_min_selected_overlap"] = min_selected_overlap
    payload["policy_suppress_for_temporal_query"] = bool(suppress_for_temporal_query)
    payload["policy_min_selected_semantic_similarity"] = round(
        float(min_selected_semantic_similarity or 0.0),
        4,
    )
    payload["policy_max_selected_semantic_similarity"] = (
        round(max_selected_semantic_similarity, 4)
        if max_selected_semantic_similarity is not None
        else None
    )
    payload["policy_suppress_multi_route"] = bool(suppress_multi_route)
    payload["suppressed_event_ids"] = []
    payload["suppressed_atom_ids"] = []
    payload["suppressed_raw_span_count"] = 0

    if policy in {"", "always"}:
        return payload
    if policy not in {
        "selected_overlap_and_candidate_count_v0",
        "text_temporal_suppressed_v0",
        "route_uncertainty_semantic_v0",
    }:
        raise ValueError(f"Unknown topic-soft policy: {policy}")

    reason = "selected"
    if not event_ids:
        reason = "no_topic_event"
    elif policy in {"text_temporal_suppressed_v0", "route_uncertainty_semantic_v0"} and suppress_for_temporal_query:
        reason = "temporal_query_text_suppressed"
    elif (
        policy == "route_uncertainty_semantic_v0"
        and bool(suppress_multi_route)
        and active_route_count > 1
    ):
        reason = "multi_route_uncertain"
    elif (
        policy == "route_uncertainty_semantic_v0"
        and float(min_selected_semantic_similarity or 0.0) > 0.0
        and max_selected_semantic_similarity is not None
        and max_selected_semantic_similarity < float(min_selected_semantic_similarity)
    ):
        reason = "low_selected_semantic_similarity"
    elif max_selected_overlap < max(0, int(min_selected_overlap)):
        reason = "low_selected_content_overlap"
    elif max_candidate_atom_count > 0 and candidate_atom_count > max_candidate_atom_count:
        reason = "too_many_candidate_atoms"

    applied = reason == "selected"
    payload["policy_applied"] = applied
    payload["policy_reason"] = reason
    if not applied:
        payload["suppressed_event_ids"] = event_ids
        payload["suppressed_atom_ids"] = atom_ids
        payload["suppressed_raw_span_count"] = len(payload.get("raw_spans") or [])
        payload["event_ids"] = []
        payload["atom_ids"] = []
        payload["raw_spans"] = []
    return payload


def merge_topic_soft_evidence(
    evidence: dict[str, Any],
    expansion: dict[str, Any],
) -> dict[str, Any]:
    if not expansion.get("event_ids"):
        return {**evidence, "topic_soft": expansion}
    merged = dict(evidence)
    selected_event_ids = [str(value) for value in (evidence.get("selected_event_ids") or []) if str(value).strip()]
    for event_id in expansion.get("event_ids") or []:
        if event_id not in selected_event_ids:
            selected_event_ids.append(event_id)
    merged["selected_event_ids"] = selected_event_ids

    seen_span_ids = {str(span.get("span_id") or "") for span in (evidence.get("raw_spans") or [])}
    raw_spans = list(evidence.get("raw_spans") or [])
    for span in expansion.get("raw_spans") or []:
        span_id = str(span.get("span_id") or "")
        if span_id and span_id in seen_span_ids:
            continue
        seen_span_ids.add(span_id)
        raw_spans.append(span)
    merged["raw_spans"] = raw_spans
    merged["topic_soft"] = expansion
    return merged
