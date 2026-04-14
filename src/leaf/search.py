from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any

from .clients import EmbeddingClient, cosine_similarity
from .extract import extract_entities, extract_semantic_references
from .grounding import (
    is_inference_query,
    is_temporal_query,
    parse_anchor_datetime,
    query_tokens as make_query_tokens,
    split_text_fragments,
)
from .store import SQLiteMemoryStore

QUERY_STOPWORDS = {
    "a",
    "an",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "would",
    "should",
    "has",
    "have",
    "had",
    "get",
    "got",
    "take",
    "took",
    "went",
    "go",
    "gone",
    "kind",
    "type",
    "types",
    "sort",
    "major",
    "main",
    "just",
    "both",
    "common",
    "many",
}

QUERY_ALIASES = {
    "roadtrip": {"roadtrip", "road trip", "trip", "travel"},
    "roadtrips": {"roadtrip", "road trip", "trip", "trips", "travel"},
    "accomplish": {
        "accomplish",
        "accomplished",
        "achievement",
        "finish",
        "finished",
        "complete",
        "completed",
        "print",
        "printed",
    },
    "accomplished": {
        "accomplish",
        "accomplished",
        "achievement",
        "finish",
        "finished",
        "complete",
        "completed",
        "print",
        "printed",
    },
    "achievement": {
        "achievement",
        "accomplish",
        "accomplished",
        "finish",
        "finished",
        "complete",
        "completed",
        "print",
        "printed",
    },
}

SEMANTIC_HINTS = {
    "destress": ["stress relief", "relax", "unwind", "escape"],
    "de stress": ["stress relief", "relax", "unwind", "escape"],
    "de-stress": ["stress relief", "relax", "unwind", "escape"],
    "martial arts": ["kickboxing", "taekwondo", "karate", "combat sport"],
    "paint": ["painting", "artwork", "canvas", "picture"],
    "education": ["study", "career", "field", "training"],
    "job": ["work", "career", "employment"],
}

MONTH_NUMBERS = {
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

LIST_QUERY_PATTERNS = (
    "what topics",
    "which topics",
    "what books",
    "which books",
    "what suggestions",
    "what tips",
    "what places",
    "what painters",
    "what bands",
    "what projects",
    "what sports",
    "what outdoor sports",
    "what problems",
    "what issues",
    "what are some",
    "what were they",
)

SINGLE_FACT_PATTERNS = (
    "what was its name",
    "what is its name",
    "which one is it",
    "what dish",
    "what movie",
    "what exhibition",
    "what attraction",
    "what city",
    "what book",
    "what was it",
    "where did i plan to go",
    "where did i go",
    "where did i plan",
    "what is my favorite",
    "what's my favorite",
)

YES_NO_PREFIXES = (
    "do ",
    "does ",
    "did ",
    "is ",
    "are ",
    "was ",
    "were ",
    "can ",
    "could ",
    "would ",
    "should ",
    "have ",
    "has ",
    "had ",
)

OPEN_DOMAIN_SUPPORT_HINTS = [
    {
        "patterns": [" likely ", " likely to ", " would ", " be considered ", " considered "],
        "hints": ["values", "beliefs", "preferences", "goals", "supportive", "interested in"],
    },
    {
        "patterns": [" personality ", " traits ", " attributes ", " describe "],
        "hints": ["described as", "personality", "supportive", "thoughtful", "driven", "authentic"],
    },
    {
        "patterns": [" future job ", " pursue in the future ", " career ", " field ", " education "],
        "hints": ["career", "job", "study", "training", "wants to", "goal", "interested in"],
    },
    {
        "patterns": [" political leaning ", " religious ", " ally ", " patriotic "],
        "hints": ["beliefs", "values", "support", "community", "identity"],
    },
    {
        "patterns": [" move to ", " move back ", " another country ", " home country "],
        "hints": ["plans", "goals", "family", "adoption", "future", "wants to stay"],
    },
]


def _ordered_unique_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _entity_overlap_score(query_entities: list[str], candidate_values: list[str]) -> float:
    normalized_candidates = [str(value or "").strip().lower() for value in candidate_values if str(value or "").strip()]
    if not query_entities or not normalized_candidates:
        return 0.0
    best = 0.0
    for query_entity in query_entities:
        query_text = str(query_entity or "").strip().lower()
        if not query_text:
            continue
        for candidate in normalized_candidates:
            if not candidate:
                continue
            if query_text == candidate:
                best = max(best, 1.0)
                continue
            if query_text in candidate or candidate in query_text:
                best = max(best, 0.92)
                continue
            query_tokens = set(query_text.split())
            candidate_tokens = set(candidate.split())
            if query_tokens and candidate_tokens:
                overlap = len(query_tokens.intersection(candidate_tokens)) / max(1, len(query_tokens.union(candidate_tokens)))
                best = max(best, overlap)
    return best


def _route_query_intents(query: str) -> list[str]:
    lowered = str(query or "").lower()
    padded = f" {lowered.strip()} "
    intents: list[str] = []
    if any(token in lowered for token in ["why", "because", "reason"]):
        intents.append("provenance")
    if any(token in lowered for token in ["latest", "current", "changed", "update", "newest"]):
        intents.append("temporal_update")
    if any(token in lowered for token in ["prefer", "like", "favorite"]):
        intents.append("preference")
    if is_inference_query(query):
        intents.append("persona_inference")
    if any(pattern in padded for spec in OPEN_DOMAIN_SUPPORT_HINTS for pattern in spec["patterns"]):
        intents.append("open_domain_inference")
    if any(token in lowered for token in ["who", "what", "which", "when"]):
        intents.append("factual_recall")
    if not intents:
        intents.append("task_continuity")
    return intents


def _support_pattern_hints(query: str, intents: list[str]) -> list[str]:
    padded = f" {str(query or '').strip().lower()} "
    hints: list[str] = []
    if "open_domain_inference" not in intents and "persona_inference" not in intents:
        return hints
    for spec in OPEN_DOMAIN_SUPPORT_HINTS:
        if any(pattern in padded for pattern in spec["patterns"]):
            hints.extend(spec["hints"])
    if "persona_inference" in intents:
        hints.extend(["values", "goals", "beliefs", "personality", "supportive"])
    return _ordered_unique_strings(hints)[:10]


def _semantic_query_hints(query: str) -> list[str]:
    lowered = str(query or "").lower()
    hints: list[str] = []
    for phrase, expansions in SEMANTIC_HINTS.items():
        if phrase in lowered:
            hints.extend(expansions)
    return _ordered_unique_strings(hints)[:8]


def _requires_entity_coverage(query: str, query_entities: list[str]) -> bool:
    lowered = str(query or "").lower()
    return len(query_entities) >= 2 and any(token in lowered for token in ["both", "together", " and "])


def _requires_session_diversity(query: str) -> bool:
    lowered = str(query or "").lower()
    if any(token in lowered for token in ["both", "together", "history", "ever", "used to"]):
        return True
    return bool(re.search(r"^(what|which)\b.*\b(has|have|did)\b", lowered))


def snapshot_to_public_page(snapshot: Any) -> dict[str, Any]:
    payload = snapshot.to_dict()
    payload.pop("embedding", None)
    anchor_span_ids = payload.get("raw_refs") or []
    return {
        "page_id": payload["snapshot_id"],
        "corpus_id": payload["corpus_id"],
        "parent_id": payload["parent_id"],
        "level": payload["snapshot_kind"],
        "title": payload["title"],
        "synopsis": payload["synopsis"],
        "summary": payload["summary"],
        "page_kind": payload["snapshot_kind"],
        "entity_refs": payload.get("entity_refs") or [],
        "canonical_entity_refs": payload.get("canonical_entity_refs") or [],
        "raw_refs": payload.get("raw_refs") or [],
        "anchor_span_ids": anchor_span_ids,
        "child_ids": payload.get("child_ids") or [],
        "time_range": payload.get("time_range"),
        "metadata": payload.get("metadata") or {},
    }


def event_to_raw_spans(event: Any) -> list[dict[str, Any]]:
    base_span_id = event.raw_span_id or event.event_id
    metadata = dict(event.metadata or {})
    fragments = split_text_fragments(str(event.text or ""))
    if len(fragments) <= 1:
        return [
            {
                "span_id": base_span_id,
                "corpus_id": event.corpus_id,
                "session_id": event.session_id,
                "speaker": event.speaker,
                "text": event.text,
                "turn_index": event.turn_index,
                "timestamp": event.timestamp,
                "metadata": metadata,
            }
        ]
    raw_spans: list[dict[str, Any]] = []
    fragment_count = len(fragments)
    for index, fragment in enumerate(fragments, start=1):
        fragment_metadata = dict(metadata)
        fragment_metadata.update(
            {
                "original_span_id": base_span_id,
                "fragment_index": index,
                "fragment_count": fragment_count,
            }
        )
        raw_spans.append(
            {
                "span_id": f"{base_span_id}#frag{index}",
                "corpus_id": event.corpus_id,
                "session_id": event.session_id,
                "speaker": event.speaker,
                "text": fragment,
                "turn_index": event.turn_index,
                "timestamp": event.timestamp,
                "metadata": fragment_metadata,
            }
        )
    return raw_spans


def version_to_atom(version: Any, obj: Any) -> dict[str, Any]:
    metadata = dict(version.metadata or {})
    metadata.setdefault("subject", obj.subject)
    metadata.setdefault("slot", obj.slot)
    return {
        "atom_id": version.version_id,
        "event_id": version.event_id,
        "span_id": version.event_id or version.version_id,
        "atom_type": obj.slot,
        "content": version.summary,
        "entities": [obj.subject] + list(obj.aliases or []),
        "canonical_entities": list(obj.canonical_entities or []),
        "support_span_ids": [],
        "memory_kind": obj.memory_kind,
        "status": version.status,
        "time_range": version.valid_from,
        "confidence": version.confidence,
        "metadata": metadata,
    }


def query_terms(question: str) -> set[str]:
    refs = extract_entities(question) + extract_semantic_references(question)
    refs.extend(make_query_tokens(question))
    normalized: set[str] = set()
    for item in refs:
        text = str(item).strip().lower()
        if not text or text in QUERY_STOPWORDS:
            continue
        normalized.add(text)
        collapsed = text.replace(" ", "")
        if collapsed and collapsed != text:
            normalized.add(collapsed)
        if text.endswith("s") and len(text) > 4:
            normalized.add(text[:-1])
        for alias in QUERY_ALIASES.get(text, set()):
            normalized.add(alias)
        for alias in QUERY_ALIASES.get(collapsed, set()):
            normalized.add(alias)
    return normalized


def query_temporal_hints(question: str) -> dict[str, int]:
    lowered = str(question or "").lower()
    hints: dict[str, int] = {}
    year_match = re.search(r"\b(19|20)\d{2}\b", lowered)
    if year_match:
        hints["year"] = int(year_match.group(0))
    for month_name, month_number in MONTH_NUMBERS.items():
        if re.search(rf"\b{month_name}\b", lowered):
            hints["month"] = month_number
            day_match = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_name}\b", lowered)
            if not day_match:
                day_match = re.search(rf"\b{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", lowered)
            if day_match:
                hints["day"] = int(day_match.group(1))
            break
    return hints


def is_list_query(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    return any(pattern in lowered for pattern in LIST_QUERY_PATTERNS)


def is_single_fact_query(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    if not lowered:
        return False
    if is_list_query(lowered):
        return False
    if lowered.startswith(YES_NO_PREFIXES) or lowered.endswith("right?") or ", right?" in lowered:
        return True
    if any(pattern in lowered for pattern in SINGLE_FACT_PATTERNS):
        return True
    return bool(re.search(r"\b(what\s+(is|was)|where|which|who|when)\b", lowered))


def target_date_key(temporal_hints: dict[str, int]) -> str | None:
    month = temporal_hints.get("month")
    day = temporal_hints.get("day")
    if month is None or day is None:
        return None
    return f"{int(month):02d}-{int(day):02d}"


def event_date_key(timestamp: str | None) -> str | None:
    anchor = parse_anchor_datetime(timestamp)
    if anchor is None:
        return None
    return anchor.strftime("%m-%d")


def _event_tokens(event: Any) -> set[str]:
    return query_terms(
        "\n".join(
            [
                str(event.speaker or ""),
                str(event.text or ""),
                " ".join(str(item) for item in (event.canonical_entity_refs or [])),
                str(event.metadata or {}),
            ]
        )
    )


def _matched_query_entities_for_event(query_entities: list[str], event: Any) -> set[str]:
    matched: set[str] = set()
    candidate_values = [str(event.speaker or ""), str(event.text or "")]
    candidate_values.extend(str(item) for item in (event.canonical_entity_refs or []))
    for query_entity in query_entities:
        if _entity_overlap_score([query_entity], candidate_values) >= 0.85:
            matched.add(query_entity)
    return matched


def _is_anchor_style_event(event: Any) -> bool:
    metadata = dict(event.metadata or {})
    if metadata.get("blip_caption"):
        return True
    lowered = str(event.text or "").lower()
    return any(
        marker in lowered
        for marker in ["this", "that", "take a look", "look at", "check this", "here it is", "photo", "picture"]
    )


def retrieve_leaf_memory(
    *,
    store: SQLiteMemoryStore,
    corpus_id: str,
    question: str,
    embedding: EmbeddingClient,
    snapshot_limit: int = 6,
    raw_span_limit: int = 8,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    intents = _route_query_intents(question)
    support_hints = _support_pattern_hints(question, intents)
    semantic_hints = _semantic_query_hints(question)
    expanded_query = "\n".join(_ordered_unique_strings([question] + semantic_hints + support_hints))
    query_embedding = embedding.embed(expanded_query)
    normalized_query_terms = query_terms(expanded_query)
    lowered_question = str(question or "").strip().lower()
    query_entities = _ordered_unique_strings(
        [str(item).strip().lower() for item in extract_entities(question) if str(item).strip()]
        + [str(item).strip().lower() for item in extract_semantic_references(question) if str(item).strip()]
    )
    temporal_hints = query_temporal_hints(question)
    explicit_target_date = target_date_key(temporal_hints)
    single_fact_query = is_single_fact_query(question)
    prefer_temporal_focus = explicit_target_date is not None
    prefer_temporal_diversity = (
        not prefer_temporal_focus
        and (bool(temporal_hints) or lowered_question.startswith("how many "))
    )
    focus_first_conversation = "first conversation" in lowered_question
    focus_last_conversation = any(token in lowered_question for token in ["last conversation", "most recent conversation", "latest conversation"])
    require_entity_coverage = _requires_entity_coverage(question, query_entities)
    prefer_session_diversity = _requires_session_diversity(question)
    prefer_inference_support = "open_domain_inference" in intents or "persona_inference" in intents

    def score_embedding(vector: list[float] | None) -> float:
        if not vector:
            return 0.0
        return cosine_similarity(query_embedding, vector)

    def text_overlap_score(*texts: str) -> float:
        merged_terms = query_terms(" ".join(texts))
        if not normalized_query_terms or not merged_terms:
            return 0.0
        return len(normalized_query_terms.intersection(merged_terms)) / max(1, len(normalized_query_terms))

    def score_temporal(timestamp: str | None) -> float:
        if not temporal_hints or not timestamp:
            return 0.0
        anchor = parse_anchor_datetime(timestamp)
        if anchor is None:
            return 0.0
        score = 0.0
        if temporal_hints.get("year") == anchor.year:
            score += 0.18
        if temporal_hints.get("month") == anchor.month:
            score += 0.28
        if temporal_hints.get("day") == anchor.day:
            score += 0.36
        if explicit_target_date is not None:
            if anchor.strftime("%m-%d") == explicit_target_date:
                score += 0.32
            else:
                score -= 0.1
        return score

    def score_speaker(speaker: str | None) -> float:
        speaker_text = str(speaker or "").strip().lower()
        if not speaker_text or not query_entities:
            return 0.0
        return 0.22 if speaker_text in query_entities else 0.0

    def score_snapshot(snapshot: Any) -> float:
        metadata = dict(snapshot.metadata or {})
        semantic_role = str(metadata.get("semantic_role") or "")
        snapshot_entities = [str(item).lower() for item in (snapshot.canonical_entity_refs or snapshot.entity_refs or [])]
        score = 0.72 * score_embedding(snapshot.embedding)
        score += 0.48 * text_overlap_score(
            snapshot.title,
            snapshot.synopsis,
            snapshot.summary,
            " ".join(snapshot.entity_refs),
            " ".join(snapshot.canonical_entity_refs),
            semantic_role,
        )
        score += min(0.38, _entity_overlap_score(query_entities, snapshot_entities + [str(snapshot.scope_id)]) * 0.38)
        score += min(0.18, text_overlap_score(snapshot.title) * 0.24)
        if snapshot.snapshot_kind == "entity" and str(snapshot.scope_id).lower() in set(query_entities).union(normalized_query_terms):
            score += 0.42
        if snapshot.snapshot_kind == "session_page":
            score += 0.12
        if snapshot.snapshot_kind == "session_block":
            depth = int(metadata.get("depth") or 0)
            score += max(0.0, 0.18 - 0.03 * depth)
        if snapshot.snapshot_kind == "session" and semantic_role == "session_snapshot":
            score += 0.08
        if "open_domain_inference" in intents or "persona_inference" in intents:
            if snapshot.snapshot_kind in {"session", "entity"}:
                score += 0.08
            if semantic_role in {"session_snapshot", "entity_snapshot", "session_block"}:
                score += 0.06
        if "factual_recall" in intents and snapshot.snapshot_kind in {"session_page", "session_block"}:
            score += 0.05
        if "temporal_update" in intents and snapshot.time_range:
            score += 0.08
        return score

    snapshots: list[Any] = []
    root_snapshot = store.get_snapshot(corpus_id=corpus_id, snapshot_kind="root", scope_id=corpus_id)
    if root_snapshot is not None:
        snapshots.append(root_snapshot)
    entity_snapshots = store.list_snapshots(corpus_id=corpus_id, snapshot_kind="entity")
    session_snapshots = store.list_snapshots(corpus_id=corpus_id, snapshot_kind="session")
    session_pages = store.list_snapshots(corpus_id=corpus_id, snapshot_kind="session_page")
    session_blocks = store.list_snapshots(corpus_id=corpus_id, snapshot_kind="session_block")
    snapshots.extend(entity_snapshots)
    snapshots.extend(session_snapshots)
    snapshots.extend(session_pages)
    snapshots.extend(session_blocks)
    snapshot_by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}

    scored_all_snapshots = sorted(
        ((score_snapshot(snapshot), snapshot) for snapshot in snapshots),
        key=lambda item: item[0],
        reverse=True,
    )

    top_level_snapshots: list[Any] = []
    if root_snapshot is not None:
        top_level_snapshots.append(root_snapshot)
    top_level_snapshots.extend(entity_snapshots)
    top_level_snapshots.extend(session_snapshots)
    scored_top_level = sorted(
        ((score_snapshot(snapshot), snapshot) for snapshot in top_level_snapshots),
        key=lambda item: item[0],
        reverse=True,
    )
    entry_limit = max(2, min(snapshot_limit, 4))
    frontier = [snapshot for _, snapshot in scored_top_level[:entry_limit]]

    traversed_snapshots: list[Any] = []
    traversed_ids: set[str] = set()
    leaf_snapshots: list[Any] = []

    def append_traversed(snapshot: Any) -> None:
        if snapshot.snapshot_id in traversed_ids:
            return
        traversed_snapshots.append(snapshot)
        traversed_ids.add(snapshot.snapshot_id)

    max_traversal_nodes = max(snapshot_limit * 3, 12)
    while frontier and len(traversed_snapshots) < max_traversal_nodes:
        next_frontier: list[Any] = []
        for snapshot in frontier:
            if len(traversed_snapshots) >= max_traversal_nodes:
                break
            append_traversed(snapshot)
            child_snapshots = [
                snapshot_by_id[child_id]
                for child_id in snapshot.child_ids
                if child_id in snapshot_by_id
            ]
            if not child_snapshots:
                leaf_snapshots.append(snapshot)
                continue
            scored_children = sorted(
                ((score_snapshot(child), child) for child in child_snapshots),
                key=lambda item: item[0],
                reverse=True,
            )
            if snapshot.snapshot_kind in {"root", "session"}:
                child_branch_limit = 3
            elif snapshot.snapshot_kind == "session_block":
                child_branch_limit = 2
            else:
                child_branch_limit = 1
            for _, child in scored_children[:child_branch_limit]:
                next_frontier.append(child)
        frontier = next_frontier

    if not traversed_snapshots:
        traversed_snapshots = [snapshot for _, snapshot in scored_top_level[:snapshot_limit]]
        traversed_ids = {snapshot.snapshot_id for snapshot in traversed_snapshots}
    if not leaf_snapshots:
        leaf_snapshots = list(traversed_snapshots)

    fallback_snapshots: list[Any] = []
    fallback_limit = max(2, snapshot_limit // 4)
    for _, snapshot in scored_all_snapshots:
        if snapshot.snapshot_id in traversed_ids:
            continue
        fallback_snapshots.append(snapshot)
        traversed_ids.add(snapshot.snapshot_id)
        if len(fallback_snapshots) >= fallback_limit:
            break
    traversed_snapshots.extend(fallback_snapshots)

    snapshot_event_ids: list[str] = []
    selected_object_ids: list[str] = []
    for snapshot in traversed_snapshots:
        for event_id in snapshot.event_ids:
            if event_id not in snapshot_event_ids:
                snapshot_event_ids.append(event_id)
        for object_id in snapshot.object_ids:
            if object_id not in selected_object_ids:
                selected_object_ids.append(object_id)
    for snapshot in leaf_snapshots:
        for event_id in snapshot.event_ids:
            if event_id not in snapshot_event_ids:
                snapshot_event_ids.append(event_id)

    entity_scope_ids = [snapshot.scope_id for snapshot in traversed_snapshots if snapshot.snapshot_kind == "entity"]
    for entity in query_entities:
        if entity and entity not in entity_scope_ids:
            entity_scope_ids.append(entity)

    all_events = store.get_events(corpus_id=corpus_id)
    event_lookup = {event.event_id: event for event in all_events}
    session_turn_lookup: dict[str, dict[int, Any]] = defaultdict(dict)
    speaker_events: dict[str, list[Any]] = defaultdict(list)
    ordered_session_ids: list[str] = []
    event_base_scores: dict[str, float] = {}
    scored_events: list[tuple[float, Any]] = []
    for event in all_events:
        session_id = str(event.session_id)
        session_turn_lookup[session_id][int(event.turn_index)] = event
        speaker_events[str(event.speaker)].append(event)
        if session_id not in ordered_session_ids:
            ordered_session_ids.append(session_id)
        score = 0.84 * score_embedding(event.embedding)
        score += 0.58 * text_overlap_score(
            event.text,
            " ".join(event.canonical_entity_refs),
            str(event.metadata),
            str(event.speaker),
        )
        score += score_temporal(event.timestamp)
        score += score_speaker(event.speaker)
        if single_fact_query:
            score += 0.12 * text_overlap_score(event.text)
        if event.event_id in snapshot_event_ids:
            score += 0.18
        if prefer_inference_support and event.canonical_entity_refs:
            score += 0.05
        event_base_scores[event.event_id] = score
        scored_events.append((score, event))
    scored_events.sort(key=lambda item: item[0], reverse=True)

    candidate_events: list[Any] = []
    candidate_event_ids: set[str] = set()
    candidate_bonus_by_event: dict[str, float] = defaultdict(float)

    def append_candidate(event: Any | None) -> None:
        if event is None or event.event_id in candidate_event_ids:
            return
        candidate_event_ids.add(event.event_id)
        candidate_events.append(event)

    for event_id in snapshot_event_ids:
        append_candidate(event_lookup.get(event_id))

    direct_seed_limit = max(raw_span_limit * 6, 24)
    direct_seed_events = [event for _, event in scored_events[:direct_seed_limit]]
    for event in direct_seed_events:
        append_candidate(event)

    if prefer_temporal_focus and explicit_target_date is not None:
        for event in all_events:
            if event_date_key(event.timestamp) == explicit_target_date:
                append_candidate(event)
                candidate_bonus_by_event[event.event_id] += 0.42

    if focus_first_conversation and ordered_session_ids:
        first_session_id = sorted(ordered_session_ids)[0]
        for _, event in sorted(session_turn_lookup.get(first_session_id, {}).items()):
            append_candidate(event)
            candidate_bonus_by_event[event.event_id] += 0.38

    if focus_last_conversation and ordered_session_ids:
        last_session_id = sorted(ordered_session_ids)[-1]
        for _, event in sorted(session_turn_lookup.get(last_session_id, {}).items()):
            append_candidate(event)
            candidate_bonus_by_event[event.event_id] += 0.38

    for entity in entity_scope_ids:
        for event in store.get_events_for_entity(corpus_id=corpus_id, entity=entity, limit=12):
            append_candidate(event)
            candidate_bonus_by_event[event.event_id] += 0.08

    neighbor_seed_limit = max(8, raw_span_limit * 2)
    for event in direct_seed_events[:neighbor_seed_limit]:
        session_events = session_turn_lookup.get(str(event.session_id), {})
        for delta in (-2, -1, 1, 2):
            neighbor = session_events.get(int(event.turn_index) + delta)
            if neighbor is None:
                continue
            append_candidate(neighbor)
            distance = abs(delta)
            bonus = float(event_base_scores.get(event.event_id, 0.0)) * (0.22 if distance == 1 else 0.12)
            if _is_anchor_style_event(event):
                bonus += 0.18 if distance == 1 else 0.08
            if normalized_query_terms.intersection(_event_tokens(neighbor)):
                bonus += 0.08
            seed_entities = _matched_query_entities_for_event(query_entities, event)
            neighbor_entities = _matched_query_entities_for_event(query_entities, neighbor)
            if neighbor_entities.difference(seed_entities):
                bonus += 0.18
            candidate_bonus_by_event[neighbor.event_id] += bonus

    target_speakers = [
        speaker
        for speaker in _ordered_unique_strings([str(event.speaker).strip().lower() for event in direct_seed_events])
        if any(_entity_overlap_score([query_entity], [speaker]) >= 0.85 for query_entity in query_entities)
    ]
    if target_speakers:
        best_by_speaker_session: dict[tuple[str, str], tuple[Any, int, int, float]] = {}
        for event in all_events:
            speaker = str(event.speaker).strip().lower()
            if speaker not in target_speakers:
                continue
            matched = _matched_query_entities_for_event(query_entities, event)
            semantic_hits = {
                hit
                for hit in matched
                if _entity_overlap_score([hit], [speaker]) < 0.85
            }
            lexical_hits = len(normalized_query_terms.intersection(_event_tokens(event)))
            if not semantic_hits and lexical_hits < 2:
                continue
            key = (speaker, str(event.session_id))
            rank = (
                len(semantic_hits),
                lexical_hits,
                float(event_base_scores.get(event.event_id, 0.0)),
            )
            current = best_by_speaker_session.get(key)
            if current is None or rank > (current[1], current[2], current[3]):
                best_by_speaker_session[key] = (event, len(semantic_hits), lexical_hits, rank[2])
        for event, semantic_hit_count, lexical_hit_count, _ in best_by_speaker_session.values():
            append_candidate(event)
            candidate_bonus_by_event[event.event_id] += (
                0.34 + semantic_hit_count * 0.34 + min(0.18, lexical_hit_count * 0.04)
            )

    candidate_pool_limit = max(raw_span_limit * 10, 40)
    for _, event in scored_events:
        append_candidate(event)
        if len(candidate_events) >= candidate_pool_limit:
            break

    candidate_atoms = store.get_atoms_for_events([event.event_id for event in candidate_events])
    atom_score_by_event: dict[str, float] = defaultdict(float)
    atom_score_rows: list[tuple[float, Any]] = []
    for atom in candidate_atoms:
        atom_text = "\n".join(
            [
                str(atom.content or ""),
                " ".join(str(item) for item in (atom.canonical_entities or atom.entities or [])),
                str(atom.metadata or {}),
            ]
        )
        atom_score = 0.0
        atom_score += 0.52 * text_overlap_score(atom_text)
        atom_score += min(0.24, _entity_overlap_score(query_entities, list(atom.canonical_entities or atom.entities or [])) * 0.24)
        if atom.status == "active":
            atom_score += 0.08
        elif atom.status == "superseded":
            atom_score -= 0.06
        if atom.memory_kind in {"state", "preference", "plan", "relation"}:
            atom_score += 0.06
        if prefer_inference_support and atom.memory_kind in {"state", "preference", "plan", "relation"}:
            atom_score += 0.1
        if temporal_hints and atom.time_range:
            atom_score += 0.08
        atom_score_by_event[atom.event_id] = max(atom_score_by_event.get(atom.event_id, 0.0), atom_score)
        atom_score_rows.append((atom_score, atom))

    if prefer_temporal_focus and explicit_target_date is not None:
        exact_date_candidates = [event for event in candidate_events if event_date_key(event.timestamp) == explicit_target_date]
        if exact_date_candidates:
            candidate_events = exact_date_candidates + [event for event in candidate_events if event.event_id not in {item.event_id for item in exact_date_candidates}]

    selected_events: list[Any] = []
    selected_event_ids: set[str] = set()
    covered_entities: set[str] = set()
    covered_sessions: set[str] = set()
    covered_dates: set[str] = set()
    candidate_queue = list(candidate_events)
    while candidate_queue and len(selected_events) < max(raw_span_limit * 2, 8):
        best_index = 0
        best_score = float("-inf")
        selected_token_sets = [_event_tokens(existing) for existing in selected_events]
        for index, event in enumerate(candidate_queue):
            score = (
                float(event_base_scores.get(event.event_id, 0.0))
                + float(atom_score_by_event.get(event.event_id, 0.0))
                + float(candidate_bonus_by_event.get(event.event_id, 0.0))
            )
            matched_entities = _matched_query_entities_for_event(query_entities, event)
            if require_entity_coverage:
                score += len([entity for entity in matched_entities if entity not in covered_entities]) * 0.28
            if prefer_session_diversity and event.session_id not in covered_sessions:
                score += 0.12
            date_key = event_date_key(event.timestamp) or str(event.timestamp or "")
            if prefer_temporal_diversity and date_key and date_key not in covered_dates:
                score += 0.08
            if explicit_target_date is not None:
                if date_key == explicit_target_date:
                    score += 0.34
                else:
                    score -= 0.18
            score += len(normalized_query_terms.intersection(_event_tokens(event))) * 0.03
            redundancy = 0.0
            event_tokens = _event_tokens(event)
            for existing_tokens in selected_token_sets:
                overlap = len(event_tokens.intersection(existing_tokens))
                redundancy = max(redundancy, overlap * 0.03)
            candidate_score = score - redundancy
            if candidate_score > best_score:
                best_score = candidate_score
                best_index = index
        event = candidate_queue.pop(best_index)
        if event.event_id in selected_event_ids:
            continue
        selected_events.append(event)
        selected_event_ids.add(event.event_id)
        covered_entities.update(_matched_query_entities_for_event(query_entities, event))
        covered_sessions.add(str(event.session_id))
        date_key = event_date_key(event.timestamp) or str(event.timestamp or "")
        if date_key:
            covered_dates.add(date_key)
        if len(selected_events) >= max(raw_span_limit, 4):
            event_span_count = sum(len(event_to_raw_spans(item)) for item in selected_events)
            if event_span_count >= raw_span_limit:
                break

    raw_spans: list[dict[str, Any]] = []
    seen_span_ids: set[str] = set()

    def append_raw_span(event: Any) -> None:
        for span in event_to_raw_spans(event):
            span_id = str(span.get("span_id") or "").strip()
            if not span_id or span_id in seen_span_ids:
                continue
            raw_spans.append(span)
            seen_span_ids.add(span_id)
            if len(raw_spans) >= raw_span_limit:
                return

    for event in selected_events:
        append_raw_span(event)
        if len(raw_spans) >= raw_span_limit:
            break
    if len(raw_spans) < raw_span_limit:
        for _, event in scored_events:
            append_raw_span(event)
            if len(raw_spans) >= raw_span_limit:
                break

    atoms: list[dict[str, Any]] = []
    emitted_atom_ids: set[str] = set()
    selected_event_id_set = {event.event_id for event in selected_events}
    event_atom_rows = [
        (score, atom)
        for score, atom in atom_score_rows
        if atom.event_id in selected_event_id_set
    ]
    event_atom_rows.sort(key=lambda item: item[0], reverse=True)
    for _, atom in event_atom_rows:
        payload = atom.to_dict()
        if not payload.get("support_span_ids") and payload.get("span_id"):
            payload["support_span_ids"] = [payload["span_id"]]
        if payload["atom_id"] in emitted_atom_ids:
            continue
        emitted_atom_ids.add(payload["atom_id"])
        atoms.append(payload)
        if len(atoms) >= 12:
            break

    version_atom_rows: list[tuple[float, dict[str, Any]]] = []
    seen_object_ids: set[str] = set()
    for object_id in selected_object_ids[:24]:
        if object_id in seen_object_ids:
            continue
        seen_object_ids.add(object_id)
        obj = store.get_object(object_id)
        if obj is None:
            continue
        latest_version = store.get_latest_version(object_id)
        if latest_version is None:
            continue
        payload = version_to_atom(latest_version, obj)
        version_score = 0.45 * text_overlap_score(payload["content"], " ".join(payload.get("canonical_entities") or []))
        version_score += min(
            0.22,
            _entity_overlap_score(query_entities, list(payload.get("canonical_entities") or payload.get("entities") or [])) * 0.22,
        )
        if prefer_inference_support and payload.get("memory_kind") in {"state", "preference", "plan", "relation"}:
            version_score += 0.08
        version_atom_rows.append((version_score, payload))
    for entity in entity_scope_ids:
        for obj in store.get_objects_for_subject(corpus_id=corpus_id, subject=entity):
            if obj.object_id in seen_object_ids:
                continue
            seen_object_ids.add(obj.object_id)
            latest_version = store.get_latest_version(obj.object_id)
            if latest_version is None:
                continue
            payload = version_to_atom(latest_version, obj)
            version_score = 0.45 * text_overlap_score(payload["content"], " ".join(payload.get("canonical_entities") or []))
            version_score += min(
                0.22,
                _entity_overlap_score(query_entities, list(payload.get("canonical_entities") or payload.get("entities") or [])) * 0.22,
            )
            version_atom_rows.append((version_score, payload))
    version_atom_rows.sort(key=lambda item: item[0], reverse=True)
    for _, payload in version_atom_rows:
        atom_id = str(payload.get("atom_id") or "").strip()
        if not atom_id or atom_id in emitted_atom_ids:
            continue
        emitted_atom_ids.add(atom_id)
        atoms.append(payload)
        if len(atoms) >= 18:
            break

    return {
        "traversal_path": [snapshot.snapshot_id for snapshot in traversed_snapshots],
        "pages": [snapshot_to_public_page(snapshot) for snapshot in traversed_snapshots[:snapshot_limit]],
        "atoms": atoms,
        "edges": [],
        "raw_spans": raw_spans,
        "selected_event_ids": [event.event_id for event in selected_events],
        "selected_session_ids": _ordered_unique_strings([str(event.session_id) for event in selected_events]),
        "timing": {
            "search_total_ms": (time.perf_counter() - started_at) * 1000.0,
        },
    }
