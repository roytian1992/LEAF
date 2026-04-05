from __future__ import annotations

import re
import time
from typing import Any

from .clients import EmbeddingClient, cosine_similarity
from .extract import extract_entities, extract_semantic_references
from .grounding import parse_anchor_datetime, query_tokens as make_query_tokens
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


def snapshot_to_public_page(snapshot: Any) -> dict[str, Any]:
    payload = snapshot.to_dict()
    payload.pop("embedding", None)
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
        "child_ids": payload.get("child_ids") or [],
        "time_range": payload.get("time_range"),
        "metadata": payload.get("metadata") or {},
    }


def event_to_raw_span(event: Any) -> dict[str, Any]:
    return {
        "span_id": event.raw_span_id or event.event_id,
        "corpus_id": event.corpus_id,
        "session_id": event.session_id,
        "speaker": event.speaker,
        "text": event.text,
        "turn_index": event.turn_index,
        "timestamp": event.timestamp,
        "metadata": event.metadata or {},
    }


def version_to_atom(version: Any, obj: Any) -> dict[str, Any]:
    return {
        "atom_id": version.version_id,
        "span_id": version.event_id or version.version_id,
        "atom_type": obj.slot,
        "content": version.summary,
        "memory_kind": obj.memory_kind,
        "status": version.status,
        "time_range": version.valid_from,
        "confidence": version.confidence,
        "metadata": version.metadata or {},
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


def retrieve_leaf_memory(
    *,
    store: SQLiteMemoryStore,
    corpus_id: str,
    question: str,
    embedding: EmbeddingClient,
    raw_span_limit: int = 8,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    query_embedding = embedding.embed(question)
    normalized_query_terms = query_terms(question)
    query_entities = [str(item).strip().lower() for item in extract_entities(question) if str(item).strip()]
    temporal_hints = query_temporal_hints(question)
    prefer_temporal_diversity = bool(temporal_hints) or question.lower().strip().startswith("how many ")

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
        return score

    def score_speaker(speaker: str | None) -> float:
        speaker_text = str(speaker or "").strip().lower()
        if not speaker_text or not query_entities:
            return 0.0
        return 0.22 if speaker_text in query_entities else 0.0

    snapshots = []
    root_snapshot = store.get_snapshot(corpus_id=corpus_id, snapshot_kind="root", scope_id=corpus_id)
    if root_snapshot is not None:
        snapshots.append(root_snapshot)
    snapshots.extend(store.list_snapshots(corpus_id=corpus_id, snapshot_kind="entity"))
    snapshots.extend(store.list_snapshots(corpus_id=corpus_id, snapshot_kind="session"))

    scored_snapshots: list[tuple[float, Any]] = []
    for snapshot in snapshots:
        score = 0.8 * score_embedding(snapshot.embedding)
        score += 0.35 * text_overlap_score(
            snapshot.title,
            snapshot.synopsis,
            snapshot.summary,
            " ".join(snapshot.entity_refs),
        )
        if snapshot.snapshot_kind == "entity" and snapshot.scope_id.lower() in set(query_entities).union(normalized_query_terms):
            score += 0.4
        scored_snapshots.append((score, snapshot))
    scored_snapshots.sort(key=lambda item: item[0], reverse=True)
    top_snapshots = [snapshot for _, snapshot in scored_snapshots[:6]]

    selected_event_ids: list[str] = []
    selected_object_ids: list[str] = []
    for snapshot in top_snapshots:
        for event_id in snapshot.event_ids:
            if event_id not in selected_event_ids:
                selected_event_ids.append(event_id)
        for object_id in snapshot.object_ids:
            if object_id not in selected_object_ids:
                selected_object_ids.append(object_id)

    entity_scope_ids = [snapshot.scope_id for snapshot in top_snapshots if snapshot.snapshot_kind == "entity"]
    for entity in query_entities:
        if entity and entity not in entity_scope_ids:
            entity_scope_ids.append(entity)
    for entity in entity_scope_ids:
        for event in store.get_events_for_entity(corpus_id=corpus_id, entity=entity, limit=12):
            if event.event_id not in selected_event_ids:
                selected_event_ids.append(event.event_id)

    all_events = store.get_events(corpus_id=corpus_id)
    scored_events: list[tuple[float, Any]] = []
    for event in all_events:
        score = 0.9 * score_embedding(event.embedding)
        score += 0.55 * text_overlap_score(event.text, " ".join(event.canonical_entity_refs), str(event.metadata))
        score += score_temporal(event.timestamp)
        score += score_speaker(event.speaker)
        if event.event_id in selected_event_ids:
            score += 0.15
        scored_events.append((score, event))
    scored_events.sort(key=lambda item: item[0], reverse=True)

    raw_spans: list[dict[str, Any]] = []
    seen_span_ids: set[str] = set()

    def append_raw_span(event: Any) -> None:
        span_id = event.raw_span_id or event.event_id
        if span_id in seen_span_ids:
            return
        raw_spans.append(event_to_raw_span(event))
        seen_span_ids.add(span_id)

    if prefer_temporal_diversity:
        seen_time_keys: set[str] = set()
        for _, event in scored_events:
            anchor = parse_anchor_datetime(event.timestamp)
            time_key = anchor.strftime("%Y-%m-%d") if anchor is not None else ""
            if time_key and time_key in seen_time_keys:
                continue
            append_raw_span(event)
            if time_key:
                seen_time_keys.add(time_key)
            if len(raw_spans) >= raw_span_limit:
                break
    if len(raw_spans) < raw_span_limit:
        for _, event in scored_events:
            append_raw_span(event)
            if len(raw_spans) >= raw_span_limit:
                break

    atoms: list[dict[str, Any]] = []
    seen_version_ids: set[str] = set()
    for object_id in selected_object_ids[:24]:
        obj = store.get_object(object_id)
        if obj is None:
            continue
        latest_version = store.get_latest_version(object_id)
        if latest_version is None or latest_version.version_id in seen_version_ids:
            continue
        atoms.append(version_to_atom(latest_version, obj))
        seen_version_ids.add(latest_version.version_id)
    for entity in entity_scope_ids:
        for obj in store.get_objects_for_subject(corpus_id=corpus_id, subject=entity):
            latest_version = store.get_latest_version(obj.object_id)
            if latest_version is None or latest_version.version_id in seen_version_ids:
                continue
            atoms.append(version_to_atom(latest_version, obj))
            seen_version_ids.add(latest_version.version_id)
            if len(atoms) >= 12:
                break
        if len(atoms) >= 12:
            break

    return {
        "traversal_path": [snapshot.snapshot_id for snapshot in top_snapshots],
        "pages": [snapshot_to_public_page(snapshot) for snapshot in top_snapshots],
        "atoms": atoms,
        "edges": [],
        "raw_spans": raw_spans,
        "timing": {
            "search_total_ms": (time.perf_counter() - started_at) * 1000.0,
        },
    }
