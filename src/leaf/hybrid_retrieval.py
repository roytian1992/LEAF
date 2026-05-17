from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .extract import extract_entities, extract_semantic_references
from .grounding import is_temporal_query, parse_anchor_datetime
from .normalize import language_aware_stemmed_content_terms


ENTITY_BOOST_WEIGHT = 0.5


@dataclass(slots=True)
class HybridDocument:
    doc_id: str
    event_id: str
    text: str
    terms: set[str]
    entities: set[str]
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HybridCorpusIndex:
    documents: list[HybridDocument]
    document_by_id: dict[str, HybridDocument]
    event_doc_ids: dict[str, list[str]]
    term_df: dict[str, int]
    entity_doc_ids: dict[str, list[str]]
    entity_event_ids: dict[str, list[str]]


def build_hybrid_corpus_index(
    *,
    events: list[Any],
    atoms: list[Any],
    additive_memories: list[Any] | None = None,
    additive_sources: set[str] | None = None,
    language_mode: str = "auto",
) -> HybridCorpusIndex:
    docs: list[HybridDocument] = []
    event_doc_ids: dict[str, list[str]] = defaultdict(list)
    term_doc_ids: dict[str, set[str]] = defaultdict(set)
    entity_doc_ids: dict[str, list[str]] = defaultdict(list)
    entity_event_ids: dict[str, list[str]] = defaultdict(list)

    atom_texts_by_event: dict[str, list[str]] = defaultdict(list)
    atom_entities_by_event: dict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        event_id = str(getattr(atom, "event_id", "") or "").strip()
        if not event_id:
            continue
        atom_text = str(getattr(atom, "content", "") or "").strip()
        if atom_text:
            atom_texts_by_event[event_id].append(atom_text)
        for entity in (getattr(atom, "canonical_entities", []) or getattr(atom, "entities", []) or []):
            normalized_entity = _normalize_entity(entity)
            if normalized_entity:
                atom_entities_by_event[event_id].add(normalized_entity)

    for event in events:
        event_id = str(getattr(event, "event_id", "") or "").strip()
        if not event_id:
            continue
        metadata = dict(getattr(event, "metadata", None) or {})
        event_entities = {_normalize_entity(item) for item in (getattr(event, "canonical_entity_refs", []) or [])}
        event_entities.update(_normalize_entity(item) for item in (getattr(event, "entity_refs", []) or []))
        event_entities.update(atom_entities_by_event.get(event_id, set()))
        event_entities.update(_extract_query_entities(str(getattr(event, "text", "") or ""), language_mode=language_mode))
        event_entities = {item for item in event_entities if item}
        semantic_refs = [
            str(item).strip()
            for item in (metadata.get("semantic_refs") or [])
            if str(item).strip()
        ]
        doc_text = "\n".join(
            item
            for item in [
                str(getattr(event, "speaker", "") or ""),
                str(getattr(event, "text", "") or ""),
                str(metadata.get("blip_caption") or ""),
                " ".join(semantic_refs),
                " ".join(atom_texts_by_event.get(event_id, [])[:6]),
                " ".join(sorted(event_entities)),
            ]
            if item
        )
        terms = set(_terms(doc_text, language_mode=language_mode))
        doc = HybridDocument(
            doc_id=f"event:{event_id}",
            event_id=event_id,
            text=doc_text,
            terms=terms,
            entities=event_entities,
            timestamp=getattr(event, "timestamp", None),
            metadata={"kind": "event"},
        )
        docs.append(doc)
        event_doc_ids[event_id].append(doc.doc_id)
        for term in terms:
            term_doc_ids[term].add(doc.doc_id)
        for entity in event_entities:
            entity_doc_ids[entity].append(doc.doc_id)
            entity_event_ids[entity].append(event_id)

    for memory in additive_memories or []:
        event_id = str(getattr(memory, "event_id", "") or "").strip()
        memory_id = str(getattr(memory, "memory_id", "") or "").strip()
        text = str(getattr(memory, "text", "") or "").strip()
        if not event_id or not memory_id or not text:
            continue
        metadata = dict(getattr(memory, "metadata", None) or {})
        source = str(metadata.get("source") or "").strip()
        if source == "event_atom_derived_v1":
            continue
        if additive_sources and source not in additive_sources:
            continue
        memory_entities = {_normalize_entity(item) for item in (getattr(memory, "canonical_entities", []) or [])}
        memory_entities.update(_normalize_entity(item) for item in (getattr(memory, "entities", []) or []))
        memory_entities.update(_extract_query_entities(text, language_mode=language_mode))
        memory_entities = {item for item in memory_entities if item}
        terms = set(getattr(memory, "terms", []) or []) or set(_terms(text, language_mode=language_mode))
        doc = HybridDocument(
            doc_id=f"additive:{memory_id}",
            event_id=event_id,
            text=text,
            terms=set(terms),
            entities=memory_entities,
            timestamp=getattr(memory, "timestamp", None),
            metadata={"kind": "additive_memory", "memory_id": memory_id, "source": source},
        )
        docs.append(doc)
        event_doc_ids[event_id].append(doc.doc_id)
        for term in doc.terms:
            term_doc_ids[term].add(doc.doc_id)
        for entity in memory_entities:
            entity_doc_ids[entity].append(doc.doc_id)
            entity_event_ids[entity].append(event_id)

    document_by_id = {doc.doc_id: doc for doc in docs}
    return HybridCorpusIndex(
        documents=docs,
        document_by_id=document_by_id,
        event_doc_ids={key: list(dict.fromkeys(value)) for key, value in event_doc_ids.items()},
        term_df={term: len(doc_ids) for term, doc_ids in term_doc_ids.items()},
        entity_doc_ids={key: list(dict.fromkeys(value)) for key, value in entity_doc_ids.items()},
        entity_event_ids={key: list(dict.fromkeys(value)) for key, value in entity_event_ids.items()},
    )


def hybrid_rank_event_boosts(
    *,
    question: str,
    index: HybridCorpusIndex,
    candidate_event_ids: list[str] | None = None,
    language_mode: str = "auto",
    top_k: int = 80,
) -> dict[str, Any]:
    query_terms = set(_terms(question, language_mode=language_mode))
    query_entities = _extract_query_entities(question, language_mode=language_mode)
    if not query_terms and not query_entities:
        return {
            "enabled": True,
            "query_terms": [],
            "query_entities": [],
            "ranked_event_ids": [],
            "boost_by_event": {},
            "matched_event_count": 0,
        }

    candidate_set = {str(item) for item in (candidate_event_ids or []) if str(item).strip()}
    candidate_docs = [
        doc
        for doc in index.documents
        if not candidate_set or doc.event_id in candidate_set
    ]
    if not candidate_docs:
        return {
            "enabled": True,
            "query_terms": sorted(query_terms),
            "query_entities": sorted(query_entities),
            "ranked_event_ids": [],
            "boost_by_event": {},
            "matched_event_count": 0,
        }

    avg_doc_len = sum(len(doc.terms) for doc in candidate_docs) / max(1, len(candidate_docs))
    doc_count = max(1, len(candidate_docs))
    query_term_counts = Counter(term for term in query_terms if term)
    entity_boost_by_event = _entity_boosts(query_entities=query_entities, index=index, candidate_set=candidate_set)
    temporal_query = is_temporal_query(question)

    rows: list[tuple[float, str, dict[str, Any]]] = []
    for doc in candidate_docs:
        bm25 = _bm25_score(
            query_term_counts=query_term_counts,
            doc_terms=doc.terms,
            term_df=index.term_df,
            doc_count=doc_count,
            avg_doc_len=avg_doc_len,
        )
        if bm25 <= 0.0 and not entity_boost_by_event.get(doc.event_id):
            continue
        normalized_bm25 = _normalize_bm25(bm25, len(query_terms))
        entity_boost = float(entity_boost_by_event.get(doc.event_id, 0.0))
        temporal_boost = _temporal_soft_boost(doc.timestamp) if temporal_query else 0.0
        combined = normalized_bm25 + entity_boost + temporal_boost
        if combined <= 0.0:
            continue
        rows.append(
            (
                combined,
                doc.event_id,
                {
                    "bm25": round(normalized_bm25, 4),
                    "entity": round(entity_boost, 4),
                    "temporal": round(temporal_boost, 4),
                    "raw_bm25": round(bm25, 4),
                    "matched_terms": sorted(query_terms.intersection(doc.terms))[:10],
                    "matched_entities": sorted(query_entities.intersection(doc.entities))[:10],
                },
            )
        )

    best_by_event: dict[str, tuple[float, dict[str, Any]]] = {}
    for score, event_id, diagnostic in rows:
        current = best_by_event.get(event_id)
        if current is None or score > current[0]:
            best_by_event[event_id] = (score, diagnostic)

    ranked = sorted(best_by_event.items(), key=lambda item: item[1][0], reverse=True)[: max(1, int(top_k))]
    if not ranked:
        return {
            "enabled": True,
            "query_terms": sorted(query_terms),
            "query_entities": sorted(query_entities),
            "ranked_event_ids": [],
            "boost_by_event": {},
            "matched_event_count": 0,
        }

    max_score = max(score for _, (score, _) in ranked) or 1.0
    boost_by_event = {
        event_id: round(min(0.42, (score / max_score) * 0.42), 6)
        for event_id, (score, _) in ranked
    }
    return {
        "enabled": True,
        "query_terms": sorted(query_terms)[:16],
        "query_entities": sorted(query_entities)[:16],
        "ranked_event_ids": [event_id for event_id, _ in ranked],
        "boost_by_event": boost_by_event,
        "diagnostics_by_event": {event_id: diagnostic for event_id, (_score, diagnostic) in ranked[:16]},
        "matched_event_count": len(ranked),
        "max_raw_score": round(max_score, 4),
    }


def _terms(text: str, *, language_mode: str) -> tuple[str, ...]:
    return language_aware_stemmed_content_terms(
        str(text or ""),
        mode=language_mode or "auto",
        include_cjk_subgrams=True,
        max_cjk_ngram=4,
    )


def _normalize_entity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def _extract_query_entities(text: str, *, language_mode: str) -> set[str]:
    entities: set[str] = set()
    for item in extract_entities(str(text or ""), mode=language_mode):
        normalized = _normalize_entity(item)
        if normalized:
            entities.add(normalized)
    for item in extract_semantic_references(str(text or ""), mode=language_mode):
        normalized = _normalize_entity(item)
        if normalized:
            entities.add(normalized)
    return entities


def _bm25_score(
    *,
    query_term_counts: Counter[str],
    doc_terms: set[str],
    term_df: dict[str, int],
    doc_count: int,
    avg_doc_len: float,
) -> float:
    if not query_term_counts or not doc_terms:
        return 0.0
    k1 = 1.4
    b = 0.72
    doc_len = max(1.0, float(len(doc_terms)))
    avg_len = max(1.0, avg_doc_len)
    score = 0.0
    for term, qtf in query_term_counts.items():
        if term not in doc_terms:
            continue
        df = max(0, int(term_df.get(term, 0)))
        idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
        tf = 1.0
        denom = tf + k1 * (1.0 - b + b * doc_len / avg_len)
        score += float(qtf) * idf * ((tf * (k1 + 1.0)) / denom)
    return score


def _normalize_bm25(raw_score: float, query_term_count: int) -> float:
    if raw_score <= 0.0:
        return 0.0
    term_count = max(1, int(query_term_count))
    if term_count <= 3:
        midpoint, steepness = 2.0, 0.85
    elif term_count <= 6:
        midpoint, steepness = 3.2, 0.7
    elif term_count <= 10:
        midpoint, steepness = 4.6, 0.55
    else:
        midpoint, steepness = 6.0, 0.45
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


def _entity_boosts(
    *,
    query_entities: set[str],
    index: HybridCorpusIndex,
    candidate_set: set[str],
) -> dict[str, float]:
    if not query_entities:
        return {}
    event_boosts: dict[str, float] = {}
    for query_entity in sorted(query_entities)[:8]:
        matched_entities = [
            entity
            for entity in index.entity_event_ids
            if _entity_similarity(query_entity, entity) >= 0.72
        ]
        for entity in matched_entities[:24]:
            similarity = _entity_similarity(query_entity, entity)
            linked_event_ids = index.entity_event_ids.get(entity) or []
            if candidate_set:
                linked_event_ids = [event_id for event_id in linked_event_ids if event_id in candidate_set]
            if not linked_event_ids:
                continue
            spread = len(linked_event_ids)
            spread_weight = 1.0 / (1.0 + 0.001 * ((spread - 1) ** 2))
            boost = similarity * ENTITY_BOOST_WEIGHT * spread_weight
            for event_id in linked_event_ids:
                event_boosts[event_id] = max(event_boosts.get(event_id, 0.0), boost)
    return event_boosts


def _entity_similarity(left: str, right: str) -> float:
    left_text = _normalize_entity(left)
    right_text = _normalize_entity(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    if left_text in right_text or right_text in left_text:
        return 0.9
    left_terms = set(_terms(left_text, language_mode="auto"))
    right_terms = set(_terms(right_text, language_mode="auto"))
    if left_terms and right_terms:
        return len(left_terms.intersection(right_terms)) / max(1, len(left_terms.union(right_terms)))
    return 0.0


def _temporal_soft_boost(timestamp: str | None) -> float:
    anchor = parse_anchor_datetime(timestamp)
    if anchor is None:
        return 0.0
    return 0.04
