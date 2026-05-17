from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .clients import EmbeddingClient, cosine_similarity
from .normalize import language_aware_content_terms
from .store import SQLiteMemoryStore

PROFILE_NOISE_TERMS = {
    "actually",
    "amazing",
    "appreciate",
    "awesome",
    "away",
    "beautiful",
    "better",
    "big",
    "bit",
    "brings",
    "called",
    "chat",
    "come",
    "comes",
    "coming",
    "context",
    "cool",
    "day",
    "days",
    "definitely",
    "did",
    "does",
    "doing",
    "done",
    "event",
    "events",
    "feel",
    "feeling",
    "feels",
    "felt",
    "finally",
    "forward",
    "free",
    "glad",
    "going",
    "good",
    "got",
    "great",
    "guess",
    "happy",
    "heard",
    "help",
    "helped",
    "helpful",
    "huge",
    "idea",
    "important",
    "incredible",
    "just",
    "later",
    "life",
    "like",
    "long",
    "look",
    "lot",
    "love",
    "loved",
    "loves",
    "made",
    "make",
    "makes",
    "means",
    "month",
    "near",
    "nice",
    "people",
    "pretty",
    "really",
    "said",
    "says",
    "shared",
    "something",
    "sounds",
    "started",
    "stay",
    "thing",
    "things",
    "time",
    "times",
    "today",
    "told",
    "want",
    "wanted",
    "wants",
    "way",
    "week",
    "yesterday",
}


def profile_terms(text: str) -> set[str]:
    return {
        term
        for term in language_aware_content_terms(text, mode="auto", include_cjk_subgrams=True)
        if term not in PROFILE_NOISE_TERMS
    }


def mean_vector(vectors: list[list[float]]) -> list[float] | None:
    valid = [vector for vector in vectors if vector]
    if not valid:
        return None
    width = len(valid[0])
    same_width = [vector for vector in valid if len(vector) == width]
    if not same_width:
        return None
    return [sum(vector[index] for vector in same_width) / len(same_width) for index in range(width)]


def vector_norm(vector: list[float] | None) -> float:
    if not vector:
        return 0.0
    return math.sqrt(sum(value * value for value in vector))


def topic_profile_text(node: dict[str, Any], *, top_terms: list[str] | None = None, entity_signature: list[str] | None = None) -> str:
    pieces = [
        str(node.get("name") or ""),
        str(node.get("description") or ""),
        " ".join(str(item) for item in (node.get("keywords") or [])),
        " ".join(top_terms or []),
        " ".join(entity_signature or []),
    ]
    return "\n".join(piece for piece in pieces if piece.strip())


def _assignment_source(reason: dict[str, Any]) -> str:
    if reason.get("evolution_reason") == "proposal_evidence_atom":
        return "proposal_evidence_atom"
    if reason.get("strategy") == "evolved_keyword_match":
        return "evolved_keyword_match"
    if reason.get("evolution_secondary_assignment"):
        return "evolved_secondary"
    if reason.get("evolution_reassignment"):
        return "evolved_reassignment"
    return str(reason.get("strategy") or "base_assignment")


def build_topic_profiles(
    store: SQLiteMemoryStore,
    *,
    view_id: str,
    embedding: EmbeddingClient | None = None,
    max_exemplars: int = 12,
    max_terms: int = 24,
    max_entities: int = 16,
    max_document_frequency_ratio: float = 0.5,
    write: bool = True,
) -> dict[str, Any]:
    view = store.get_memory_view(view_id)
    if view is None:
        raise RuntimeError(f"Memory view not found: {view_id}")
    corpus_id = str(view["corpus_id"])
    nodes = store.list_topic_nodes(view_id)
    assignments = [
        item
        for item in store.list_topic_assignments(view_id)
        if str(item.get("item_kind") or "") in {"atom", "atom_secondary"}
    ]
    atom_ids = [str(item.get("item_id") or "") for item in assignments if str(item.get("item_id") or "").strip()]
    atoms_by_id = {atom.atom_id: atom for atom in store.get_atoms_by_ids(atom_ids)}
    event_ids = list({atom.event_id for atom in atoms_by_id.values() if atom.event_id})
    events_by_id = store.get_events_by_ids(event_ids)
    assignments_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_topic[str(assignment["topic_id"])].append(assignment)

    topic_term_counts: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    for node in nodes:
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            continue
        topic_id = str(node["topic_id"])
        term_counts: Counter[str] = Counter()
        for assignment in assignments_by_topic.get(topic_id, []):
            atom = atoms_by_id.get(str(assignment.get("item_id") or ""))
            if atom is None:
                continue
            terms = profile_terms(
                atom.content or ""
            )
            term_counts.update(terms)
        topic_term_counts[topic_id] = term_counts
        document_frequency.update(term_counts.keys())
    topic_count_for_idf = max(1, len(topic_term_counts))

    profiles: list[dict[str, Any]] = []
    updated_count = 0
    embedded_count = 0
    for node in nodes:
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            continue
        topic_id = str(node["topic_id"])
        topic_assignments = assignments_by_topic.get(topic_id, [])
        primary_assignments = [item for item in topic_assignments if str(item.get("item_kind") or "") == "atom"]
        secondary_assignments = [
            item for item in topic_assignments if str(item.get("item_kind") or "") == "atom_secondary"
        ]
        term_counts: Counter[str] = Counter(topic_term_counts.get(topic_id) or {})
        entity_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        confidence_values: list[float] = []
        event_vectors: list[list[float]] = []
        exemplar_rows: list[tuple[float, str]] = []
        for assignment in topic_assignments:
            atom_id = str(assignment.get("item_id") or "")
            atom = atoms_by_id.get(atom_id)
            if atom is None:
                continue
            reason = dict(assignment.get("reason") or {})
            source_counts[_assignment_source(reason)] += 1
            confidence = float(assignment.get("confidence") or 0.0)
            confidence_values.append(confidence)
            terms = profile_terms(
                atom.content or ""
            )
            entity_counts.update(str(item).strip() for item in (atom.canonical_entities or atom.entities or []) if str(item).strip())
            event = events_by_id.get(atom.event_id)
            if event is not None and event.embedding:
                event_vectors.append(event.embedding)
            exemplar_rows.append((confidence + min(1.0, len(terms) / 12.0), atom_id))

        centroid = mean_vector(event_vectors)
        scored_terms = []
        total_terms = max(1, sum(term_counts.values()))
        max_document_frequency = max(1, int(math.floor(topic_count_for_idf * float(max_document_frequency_ratio))))
        filtered_high_df_count = 0
        for term, count in term_counts.items():
            if int(document_frequency.get(term) or 0) > max_document_frequency:
                filtered_high_df_count += 1
                continue
            idf = math.log((1.0 + topic_count_for_idf) / (1.0 + float(document_frequency.get(term) or 0))) + 1.0
            score = (float(count) / total_terms) * idf
            scored_terms.append((score, count, term))
        scored_terms.sort(key=lambda item: (-item[0], -item[1], item[2]))
        top_terms = [term for _score, _count, term in scored_terms[: max(0, int(max_terms))]]
        entity_signature = [term for term, _count in entity_counts.most_common(max(0, int(max_entities)))]
        exemplar_rows.sort(key=lambda item: (-item[0], item[1]))
        exemplar_ids = [atom_id for _score, atom_id in exemplar_rows[: max(1, int(max_exemplars))]]
        coherence_values: list[float] = []
        if centroid:
            for vector in event_vectors:
                coherence_values.append(cosine_similarity(centroid, vector))
        coherence = sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        stats = {
            **dict(node.get("stats") or {}),
            "profile_version": "topic_profile_v1",
            "assignment_count": len(topic_assignments),
            "primary_assignment_count": len(primary_assignments),
            "secondary_assignment_count": len(secondary_assignments),
            "avg_assignment_confidence": round(avg_confidence, 4),
            "event_embedding_count": len(event_vectors),
            "centroid_norm": round(vector_norm(centroid), 6),
            "coherence": round(coherence, 4),
            "top_terms": top_terms,
            "term_ranker": "ctfidf_v0",
            "max_document_frequency_ratio": round(float(max_document_frequency_ratio), 4),
            "max_document_frequency": max_document_frequency,
            "filtered_high_df_term_count": filtered_high_df_count,
            "entity_signature": entity_signature,
            "assignment_source_counts": dict(sorted(source_counts.items())),
        }
        profile_embedding = centroid or node.get("embedding")
        if profile_embedding is None and embedding is not None:
            text = topic_profile_text(node, top_terms=top_terms, entity_signature=entity_signature)
            if text.strip():
                profile_embedding = embedding.embed(text)
                embedded_count += 1
        if write:
            store.upsert_topic_node(
                topic_id=topic_id,
                view_id=view_id,
                parent_id=node.get("parent_id"),
                name=str(node.get("name") or ""),
                description=str(node.get("description") or ""),
                level=int(node.get("level") or 0),
                keywords=list(node.get("keywords") or []),
                exemplar_ids=exemplar_ids,
                stats=stats,
                embedding=profile_embedding,
                metadata={
                    **metadata,
                    "profile_version": "topic_profile_v1",
                    "profile_term_count": len(top_terms),
                    "profile_entity_count": len(entity_signature),
                    "profile_max_document_frequency_ratio": round(float(max_document_frequency_ratio), 4),
                },
            )
            updated_count += 1
        profiles.append(
            {
                "topic_id": topic_id,
                "slug": metadata.get("topic_slug") or metadata.get("evolved_slug") or metadata.get("seed_slug"),
                "name": node.get("name"),
                "assignment_count": len(topic_assignments),
                "primary_assignment_count": len(primary_assignments),
                "secondary_assignment_count": len(secondary_assignments),
                "coherence": round(coherence, 4),
                "avg_assignment_confidence": round(avg_confidence, 4),
                "event_embedding_count": len(event_vectors),
                "top_terms": top_terms[:10],
                "entity_signature": entity_signature[:10],
                "exemplar_ids": exemplar_ids,
            }
        )
    if write:
        store.commit()
    return {
        "view_id": view_id,
        "corpus_id": corpus_id,
        "profile_version": "topic_profile_v1",
        "max_document_frequency_ratio": round(float(max_document_frequency_ratio), 4),
        "topic_count": len(profiles),
        "updated_topic_count": updated_count,
        "embedded_topic_count": embedded_count,
        "profiles": profiles,
    }
