from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .clients import cosine_similarity
from .memory_overlay import overlay_query_features
from .normalize import STOPWORDS, ZH_STOPWORDS, language_aware_content_terms
from .records import MemoryAtomRecord
from .store import SQLiteMemoryStore


TOKEN_RE = re.compile(r"[a-z0-9]+")
TOPIC_GROWTH_STOPWORDS = {
    "about",
    "able",
    "after",
    "again",
    "agree",
    "also",
    "amazing",
    "appreciate",
    "awesome",
    "buddy",
    "because",
    "been",
    "before",
    "being",
    "best",
    "big",
    "brought",
    "called",
    "care",
    "chat",
    "congrats",
    "context",
    "cool",
    "could",
    "day",
    "days",
    "definitely",
    "doing",
    "did",
    "done",
    "especially",
    "experience",
    "feel",
    "feels",
    "felt",
    "from",
    "glad",
    "going",
    "gonna",
    "good",
    "got",
    "gotcha",
    "great",
    "have",
    "help",
    "helped",
    "helpful",
    "hey",
    "huge",
    "important",
    "into",
    "just",
    "like",
    "made",
    "make",
    "makes",
    "many",
    "more",
    "most",
    "much",
    "need",
    "nice",
    "oops",
    "options",
    "people",
    "really",
    "said",
    "says",
    "shared",
    "some",
    "something",
    "sorry",
    "sounds",
    "started",
    "still",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "time",
    "times",
    "today",
    "turn",
    "thanks",
    "thank",
    "want",
    "what",
    "when",
    "which",
    "with",
    "would",
}

TOPIC_GROWTH_GENERIC_ENTITY_TOKENS = {
    "assistant",
    "bot",
    "buddy",
    "cheers",
    "gpt",
    "gotcha",
    "hey",
    "human",
    "thanks",
    "thank",
    "user",
    "wow",
}


@dataclass(frozen=True, slots=True)
class SeedTopic:
    slug: str
    name: str
    description: str
    keywords: tuple[str, ...]


SEED_TOPICS: tuple[SeedTopic, ...] = (
    SeedTopic(
        slug="personal_profile",
        name="Personal Profile",
        description="Stable identity, background, personality, self-description, and profile facts.",
        keywords=(
            "identity",
            "background",
            "personality",
            "profile",
            "age",
            "birthday",
            "family",
            "parent",
            "child",
            "children",
            "transgender",
            "authentic",
        ),
    ),
    SeedTopic(
        slug="relationships",
        name="Relationships",
        description="Friends, partners, family relationships, social support, and interpersonal events.",
        keywords=(
            "friend",
            "friends",
            "partner",
            "relationship",
            "married",
            "single",
            "mom",
            "mother",
            "father",
            "support",
            "supportive",
            "breakup",
        ),
    ),
    SeedTopic(
        slug="work_education",
        name="Work And Education",
        description="Jobs, work plans, career interests, education, study, school, and training.",
        keywords=(
            "work",
            "job",
            "career",
            "occupation",
            "education",
            "study",
            "school",
            "class",
            "training",
            "certification",
            "psychology",
            "counseling",
            "mental",
            "health",
        ),
    ),
    SeedTopic(
        slug="health_emotion",
        name="Health And Emotion",
        description="Physical health, mental health, stress, feelings, mood, coping, and wellbeing.",
        keywords=(
            "health",
            "stress",
            "anxious",
            "anxiety",
            "therapy",
            "emotion",
            "feeling",
            "felt",
            "depress",
            "relax",
            "destress",
            "wellbeing",
        ),
    ),
    SeedTopic(
        slug="hobbies_media",
        name="Hobbies And Media",
        description="Hobbies, arts, music, books, movies, games, sports, photos, and leisure activities.",
        keywords=(
            "hobby",
            "music",
            "movie",
            "book",
            "reading",
            "painting",
            "paint",
            "photo",
            "picture",
            "game",
            "sport",
            "dancing",
            "concert",
        ),
    ),
    SeedTopic(
        slug="travel_places",
        name="Travel And Places",
        description="Trips, places, cities, restaurants, local venues, travel plans, and outdoor locations.",
        keywords=(
            "travel",
            "trip",
            "roadtrip",
            "road",
            "city",
            "place",
            "restaurant",
            "hike",
            "hiking",
            "camping",
            "beach",
            "park",
            "museum",
        ),
    ),
    SeedTopic(
        slug="plans_tasks",
        name="Plans And Tasks",
        description="Future plans, goals, tasks, reminders, scheduling, projects, and intended actions.",
        keywords=(
            "plan",
            "plans",
            "goal",
            "task",
            "project",
            "schedule",
            "appointment",
            "tomorrow",
            "next",
            "future",
            "want",
            "wants",
            "decided",
        ),
    ),
    SeedTopic(
        slug="preferences_opinions",
        name="Preferences And Opinions",
        description="Likes, dislikes, favorites, preferences, beliefs, opinions, and taste.",
        keywords=(
            "like",
            "likes",
            "liked",
            "love",
            "favorite",
            "prefer",
            "preference",
            "opinion",
            "belief",
            "values",
            "interested",
            "enjoy",
        ),
    ),
    SeedTopic(
        slug="events_timeline",
        name="Events And Timeline",
        description="Time-anchored episodes, date-specific events, changes, updates, and sequences.",
        keywords=(
            "today",
            "yesterday",
            "week",
            "month",
            "year",
            "last",
            "before",
            "after",
            "then",
            "again",
            "recent",
            "event",
            "update",
        ),
    ),
    SeedTopic(
        slug="misc",
        name="Miscellaneous",
        description="Memories that do not fit a stronger seed topic yet.",
        keywords=(),
    ),
)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: str, length: int = 16) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def topic_node_id(view_id: str, slug: str) -> str:
    return f"topic_{stable_hash(view_id, slug, length=20)}"


def tokenize(text: str) -> set[str]:
    terms = {
        term
        for term in language_aware_content_terms(text, mode="auto", include_cjk_subgrams=True)
        if term not in STOPWORDS and term not in ZH_STOPWORDS and term not in TOPIC_GROWTH_STOPWORDS
    }
    if terms:
        return terms
    return {
        match.group(0)
        for match in TOKEN_RE.finditer(str(text or "").lower())
        if match.group(0) not in STOPWORDS and match.group(0) not in TOPIC_GROWTH_STOPWORDS
    }


def _topic_route_keywords(node: dict[str, Any]) -> list[str]:
    metadata = dict(node.get("metadata") or {})
    if metadata.get("route_exposure") == "inactive":
        return []
    route_keywords = [str(item).strip() for item in (metadata.get("route_keywords") or []) if str(item).strip()]
    if route_keywords:
        return route_keywords
    return [str(keyword).strip() for keyword in (node.get("keywords") or []) if str(keyword).strip()]


def _topic_allows_primary_assignment(node: dict[str, Any]) -> bool:
    metadata = dict(node.get("metadata") or {})
    return metadata.get("primary_assignment_exposure") != "inactive"


def _topic_is_evolved(node: dict[str, Any]) -> bool:
    metadata = dict(node.get("metadata") or {})
    return bool(metadata.get("evolved_slug") or metadata.get("evolution_source"))


def _topic_profile_quality(
    node: dict[str, Any],
    *,
    embedding_score: float,
    matched_keywords: list[str],
    matched_profile_terms: list[str],
) -> dict[str, Any]:
    stats = dict(node.get("stats") or {})
    metadata = dict(node.get("metadata") or {})
    try:
        assignment_count = int(stats.get("assignment_count") or 0)
    except (TypeError, ValueError):
        assignment_count = 0
    try:
        coherence = float(stats.get("coherence") or 0.0)
    except (TypeError, ValueError):
        coherence = 0.0
    try:
        avg_confidence = float(stats.get("avg_assignment_confidence") or 0.0)
    except (TypeError, ValueError):
        avg_confidence = 0.0
    try:
        growth_distinct_count = int(metadata.get("growth_distinct_count") or 0)
    except (TypeError, ValueError):
        growth_distinct_count = 0
    try:
        growth_score = float(metadata.get("growth_score") or 0.0)
    except (TypeError, ValueError):
        growth_score = 0.0
    return {
        "assignment_count": assignment_count,
        "coherence": round(coherence, 4),
        "avg_assignment_confidence": round(avg_confidence, 4),
        "growth_distinct_count": growth_distinct_count,
        "growth_score": round(growth_score, 4),
        "embedding_score": round(float(embedding_score or 0.0), 4),
        "keyword_match_count": len(matched_keywords),
        "profile_match_count": len(matched_profile_terms),
        "evolved": _topic_is_evolved(node),
    }


def _allow_profile_quality_route(
    node: dict[str, Any],
    *,
    embedding_score: float,
    matched_keywords: list[str],
    matched_profile_terms: list[str],
) -> tuple[bool, str]:
    if matched_keywords:
        return True, "keyword_match"
    if not matched_profile_terms:
        return False, "no_profile_match"

    stats = dict(node.get("stats") or {})
    metadata = dict(node.get("metadata") or {})
    assignment_count = int(stats.get("assignment_count") or 0)
    primary_count = int(stats.get("primary_assignment_count") or 0)
    secondary_count = int(stats.get("secondary_assignment_count") or 0)
    coherence = float(stats.get("coherence") or 0.0)
    growth_distinct_count = int(metadata.get("growth_distinct_count") or 0)
    profile_match_count = len(matched_profile_terms)
    is_evolved = _topic_is_evolved(node)

    if is_evolved:
        if growth_distinct_count < 5:
            return False, "low_growth_support"
        if profile_match_count < 2 and float(embedding_score or 0.0) < 0.42:
            return False, "weak_evolved_profile_match"
        return True, "evolved_profile_quality"

    if assignment_count < 12 and primary_count < 6:
        return False, "low_seed_assignment_support"
    if profile_match_count < 2 and float(embedding_score or 0.0) < 0.4:
        return False, "weak_seed_profile_match"
    if secondary_count > primary_count * 2 and profile_match_count < 3:
        return False, "secondary_dominated_seed_profile"
    if coherence > 0.0 and coherence < 0.68 and profile_match_count < 3:
        return False, "low_seed_profile_coherence"
    return True, "seed_profile_quality"


def _term_is_noisy_topic_label(term: str) -> bool:
    tokens = tokenize(str(term).replace("_", " "))
    if not tokens:
        return True
    if any(token in TOPIC_GROWTH_STOPWORDS or token in STOPWORDS or token in ZH_STOPWORDS for token in tokens):
        return True
    return False


def assign_seed_topic(text: str, entities: list[str] | None = None) -> dict[str, Any]:
    tokens = tokenize(" ".join([text or "", " ".join(entities or [])]))
    scored: list[tuple[int, int, SeedTopic, list[str]]] = []
    for index, topic in enumerate(SEED_TOPICS):
        if topic.slug == "misc":
            continue
        matched = sorted(tokens.intersection(topic.keywords))
        scored.append((len(matched), index, topic, matched))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] <= 0:
        topic = next(seed for seed in SEED_TOPICS if seed.slug == "misc")
        return {
            "slug": topic.slug,
            "confidence": 0.2,
            "reason": {"matched_keywords": [], "strategy": "seed_keyword_fallback"},
        }
    score, _, topic, matched = scored[0]
    confidence = min(0.9, 0.35 + 0.12 * score)
    return {
        "slug": topic.slug,
        "confidence": round(confidence, 3),
        "reason": {"matched_keywords": matched, "strategy": "seed_keyword_match"},
    }


def topic_slug(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    return str(
        metadata.get("topic_slug")
        or metadata.get("evolved_slug")
        or metadata.get("seed_slug")
        or node.get("name")
        or node.get("topic_id")
    )


def topic_node_role(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    if metadata.get("seed_role") == "root":
        return "root"
    if metadata.get("evolved_slug") or metadata.get("evolution_source"):
        return "evolved"
    if metadata.get("seed_slug"):
        return "seed"
    return "unknown"


def topic_node_matches_scope(node: dict[str, Any], topic_scope: str) -> bool:
    scope = str(topic_scope or "all").strip().lower()
    if scope in {"", "all"}:
        return True
    role = topic_node_role(node)
    if scope == "evolved":
        return role == "evolved"
    if scope == "seed":
        return role == "seed"
    if scope == "non_seed":
        return role != "seed"
    return True


def normalize_topic_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or "topic"


def assign_atom_to_topic_view(
    atom: MemoryAtomRecord,
    topic_nodes: list[dict[str, Any]],
    *,
    fallback_slug: str = "misc",
) -> dict[str, Any] | None:
    tokens = tokenize(_atom_assignment_text(atom))
    scored: list[tuple[int, float, int, dict[str, Any], list[str], str]] = []
    fallback_node: dict[str, Any] | None = None
    for index, node in enumerate(topic_nodes):
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            continue
        slug = topic_slug(node)
        if slug == fallback_slug:
            fallback_node = node
        if not _topic_allows_primary_assignment(node):
            continue
        keywords = _topic_route_keywords(node)
        keyword_tokens = tokenize(" ".join(keywords))
        if not keyword_tokens:
            continue
        matched = sorted(tokens.intersection(keyword_tokens))
        if not matched:
            continue
        normalized = len(matched) / max(len(keyword_tokens), 1)
        scored.append((len(matched), normalized, index, node, matched, slug))
    if scored:
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        match_count, normalized, _, node, matched, slug = scored[0]
        confidence = min(0.95, 0.35 + 0.12 * float(match_count) + 0.15 * float(normalized))
        return {
            "topic_id": str(node["topic_id"]),
            "slug": slug,
            "confidence": round(confidence, 3),
            "reason": {
                "matched_keywords": matched,
                "strategy": "topic_view_keyword_match",
                "topic_slug": slug,
            },
        }
    if fallback_node is None:
        return None
    return {
        "topic_id": str(fallback_node["topic_id"]),
        "slug": fallback_slug,
        "confidence": 0.2,
        "reason": {
            "matched_keywords": [],
            "strategy": "topic_view_keyword_fallback",
            "topic_slug": fallback_slug,
        },
    }


def assign_atoms_to_topic_view(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    view_id: str,
    atom_ids: list[str] | None = None,
    limit: int | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    topic_nodes = store.list_topic_nodes(view_id)
    atoms = store.get_atoms_by_ids(atom_ids) if atom_ids is not None else store.list_atoms(corpus_id, limit=limit)
    counts: Counter[str] = Counter()
    confidence_totals: defaultdict[str, float] = defaultdict(float)
    assignments_written = 0
    skipped_atoms: list[str] = []
    for atom in atoms:
        assignment = assign_atom_to_topic_view(atom, topic_nodes)
        if assignment is None:
            skipped_atoms.append(atom.atom_id)
            continue
        slug = str(assignment["slug"])
        assignment_id = f"assign_{stable_hash(view_id, atom.atom_id, length=24)}"
        store.upsert_topic_assignment(
            assignment_id=assignment_id,
            view_id=view_id,
            corpus_id=corpus_id,
            item_kind="atom",
            item_id=atom.atom_id,
            topic_id=str(assignment["topic_id"]),
            confidence=float(assignment["confidence"]),
            reason={
                **dict(assignment.get("reason") or {}),
                "event_id": atom.event_id,
                "incremental": atom_ids is not None,
                "seed_slug": slug,
            },
        )
        assignments_written += 1
        counts[slug] += 1
        confidence_totals[slug] += float(assignment["confidence"])
    if commit:
        store.commit()
    return {
        "view_id": view_id,
        "corpus_id": corpus_id,
        "atoms_seen": len(atoms),
        "assignments_written": assignments_written,
        "skipped_atom_count": len(skipped_atoms),
        "skipped_atom_ids": skipped_atoms[:50],
        "incremental": atom_ids is not None,
        "topic_counts": dict(sorted(counts.items())),
        "avg_confidence_by_topic": {
            slug: round(confidence_totals[slug] / max(count, 1), 4)
            for slug, count in sorted(counts.items())
        },
    }


def route_query_to_topics(
    store: SQLiteMemoryStore,
    *,
    view_id: str,
    query: str,
    top_k: int = 3,
    query_embedding: list[float] | None = None,
    router: str = "keyword",
    topic_scope: str = "all",
) -> list[dict[str, Any]]:
    nodes = store.list_topic_nodes(view_id)
    tokens = tokenize(query)
    profile_query_terms = set(language_aware_content_terms(query, mode="auto", include_cjk_subgrams=True))
    if top_k <= 0:
        return []

    scored: list[dict[str, Any]] = []
    fallback: dict[str, Any] | None = None
    use_overlay = router == "overlay_facet_hybrid"
    query_overlay_features = overlay_query_features(query) if use_overlay else {}
    use_profile = router in {"profile_hybrid", "profile_quality", "overlay_facet_hybrid"}
    use_profile_quality = router == "profile_quality"
    for node in nodes:
        slug = topic_slug(node)
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            continue
        if not topic_node_matches_scope(node, topic_scope):
            continue
        keywords = _topic_route_keywords(node)
        keyword_tokens = tokenize(" ".join(keywords))
        stats = dict(node.get("stats") or {})
        topic_profile_terms = {
            str(term).strip().lower()
            for term in list(stats.get("top_terms") or [])
            if str(term).strip()
        }
        topic_profile_terms.update(
            str(term).strip().lower()
            for term in list(metadata.get("profile_terms") or [])
            if str(term).strip()
        )
        overlay_profile_terms = {
            str(term).strip().lower()
            for term in list(metadata.get("overlay_profile_terms") or [])
            if str(term).strip()
        }
        overlay_facets = {
            str(term).strip().lower()
            for term in (dict(metadata.get("overlay_facets") or {}).keys())
            if str(term).strip()
        }
        if use_overlay and overlay_profile_terms:
            topic_profile_terms.update(overlay_profile_terms)
        language_topic_terms = set(
            language_aware_content_terms(
                " ".join(
                    [
                        str(node.get("name") or ""),
                        str(node.get("description") or ""),
                        " ".join(keywords),
                        " ".join(topic_profile_terms),
                    ]
                ),
                mode="auto",
                include_cjk_subgrams=True,
            )
        )
        if not keywords and not (use_profile and (language_topic_terms or topic_profile_terms)):
            if slug == "misc":
                fallback = _topic_route_payload(node, slug=slug, matched_keywords=[], score=0.0)
            continue
        matched = sorted(tokens.intersection(keyword_tokens))
        profile_matched = sorted(profile_query_terms.intersection(language_topic_terms | topic_profile_terms))
        overlay_facet_matched = sorted(
            facet for facet in overlay_facets if bool(query_overlay_features.get(facet))
        )
        if not matched and not (use_profile and profile_matched):
            continue
        if use_overlay and not matched and not overlay_facet_matched and len(profile_matched) < 2:
            continue
        normalized = len(matched) / max(len(keyword_tokens), 1)
        score = float(len(matched)) + normalized
        embedding_score = 0.0
        if use_profile:
            embedding_score = cosine_similarity(query_embedding or [], node.get("embedding") or [])
            profile_overlap = len(profile_matched) / max(len(language_topic_terms | topic_profile_terms), 1)
            avg_confidence = float(stats.get("avg_assignment_confidence") or 0.0)
            coherence = float(stats.get("coherence") or 0.0)
            coverage = min(1.0, float(stats.get("assignment_count") or 0.0) / 24.0)
            score += (
                len(profile_matched) * 0.8
                + profile_overlap * 2.0
                + max(0.0, embedding_score) * 2.0
                + avg_confidence * 0.35
                + coherence * 0.25
                + coverage * 0.15
            )
            if use_overlay:
                score += len(overlay_facet_matched) * 0.65
                if overlay_facet_matched:
                    score += min(0.55, len(profile_matched) * 0.08)
                elif not matched:
                    score -= 0.55
            if use_profile_quality:
                allowed, quality_reason = _allow_profile_quality_route(
                    node,
                    embedding_score=embedding_score,
                    matched_keywords=matched,
                    matched_profile_terms=profile_matched,
                )
                if not allowed:
                    continue
        payload = _topic_route_payload(node, slug=slug, matched_keywords=matched, score=score)
        payload["topic_role"] = topic_node_role(node)
        payload["topic_scope"] = str(topic_scope or "all")
        if use_profile:
            payload["router"] = (
                "overlay_facet_hybrid_v0"
                if use_overlay
                else "profile_quality_v0"
                if use_profile_quality
                else "profile_hybrid_v0"
            )
            payload["matched_profile_terms"] = profile_matched
            payload["matched_overlay_facets"] = overlay_facet_matched
            payload["profile"] = {
                "coherence": round(float(stats.get("coherence") or 0.0), 4),
                "avg_assignment_confidence": round(float(stats.get("avg_assignment_confidence") or 0.0), 4),
                "assignment_count": int(stats.get("assignment_count") or 0),
                "has_embedding": bool(node.get("embedding")),
                "embedding_score": round(embedding_score, 4),
            }
            payload["profile_quality"] = _topic_profile_quality(
                node,
                embedding_score=embedding_score,
                matched_keywords=matched,
                matched_profile_terms=profile_matched,
            )
            if use_profile_quality:
                payload["profile_quality"]["decision_reason"] = quality_reason
        scored.append(payload)

    scored.sort(key=lambda row: (-float(row["score"]), str(row["name"]), str(row["topic_id"])))
    if scored:
        return scored[:top_k]
    if fallback is not None:
        return [fallback]
    return []


def _topic_route_payload(
    node: dict[str, Any],
    *,
    slug: str,
    matched_keywords: list[str],
    score: float,
) -> dict[str, Any]:
    confidence = 0.2 if score <= 0 else min(0.95, 0.35 + 0.15 * len(matched_keywords))
    return {
        "topic_id": str(node["topic_id"]),
        "slug": slug,
        "name": str(node.get("name") or ""),
        "level": int(node.get("level") or 0),
        "matched_keywords": matched_keywords,
        "score": round(float(score), 4),
        "confidence": round(float(confidence), 3),
        "router": "keyword_shadow_v0",
    }


def topic_tree_outline(store: SQLiteMemoryStore, *, view_id: str) -> dict[str, Any]:
    nodes = store.list_topic_nodes(view_id)
    children_by_parent: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        children_by_parent[node.get("parent_id")].append(node)

    def node_payload(node: dict[str, Any]) -> dict[str, Any]:
        topic_id = str(node["topic_id"])
        children = sorted(
            children_by_parent.get(topic_id, []),
            key=lambda item: (int(item.get("level") or 0), str(item.get("name") or ""), str(item.get("topic_id") or "")),
        )
        return {
            "topic_id": topic_id,
            "slug": topic_slug(node),
            "name": str(node.get("name") or ""),
            "level": int(node.get("level") or 0),
            "keywords": list(node.get("keywords") or []),
            "children": [node_payload(child) for child in children],
        }

    roots = sorted(
        children_by_parent.get(None, []),
        key=lambda item: (int(item.get("level") or 0), str(item.get("name") or ""), str(item.get("topic_id") or "")),
    )
    return {
        "view_id": view_id,
        "node_count": len(nodes),
        "roots": [node_payload(root) for root in roots],
    }


def active_topic_hints_for_text(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    text: str,
    limit: int = 5,
) -> dict[str, Any] | None:
    active_view = store.get_active_memory_view(corpus_id)
    if active_view is None:
        return None
    tokens = tokenize(text)
    scored: list[tuple[int, int, dict[str, Any], list[str]]] = []
    fallback: dict[str, Any] | None = None
    for index, node in enumerate(store.list_topic_nodes(str(active_view["view_id"]))):
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            continue
        slug = topic_slug(node)
        keywords = _topic_route_keywords(node)
        keyword_tokens = tokenize(" ".join(keywords))
        if slug == "misc":
            fallback = node
        matched = sorted(tokens.intersection(keyword_tokens))
        if matched:
            scored.append((len(matched), index, node, matched))
    scored.sort(key=lambda item: (-item[0], item[1]))
    hint_nodes = scored[: max(1, int(limit))]
    if not hint_nodes and fallback is not None:
        hint_nodes = [(0, 0, fallback, [])]
    return {
        "active_view_id": active_view["view_id"],
        "view_name": active_view["name"],
        "hints": [
            {
                "topic_id": str(node["topic_id"]),
                "slug": topic_slug(node),
                "name": str(node.get("name") or ""),
                "description": str(node.get("description") or ""),
                "keywords": list(node.get("keywords") or [])[:12],
                "matched_keywords": matched,
                "level": int(node.get("level") or 0),
            }
            for _score, _index, node, matched in hint_nodes
        ],
    }


def _make_view_id(corpus_id: str, name: str) -> str:
    suffix = uuid.uuid4().hex[:10]
    return f"view_{stable_hash(corpus_id, name, suffix, length=18)}"


def _root_topic_payload(view_id: str, corpus_id: str) -> dict[str, Any]:
    return {
        "topic_id": topic_node_id(view_id, "root"),
        "view_id": view_id,
        "parent_id": None,
        "name": "Memory Root",
        "description": f"Root topic for corpus {corpus_id}.",
        "level": 0,
        "keywords": [],
        "metadata": {"seed_role": "root"},
    }


def create_seed_topic_tree(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    name: str = "seed-topic-tree-v0",
    parent_view_id: str | None = None,
    activate: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = utc_now_iso()
    view_id = _make_view_id(corpus_id, name)
    store.upsert_memory_view(
        view_id=view_id,
        corpus_id=corpus_id,
        parent_view_id=parent_view_id,
        name=name,
        status="active" if activate else "candidate",
        active=False,
        created_at=created_at,
        metadata={
            "view_type": "seed_topic_tree",
            "topic_model": "seeded_adaptive_v0",
            **(metadata or {}),
        },
    )
    root = _root_topic_payload(view_id, corpus_id)
    store.upsert_topic_node(**root)
    root_id = str(root["topic_id"])
    for seed in SEED_TOPICS:
        store.upsert_topic_node(
            topic_id=topic_node_id(view_id, seed.slug),
            view_id=view_id,
            parent_id=root_id,
            name=seed.name,
            description=seed.description,
            level=1,
            keywords=list(seed.keywords),
            metadata={"seed_slug": seed.slug},
        )
    if activate:
        store.promote_memory_view(view_id, promoted_at=created_at)
    store.commit()
    return {
        "view_id": view_id,
        "corpus_id": corpus_id,
        "name": name,
        "active": activate,
        "topic_nodes_written": len(SEED_TOPICS) + 1,
        "root_topic_id": root_id,
    }


def assign_atoms_to_seed_topics(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    view_id: str,
    limit: int | None = None,
) -> dict[str, Any]:
    topic_nodes = store.list_topic_nodes(view_id)
    topic_slugs = {topic_slug(node) for node in topic_nodes}
    if "misc" not in topic_slugs:
        raise ValueError(f"View {view_id} does not contain seed topic nodes.")
    return assign_atoms_to_topic_view(
        store,
        corpus_id=corpus_id,
        view_id=view_id,
        limit=limit,
        commit=True,
    )


def bootstrap_seed_memory_view(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    name: str = "seed-topic-tree-v0",
    parent_view_id: str | None = None,
    activate: bool = False,
    assign_atoms: bool = True,
    assignment_limit: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    view = create_seed_topic_tree(
        store,
        corpus_id=corpus_id,
        name=name,
        parent_view_id=parent_view_id,
        activate=activate,
        metadata=metadata,
    )
    assignment = (
        assign_atoms_to_seed_topics(
            store,
            corpus_id=corpus_id,
            view_id=str(view["view_id"]),
            limit=assignment_limit,
        )
        if assign_atoms
        else {"assignments_written": 0, "topic_counts": {}}
    )
    return {**view, "assignment": assignment}


def grow_topic_tree_from_recent_atoms(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    base_view: dict[str, Any],
    recent_atom_ids: list[str] | None = None,
    name: str = "online-topic-growth-v1",
    max_new_topics: int = 3,
    max_depth: int = 4,
    min_cluster_atoms: int = 3,
    window_atom_limit: int = 80,
    low_confidence_threshold: float = 0.45,
    growth_strategy: str = "global_terms",
    evolved_primary_assignment_enabled: bool = True,
    evolved_primary_assignment_mode: str = "all",
    activate: bool = True,
    trigger: dict[str, Any] | None = None,
    secondary_assignment_enabled: bool = True,
    secondary_max_assignments: int = 50,
    secondary_min_score: float = 3.0,
    secondary_min_term_overlap: int = 2,
    secondary_min_embedding_score: float = 0.0,
    secondary_text_mode: str = "content_entities",
    secondary_max_profile_terms: int = 0,
    secondary_min_score_margin: float = 0.0,
    secondary_min_score_ratio: float = 0.0,
) -> dict[str, Any]:
    base_view_id = str(base_view["view_id"])
    base_nodes = store.list_topic_nodes(base_view_id)
    if not base_nodes:
        return {"status": "no_base_topics", "base_view_id": base_view_id, "added_topic_count": 0}
    atoms = (
        store.get_atoms_by_ids(recent_atom_ids)
        if recent_atom_ids is not None
        else store.list_recent_atoms(corpus_id, limit=window_atom_limit)
    )
    atoms = atoms[-max(1, int(window_atom_limit)) :]
    if not atoms:
        return {"status": "no_recent_atoms", "base_view_id": base_view_id, "added_topic_count": 0}

    node_by_id = {str(node["topic_id"]): node for node in base_nodes}
    existing_slugs = {normalize_topic_slug(topic_slug(node)) for node in base_nodes}
    existing_keyword_tokens = _existing_topic_keyword_tokens(base_nodes)
    assignments = store.list_topic_assignments(base_view_id, item_kind="atom")
    assignment_by_atom = {str(item["item_id"]): item for item in assignments}
    resolved_primary_assignment_mode = _resolve_evolved_primary_assignment_mode(
        evolved_primary_assignment_mode,
        evolved_primary_assignment_enabled=bool(evolved_primary_assignment_enabled),
    )

    candidate_atoms: list[MemoryAtomRecord] = []
    growth_terms_by_atom: dict[str, list[str]] = {}
    for atom in atoms:
        assignment = assignment_by_atom.get(atom.atom_id)
        topic = node_by_id.get(str(assignment.get("topic_id"))) if assignment else None
        slug = normalize_topic_slug(topic_slug(topic)) if topic else "misc"
        confidence = float(assignment.get("confidence") or 0.0) if assignment else 0.0
        growth_terms = _topic_growth_terms(atom, existing_keyword_tokens=existing_keyword_tokens)
        growth_terms_by_atom[atom.atom_id] = growth_terms
        if slug == "misc" or confidence <= low_confidence_threshold or assignment is None or growth_terms:
            candidate_atoms.append(atom)
    if not candidate_atoms:
        return {
            "status": "no_growth_candidates",
            "base_view_id": base_view_id,
            "recent_atom_count": len(atoms),
            "added_topic_count": 0,
        }

    atom_by_id = {atom.atom_id: atom for atom in candidate_atoms}
    window_speaker_tokens = _speaker_tokens_for_atoms(atoms)
    resolved_growth_strategy = str(growth_strategy or "global_terms").strip().lower()
    if resolved_growth_strategy == "node_local":
        ranked_clusters = _rank_node_local_growth_clusters(
            candidate_atoms=candidate_atoms,
            growth_terms_by_atom=growth_terms_by_atom,
            assignment_by_atom=assignment_by_atom,
            node_by_id=node_by_id,
            min_cluster_atoms=min_cluster_atoms,
            low_confidence_threshold=low_confidence_threshold,
            max_depth=max_depth,
        )
    else:
        ranked_clusters = _rank_global_growth_clusters(
            candidate_atoms=candidate_atoms,
            growth_terms_by_atom=growth_terms_by_atom,
            min_cluster_atoms=min_cluster_atoms,
        )

    proposed_topics: list[dict[str, Any]] = []
    used_slugs = set(existing_slugs)
    used_evidence_sets: list[set[str]] = []
    for cluster in ranked_clusters:
        term = str(cluster.get("term") or "")
        atom_ids = list(cluster.get("atom_ids") or [])
        slug = normalize_topic_slug(term)
        if slug in used_slugs:
            continue
        exemplar_ids = list(dict.fromkeys(atom_ids))[:12]
        exemplar_set = set(exemplar_ids)
        if _is_redundant_growth_topic(exemplar_set, used_evidence_sets):
            continue
        parent_topic_id = str(cluster.get("parent_topic_id") or "") or _dominant_parent_topic_id(
            exemplar_ids,
            assignment_by_atom,
            node_by_id,
            max_depth=max_depth,
        )
        keywords = _growth_keywords_for_atoms(
            [atom_by_id[atom_id] for atom_id in exemplar_ids if atom_id in atom_by_id],
            primary=term,
            existing_keyword_tokens=existing_keyword_tokens,
            suppressed_label_tokens=window_speaker_tokens,
            parent_tokens=_node_keyword_tokens(node_by_id[parent_topic_id]) if parent_topic_id in node_by_id else set(),
        )
        if not keywords:
            continue
        route_keywords = _growth_route_keywords(keywords, primary=term)
        used_slugs.add(slug)
        used_evidence_sets.append(exemplar_set)
        proposed_topics.append(
            {
                "slug": slug,
                "name": term.replace("_", " ").title(),
                "description": f"Incrementally grown topic for memories involving {term.replace('_', ' ')}.",
                "keywords": keywords,
                "route_keywords": route_keywords,
                "evidence_atom_ids": exemplar_ids,
                "corpus_id": corpus_id,
                "parent_base_topic_id": parent_topic_id,
                "rationale": str(
                    cluster.get("rationale")
                    or "Repeated or low-confidence recent atoms formed a stable topic cluster."
                ),
                "growth_strategy": resolved_growth_strategy,
                "growth_action": cluster.get("growth_action") or "global_cluster",
                "growth_score": cluster.get("score"),
                "node_density": cluster.get("node_density"),
                "unmapped_density": cluster.get("unmapped_density"),
                "growth_distinct_count": cluster.get("distinct_count"),
                "growth_mention_count": cluster.get("mention_count"),
                "growth_local_ratio": cluster.get("local_ratio"),
            }
        )
        if len(proposed_topics) >= max(0, int(max_new_topics)):
            break
    if not proposed_topics:
        return {
            "status": "no_stable_clusters",
            "base_view_id": base_view_id,
            "recent_atom_count": len(atoms),
            "candidate_atom_count": len(candidate_atoms),
            "added_topic_count": 0,
            "growth_strategy": resolved_growth_strategy,
        }

    return _create_topic_growth_view(
        store,
        corpus_id=corpus_id,
        base_view=base_view,
        base_nodes=base_nodes,
        name=name,
        proposed_topics=proposed_topics,
        activate=activate,
        trigger=trigger or {},
        topic_model=f"online_incremental_growth_{resolved_growth_strategy}_v0",
        evolved_primary_assignment_enabled=bool(evolved_primary_assignment_enabled),
        evolved_primary_assignment_mode=resolved_primary_assignment_mode,
        secondary_assignment_enabled=secondary_assignment_enabled,
        secondary_max_assignments=secondary_max_assignments,
        secondary_min_score=secondary_min_score,
        secondary_min_term_overlap=secondary_min_term_overlap,
        secondary_min_embedding_score=secondary_min_embedding_score,
        secondary_text_mode=secondary_text_mode,
        secondary_max_profile_terms=secondary_max_profile_terms,
        secondary_min_score_margin=secondary_min_score_margin,
        secondary_min_score_ratio=secondary_min_score_ratio,
    )


def _is_redundant_growth_topic(exemplar_ids: set[str], used_evidence_sets: list[set[str]]) -> bool:
    if not exemplar_ids:
        return True
    for used in used_evidence_sets:
        overlap = len(exemplar_ids.intersection(used))
        if overlap <= 0:
            continue
        if overlap / max(1, min(len(exemplar_ids), len(used))) >= 0.75:
            return True
    return False


def _rank_global_growth_clusters(
    *,
    candidate_atoms: list[MemoryAtomRecord],
    growth_terms_by_atom: dict[str, list[str]],
    min_cluster_atoms: int,
) -> list[dict[str, Any]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    cluster_mentions: Counter[str] = Counter()
    for atom in candidate_atoms:
        for term in growth_terms_by_atom.get(atom.atom_id, [])[:10]:
            if _is_low_specificity_growth_term(term):
                continue
            if _term_is_noisy_topic_label(term):
                continue
            if atom.atom_id not in clusters[term]:
                clusters[term].append(atom.atom_id)
            cluster_mentions[term] += _topic_growth_term_count(atom, term)
    ranked = sorted(
        (
            {
                "term": term,
                "atom_ids": atom_ids,
                "score": float(len(set(atom_ids)) * 4 + cluster_mentions[term]),
                "growth_action": "global_cluster",
                "distinct_count": len(set(atom_ids)),
                "mention_count": int(cluster_mentions[term]),
                "local_ratio": 0.0,
                "rationale": "Global repeated recent atoms formed a stable topic cluster.",
            }
            for term, atom_ids in clusters.items()
            if len(set(atom_ids)) >= max(1, int(min_cluster_atoms))
            or cluster_mentions[term] >= max(1, int(min_cluster_atoms))
        ),
        key=lambda item: (-float(item["score"]), str(item["term"])),
    )
    return ranked


def _rank_node_local_growth_clusters(
    *,
    candidate_atoms: list[MemoryAtomRecord],
    growth_terms_by_atom: dict[str, list[str]],
    assignment_by_atom: dict[str, dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    min_cluster_atoms: int,
    low_confidence_threshold: float,
    max_depth: int,
) -> list[dict[str, Any]]:
    atoms_by_node: dict[str, list[MemoryAtomRecord]] = defaultdict(list)
    low_confidence_atoms_by_node: dict[str, set[str]] = defaultdict(set)
    for atom in candidate_atoms:
        assignment = assignment_by_atom.get(atom.atom_id)
        topic_id = str((assignment or {}).get("topic_id") or "")
        if topic_id not in node_by_id:
            misc = next((node for node in node_by_id.values() if normalize_topic_slug(topic_slug(node)) == "misc"), None)
            topic_id = str(misc["topic_id"]) if misc else ""
        if not topic_id:
            continue
        parent_id = _bounded_growth_parent_topic_id(topic_id, node_by_id, max_depth=max_depth)
        if not parent_id:
            continue
        atoms_by_node[parent_id].append(atom)
        confidence = float((assignment or {}).get("confidence") or 0.0)
        if confidence <= float(low_confidence_threshold):
            low_confidence_atoms_by_node[parent_id].add(atom.atom_id)

    child_ids_by_parent: dict[str, set[str]] = defaultdict(set)
    for node_id, node in node_by_id.items():
        parent_id = str(node.get("parent_id") or "")
        if parent_id:
            child_ids_by_parent[parent_id].add(node_id)

    ranked: list[dict[str, Any]] = []
    min_atoms = max(1, int(min_cluster_atoms))
    for parent_id, atoms in atoms_by_node.items():
        parent_node = node_by_id.get(parent_id)
        if parent_node is None:
            continue
        parent_level = int(parent_node.get("level") or 0)
        parent_tokens = _node_keyword_tokens(parent_node)
        sibling_tokens = set().union(
            *[_node_keyword_tokens(node_by_id[child_id]) for child_id in child_ids_by_parent.get(parent_id, set())]
        ) if child_ids_by_parent.get(parent_id) else set()
        speaker_tokens = _speaker_tokens_for_atoms(atoms)
        term_atom_ids: dict[str, list[str]] = defaultdict(list)
        term_mentions: Counter[str] = Counter()
        for atom in atoms:
            for term in growth_terms_by_atom.get(atom.atom_id, [])[:12]:
                if _is_low_specificity_growth_term(term):
                    continue
                if _term_is_noisy_topic_label(term):
                    continue
                if _is_parent_or_sibling_redundant_term(term, parent_tokens=parent_tokens, sibling_tokens=sibling_tokens):
                    continue
                term_tokens = tokenize(str(term).replace("_", " "))
                if term_tokens and term_tokens.intersection(speaker_tokens):
                    continue
                if atom.atom_id not in term_atom_ids[term]:
                    term_atom_ids[term].append(atom.atom_id)
                term_mentions[term] += _topic_growth_term_count(atom, term)
        if not term_atom_ids:
            continue
        node_density = len(atoms)
        unmapped_density = len(low_confidence_atoms_by_node.get(parent_id, set())) / max(1, node_density)
        growth_action = (
            "depth_expansion"
            if not child_ids_by_parent.get(parent_id) or parent_level >= 2
            else "width_expansion"
        )
        for term, atom_ids in term_atom_ids.items():
            distinct_count = len(set(atom_ids))
            mentions = int(term_mentions[term])
            if distinct_count < min_atoms and mentions < min_atoms:
                continue
            local_ratio = distinct_count / max(1, node_density)
            score = distinct_count * 5.0 + mentions + local_ratio * 3.0 + unmapped_density * 2.0
            ranked.append(
                {
                    "term": term,
                    "atom_ids": atom_ids,
                    "parent_topic_id": parent_id,
                    "growth_action": growth_action,
                    "score": round(score, 4),
                    "node_density": node_density,
                    "unmapped_density": round(unmapped_density, 4),
                    "distinct_count": distinct_count,
                    "mention_count": mentions,
                    "local_ratio": round(local_ratio, 4),
                    "rationale": (
                        f"Node-local {growth_action} under {normalize_topic_slug(topic_slug(parent_node))}; "
                        f"density={node_density}, unmapped_density={unmapped_density:.3f}."
                    ),
                }
            )
    ranked.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            -int(item.get("node_density") or 0),
            str(item.get("parent_topic_id") or ""),
            str(item.get("term") or ""),
        )
    )
    return ranked


def _speaker_tokens_for_atoms(atoms: list[MemoryAtomRecord]) -> set[str]:
    tokens: set[str] = set()
    for atom in atoms:
        metadata = dict(atom.metadata or {})
        speaker_text = " ".join(
            str(value or "")
            for value in [
                metadata.get("source_speaker"),
                metadata.get("speaker"),
                " ".join(str(item) for item in (metadata.get("source_speakers") or [])),
            ]
        )
        tokens.update(token for token in tokenize(speaker_text) if token)
    return tokens


def _node_keyword_tokens(node: dict[str, Any]) -> set[str]:
    pieces = [topic_slug(node), str(node.get("name") or ""), " ".join(str(item) for item in (node.get("keywords") or []))]
    return {token for token in tokenize(" ".join(pieces)) if token}


def _is_low_specificity_growth_term(term: str) -> bool:
    tokens = sorted(tokenize(str(term).replace("_", " ")))
    if not tokens:
        return True
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) <= 3:
            return True
        if token.endswith(("ing", "ed")) and len(token) <= 7:
            return True
    return False


def _is_parent_or_sibling_redundant_term(
    term: str,
    *,
    parent_tokens: set[str],
    sibling_tokens: set[str],
) -> bool:
    tokens = set(tokenize(str(term).replace("_", " ")))
    if not tokens:
        return True
    if tokens.issubset(parent_tokens):
        return True
    if len(tokens) == 1 and tokens.issubset(sibling_tokens):
        return True
    return False


def _existing_topic_keyword_tokens(nodes: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for node in nodes:
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            continue
        pieces = [topic_slug(node), str(node.get("name") or ""), " ".join(str(item) for item in (node.get("keywords") or []))]
        tokens.update(token for token in tokenize(" ".join(pieces)) if token)
    return tokens


def _topic_growth_terms(atom: MemoryAtomRecord, *, existing_keyword_tokens: set[str]) -> list[str]:
    metadata = dict(atom.metadata or {})
    speaker_text = " ".join(
        str(value or "")
        for value in [
            metadata.get("source_speaker"),
            metadata.get("speaker"),
            " ".join(str(item) for item in (metadata.get("source_speakers") or [])),
        ]
    )
    speaker_tokens = tokenize(speaker_text) | TOPIC_GROWTH_GENERIC_ENTITY_TOKENS
    terms: list[str] = []
    for entity in list(atom.entities or []) + list(atom.canonical_entities or []):
        entity_slug = normalize_topic_slug(str(entity))
        entity_tokens = set(tokenize(entity_slug))
        if not entity_slug or entity_slug in terms:
            continue
        if entity_slug in TOPIC_GROWTH_GENERIC_ENTITY_TOKENS:
            continue
        if any(token in TOPIC_GROWTH_GENERIC_ENTITY_TOKENS for token in entity_tokens):
            continue
        if entity_tokens and (entity_tokens.issubset(existing_keyword_tokens) or entity_tokens.issubset(speaker_tokens)):
            continue
        if 3 <= len(entity_slug) <= 48:
            terms.append(entity_slug)
        if len(terms) >= 8:
            break
    cleaned_content = _clean_topic_growth_text(atom.content or "")
    raw_tokens = [
        token
        for token in language_aware_content_terms(cleaned_content, mode="auto", include_cjk_subgrams=True)
        if len(token) >= 2
        and token not in TOPIC_GROWTH_STOPWORDS
        and token not in existing_keyword_tokens
        and token not in speaker_tokens
    ]
    for token, _count in Counter(raw_tokens).most_common(12):
        if token not in terms:
            terms.append(token)
        if len(terms) >= 16:
            break
    return terms


def _clean_topic_growth_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("context overlap:"):
            continue
        line = re.sub(r"\b(context overlap|speaker|unknown)\b", " ", line, flags=re.IGNORECASE)
        lines.append(line)
    return "\n".join(lines)


def _topic_growth_term_count(atom: MemoryAtomRecord, term: str) -> int:
    term_tokens = tokenize(str(term).replace("_", " "))
    if not term_tokens:
        return 0
    text = " ".join(
        [
            str(atom.content or ""),
            " ".join(str(entity) for entity in (atom.entities or [])),
            " ".join(str(entity) for entity in (atom.canonical_entities or [])),
        ]
    )
    text_tokens = tokenize(text)
    return sum(1 for token in term_tokens if token in text_tokens)


def _dominant_parent_topic_id(
    atom_ids: list[str],
    assignment_by_atom: dict[str, dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    *,
    max_depth: int,
) -> str | None:
    counts: Counter[str] = Counter()
    for atom_id in atom_ids:
        assignment = assignment_by_atom.get(atom_id)
        if not assignment:
            continue
        topic_id = str(assignment.get("topic_id") or "")
        if topic_id and topic_id in node_by_id:
            counts[topic_id] += 1
    if counts:
        return _bounded_growth_parent_topic_id(counts.most_common(1)[0][0], node_by_id, max_depth=max_depth)
    misc = next((node for node in node_by_id.values() if normalize_topic_slug(topic_slug(node)) == "misc"), None)
    return str(misc["topic_id"]) if misc else None


def _bounded_growth_parent_topic_id(
    topic_id: str,
    node_by_id: dict[str, dict[str, Any]],
    *,
    max_depth: int,
) -> str | None:
    max_parent_level = max(0, int(max_depth) - 1)
    current_id = str(topic_id or "")
    seen: set[str] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        node = node_by_id.get(current_id)
        if node is None:
            break
        metadata = dict(node.get("metadata") or {})
        level = int(node.get("level") or 0)
        if metadata.get("seed_role") == "root":
            break
        if level <= max_parent_level:
            return current_id
        current_id = str(node.get("parent_id") or "")
    misc = next((node for node in node_by_id.values() if normalize_topic_slug(topic_slug(node)) == "misc"), None)
    return str(misc["topic_id"]) if misc else None


def _growth_keywords_for_atoms(
    atoms: list[MemoryAtomRecord],
    *,
    primary: str,
    existing_keyword_tokens: set[str],
    suppressed_label_tokens: set[str] | None = None,
    parent_tokens: set[str] | None = None,
) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    parent_tokens = parent_tokens or set()
    suppressed_label_tokens = suppressed_label_tokens or set()

    def add(value: str) -> None:
        cleaned = normalize_topic_slug(value).replace("_", " ")
        if not cleaned or cleaned in seen:
            return
        tokens = tokenize(cleaned)
        if tokens and tokens.issubset(existing_keyword_tokens):
            return
        if tokens and tokens.issubset(parent_tokens):
            return
        if tokens and tokens.intersection(suppressed_label_tokens):
            return
        if _is_low_specificity_growth_term(cleaned):
            return
        if _term_is_noisy_topic_label(cleaned):
            return
        seen.add(cleaned)
        keywords.append(cleaned)

    add(primary)
    counts: Counter[str] = Counter()
    for atom in atoms:
        atom_terms = set(_topic_growth_terms(atom, existing_keyword_tokens=existing_keyword_tokens))
        for term in atom_terms:
            counts[term.replace("_", " ")] += 1
    for term, _count in counts.most_common(16):
        if _count < 2 and len(atoms) >= 5:
            continue
        add(term)
        if len(keywords) >= 12:
            break
    return keywords[:12]


def _growth_route_keywords(keywords: list[str], *, primary: str, limit: int = 1) -> list[str]:
    route_keywords: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        cleaned = normalize_topic_slug(value).replace("_", " ")
        if not cleaned or cleaned in seen:
            return
        if _is_low_specificity_growth_term(cleaned):
            return
        seen.add(cleaned)
        route_keywords.append(cleaned)

    add(primary)
    for keyword in keywords:
        add(keyword)
        if len(route_keywords) >= max(1, int(limit)):
            break
    return route_keywords or [normalize_topic_slug(primary).replace("_", " ")]


def _resolve_evolved_primary_assignment_mode(
    mode: str,
    *,
    evolved_primary_assignment_enabled: bool,
) -> str:
    resolved = str(mode or "all").strip().lower()
    if resolved not in {"all", "none", "quality_v0", "quality_v1"}:
        resolved = "all"
    if not evolved_primary_assignment_enabled and resolved == "all":
        resolved = "none"
    return resolved


def _evidence_label_support(
    atoms: list[MemoryAtomRecord],
    *,
    label_tokens: set[str],
) -> dict[str, Any]:
    content_support = 0
    entity_support = 0
    standalone_entity_support = 0
    speaker_mentions = 0
    source_speaker_mentions = 0
    entity_values: list[str] = []
    for atom in atoms:
        content_tokens = tokenize(_atom_assignment_text(atom))
        if label_tokens.issubset(content_tokens):
            content_support += 1
        metadata = dict(atom.metadata or {})
        speakers = {
            normalize_topic_slug(str(value))
            for value in [
                metadata.get("source_speaker"),
                metadata.get("speaker"),
                *list(metadata.get("source_speakers") or []),
            ]
            if str(value or "").strip()
        }
        entities = [
            str(entity or "")
            for entity in list(atom.entities or []) + list(atom.canonical_entities or [])
            if str(entity or "").strip()
        ]
        entity_values.extend(entities)
        entity_tokens = tokenize(" ".join(entities))
        if label_tokens and label_tokens.issubset(entity_tokens):
            entity_support += 1
        for entity in entities:
            normalized_entity = normalize_topic_slug(entity).replace("_", " ")
            entity_token_set = tokenize(normalized_entity)
            if label_tokens and label_tokens == entity_token_set:
                standalone_entity_support += 1
                break
        label_slugs = {normalize_topic_slug(token) for token in label_tokens if token}
        if label_slugs and label_slugs.issubset(speakers):
            source_speaker_mentions += 1
        if label_tokens and label_tokens.issubset(tokenize(" ".join(speakers))):
            speaker_mentions += 1
    return {
        "content_support_count": content_support,
        "entity_support_count": entity_support,
        "standalone_entity_support_count": standalone_entity_support,
        "speaker_mention_count": speaker_mentions,
        "source_speaker_mention_count": source_speaker_mentions,
        "entity_values_sample": sorted(set(entity_values))[:20],
    }


def _corpus_label_support(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    label_tokens: set[str],
) -> dict[str, Any]:
    atoms = store.list_atoms(corpus_id)
    entity_support = 0
    exact_entity_support = 0
    content_support = 0
    for atom in atoms:
        entities = [
            str(entity or "")
            for entity in list(atom.entities or []) + list(atom.canonical_entities or [])
            if str(entity or "").strip()
        ]
        if label_tokens and label_tokens.issubset(tokenize(" ".join(entities))):
            entity_support += 1
        if label_tokens and label_tokens.issubset(tokenize(_atom_assignment_text(atom))):
            content_support += 1
        for entity in entities:
            if label_tokens == tokenize(normalize_topic_slug(entity).replace("_", " ")):
                exact_entity_support += 1
                break
    atom_count = len(atoms)
    return {
        "corpus_atom_count": atom_count,
        "corpus_entity_support_count": entity_support,
        "corpus_exact_entity_support_count": exact_entity_support,
        "corpus_content_support_count": content_support,
        "corpus_entity_support_ratio": round(entity_support / max(1, atom_count), 4),
        "corpus_exact_entity_support_ratio": round(exact_entity_support / max(1, atom_count), 4),
        "corpus_content_support_ratio": round(content_support / max(1, atom_count), 4),
    }


def _evolved_primary_assignment_gate(
    store: SQLiteMemoryStore,
    topic: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    resolved_mode = str(mode or "all").strip().lower()
    route_keywords = [str(item).strip() for item in (topic.get("route_keywords") or []) if str(item).strip()]
    label = route_keywords[0] if route_keywords else str(topic.get("slug") or topic.get("name") or "")
    label_text = normalize_topic_slug(label).replace("_", " ")
    label_tokens = tokenize(label_text)
    evidence_atom_ids = [str(atom_id) for atom_id in (topic.get("evidence_atom_ids") or []) if str(atom_id)]
    if resolved_mode == "all":
        return {
            "mode": resolved_mode,
            "decision": "active",
            "reason": "primary_assignment_mode_all",
            "label": label_text,
            "evidence_atom_count": len(evidence_atom_ids),
        }
    if resolved_mode == "none":
        return {
            "mode": resolved_mode,
            "decision": "inactive",
            "reason": "primary_assignment_mode_none",
            "label": label_text,
            "evidence_atom_count": len(evidence_atom_ids),
        }
    if not label_tokens or _is_low_specificity_growth_term(label_text) or _term_is_noisy_topic_label(label_text):
        return {
            "mode": resolved_mode,
            "decision": "inactive",
            "reason": "label_not_specific_enough",
            "label": label_text,
            "evidence_atom_count": len(evidence_atom_ids),
        }

    fetched_atoms = store.get_atoms_by_ids(evidence_atom_ids)
    atom_by_id = {atom.atom_id: atom for atom in fetched_atoms}
    ordered_atoms = [atom_by_id[atom_id] for atom_id in evidence_atom_ids if atom_id in atom_by_id]
    evidence_count = len(ordered_atoms)
    support = _evidence_label_support(ordered_atoms, label_tokens=label_tokens)
    corpus_support = _corpus_label_support(
        store,
        corpus_id=str(topic.get("corpus_id") or ""),
        label_tokens=label_tokens,
    ) if str(topic.get("corpus_id") or "").strip() else {
        "corpus_atom_count": 0,
        "corpus_entity_support_count": 0,
        "corpus_exact_entity_support_count": 0,
        "corpus_content_support_count": 0,
        "corpus_entity_support_ratio": 0.0,
        "corpus_exact_entity_support_ratio": 0.0,
        "corpus_content_support_ratio": 0.0,
    }
    content_support = int(support["content_support_count"])
    entity_support = int(support["entity_support_count"])
    standalone_entity_support = int(support["standalone_entity_support_count"])
    source_speaker_mentions = int(support["source_speaker_mention_count"])
    support_ratio = content_support / max(1, evidence_count)
    entity_support_ratio = entity_support / max(1, evidence_count)
    distinct_count = int(topic.get("growth_distinct_count") or evidence_count)
    mention_count = int(topic.get("growth_mention_count") or 0)
    score = float(topic.get("growth_score") or 0.0)
    min_content_support = max(4, math.ceil(evidence_count * 0.55))
    stable_evidence = evidence_count >= 5 and distinct_count >= 4 and score >= 20.0
    corpus_entity_ratio = float(corpus_support.get("corpus_exact_entity_support_ratio") or 0.0)
    globally_dominant_entity = corpus_entity_ratio >= 0.5 and len(label_tokens) == 1
    if resolved_mode == "quality_v1":
        decision_active = (
            stable_evidence
            and not globally_dominant_entity
            and (
                standalone_entity_support >= 1
                or entity_support_ratio >= 0.35
                or (len(label_tokens) > 1 and entity_support >= 2)
            )
        )
    else:
        decision_active = (
            stable_evidence
            and (
                entity_support >= 2
                or content_support >= min_content_support
                or (len(label_tokens) > 1 and content_support >= 3)
            )
        )
    if decision_active:
        reason = "stable_evidence_supported_topic"
    elif evidence_count < 5:
        reason = "insufficient_evidence_atoms"
    elif distinct_count < 4:
        reason = "insufficient_distinct_atoms"
    elif score < 20.0:
        reason = "growth_score_below_threshold"
    elif resolved_mode == "quality_v1" and globally_dominant_entity:
        reason = "label_is_globally_dominant_entity"
    elif resolved_mode == "quality_v1":
        reason = "weak_entity_support_for_primary_assignment"
    else:
        reason = "weak_label_support_in_evidence"
    return {
        "mode": resolved_mode,
        "decision": "active" if decision_active else "inactive",
        "reason": reason,
        "label": label_text,
        "label_tokens": sorted(label_tokens),
        "evidence_atom_count": evidence_count,
        "content_support_count": content_support,
        "entity_support_count": entity_support,
        "standalone_entity_support_count": standalone_entity_support,
        "source_speaker_mention_count": source_speaker_mentions,
        "support_ratio": round(support_ratio, 4),
        "entity_support_ratio": round(entity_support_ratio, 4),
        "growth_distinct_count": distinct_count,
        "growth_mention_count": mention_count,
        "growth_score": score,
        "entity_values_sample": support["entity_values_sample"],
        **corpus_support,
    }


def _create_topic_growth_view(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    base_view: dict[str, Any],
    base_nodes: list[dict[str, Any]],
    name: str,
    proposed_topics: list[dict[str, Any]],
    activate: bool,
    trigger: dict[str, Any],
    topic_model: str,
    evolved_primary_assignment_enabled: bool,
    evolved_primary_assignment_mode: str,
    secondary_assignment_enabled: bool,
    secondary_max_assignments: int,
    secondary_min_score: float,
    secondary_min_term_overlap: int,
    secondary_min_embedding_score: float,
    secondary_text_mode: str,
    secondary_max_profile_terms: int,
    secondary_min_score_margin: float,
    secondary_min_score_ratio: float,
) -> dict[str, Any]:
    now = utc_now_iso()
    base_view_id = str(base_view["view_id"])
    view_id = f"view_{stable_hash(corpus_id, name, uuid.uuid4().hex[:10], length=18)}"
    store.upsert_memory_view(
        view_id=view_id,
        corpus_id=corpus_id,
        parent_view_id=base_view_id,
        name=name,
        status="active" if activate else "candidate",
        active=False,
        created_at=now,
        metadata={
            "view_type": "evolved_topic_tree",
            "topic_model": topic_model,
            "base_view_id": base_view_id,
            "trigger": trigger,
        },
    )

    base_to_candidate: dict[str, str] = {}
    for node in sorted(base_nodes, key=lambda item: (int(item.get("level") or 0), str(item.get("topic_id") or ""))):
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            candidate_topic_id = topic_node_id(view_id, "root")
        else:
            candidate_topic_id = topic_node_id(view_id, normalize_topic_slug(topic_slug(node)))
        base_to_candidate[str(node["topic_id"])] = candidate_topic_id

    copied_nodes = 0
    for node in sorted(base_nodes, key=lambda item: (int(item.get("level") or 0), str(item.get("topic_id") or ""))):
        metadata = dict(node.get("metadata") or {})
        candidate_topic_id = base_to_candidate[str(node["topic_id"])]
        parent_id = base_to_candidate.get(str(node.get("parent_id"))) if node.get("parent_id") else None
        node_metadata = {**metadata, "base_topic_id": node["topic_id"], "base_view_id": base_view_id}
        if metadata.get("seed_role") != "root":
            node_metadata.setdefault("topic_slug", normalize_topic_slug(topic_slug(node)))
        store.upsert_topic_node(
            topic_id=candidate_topic_id,
            view_id=view_id,
            parent_id=parent_id,
            name=str(node.get("name") or ""),
            description=str(node.get("description") or ""),
            level=int(node.get("level") or 0),
            keywords=list(node.get("keywords") or []),
            exemplar_ids=list(node.get("exemplar_ids") or []),
            stats=dict(node.get("stats") or {}),
            embedding=node.get("embedding"),
            metadata=node_metadata,
        )
        copied_nodes += 1

    misc_base_topic_id = next(
        (str(node["topic_id"]) for node in base_nodes if normalize_topic_slug(topic_slug(node)) == "misc"),
        "",
    )
    resolved_primary_assignment_mode = _resolve_evolved_primary_assignment_mode(
        evolved_primary_assignment_mode,
        evolved_primary_assignment_enabled=bool(evolved_primary_assignment_enabled),
    )
    slug_to_topic_id: dict[str, str] = {}
    added_topics: list[dict[str, Any]] = []
    for topic in proposed_topics:
        slug = normalize_topic_slug(str(topic.get("slug") or topic.get("name") or ""))
        topic_id = topic_node_id(view_id, slug)
        parent_base_topic_id = str(topic.get("parent_base_topic_id") or misc_base_topic_id)
        parent_id = base_to_candidate.get(parent_base_topic_id) or topic_node_id(view_id, "root")
        parent_node = next((node for node in base_nodes if base_to_candidate.get(str(node["topic_id"])) == parent_id), None)
        level = int(parent_node.get("level") or 0) + 1 if parent_node else 1
        primary_assignment_gate = _evolved_primary_assignment_gate(
            store,
            topic,
            mode=resolved_primary_assignment_mode,
        )
        store.upsert_topic_node(
            topic_id=topic_id,
            view_id=view_id,
            parent_id=parent_id,
            name=str(topic.get("name") or slug.replace("_", " ").title()),
            description=str(topic.get("description") or ""),
            level=level,
            keywords=list(topic.get("keywords") or []),
            exemplar_ids=list(topic.get("evidence_atom_ids") or []),
            stats={"evidence_atom_count": len(topic.get("evidence_atom_ids") or [])},
            metadata={
                "topic_slug": slug,
                "evolved_slug": slug,
                "evolution_source": "online_incremental_growth",
                "base_view_id": base_view_id,
                "parent_base_topic_id": parent_base_topic_id or None,
                "rationale": topic.get("rationale") or "",
                "route_keywords": list(topic.get("route_keywords") or []),
                "route_exposure": "active" if list(topic.get("route_keywords") or []) else "inactive",
                "primary_assignment_exposure": str(primary_assignment_gate.get("decision") or "inactive"),
                "primary_assignment_gate": primary_assignment_gate,
                "growth_strategy": topic.get("growth_strategy") or "",
                "growth_action": topic.get("growth_action") or "",
                "growth_score": topic.get("growth_score"),
                "node_density": topic.get("node_density"),
                "unmapped_density": topic.get("unmapped_density"),
                "growth_distinct_count": topic.get("growth_distinct_count"),
                "growth_mention_count": topic.get("growth_mention_count"),
                "growth_local_ratio": topic.get("growth_local_ratio"),
            },
        )
        slug_to_topic_id[slug] = topic_id
        added_topics.append(
            {
                "slug": slug,
                "topic_id": topic_id,
                "parent_topic_id": parent_id,
                "level": level,
                "keywords": list(topic.get("keywords") or []),
                "evidence_atom_ids": list(topic.get("evidence_atom_ids") or []),
            }
        )

    reassignment_by_atom: dict[str, str] = {}
    for topic in proposed_topics:
        slug = normalize_topic_slug(str(topic.get("slug") or topic.get("name") or ""))
        for atom_id in topic.get("evidence_atom_ids") or []:
            if slug in slug_to_topic_id:
                reassignment_by_atom[str(atom_id)] = slug

    copied_primary_assignments = 0
    copied_secondary_assignments = 0
    explicit_secondary_atoms: set[str] = set()
    for assignment in store.list_topic_assignments(base_view_id, item_kind="atom"):
        atom_id = str(assignment["item_id"])
        target_topic_id = base_to_candidate.get(str(assignment["topic_id"]))
        reason = dict(assignment.get("reason") or {})
        if atom_id in reassignment_by_atom:
            slug = reassignment_by_atom[atom_id]
            evolved_topic_id = slug_to_topic_id.get(slug)
            if evolved_topic_id:
                store.upsert_topic_assignment(
                    assignment_id=f"assign_{stable_hash(view_id, 'secondary', atom_id, evolved_topic_id, length=24)}",
                    view_id=view_id,
                    corpus_id=corpus_id,
                    item_kind="atom_secondary",
                    item_id=atom_id,
                    topic_id=evolved_topic_id,
                    confidence=max(0.72, float(assignment.get("confidence") or 0.0)),
                    reason={
                        "evolution_secondary_assignment": True,
                        "evolved_slug": slug,
                        "strategy": "online_incremental_topic_growth_evidence",
                        "base_assignment_id": assignment.get("assignment_id"),
                        "base_view_id": base_view_id,
                    },
                )
                explicit_secondary_atoms.add(atom_id)
        if not target_topic_id:
            continue
        store.upsert_topic_assignment(
            assignment_id=f"assign_{stable_hash(view_id, atom_id, length=24)}",
            view_id=view_id,
            corpus_id=corpus_id,
            item_kind="atom",
            item_id=atom_id,
            topic_id=target_topic_id,
            confidence=float(assignment.get("confidence") or 0.0),
            reason={**reason, "base_assignment_id": assignment.get("assignment_id")},
        )
        copied_primary_assignments += 1

    for assignment in store.list_topic_assignments(base_view_id, item_kind="atom_secondary"):
        atom_id = str(assignment["item_id"])
        target_topic_id = base_to_candidate.get(str(assignment["topic_id"]))
        if not target_topic_id:
            continue
        reason = dict(assignment.get("reason") or {})
        store.upsert_topic_assignment(
            assignment_id=f"assign_{stable_hash(view_id, 'secondary', atom_id, target_topic_id, length=24)}",
            view_id=view_id,
            corpus_id=corpus_id,
            item_kind="atom_secondary",
            item_id=atom_id,
            topic_id=target_topic_id,
            confidence=float(assignment.get("confidence") or 0.0),
            reason={
                **reason,
                "base_assignment_id": assignment.get("assignment_id"),
                "base_view_id": base_view_id,
                "copied_from_base_secondary": True,
            },
        )
        copied_secondary_assignments += 1

    secondary_assigned_atoms: set[str] = set(explicit_secondary_atoms)
    profile_secondary_atoms: set[str] = set()
    profile_candidate_count = 0
    if secondary_assignment_enabled and int(secondary_max_assignments) > 0:
        profile_result = _assign_online_growth_secondary_profiles(
            store,
            corpus_id=corpus_id,
            view_id=view_id,
            base_view_id=base_view_id,
            added_topics=added_topics,
            slug_to_topic_id=slug_to_topic_id,
            already_assigned_atom_ids=explicit_secondary_atoms,
            max_assignments=int(secondary_max_assignments),
            min_score=float(secondary_min_score),
            min_term_overlap=int(secondary_min_term_overlap),
            min_embedding_score=float(secondary_min_embedding_score),
            text_mode=str(secondary_text_mode or "content_entities"),
            max_profile_terms=int(secondary_max_profile_terms),
            min_score_margin=float(secondary_min_score_margin),
            min_score_ratio=float(secondary_min_score_ratio),
        )
        profile_secondary_atoms = set(profile_result["secondary_assigned_atom_ids"])
        secondary_assigned_atoms.update(profile_secondary_atoms)
        profile_candidate_count = int(profile_result["secondary_candidate_count"])

    if activate:
        store.promote_memory_view(view_id, promoted_at=now)
    store.commit()
    return {
        "status": "promoted" if activate else "candidate_created",
        "view_id": view_id,
        "base_view_id": base_view_id,
        "name": name,
        "active": activate,
        "copied_topic_nodes": copied_nodes,
        "added_topic_count": len(added_topics),
        "added_topics": added_topics,
        "assignment_mode": "primary_preserved_secondary_evolved",
        "assignments_written": copied_primary_assignments
        + copied_secondary_assignments
        + len(explicit_secondary_atoms)
        + len(profile_secondary_atoms),
        "primary_assignments_written": copied_primary_assignments,
        "copied_secondary_assignments_written": copied_secondary_assignments,
        "explicit_secondary_assignment_count": len(explicit_secondary_atoms),
        "profile_secondary_assignment_count": len(profile_secondary_atoms),
        "reassigned_atom_count": 0,
        "reassigned_atom_ids": [],
        "secondary_assignment_enabled": bool(secondary_assignment_enabled),
        "secondary_assignment_count": len(secondary_assigned_atoms),
        "secondary_candidate_count": profile_candidate_count,
        "evolved_primary_assignment_mode": resolved_primary_assignment_mode,
        "secondary_assigned_atom_ids": sorted(secondary_assigned_atoms),
        "secondary_assignment_config": {
            "max_assignments": int(secondary_max_assignments),
            "min_score": float(secondary_min_score),
            "min_term_overlap": int(secondary_min_term_overlap),
            "min_embedding_score": float(secondary_min_embedding_score),
            "text_mode": str(secondary_text_mode or "content_entities"),
            "max_profile_terms": int(secondary_max_profile_terms),
            "min_score_margin": float(secondary_min_score_margin),
            "min_score_ratio": float(secondary_min_score_ratio),
        },
        "created_at": now,
    }


def _atom_assignment_text(atom: MemoryAtomRecord) -> str:
    return " ".join(
        [
            str(atom.atom_type or ""),
            str(atom.content or ""),
            " ".join(str(entity) for entity in (atom.entities or [])),
            " ".join(str(entity) for entity in (atom.canonical_entities or [])),
        ]
    )


def _atom_profile_text(atom: MemoryAtomRecord, *, text_mode: str = "content_entities") -> str:
    if text_mode == "content_only":
        return atom.content or ""
    return " ".join(
        [
            atom.content or "",
            " ".join(atom.entities or []),
            " ".join(atom.canonical_entities or []),
        ]
    )


def _topic_profile_terms(text: str) -> set[str]:
    return {
        term
        for term in language_aware_content_terms(text, mode="auto", include_cjk_subgrams=True)
        if term and term not in TOPIC_GROWTH_STOPWORDS
    }


def _mean_vector(vectors: list[list[float]]) -> list[float] | None:
    valid = [vector for vector in vectors if vector]
    if not valid:
        return None
    width = len(valid[0])
    same_width = [vector for vector in valid if len(vector) == width]
    if not same_width:
        return None
    return [sum(vector[index] for vector in same_width) / len(same_width) for index in range(width)]


def _build_online_growth_topic_profiles(
    store: SQLiteMemoryStore,
    *,
    added_topics: list[dict[str, Any]],
    text_mode: str,
    max_profile_terms: int,
) -> dict[str, dict[str, Any]]:
    evidence_atom_ids = sorted(
        {
            str(atom_id).strip()
            for topic in added_topics
            for atom_id in (topic.get("evidence_atom_ids") or [])
            if str(atom_id).strip()
        }
    )
    atoms_by_id = {atom.atom_id: atom for atom in store.get_atoms_by_ids(evidence_atom_ids)}
    events_by_id = store.get_events_by_ids(
        sorted({atom.event_id for atom in atoms_by_id.values() if atom.event_id})
    )
    profiles: dict[str, dict[str, Any]] = {}
    for topic in added_topics:
        slug = str(topic.get("slug") or "").strip()
        if not slug:
            continue
        term_counts: Counter[str] = Counter()
        for keyword in topic.get("keywords") or []:
            for term in _topic_profile_terms(str(keyword)):
                term_counts[term] += 3
        event_vectors: list[list[float]] = []
        exemplar_ids: list[str] = []
        for atom_id in topic.get("evidence_atom_ids") or []:
            atom = atoms_by_id.get(str(atom_id))
            if atom is None:
                continue
            exemplar_ids.append(atom.atom_id)
            for term in _topic_profile_terms(_atom_profile_text(atom, text_mode=text_mode)):
                term_counts[term] += 1
            event = events_by_id.get(atom.event_id)
            if event is not None and event.embedding:
                event_vectors.append(event.embedding)
        selected_items = sorted(term_counts.items(), key=lambda item: (-item[1], item[0]))
        max_terms = max(0, int(max_profile_terms))
        if max_terms > 0:
            selected_items = selected_items[:max_terms]
        selected_terms = {term for term, _weight in selected_items}
        profiles[slug] = {
            "terms": selected_terms,
            "term_weights": {term: weight for term, weight in term_counts.items() if term in selected_terms},
            "centroid": _mean_vector(event_vectors),
            "exemplar_ids": exemplar_ids,
            "event_embedding_count": len(event_vectors),
            "raw_term_count": len(term_counts),
            "profile_term_count": len(selected_terms),
            "max_profile_terms": max_terms,
        }
    return profiles


def _score_online_growth_profile_assignment(
    *,
    atom_terms: set[str],
    event_embedding: list[float] | None,
    profile: dict[str, Any],
) -> tuple[float, int, float, list[str]]:
    profile_terms = set(profile.get("terms") or set())
    matched = sorted(atom_terms.intersection(profile_terms))
    if not matched:
        return 0.0, 0, 0.0, []
    term_weights = dict(profile.get("term_weights") or {})
    weighted_overlap = sum(float(term_weights.get(term) or 1.0) for term in matched)
    overlap_norm = len(matched) / max(1, min(len(profile_terms), 16))
    centroid = profile.get("centroid")
    embedding_score = cosine_similarity(event_embedding or [], centroid or []) if centroid and event_embedding else 0.0
    score = weighted_overlap + overlap_norm * 2.0 + max(0.0, embedding_score) * 1.5
    return score, len(matched), embedding_score, matched


def _assign_online_growth_secondary_profiles(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    view_id: str,
    base_view_id: str,
    added_topics: list[dict[str, Any]],
    slug_to_topic_id: dict[str, str],
    already_assigned_atom_ids: set[str],
    max_assignments: int,
    min_score: float,
    min_term_overlap: int,
    min_embedding_score: float,
    text_mode: str,
    max_profile_terms: int,
    min_score_margin: float,
    min_score_ratio: float,
) -> dict[str, Any]:
    profiles = _build_online_growth_topic_profiles(
        store,
        added_topics=added_topics,
        text_mode=text_mode,
        max_profile_terms=max_profile_terms,
    )
    atoms = store.list_atoms(corpus_id)
    events_by_id = store.get_events_by_ids(sorted({atom.event_id for atom in atoms if atom.event_id}))
    candidates: list[tuple[float, str, list[str], str, dict[str, Any]]] = []
    for atom in atoms:
        if atom.atom_id in already_assigned_atom_ids:
            continue
        atom_terms = _topic_profile_terms(_atom_profile_text(atom, text_mode=text_mode))
        if not atom_terms:
            continue
        event = events_by_id.get(atom.event_id)
        event_embedding = event.embedding if event is not None else None
        best: tuple[float, str, list[str], dict[str, Any]] | None = None
        second_best: tuple[float, str] | None = None
        for topic in added_topics:
            slug = str(topic.get("slug") or "").strip()
            profile = profiles.get(slug) or {}
            score, term_overlap, embedding_score, matched = _score_online_growth_profile_assignment(
                atom_terms=atom_terms,
                event_embedding=event_embedding,
                profile=profile,
            )
            if term_overlap < max(1, int(min_term_overlap)):
                continue
            if score < float(min_score):
                continue
            if (
                float(min_embedding_score) > 0.0
                and profile.get("centroid")
                and event_embedding
                and embedding_score < float(min_embedding_score)
            ):
                continue
            detail = {
                "profile_score": round(float(score), 4),
                "profile_term_overlap": int(term_overlap),
                "profile_embedding_score": round(float(embedding_score), 4),
                "profile_event_embedding_count": int(profile.get("event_embedding_count") or 0),
                "profile_term_count": int(profile.get("profile_term_count") or len(profile.get("terms") or [])),
                "profile_raw_term_count": int(profile.get("raw_term_count") or len(profile.get("terms") or [])),
            }
            if best is None or score > best[0]:
                if best is not None:
                    second_best = (best[0], best[1])
                best = (score, slug, matched, detail)
            elif second_best is None or score > second_best[0]:
                second_best = (score, slug)
        if best is None:
            continue
        score, slug, matched, detail = best
        second_score = float(second_best[0]) if second_best is not None else 0.0
        score_margin = float(score) - second_score
        score_ratio = (float(score) / second_score) if second_score > 0.0 else math.inf
        if float(min_score_margin) > 0.0 and score_margin < float(min_score_margin):
            continue
        if float(min_score_ratio) > 0.0 and second_score > 0.0 and score_ratio < float(min_score_ratio):
            continue
        candidates.append(
            (
                score,
                slug,
                matched,
                atom.atom_id,
                {
                    **detail,
                    "profile_second_topic_slug": second_best[1] if second_best is not None else "",
                    "profile_second_score": round(second_score, 4),
                    "profile_score_margin": round(score_margin, 4),
                    "profile_score_ratio": None if math.isinf(score_ratio) else round(score_ratio, 4),
                    "profile_auto_text_mode": text_mode,
                    "profile_auto_min_score_margin": float(min_score_margin),
                    "profile_auto_min_score_ratio": float(min_score_ratio),
                    "profile_auto_max_profile_terms": int(max_profile_terms),
                },
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
    assigned_atom_ids: list[str] = []
    assigned_by_topic: Counter[str] = Counter()
    for score, slug, matched, atom_id, detail in candidates[: max(0, int(max_assignments))]:
        topic_id = slug_to_topic_id.get(slug)
        if not topic_id:
            continue
        store.upsert_topic_assignment(
            assignment_id=f"assign_{stable_hash(view_id, 'secondary', atom_id, topic_id, length=24)}",
            view_id=view_id,
            corpus_id=corpus_id,
            item_kind="atom_secondary",
            item_id=atom_id,
            topic_id=topic_id,
            confidence=0.7,
            reason={
                "evolution_secondary_assignment": True,
                "evolved_slug": slug,
                "matched_keywords": matched,
                "strategy": "online_growth_profile_match",
                "auto_reassignment_mode": "profile",
                "auto_reassignment_score": round(float(score), 4),
                "base_view_id": base_view_id,
                **detail,
            },
        )
        assigned_atom_ids.append(atom_id)
        assigned_by_topic[slug] += 1
    return {
        "secondary_candidate_count": len(candidates),
        "secondary_assigned_atom_ids": assigned_atom_ids,
        "secondary_counts_by_topic": dict(sorted(assigned_by_topic.items())),
    }
