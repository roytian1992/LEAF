from __future__ import annotations

import argparse
import json
import math
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import stable_hash, tokenize, topic_node_id, utc_now_iso
from leaf.clients import ChatClient, cosine_similarity, extract_json_object
from leaf.config import load_config
from leaf.normalize import language_aware_content_terms
from leaf.records import MemoryAtomRecord
from leaf.store import SQLiteMemoryStore


STOPWORDS = {
    "about",
    "after",
    "also",
    "asked",
    "because",
    "been",
    "before",
    "called",
    "could",
    "from",
    "have",
    "into",
    "many",
    "mentioned",
    "more",
    "most",
    "other",
    "really",
    "recently",
    "said",
    "shared",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "times",
    "what",
    "when",
    "which",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a candidate evolved topic view from topic-routing shadow misses.",
    )
    parser.add_argument("--config", default="", help="Required for --strategy llm.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--base-view-id", default="", help="Defaults to the active memory view.")
    parser.add_argument("--selfqa", required=True)
    parser.add_argument("--shadow-eval-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--proposal-json",
        default="",
        help="Optional previous evolve_topic_view_from_shadow output. Reuses its normalized proposal to avoid a new LLM proposal.",
    )
    parser.add_argument("--name", default="evolved-topic-shadow-v1")
    parser.add_argument("--strategy", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--max-new-topics", type=int, default=4)
    parser.add_argument("--max-miss-rows", type=int, default=12)
    parser.add_argument(
        "--max-auto-reassignments",
        type=int,
        default=50,
        help="Maximum keyword-driven atom reassignments beyond explicit proposal atoms. 0 disables auto reassignment.",
    )
    parser.add_argument(
        "--min-auto-reassign-keyword-matches",
        type=int,
        default=2,
        help="Minimum distinct keyword-token matches required for keyword-driven atom reassignment.",
    )
    parser.add_argument(
        "--auto-reassignment-mode",
        choices=["keyword", "profile"],
        default="keyword",
        help=(
            "How to expand proposal evidence atoms into additional atom assignments. "
            "keyword preserves the previous keyword-overlap behavior; profile uses proposal evidence terms "
            "and event embedding centroids to make a more conservative topic-profile match."
        ),
    )
    parser.add_argument(
        "--profile-auto-min-score",
        type=float,
        default=3.0,
        help="For --auto-reassignment-mode profile, require this minimum profile score.",
    )
    parser.add_argument(
        "--profile-auto-min-term-overlap",
        type=int,
        default=2,
        help="For --auto-reassignment-mode profile, require this many profile-term overlaps.",
    )
    parser.add_argument(
        "--profile-auto-min-embedding-score",
        type=float,
        default=0.0,
        help="For --auto-reassignment-mode profile, require this cosine score when both vectors exist. 0 disables.",
    )
    parser.add_argument(
        "--profile-auto-text-mode",
        choices=["content_entities", "content_only"],
        default="content_entities",
        help=(
            "For --auto-reassignment-mode profile, controls text used to build and match topic profiles. "
            "content_entities preserves prior behavior; content_only reduces entity-list noise."
        ),
    )
    parser.add_argument(
        "--profile-auto-min-score-margin",
        type=float,
        default=0.0,
        help=(
            "For --auto-reassignment-mode profile, require best topic score to exceed the second-best score "
            "by this absolute margin. 0 disables."
        ),
    )
    parser.add_argument(
        "--profile-auto-min-score-ratio",
        type=float,
        default=0.0,
        help=(
            "For --auto-reassignment-mode profile, require best topic score / second-best score to meet this ratio. "
            "0 disables; rows with no positive second-best score pass."
        ),
    )
    parser.add_argument(
        "--profile-auto-max-profile-terms",
        type=int,
        default=0,
        help=(
            "For --auto-reassignment-mode profile, keep only this many highest-weight terms per evolved topic profile. "
            "0 preserves prior full-profile behavior."
        ),
    )
    parser.add_argument(
        "--max-total-reassignments",
        type=int,
        default=0,
        help="Maximum total reassigned atoms. 0 disables this absolute cap.",
    )
    parser.add_argument(
        "--max-reassigned-atom-ratio",
        type=float,
        default=0.15,
        help="Maximum total reassigned atoms as a fraction of base atom assignments. 0 disables this cap.",
    )
    parser.add_argument(
        "--preserve-base-assignments",
        action="store_true",
        help=(
            "Keep copied base atom assignments as primary topic labels and write evolved reassignments as "
            "item_kind=atom_secondary. This preserves seed-topic bridges while allowing evolved-topic retrieval."
        ),
    )
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--record-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_shadow_rows(path: str | Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    misses = []
    for row in rows:
        shadow = row.get("topic_routing_shadow") or {}
        if shadow and not bool(shadow.get("topic_path_hit")):
            misses.append(row)
        if len(misses) >= limit:
            break
    return misses


def topic_slug(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    value = (
        metadata.get("topic_slug")
        or metadata.get("evolved_slug")
        or metadata.get("seed_slug")
        or node.get("name")
        or node.get("topic_id")
    )
    return normalize_slug(str(value))


def normalize_slug(value: str) -> str:
    lowered = str(value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered or "evolved_topic"


def normalize_keywords(values: list[Any], *, limit: int = 16) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        token = re.sub(r"[^a-z0-9 ]+", " ", token).strip()
        if not token or token in seen or token in STOPWORDS:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def task_by_id(selfqa_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in selfqa_rows if row.get("task_id")}


def build_miss_payload(rows: list[dict[str, Any]], selfqa_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        task = selfqa_by_id.get(str(row.get("task_id"))) or {}
        evidence = (task.get("metadata") or {}).get("source_evidence") or []
        payload.append(
            {
                "task_id": row.get("task_id"),
                "task_type": row.get("task_type"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "gold_topic_slugs": (row.get("topic_routing_shadow") or {}).get("gold_topic_slugs") or [],
                "routed_topic_slugs": (row.get("topic_routing_shadow") or {}).get("routed_topic_slugs") or [],
                "gold_atom_ids": row.get("gold_atom_ids") or [],
                "evidence": [
                    {
                        "atom_id": item.get("atom_id"),
                        "atom_content": item.get("atom_content"),
                        "event_text": item.get("event_text"),
                        "entities": item.get("entities") or [],
                    }
                    for item in evidence
                ],
            }
        )
    return payload


def propose_with_llm(
    llm: ChatClient,
    *,
    existing_topics: list[dict[str, Any]],
    misses: list[dict[str, Any]],
    max_new_topics: int,
) -> dict[str, Any]:
    topic_payload = [
        {
            "slug": topic_slug(topic),
            "name": topic.get("name"),
            "description": topic.get("description"),
            "keywords": topic.get("keywords") or [],
        }
        for topic in existing_topics
        if dict(topic.get("metadata") or {}).get("seed_role") != "root"
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You propose conservative topic-tree changes for a memory system. "
                "Return only one valid JSON object. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Given topic-routing misses, propose small candidate topic changes that would make future routing "
                "and atom assignment more coherent. Prefer adding a small number of useful topics or reassigning "
                "misplaced atoms. Do not delete existing topics.\n\n"
                "Return this JSON shape:\n"
                "{\n"
                '  "new_topics": [\n'
                '    {"slug": "snake_case", "name": "...", "description": "...", '
                '"keywords": ["..."], "evidence_atom_ids": ["..."], "rationale": "..."}\n'
                "  ],\n"
                '  "reassignments": [\n'
                '    {"atom_id": "...", "topic_slug": "snake_case", "reason": "..."}\n'
                "  ]\n"
                "}\n\n"
                f"max_new_topics: {max_new_topics}\n"
                f"existing_topics: {json.dumps(topic_payload, ensure_ascii=False)}\n"
                f"misses: {json.dumps(misses, ensure_ascii=False)}"
            ),
        },
    ]
    return extract_json_object(llm.text(messages, max_tokens=1400, temperature=0.0))


def propose_heuristic(misses: list[dict[str, Any]], max_new_topics: int) -> dict[str, Any]:
    atom_text: dict[str, str] = {}
    for miss in misses:
        for evidence in miss.get("evidence") or []:
            atom_id = str(evidence.get("atom_id") or "").strip()
            text = " ".join([str(evidence.get("atom_content") or ""), str(evidence.get("event_text") or "")])
            if atom_id and text:
                atom_text[atom_id] = text
    words = Counter(
        token
        for text in atom_text.values()
        for token in tokenize(text)
        if len(token) > 3 and token not in STOPWORDS
    )
    top_words = [word for word, _ in words.most_common(8)]
    slug = normalize_slug("_".join(top_words[:2]) or "evolved_missed_topic")
    return {
        "new_topics": [
            {
                "slug": slug,
                "name": slug.replace("_", " ").title(),
                "description": "Topic proposed from repeated topic-routing misses.",
                "keywords": top_words,
                "evidence_atom_ids": sorted(atom_text),
                "rationale": "Heuristic topic from frequent missed evidence terms.",
            }
        ][:max_new_topics],
        "reassignments": [
            {"atom_id": atom_id, "topic_slug": slug, "reason": "Heuristic reassignment from missed evidence."}
            for atom_id in sorted(atom_text)
        ],
    }


def normalize_proposal(raw: dict[str, Any], existing_slugs: set[str], max_new_topics: int) -> dict[str, Any]:
    new_topics: list[dict[str, Any]] = []
    topic_slugs = set(existing_slugs)
    for item in raw.get("new_topics") or []:
        if not isinstance(item, dict):
            continue
        slug = normalize_slug(str(item.get("slug") or item.get("name") or ""))
        if slug in topic_slugs:
            continue
        topic_slugs.add(slug)
        keywords = normalize_keywords(list(item.get("keywords") or []))
        if not keywords:
            keywords = normalize_keywords(str(item.get("description") or "").split())
        new_topics.append(
            {
                "slug": slug,
                "name": str(item.get("name") or slug.replace("_", " ").title()).strip(),
                "description": str(item.get("description") or "").strip() or "Evolved topic from shadow-routing misses.",
                "keywords": keywords,
                "evidence_atom_ids": [
                    str(atom_id).strip()
                    for atom_id in (item.get("evidence_atom_ids") or [])
                    if str(atom_id).strip()
                ],
                "rationale": str(item.get("rationale") or "").strip(),
            }
        )
        if len(new_topics) >= max_new_topics:
            break
    known_slugs = set(existing_slugs) | {topic["slug"] for topic in new_topics}
    reassignments: list[dict[str, Any]] = []
    for item in raw.get("reassignments") or []:
        if not isinstance(item, dict):
            continue
        atom_id = str(item.get("atom_id") or "").strip()
        slug = normalize_slug(str(item.get("topic_slug") or ""))
        if atom_id and slug in known_slugs:
            reassignments.append(
                {
                    "atom_id": atom_id,
                    "topic_slug": slug,
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
    return {"new_topics": new_topics, "reassignments": reassignments}


def _effective_reassignment_limit(
    *,
    base_assignment_count: int,
    max_total_reassignments: int,
    max_reassigned_atom_ratio: float,
) -> int | None:
    limits: list[int] = []
    if max_total_reassignments > 0:
        limits.append(int(max_total_reassignments))
    if max_reassigned_atom_ratio > 0 and base_assignment_count > 0:
        limits.append(max(1, math.floor(float(base_assignment_count) * float(max_reassigned_atom_ratio))))
    return min(limits) if limits else None


def _keyword_tokens(keywords: list[str]) -> set[str]:
    return {
        token
        for token in tokenize(" ".join(keywords))
        if token and len(token) > 2 and token not in STOPWORDS
    }


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


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in language_aware_content_terms(text, mode="auto", include_cjk_subgrams=True)
        if term and term not in STOPWORDS
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


def _build_added_topic_profiles(
    store: SQLiteMemoryStore,
    *,
    added_topics: list[dict[str, Any]],
    profile_auto_text_mode: str,
    profile_auto_max_profile_terms: int,
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
            for term in _content_terms(str(keyword)):
                term_counts[term] += 3
        event_vectors: list[list[float]] = []
        exemplar_ids: list[str] = []
        for atom_id in topic.get("evidence_atom_ids") or []:
            atom = atoms_by_id.get(str(atom_id))
            if atom is None:
                continue
            exemplar_ids.append(atom.atom_id)
            for term in _content_terms(_atom_profile_text(atom, text_mode=profile_auto_text_mode)):
                term_counts[term] += 1
            event = events_by_id.get(atom.event_id)
            if event is not None and event.embedding:
                event_vectors.append(event.embedding)
        selected_items = sorted(term_counts.items(), key=lambda item: (-item[1], item[0]))
        max_profile_terms = max(0, int(profile_auto_max_profile_terms))
        if max_profile_terms > 0:
            selected_items = selected_items[:max_profile_terms]
        selected_terms = {term for term, _ in selected_items}
        profiles[slug] = {
            "terms": selected_terms,
            "term_weights": {term: weight for term, weight in term_counts.items() if term in selected_terms},
            "centroid": _mean_vector(event_vectors),
            "exemplar_ids": exemplar_ids,
            "event_embedding_count": len(event_vectors),
            "raw_term_count": len(term_counts),
            "profile_term_count": len(selected_terms),
            "max_profile_terms": max_profile_terms,
        }
    return profiles


def _score_profile_auto_assignment(
    *,
    atom: MemoryAtomRecord,
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


def _add_reassignment(
    reassignments: list[dict[str, str]],
    seen_atom_ids: set[str],
    *,
    atom_id: str,
    topic_slug: str,
    reason: str,
) -> None:
    cleaned_atom_id = str(atom_id or "").strip()
    cleaned_slug = str(topic_slug or "").strip()
    if not cleaned_atom_id or not cleaned_slug or cleaned_atom_id in seen_atom_ids:
        return
    seen_atom_ids.add(cleaned_atom_id)
    reassignments.append(
        {
            "atom_id": cleaned_atom_id,
            "topic_slug": cleaned_slug,
            "reason": str(reason or "").strip() or "proposal_reassignment",
        }
    )


def create_candidate_view(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    base_view: dict[str, Any],
    name: str,
    proposal: dict[str, Any],
    activate: bool,
    max_auto_reassignments: int,
    min_auto_reassign_keyword_matches: int,
    auto_reassignment_mode: str,
    profile_auto_min_score: float,
    profile_auto_min_term_overlap: int,
    profile_auto_min_embedding_score: float,
    profile_auto_text_mode: str,
    profile_auto_min_score_margin: float,
    profile_auto_min_score_ratio: float,
    profile_auto_max_profile_terms: int,
    max_total_reassignments: int,
    max_reassigned_atom_ratio: float,
    preserve_base_assignments: bool,
) -> dict[str, Any]:
    now = utc_now_iso()
    assignment_mode = "primary_seed_secondary_evolved" if preserve_base_assignments else "replace_on_reassignment"
    view_id = f"view_{stable_hash(corpus_id, name, uuid.uuid4().hex[:10], length=18)}"
    store.upsert_memory_view(
        view_id=view_id,
        corpus_id=corpus_id,
        parent_view_id=str(base_view["view_id"]),
        name=name,
        status="active" if activate else "candidate",
        active=False,
        created_at=now,
        metadata={
            "view_type": "evolved_topic_tree",
            "topic_model": "shadow_miss_evolved_v0",
            "base_view_id": base_view["view_id"],
            "assignment_mode": assignment_mode,
            "auto_reassignment_mode": auto_reassignment_mode,
            "profile_auto_text_mode": profile_auto_text_mode,
            "profile_auto_min_score_margin": profile_auto_min_score_margin,
            "profile_auto_min_score_ratio": profile_auto_min_score_ratio,
            "profile_auto_max_profile_terms": profile_auto_max_profile_terms,
        },
    )

    base_nodes = store.list_topic_nodes(str(base_view["view_id"]))
    root_node = next((node for node in base_nodes if dict(node.get("metadata") or {}).get("seed_role") == "root"), None)
    root_topic_id = topic_node_id(view_id, "root")
    store.upsert_topic_node(
        topic_id=root_topic_id,
        view_id=view_id,
        parent_id=None,
        name=str(root_node.get("name") if root_node else "Memory Root"),
        description=str(root_node.get("description") if root_node else f"Root topic for corpus {corpus_id}."),
        level=0,
        keywords=[],
        metadata={"seed_role": "root", "base_topic_id": root_node.get("topic_id") if root_node else None},
    )

    slug_to_topic_id: dict[str, str] = {"root": root_topic_id}
    base_topic_to_candidate: dict[str, str] = {}
    copied_nodes = 1
    for node in base_nodes:
        metadata = dict(node.get("metadata") or {})
        if metadata.get("seed_role") == "root":
            base_topic_to_candidate[str(node["topic_id"])] = root_topic_id
            continue
        slug = topic_slug(node)
        candidate_topic_id = topic_node_id(view_id, slug)
        slug_to_topic_id[slug] = candidate_topic_id
        base_topic_to_candidate[str(node["topic_id"])] = candidate_topic_id
        new_metadata = {
            **metadata,
            "topic_slug": slug,
            "base_topic_id": node["topic_id"],
            "base_view_id": base_view["view_id"],
        }
        store.upsert_topic_node(
            topic_id=candidate_topic_id,
            view_id=view_id,
            parent_id=root_topic_id,
            name=str(node.get("name") or slug),
            description=str(node.get("description") or ""),
            level=int(node.get("level") or 1),
            keywords=list(node.get("keywords") or []),
            exemplar_ids=list(node.get("exemplar_ids") or []),
            stats=dict(node.get("stats") or {}),
            metadata=new_metadata,
        )
        copied_nodes += 1

    added_topics: list[dict[str, Any]] = []
    for topic in proposal.get("new_topics") or []:
        slug = str(topic["slug"])
        topic_id = topic_node_id(view_id, slug)
        slug_to_topic_id[slug] = topic_id
        store.upsert_topic_node(
            topic_id=topic_id,
            view_id=view_id,
            parent_id=root_topic_id,
            name=str(topic["name"]),
            description=str(topic["description"]),
            level=1,
            keywords=list(topic.get("keywords") or []),
            exemplar_ids=list(topic.get("evidence_atom_ids") or []),
            metadata={
                "topic_slug": slug,
                "evolved_slug": slug,
                "evolution_source": "topic_shadow_miss",
                "base_view_id": base_view["view_id"],
                "rationale": topic.get("rationale") or "",
            },
        )
        added_topics.append(
            {
                "slug": slug,
                "topic_id": topic_id,
                "keywords": topic.get("keywords") or [],
                "evidence_atom_ids": topic.get("evidence_atom_ids") or [],
            }
        )

    explicit_reassignments: list[dict[str, str]] = []
    explicit_atom_ids: set[str] = set()
    for topic in proposal.get("new_topics") or []:
        slug = str(topic["slug"])
        for atom_id in topic.get("evidence_atom_ids") or []:
            _add_reassignment(
                explicit_reassignments,
                explicit_atom_ids,
                atom_id=str(atom_id),
                topic_slug=slug,
                reason="proposal_evidence_atom",
            )
    for reassignment in proposal.get("reassignments") or []:
        _add_reassignment(
            explicit_reassignments,
            explicit_atom_ids,
            atom_id=str(reassignment["atom_id"]),
            topic_slug=str(reassignment["topic_slug"]),
            reason=str(reassignment.get("reason") or "proposal_reassignment"),
        )

    base_assignments = store.list_topic_assignments(str(base_view["view_id"]), item_kind="atom")
    reassignment_limit = _effective_reassignment_limit(
        base_assignment_count=len(base_assignments),
        max_total_reassignments=max_total_reassignments,
        max_reassigned_atom_ratio=max_reassigned_atom_ratio,
    )
    explicit_reassignment_input_count = len(explicit_reassignments)
    if reassignment_limit is not None and len(explicit_reassignments) > reassignment_limit:
        explicit_reassignments = explicit_reassignments[:reassignment_limit]
    truncated_explicit_reassignment_count = explicit_reassignment_input_count - len(explicit_reassignments)
    reassignment_by_atom: dict[str, dict[str, str]] = {
        item["atom_id"]: {"topic_slug": item["topic_slug"], "reason": item["reason"]}
        for item in explicit_reassignments
    }

    copied_assignments = 0
    secondary_assignments_written = 0
    reassigned_atoms: set[str] = set()
    for assignment in base_assignments:
        atom_id = str(assignment["item_id"])
        target_topic_id = base_topic_to_candidate.get(str(assignment["topic_id"]))
        reason = dict(assignment.get("reason") or {})
        if atom_id in reassignment_by_atom:
            slug = reassignment_by_atom[atom_id]["topic_slug"]
            evolved_topic_id = slug_to_topic_id.get(slug, target_topic_id)
            if preserve_base_assignments:
                reason = {
                    **reason,
                    "evolution_preserved_primary": True,
                    "evolved_slug": slug,
                    "evolution_reason": reassignment_by_atom[atom_id]["reason"],
                }
                if evolved_topic_id and evolved_topic_id != target_topic_id:
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
                            "evolution_reason": reassignment_by_atom[atom_id]["reason"],
                            "base_assignment_id": assignment.get("assignment_id"),
                            "base_view_id": base_view["view_id"],
                        },
                    )
                    secondary_assignments_written += 1
                    reassigned_atoms.add(atom_id)
            else:
                target_topic_id = evolved_topic_id
                reason = {
                    **reason,
                    "evolution_reassignment": True,
                    "evolved_slug": slug,
                    "evolution_reason": reassignment_by_atom[atom_id]["reason"],
                }
                reassigned_atoms.add(atom_id)
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
            reason={
                **reason,
                "base_assignment_id": assignment.get("assignment_id"),
                "base_view_id": base_view["view_id"],
            },
        )
        copied_assignments += 1

    # Reassign additional matching atoms to new topics. The default keyword mode
    # preserves prior behavior; profile mode is a more conservative memory-modeling
    # expansion from proposal evidence terms and event centroids.
    auto_reassignment_mode = str(auto_reassignment_mode or "keyword").strip()
    if auto_reassignment_mode not in {"keyword", "profile"}:
        raise ValueError(f"Unknown auto_reassignment_mode: {auto_reassignment_mode}")
    auto_budget = max(0, int(max_auto_reassignments))
    if reassignment_limit is not None:
        auto_budget = min(auto_budget, max(0, reassignment_limit - len(reassigned_atoms)))
    min_keyword_matches = max(1, int(min_auto_reassign_keyword_matches))
    topic_keyword_tokens = {
        str(topic["slug"]): _keyword_tokens(list(topic.get("keywords") or []))
        for topic in added_topics
    }
    topic_profiles = (
        _build_added_topic_profiles(
            store,
            added_topics=added_topics,
            profile_auto_text_mode=str(profile_auto_text_mode or "content_entities"),
            profile_auto_max_profile_terms=profile_auto_max_profile_terms,
        )
        if auto_reassignment_mode == "profile"
        else {}
    )
    auto_candidates: list[tuple[float, str, list[str], str, dict[str, Any]]] = []
    atoms = store.list_atoms(corpus_id)
    events_by_id = (
        store.get_events_by_ids(sorted({atom.event_id for atom in atoms if atom.event_id}))
        if auto_reassignment_mode == "profile"
        else {}
    )
    for atom in atoms:
        if atom.atom_id in reassigned_atoms:
            continue
        atom_text = _atom_profile_text(atom, text_mode=str(profile_auto_text_mode or "content_entities"))
        atom_tokens = tokenize(atom_text)
        atom_terms = _content_terms(atom_text)
        event = events_by_id.get(atom.event_id)
        event_embedding = event.embedding if event is not None else None
        best: tuple[float, str, list[str], dict[str, Any]] | None = None
        second_best: tuple[float, str] | None = None
        for topic in added_topics:
            slug = str(topic["slug"])
            if auto_reassignment_mode == "profile":
                profile = topic_profiles.get(slug) or {}
                score, term_overlap, embedding_score, matched = _score_profile_auto_assignment(
                    atom=atom,
                    atom_terms=atom_terms,
                    event_embedding=event_embedding,
                    profile=profile,
                )
                if term_overlap < max(1, int(profile_auto_min_term_overlap)):
                    continue
                if score < float(profile_auto_min_score):
                    continue
                if (
                    float(profile_auto_min_embedding_score) > 0.0
                    and profile.get("centroid")
                    and event_embedding
                    and embedding_score < float(profile_auto_min_embedding_score)
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
            else:
                keyword_tokens = topic_keyword_tokens.get(slug, set())
                matched = sorted(atom_tokens.intersection(keyword_tokens))
                if len(matched) >= min_keyword_matches:
                    score = float(len(matched))
                    detail = {"keyword_match_count": len(matched)}
                    if best is None or score > best[0]:
                        if best is not None:
                            second_best = (best[0], best[1])
                        best = (score, slug, matched, detail)
                    elif second_best is None or score > second_best[0]:
                        second_best = (score, slug)
        if best is None:
            continue
        score, slug, matched, detail = best
        if auto_reassignment_mode == "profile":
            second_score = float(second_best[0]) if second_best is not None else 0.0
            score_margin = float(score) - second_score
            score_ratio = (float(score) / second_score) if second_score > 0.0 else math.inf
            if float(profile_auto_min_score_margin) > 0.0 and score_margin < float(profile_auto_min_score_margin):
                continue
            if (
                float(profile_auto_min_score_ratio) > 0.0
                and second_score > 0.0
                and score_ratio < float(profile_auto_min_score_ratio)
            ):
                continue
            detail = {
                **detail,
                "profile_second_topic_slug": second_best[1] if second_best is not None else "",
                "profile_second_score": round(second_score, 4),
                "profile_score_margin": round(score_margin, 4),
                "profile_score_ratio": None if math.isinf(score_ratio) else round(score_ratio, 4),
                "profile_auto_text_mode": str(profile_auto_text_mode or "content_entities"),
                "profile_auto_min_score_margin": float(profile_auto_min_score_margin),
                "profile_auto_min_score_ratio": float(profile_auto_min_score_ratio),
                "profile_auto_max_profile_terms": int(profile_auto_max_profile_terms),
            }
        auto_candidates.append((score, str(slug), matched, atom.atom_id, detail))

    auto_candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
    auto_reassigned_atom_count = 0
    for score, slug, matched, atom_id, detail in auto_candidates[:auto_budget]:
        topic_id = slug_to_topic_id[slug]
        store.upsert_topic_assignment(
            assignment_id=(
                f"assign_{stable_hash(view_id, 'secondary', atom_id, topic_id, length=24)}"
                if preserve_base_assignments
                else f"assign_{stable_hash(view_id, atom_id, length=24)}"
            ),
            view_id=view_id,
            corpus_id=corpus_id,
            item_kind="atom_secondary" if preserve_base_assignments else "atom",
            item_id=atom_id,
            topic_id=topic_id,
            confidence=0.7,
            reason={
                "evolution_reassignment": not preserve_base_assignments,
                "evolution_secondary_assignment": preserve_base_assignments,
                "evolved_slug": slug,
                "matched_keywords": matched,
                "strategy": (
                    "evolved_profile_match"
                    if auto_reassignment_mode == "profile"
                    else "evolved_keyword_match"
                ),
                "auto_reassignment_mode": auto_reassignment_mode,
                "auto_reassignment_score": round(float(score), 4),
                **detail,
                "base_view_id": base_view["view_id"],
            },
        )
        reassigned_atoms.add(atom_id)
        if preserve_base_assignments:
            secondary_assignments_written += 1
        auto_reassigned_atom_count += 1

    if activate:
        store.promote_memory_view(view_id, promoted_at=now)
    store.commit()
    return {
        "view_id": view_id,
        "base_view_id": base_view["view_id"],
        "name": name,
        "active": activate,
        "copied_topic_nodes": copied_nodes,
        "added_topic_count": len(added_topics),
        "added_topics": added_topics,
        "assignment_mode": assignment_mode,
        "assignments_written": copied_assignments + secondary_assignments_written,
        "primary_assignments_written": copied_assignments,
        "secondary_assignments_written": secondary_assignments_written,
        "base_assignment_count": len(base_assignments),
        "explicit_reassignment_count": len(explicit_reassignments),
        "truncated_explicit_reassignment_count": truncated_explicit_reassignment_count,
        "auto_reassigned_atom_count": auto_reassigned_atom_count,
        "auto_reassignment_candidate_count": len(auto_candidates),
        "auto_reassignment_budget": auto_budget,
        "auto_reassignment_mode": auto_reassignment_mode,
        "min_auto_reassign_keyword_matches": min_keyword_matches,
        "profile_auto_min_score": profile_auto_min_score,
        "profile_auto_min_term_overlap": profile_auto_min_term_overlap,
        "profile_auto_min_embedding_score": profile_auto_min_embedding_score,
        "profile_auto_text_mode": str(profile_auto_text_mode or "content_entities"),
        "profile_auto_min_score_margin": profile_auto_min_score_margin,
        "profile_auto_min_score_ratio": profile_auto_min_score_ratio,
        "profile_auto_max_profile_terms": profile_auto_max_profile_terms,
        "reassignment_limit": reassignment_limit,
        "max_auto_reassignments": max_auto_reassignments,
        "max_total_reassignments": max_total_reassignments,
        "max_reassigned_atom_ratio": max_reassigned_atom_ratio,
        "reassigned_atom_count": len(reassigned_atoms),
        "reassigned_atom_ids": sorted(reassigned_atoms),
        "created_at": now,
    }


def main() -> None:
    args = parse_args()
    store = SQLiteMemoryStore(args.db)
    try:
        base_view = store.get_memory_view(args.base_view_id) if args.base_view_id else store.get_active_memory_view(args.corpus_id)
        if base_view is None:
            raise RuntimeError("No base memory view found.")
        selfqa_rows = load_jsonl(args.selfqa)
        misses = build_miss_payload(load_shadow_rows(args.shadow_eval_report, args.max_miss_rows), task_by_id(selfqa_rows))
        existing_topics = store.list_topic_nodes(str(base_view["view_id"]))
        existing_slugs = {
            topic_slug(topic)
            for topic in existing_topics
            if dict(topic.get("metadata") or {}).get("seed_role") != "root"
        }
        if args.proposal_json:
            raw_proposal = json.loads(Path(args.proposal_json).read_text(encoding="utf-8")).get("proposal") or {}
        elif args.strategy == "llm":
            if not args.config:
                raise RuntimeError("--config is required for --strategy llm")
            config = load_config(args.config)
            memory_cfg = config.memory_llm or config.llm
            if not memory_cfg.base_url:
                raise RuntimeError("No memory_llm or llm base_url configured.")
            raw_proposal = propose_with_llm(
                ChatClient(memory_cfg),
                existing_topics=existing_topics,
                misses=misses,
                max_new_topics=args.max_new_topics,
            )
        else:
            raw_proposal = propose_heuristic(misses, args.max_new_topics)
        proposal = normalize_proposal(raw_proposal, existing_slugs, args.max_new_topics)
        candidate = create_candidate_view(
            store,
            corpus_id=args.corpus_id,
            base_view=base_view,
            name=args.name,
            proposal=proposal,
            activate=args.activate,
            max_auto_reassignments=args.max_auto_reassignments,
            min_auto_reassign_keyword_matches=args.min_auto_reassign_keyword_matches,
            auto_reassignment_mode=args.auto_reassignment_mode,
            profile_auto_min_score=args.profile_auto_min_score,
            profile_auto_min_term_overlap=args.profile_auto_min_term_overlap,
            profile_auto_min_embedding_score=args.profile_auto_min_embedding_score,
            profile_auto_text_mode=args.profile_auto_text_mode,
            profile_auto_min_score_margin=args.profile_auto_min_score_margin,
            profile_auto_min_score_ratio=args.profile_auto_min_score_ratio,
            profile_auto_max_profile_terms=args.profile_auto_max_profile_terms,
            max_total_reassignments=args.max_total_reassignments,
            max_reassigned_atom_ratio=args.max_reassigned_atom_ratio,
            preserve_base_assignments=args.preserve_base_assignments,
        )
        result = {
            "corpus_id": args.corpus_id,
            "base_view_id": base_view["view_id"],
            "shadow_eval_report": str(args.shadow_eval_report),
            "miss_count": len(misses),
            "strategy": args.strategy,
            "proposal": proposal,
            "candidate": candidate,
        }
        if args.record_run:
            now = utc_now_iso()
            run_id = f"evo_{stable_hash(args.corpus_id, candidate['view_id'], now, length=24)}"
            store.add_evolution_run(
                run_id=run_id,
                corpus_id=args.corpus_id,
                base_view_id=str(base_view["view_id"]),
                candidate_view_id=str(candidate["view_id"]),
                trigger={
                    "kind": "topic_shadow_miss",
                    "shadow_eval_report": str(args.shadow_eval_report),
                    "strategy": args.strategy,
                },
                status="candidate_created",
                result=result,
                created_at=now,
                completed_at=now,
            )
            store.commit()
            result["evolution_run_id"] = run_id
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
