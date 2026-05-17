from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import (
    normalize_topic_slug,
    stable_hash,
    tokenize,
    topic_node_id,
    topic_slug,
    utc_now_iso,
)
from leaf.normalize import contains_cjk, language_aware_content_terms, normalize_surface_text
from leaf.store import SQLiteMemoryStore


CRITERIA_VERSION = "criteria_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone a topic view and patch route/profile keywords from self-QA criteria routing misses. "
            "The produced view is candidate-only unless --activate is set."
        ),
    )
    parser.add_argument("--config", default="", help="Accepted for pipeline compatibility; currently unused.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--selfqa", required=True)
    parser.add_argument("--search-eval-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-view-id", default="", help="Defaults to the active memory view.")
    parser.add_argument("--name", default="criteria-topic-profile-patch-v1")
    parser.add_argument(
        "--patch-strategy",
        choices=["question_terms_v1", "discriminative_profile_v2"],
        default="question_terms_v1",
    )
    parser.add_argument("--max-added-keywords-per-topic", type=int, default=6)
    parser.add_argument("--max-route-keywords", type=int, default=24)
    parser.add_argument("--max-node-keywords", type=int, default=32)
    parser.add_argument("--min-term-topic-count", type=int, default=2)
    parser.add_argument("--min-term-question-support", type=int, default=1)
    parser.add_argument("--max-term-background-topic-count", type=int, default=2)
    parser.add_argument(
        "--allow-retrieval-misses",
        action="store_true",
        help="By default only patch route misses where retrieved evidence already reaches the expected topic.",
    )
    parser.add_argument("--activate", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def keyword_surface(value: str) -> str:
    text = normalize_surface_text(value)
    if not text:
        return ""
    if contains_cjk(text):
        return text
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9+\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_keyword_surface(value: str) -> str:
    cleaned = keyword_surface(value)
    if not cleaned:
        return ""
    if contains_cjk(cleaned):
        return cleaned
    content_tokens = tokenize(cleaned)
    if not content_tokens:
        return ""
    ordered_tokens = [token for token in re.findall(r"[a-z0-9]+", cleaned) if token in content_tokens]
    return " ".join(ordered_tokens).strip()


def merge_unique(values: list[str], *, limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = keyword_surface(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        merged.append(cleaned)
        if len(merged) >= limit:
            break
    return merged


def criteria_for_task(task: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(task.get("metadata") or {})
    criteria = metadata.get(CRITERIA_VERSION) or task.get(CRITERIA_VERSION) or {}
    return dict(criteria) if isinstance(criteria, dict) else {}


def task_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in rows if str(row.get("task_id") or "").strip()}


def metadata_aliases(node: dict[str, Any]) -> set[str]:
    metadata = dict(node.get("metadata") or {})
    aliases = {str(node.get("topic_id") or "").strip()}
    for key in (
        "base_topic_id",
        "source_topic_id",
        "criteria_patch_clone_of_topic_id",
        "criteria_patch_source_topic_id",
    ):
        value = metadata.get(key)
        if isinstance(value, list):
            aliases.update(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            aliases.add(value.strip())
    aliases.discard("")
    return aliases


def route_keyword_seed(node: dict[str, Any]) -> list[str]:
    metadata = dict(node.get("metadata") or {})
    route_keywords = string_list(metadata.get("route_keywords"))
    if route_keywords:
        return route_keywords
    return string_list(node.get("keywords"))


def candidate_terms_for_task(task: dict[str, Any], row: dict[str, Any]) -> list[str]:
    question = str(row.get("question") or task.get("question") or "")
    answer = str(row.get("answer") or task.get("answer") or "")
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    criteria = criteria_for_task(task)
    answer_criteria = dict(criteria.get("answer_criteria") or {})
    phrase_candidates: list[str] = []
    phrase_candidates.extend(string_list(answer_criteria.get("must_contain")))
    phrase_candidates.extend(string_list(answer_criteria.get("acceptable_aliases")))
    if answer:
        phrase_candidates.append(answer)

    evidence_parts: list[str] = []
    speaker_tokens: set[str] = set()
    for evidence in (dict(task.get("metadata") or {}).get("source_evidence") or []):
        if not isinstance(evidence, dict):
            continue
        speaker_tokens.update(tokenize(str(evidence.get("speaker") or "")))
        evidence_parts.extend(
            [
                str(evidence.get("atom_content") or ""),
                " ".join(string_list(evidence.get("entities"))),
            ]
        )
        phrase_candidates.extend(string_list(evidence.get("entities")))

    evidence_tokens = tokenize(" ".join(evidence_parts + phrase_candidates))
    token_counts = Counter(token for token in query_tokens.intersection(evidence_tokens) if token)
    scored: list[tuple[float, str]] = []

    for phrase in phrase_candidates:
        cleaned = content_keyword_surface(phrase)
        if not cleaned:
            continue
        phrase_tokens = tokenize(cleaned)
        if not phrase_tokens:
            continue
        if phrase_tokens.issubset(speaker_tokens):
            continue
        if phrase_tokens.issubset(query_tokens) and phrase_tokens.intersection(evidence_tokens):
            scored.append((3.0 + len(phrase_tokens), cleaned))

    for token in sorted(token_counts):
        if token.isdigit():
            continue
        if token in speaker_tokens:
            continue
        if token in query_tokens and token in evidence_tokens:
            scored.append((1.0, token))

    scored.sort(key=lambda item: (-item[0], -len(tokenize(item[1])), item[1]))
    return merge_unique([term for _score, term in scored], limit=16)


def source_evidence_terms(task: dict[str, Any], row: dict[str, Any]) -> set[str]:
    pieces = [str(row.get("question") or task.get("question") or ""), str(row.get("answer") or task.get("answer") or "")]
    for evidence in (dict(task.get("metadata") or {}).get("source_evidence") or []):
        if not isinstance(evidence, dict):
            continue
        pieces.extend(
            [
                str(evidence.get("atom_content") or ""),
                str(evidence.get("event_text") or ""),
                " ".join(string_list(evidence.get("entities"))),
            ]
        )
    return set(language_aware_content_terms(" ".join(pieces), mode="auto", include_cjk_subgrams=True))


def atom_profile_terms(atom: Any) -> set[str]:
    pieces = [
        str(getattr(atom, "content", "") or ""),
        " ".join(str(item) for item in (getattr(atom, "entities", []) or [])),
        " ".join(str(item) for item in (getattr(atom, "canonical_entities", []) or [])),
    ]
    return {
        term
        for term in language_aware_content_terms(" ".join(pieces), mode="auto", include_cjk_subgrams=True)
        if term and not term.isdigit()
    }


def collect_discriminative_patch_terms(
    store: SQLiteMemoryStore,
    *,
    selfqa_rows: list[dict[str, Any]],
    eval_report: dict[str, Any],
    base_view_id: str,
    base_nodes: list[dict[str, Any]],
    allow_retrieval_misses: bool,
    min_term_topic_count: int,
    min_term_question_support: int,
    max_term_background_topic_count: int,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    selfqa_by_id = task_by_id(selfqa_rows)
    alias_to_topic_id: dict[str, str] = {}
    node_by_id = {str(node["topic_id"]): node for node in base_nodes}
    for node in base_nodes:
        for alias in metadata_aliases(node):
            alias_to_topic_id[alias] = str(node["topic_id"])

    assignments = [
        assignment
        for assignment in store.list_topic_assignments(base_view_id)
        if str(assignment.get("item_kind") or "") in {"atom", "atom_secondary"}
    ]
    atom_ids = [str(assignment.get("item_id") or "") for assignment in assignments if str(assignment.get("item_id") or "").strip()]
    atoms_by_id = {atom.atom_id: atom for atom in store.get_atoms_by_ids(atom_ids)}
    topic_term_counts: dict[str, Counter[str]] = defaultdict(Counter)
    term_topic_presence: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        topic_id = str(assignment.get("topic_id") or "").strip()
        if topic_id not in node_by_id:
            continue
        atom = atoms_by_id.get(str(assignment.get("item_id") or ""))
        if atom is None:
            continue
        terms = atom_profile_terms(atom)
        topic_term_counts[topic_id].update(terms)
        for term in terms:
            term_topic_presence[term].add(topic_id)

    target_question_support: dict[str, Counter[str]] = defaultdict(Counter)
    target_task_ids: dict[str, set[str]] = defaultdict(set)
    source_tasks: list[dict[str, Any]] = []
    for row in eval_report.get("rows") or []:
        shadow = dict(row.get("topic_routing_shadow") or {})
        if shadow.get("criteria_expected_topic_route_hit") is not False:
            continue
        if not allow_retrieval_misses and shadow.get("criteria_expected_topic_retrieval_hit") is not True:
            continue
        task_id = str(row.get("task_id") or "").strip()
        task = selfqa_by_id.get(task_id)
        if not task:
            continue
        expected_ids = string_list(shadow.get("criteria_expected_topic_ids"))
        if not expected_ids:
            criteria = criteria_for_task(task)
            expected_ids = string_list((dict(criteria.get("topic_criteria") or {})).get("expected_topic_ids"))
        target_topic_ids = sorted({alias_to_topic_id[item] for item in expected_ids if item in alias_to_topic_id})
        if not target_topic_ids:
            continue
        evidence_terms = source_evidence_terms(task, row)
        task_candidate_terms = candidate_terms_for_task(task, row)
        candidate_term_set = set(task_candidate_terms) | evidence_terms
        if not candidate_term_set:
            continue
        for topic_id in target_topic_ids:
            target_terms = set(topic_term_counts.get(topic_id) or {})
            supported_terms = target_terms.intersection(candidate_term_set)
            target_question_support[topic_id].update(supported_terms)
            if supported_terms:
                target_task_ids[topic_id].add(task_id)
        source_tasks.append(
            {
                "task_id": task_id,
                "task_type": row.get("task_type"),
                "question": row.get("question"),
                "expected_topic_ids": expected_ids,
                "target_base_topic_ids": target_topic_ids,
                "routed_topic_slugs": shadow.get("routed_topic_slugs") or [],
                "candidate_terms": sorted(candidate_term_set)[:32],
            }
        )

    patch_terms: dict[str, list[str]] = {}
    min_topic_count = max(1, int(min_term_topic_count))
    min_support = max(0, int(min_term_question_support))
    max_background_topics = max(0, int(max_term_background_topic_count))
    for topic_id, support_counter in target_question_support.items():
        node = node_by_id.get(topic_id) or {}
        existing_tokens = tokenize(" ".join(route_keyword_seed(node) + string_list(node.get("keywords"))))
        topic_counter = topic_term_counts.get(topic_id) or Counter()
        ranked_terms: list[tuple[float, int, int, str]] = []
        for term, support_count in support_counter.items():
            term_tokens = tokenize(term)
            if not term_tokens or term.isdigit():
                continue
            if term_tokens.issubset(existing_tokens):
                continue
            topic_count = int(topic_counter.get(term) or 0)
            if topic_count < min_topic_count:
                continue
            if support_count < min_support:
                continue
            background_topics = max(0, len(term_topic_presence.get(term) or set()) - 1)
            if background_topics > max_background_topics:
                continue
            specificity = topic_count / max(1, background_topics + 1)
            score = float(support_count) * 2.0 + specificity
            ranked_terms.append((score, topic_count, support_count, term))
        ranked_terms.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        patch_terms[topic_id] = [term for _score, _topic_count, _support_count, term in ranked_terms]
    return patch_terms, source_tasks


def collect_patch_terms(
    *,
    selfqa_rows: list[dict[str, Any]],
    eval_report: dict[str, Any],
    base_nodes: list[dict[str, Any]],
    allow_retrieval_misses: bool,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    selfqa_by_id = task_by_id(selfqa_rows)
    alias_to_topic_id: dict[str, str] = {}
    for node in base_nodes:
        for alias in metadata_aliases(node):
            alias_to_topic_id[alias] = str(node["topic_id"])

    term_counts_by_topic: dict[str, Counter[str]] = defaultdict(Counter)
    source_tasks: list[dict[str, Any]] = []
    for row in eval_report.get("rows") or []:
        shadow = dict(row.get("topic_routing_shadow") or {})
        if shadow.get("criteria_expected_topic_route_hit") is not False:
            continue
        if not allow_retrieval_misses and shadow.get("criteria_expected_topic_retrieval_hit") is not True:
            continue
        task_id = str(row.get("task_id") or "").strip()
        task = selfqa_by_id.get(task_id)
        if not task:
            continue
        expected_ids = string_list(shadow.get("criteria_expected_topic_ids"))
        if not expected_ids:
            criteria = criteria_for_task(task)
            expected_ids = string_list((dict(criteria.get("topic_criteria") or {})).get("expected_topic_ids"))
        target_topic_ids = sorted({alias_to_topic_id[item] for item in expected_ids if item in alias_to_topic_id})
        if not target_topic_ids:
            continue
        terms = candidate_terms_for_task(task, row)
        if not terms:
            continue
        for topic_id in target_topic_ids:
            term_counts_by_topic[topic_id].update(terms)
        source_tasks.append(
            {
                "task_id": task_id,
                "task_type": row.get("task_type"),
                "question": row.get("question"),
                "expected_topic_ids": expected_ids,
                "target_base_topic_ids": target_topic_ids,
                "routed_topic_slugs": shadow.get("routed_topic_slugs") or [],
                "candidate_terms": terms,
            }
        )

    topics_by_term: dict[str, set[str]] = defaultdict(set)
    for topic_id, counter in term_counts_by_topic.items():
        for term in counter:
            topics_by_term[term].add(topic_id)

    patch_terms: dict[str, list[str]] = {}
    node_by_id = {str(node["topic_id"]): node for node in base_nodes}
    for topic_id, counter in term_counts_by_topic.items():
        node = node_by_id.get(topic_id) or {}
        existing_tokens = tokenize(" ".join(route_keyword_seed(node) + string_list(node.get("keywords"))))
        ranked_terms = []
        for term, count in counter.most_common():
            term_tokens = tokenize(term)
            if not term_tokens:
                continue
            if term.isdigit():
                continue
            if len(topics_by_term.get(term) or set()) > 1:
                continue
            if term_tokens.issubset(existing_tokens):
                continue
            ranked_terms.append((count, len(term_tokens), term))
        ranked_terms.sort(key=lambda item: (-item[0], -item[1], item[2]))
        patch_terms[topic_id] = [term for _count, _width, term in ranked_terms]
    return patch_terms, source_tasks


def clone_and_patch_view(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    base_view: dict[str, Any],
    base_nodes: list[dict[str, Any]],
    patch_terms: dict[str, list[str]],
    source_tasks: list[dict[str, Any]],
    name: str,
    max_added_keywords_per_topic: int,
    max_route_keywords: int,
    max_node_keywords: int,
    search_eval_report: str,
    selfqa: str,
    activate: bool,
    patch_strategy: str,
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
            "view_type": "criteria_topic_profile_patch",
            "patch_version": patch_strategy,
            "base_view_id": base_view_id,
            "source_selfqa": str(selfqa),
            "source_search_eval_report": str(search_eval_report),
            "source_task_count": len(source_tasks),
            "patch_target": "route_keywords_and_node_keywords",
        },
        metrics={
            "criteria_route_miss_task_count": len(source_tasks),
            "patched_base_topic_count": len([terms for terms in patch_terms.values() if terms]),
        },
    )

    base_to_candidate: dict[str, str] = {}
    used_candidate_ids: set[str] = set()
    for node in sorted(base_nodes, key=lambda item: (int(item.get("level") or 0), str(item.get("topic_id") or ""))):
        metadata = dict(node.get("metadata") or {})
        slug = "root" if metadata.get("seed_role") == "root" else normalize_topic_slug(topic_slug(node))
        candidate_topic_id = topic_node_id(view_id, slug)
        if candidate_topic_id in used_candidate_ids:
            candidate_topic_id = topic_node_id(view_id, f"{slug}_{stable_hash(str(node['topic_id']), length=6)}")
        used_candidate_ids.add(candidate_topic_id)
        base_to_candidate[str(node["topic_id"])] = candidate_topic_id

    patched_topics: list[dict[str, Any]] = []
    copied_nodes = 0
    for node in sorted(base_nodes, key=lambda item: (int(item.get("level") or 0), str(item.get("topic_id") or ""))):
        base_topic_id = str(node["topic_id"])
        candidate_topic_id = base_to_candidate[base_topic_id]
        parent_id = base_to_candidate.get(str(node.get("parent_id"))) if node.get("parent_id") else None
        metadata = dict(node.get("metadata") or {})
        stats = dict(node.get("stats") or {})
        keywords = string_list(node.get("keywords"))
        route_keywords = route_keyword_seed(node)
        added_terms = merge_unique(
            patch_terms.get(base_topic_id, [])[: max_added_keywords_per_topic * 2],
            limit=max_added_keywords_per_topic,
        )
        node_metadata = {
            **metadata,
            "base_topic_id": base_topic_id,
            "base_view_id": base_view_id,
            "criteria_patch_clone_of_topic_id": base_topic_id,
        }
        if metadata.get("seed_role") != "root":
            node_metadata.setdefault("topic_slug", normalize_topic_slug(topic_slug(node)))
        if added_terms:
            route_keywords = merge_unique(route_keywords + added_terms, limit=max_route_keywords)
            keywords = merge_unique(keywords + added_terms, limit=max_node_keywords)
            node_metadata["route_keywords"] = route_keywords
            node_metadata["route_exposure"] = "active"
            node_metadata["criteria_patch_v1"] = {
                "patch_strategy": patch_strategy,
                "added_route_keywords": added_terms,
                "base_route_keyword_count": len(route_keyword_seed(node)),
                "source_task_ids": sorted(
                    {
                        str(task["task_id"])
                        for task in source_tasks
                        if base_topic_id in set(task.get("target_base_topic_ids") or [])
                    }
                ),
            }
            stats["criteria_patch_added_keyword_count"] = len(added_terms)
            patched_topics.append(
                {
                    "base_topic_id": base_topic_id,
                    "candidate_topic_id": candidate_topic_id,
                    "slug": normalize_topic_slug(topic_slug(node)),
                    "added_route_keywords": added_terms,
                    "route_keyword_count": len(route_keywords),
                }
            )
        store.upsert_topic_node(
            topic_id=candidate_topic_id,
            view_id=view_id,
            parent_id=parent_id,
            name=str(node.get("name") or ""),
            description=str(node.get("description") or ""),
            level=int(node.get("level") or 0),
            keywords=keywords,
            exemplar_ids=string_list(node.get("exemplar_ids")),
            stats=stats,
            embedding=node.get("embedding"),
            metadata=node_metadata,
        )
        copied_nodes += 1

    copied_assignments = 0
    for assignment in store.list_topic_assignments(base_view_id):
        target_topic_id = base_to_candidate.get(str(assignment.get("topic_id") or ""))
        if not target_topic_id:
            continue
        item_kind = str(assignment.get("item_kind") or "")
        item_id = str(assignment.get("item_id") or "")
        reason = dict(assignment.get("reason") or {})
        store.upsert_topic_assignment(
            assignment_id=f"assign_{stable_hash(view_id, item_kind, item_id, target_topic_id, length=24)}",
            view_id=view_id,
            corpus_id=corpus_id,
            item_kind=item_kind,
            item_id=item_id,
            topic_id=target_topic_id,
            confidence=float(assignment.get("confidence") or 0.0),
            reason={**reason, "base_assignment_id": assignment.get("assignment_id"), "base_view_id": base_view_id},
        )
        copied_assignments += 1

    copied_edges = 0
    for edge in store.conn.execute("select * from leaf_topic_edges where view_id = ?", (base_view_id,)).fetchall():
        src_topic_id = base_to_candidate.get(str(edge["src_topic_id"]))
        dst_topic_id = base_to_candidate.get(str(edge["dst_topic_id"]))
        if not src_topic_id or not dst_topic_id:
            continue
        edge_type = str(edge["edge_type"])
        edge_id = f"edge_{stable_hash(view_id, src_topic_id, dst_topic_id, edge_type, length=24)}"
        metadata = dict(json.loads(edge["metadata_json"] or "{}"))
        metadata.update({"base_edge_id": edge["edge_id"], "base_view_id": base_view_id})
        store.conn.execute(
            """
            insert or replace into leaf_topic_edges
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id,
                view_id,
                src_topic_id,
                dst_topic_id,
                edge_type,
                edge["evidence_ids_json"],
                float(edge["confidence"] or 0.0),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        copied_edges += 1

    if activate:
        store.promote_memory_view(view_id, promoted_at=now)
    run_id = f"run_{stable_hash(corpus_id, view_id, 'criteria_topic_profile_patch_v1', length=20)}"
    result = {
        "view_id": view_id,
        "base_view_id": base_view_id,
        "name": name,
        "active": activate,
        "copied_topic_nodes": copied_nodes,
        "copied_topic_assignments": copied_assignments,
        "copied_topic_edges": copied_edges,
        "patched_topic_count": len(patched_topics),
        "patched_topics": patched_topics,
        "source_task_count": len(source_tasks),
        "source_tasks": source_tasks,
    }
    store.add_evolution_run(
        run_id=run_id,
        corpus_id=corpus_id,
        base_view_id=base_view_id,
        candidate_view_id=view_id,
        trigger={
            "kind": "selfqa_criteria_route_miss",
            "criteria_version": CRITERIA_VERSION,
            "patch_strategy": patch_strategy,
            "selfqa": str(selfqa),
            "search_eval_report": str(search_eval_report),
        },
        status="completed",
        result=result,
        created_at=now,
        completed_at=utc_now_iso(),
    )
    store.commit()
    result["run_id"] = run_id
    return result


def main() -> None:
    args = parse_args()
    selfqa_rows = load_jsonl(args.selfqa)
    eval_report = load_json(args.search_eval_report)
    store = SQLiteMemoryStore(args.db)
    try:
        base_view = store.get_memory_view(args.base_view_id) if args.base_view_id else store.get_active_memory_view(args.corpus_id)
        if base_view is None or str(base_view.get("corpus_id") or "") != args.corpus_id:
            raise SystemExit(f"No base topic view found for corpus_id={args.corpus_id!r}.")
        base_nodes = store.list_topic_nodes(str(base_view["view_id"]))
        if args.patch_strategy == "discriminative_profile_v2":
            patch_terms, source_tasks = collect_discriminative_patch_terms(
                store,
                selfqa_rows=selfqa_rows,
                eval_report=eval_report,
                base_view_id=str(base_view["view_id"]),
                base_nodes=base_nodes,
                allow_retrieval_misses=bool(args.allow_retrieval_misses),
                min_term_topic_count=int(args.min_term_topic_count),
                min_term_question_support=int(args.min_term_question_support),
                max_term_background_topic_count=int(args.max_term_background_topic_count),
            )
        else:
            patch_terms, source_tasks = collect_patch_terms(
                selfqa_rows=selfqa_rows,
                eval_report=eval_report,
                base_nodes=base_nodes,
                allow_retrieval_misses=bool(args.allow_retrieval_misses),
            )
        result = clone_and_patch_view(
            store,
            corpus_id=args.corpus_id,
            base_view=base_view,
            base_nodes=base_nodes,
            patch_terms=patch_terms,
            source_tasks=source_tasks,
            name=args.name,
            max_added_keywords_per_topic=args.max_added_keywords_per_topic,
            max_route_keywords=args.max_route_keywords,
            max_node_keywords=args.max_node_keywords,
            search_eval_report=args.search_eval_report,
            selfqa=args.selfqa,
            activate=bool(args.activate),
            patch_strategy=str(args.patch_strategy),
        )
    finally:
        store.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
