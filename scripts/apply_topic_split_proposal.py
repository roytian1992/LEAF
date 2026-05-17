from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import normalize_topic_slug, stable_hash, tokenize, topic_node_id, topic_slug, utc_now_iso
from leaf.normalize import language_aware_content_terms
from leaf.store import SQLiteMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create candidate memory views with shadow child topics from topic_split proposals. "
            "The script does not activate the candidate view unless --activate is set."
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="evolvemem-topic-split-shadow")
    parser.add_argument("--corpus-id", action="append", default=[])
    parser.add_argument("--patch-id", action="append", default=[])
    parser.add_argument("--min-child-score", type=float, default=1.0)
    parser.add_argument("--max-child-assignments-per-topic", type=int, default=80)
    parser.add_argument("--child-assignment-kind", choices=["atom_secondary", "atom"], default="atom_secondary")
    parser.add_argument("--activate", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def merge_unique(values: list[str], *, limit: int = 80) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def slug_for_node(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    if metadata.get("seed_role") == "root":
        return "root"
    return normalize_topic_slug(topic_slug(node))


def child_slug(parent_slug: str, child_name: str, index: int) -> str:
    cleaned = normalize_topic_slug(child_name)
    if cleaned.startswith(parent_slug):
        return cleaned
    return normalize_topic_slug(f"{parent_slug}_{index}_{cleaned}")


def scope_corpus(patch: dict[str, Any]) -> str:
    return str((patch.get("scope") or {}).get("corpus_id") or "").strip()


def selected_split_patches(proposal: dict[str, Any], *, corpus_ids: set[str], patch_ids: set[str]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    for patch in proposal.get("patches") or []:
        if not isinstance(patch, dict):
            continue
        if str(patch.get("patch_type") or "") != "topic_split":
            continue
        patch_id = str(patch.get("patch_id") or "").strip()
        if patch_ids and patch_id not in patch_ids:
            continue
        corpus_id = scope_corpus(patch)
        if corpus_ids and corpus_id not in corpus_ids:
            continue
        if patch.get("invalid_evidence_failure_ids"):
            continue
        children = (patch.get("change") or {}).get("candidate_child_topics")
        if not isinstance(children, list) or not children:
            continue
        patches.append(patch)
    return patches


def node_aliases(node: dict[str, Any]) -> set[str]:
    metadata = dict(node.get("metadata") or {})
    aliases = {
        str(node.get("topic_id") or "").strip(),
        str(metadata.get("base_topic_id") or "").strip(),
        str(metadata.get("source_topic_id") or "").strip(),
        slug_for_node(node),
    }
    aliases.discard("")
    return aliases


def find_parent_node(nodes: list[dict[str, Any]], patch: dict[str, Any]) -> dict[str, Any] | None:
    scope = dict(patch.get("scope") or {})
    wanted_id = str(scope.get("topic_id") or "").strip()
    wanted_slug = normalize_topic_slug(str(scope.get("topic_slug") or "").strip())
    for node in nodes:
        aliases = node_aliases(node)
        if wanted_id and wanted_id in aliases:
            return node
        if wanted_slug and wanted_slug in aliases:
            return node
    return None


def atom_text(atom: Any) -> str:
    return " ".join(
        [
            str(getattr(atom, "content", "") or ""),
            " ".join(str(item) for item in (getattr(atom, "entities", []) or [])),
            " ".join(str(item) for item in (getattr(atom, "canonical_entities", []) or [])),
        ]
    )


def child_terms(child: dict[str, Any]) -> set[str]:
    pieces = [
        str(child.get("name") or ""),
        str(child.get("description") or ""),
        " ".join(string_list(child.get("route_keywords"))),
        " ".join(string_list(child.get("profile_terms"))),
    ]
    terms = set(language_aware_content_terms(" ".join(pieces), mode="auto", include_cjk_subgrams=True))
    terms.update(tokenize(" ".join(string_list(child.get("route_keywords")))))
    return {term for term in terms if term}


def score_atom_for_child(atom: Any, child: dict[str, Any]) -> tuple[float, list[str]]:
    terms = child_terms(child)
    if not terms:
        return 0.0, []
    atom_terms = set(language_aware_content_terms(atom_text(atom), mode="auto", include_cjk_subgrams=True))
    atom_terms.update(tokenize(atom_text(atom)))
    matched = sorted(atom_terms.intersection(terms))
    route_keyword_tokens = tokenize(" ".join(string_list(child.get("route_keywords"))))
    route_hits = sorted(atom_terms.intersection(route_keyword_tokens))
    score = float(len(matched)) + 0.75 * float(len(route_hits))
    return score, matched


def clone_base_view(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    active_view: dict[str, Any],
    candidate_view_id: str,
    name: str,
    proposal_path: str,
    patches: list[dict[str, Any]],
    activate: bool,
) -> tuple[dict[str, str], list[dict[str, Any]], int, int]:
    base_view_id = str(active_view["view_id"])
    base_nodes = store.list_topic_nodes(base_view_id)
    base_to_candidate: dict[str, str] = {}
    used_ids: set[str] = set()
    for node in sorted(base_nodes, key=lambda item: (int(item.get("level") or 0), str(item.get("topic_id") or ""))):
        slug = slug_for_node(node)
        candidate_topic_id = topic_node_id(candidate_view_id, slug)
        if candidate_topic_id in used_ids:
            candidate_topic_id = topic_node_id(candidate_view_id, f"{slug}_{stable_hash(str(node['topic_id']), length=6)}")
        used_ids.add(candidate_topic_id)
        base_to_candidate[str(node["topic_id"])] = candidate_topic_id

    copied_nodes = 0
    for node in sorted(base_nodes, key=lambda item: (int(item.get("level") or 0), str(item.get("topic_id") or ""))):
        base_topic_id = str(node["topic_id"])
        metadata = dict(node.get("metadata") or {})
        metadata.update(
            {
                "base_topic_id": base_topic_id,
                "base_view_id": base_view_id,
                "evolvemem_split_clone_of_topic_id": base_topic_id,
            }
        )
        if metadata.get("seed_role") != "root":
            metadata.setdefault("topic_slug", slug_for_node(node))
        parent_id = base_to_candidate.get(str(node.get("parent_id"))) if node.get("parent_id") else None
        store.upsert_topic_node(
            topic_id=base_to_candidate[base_topic_id],
            view_id=candidate_view_id,
            parent_id=parent_id,
            name=str(node.get("name") or ""),
            description=str(node.get("description") or ""),
            level=int(node.get("level") or 0),
            keywords=string_list(node.get("keywords")),
            exemplar_ids=string_list(node.get("exemplar_ids")),
            stats=dict(node.get("stats") or {}),
            embedding=node.get("embedding"),
            metadata=metadata,
        )
        copied_nodes += 1

    copied_assignments = 0
    for assignment in store.list_topic_assignments(base_view_id):
        candidate_topic_id = base_to_candidate.get(str(assignment.get("topic_id") or ""))
        if not candidate_topic_id:
            continue
        item_kind = str(assignment.get("item_kind") or "")
        item_id = str(assignment.get("item_id") or "")
        store.upsert_topic_assignment(
            assignment_id=f"assign_{stable_hash(candidate_view_id, item_kind, item_id, candidate_topic_id, length=24)}",
            view_id=candidate_view_id,
            corpus_id=corpus_id,
            item_kind=item_kind,
            item_id=item_id,
            topic_id=candidate_topic_id,
            confidence=float(assignment.get("confidence") or 0.0),
            reason={
                **dict(assignment.get("reason") or {}),
                "base_assignment_id": assignment.get("assignment_id"),
                "base_view_id": base_view_id,
            },
        )
        copied_assignments += 1

    copied_edges = 0
    for edge in store.conn.execute("select * from leaf_topic_edges where view_id = ?", (base_view_id,)).fetchall():
        src_topic_id = base_to_candidate.get(str(edge["src_topic_id"]))
        dst_topic_id = base_to_candidate.get(str(edge["dst_topic_id"]))
        if not src_topic_id or not dst_topic_id:
            continue
        edge_type = str(edge["edge_type"])
        metadata = dict(json.loads(edge["metadata_json"] or "{}"))
        metadata.update({"base_edge_id": edge["edge_id"], "base_view_id": base_view_id})
        store.conn.execute(
            """
            insert or replace into leaf_topic_edges
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"edge_{stable_hash(candidate_view_id, src_topic_id, dst_topic_id, edge_type, length=24)}",
                candidate_view_id,
                src_topic_id,
                dst_topic_id,
                edge_type,
                edge["evidence_ids_json"],
                float(edge["confidence"] or 0.0),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        copied_edges += 1

    store.upsert_memory_view(
        view_id=candidate_view_id,
        corpus_id=corpus_id,
        parent_view_id=base_view_id,
        name=name,
        status="active" if activate else "candidate",
        active=False,
        created_at=utc_now_iso(),
        metadata={
            "view_type": "evolvemem_topic_split_shadow",
            "base_view_id": base_view_id,
            "source_proposal": str(proposal_path),
            "source_patch_ids": [patch.get("patch_id") for patch in patches],
        },
        metrics={},
    )
    return base_to_candidate, base_nodes, copied_nodes, copied_assignments + copied_edges


def apply_split_children(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    candidate_view_id: str,
    base_to_candidate: dict[str, str],
    base_nodes: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    min_child_score: float,
    max_child_assignments_per_topic: int,
    child_assignment_kind: str,
) -> list[dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    base_node_by_id = {str(node["topic_id"]): node for node in base_nodes}
    atoms_by_id = {atom.atom_id: atom for atom in store.list_atoms(corpus_id)}
    base_view_id = next(iter({str(node.get("view_id") or "") for node in base_nodes if node.get("view_id")}), "")
    assignments_by_topic: dict[str, list[dict[str, Any]]] = {}
    for assignment in store.list_topic_assignments(base_view_id):
        assignments_by_topic.setdefault(str(assignment.get("topic_id") or ""), []).append(assignment)

    for patch in patches:
        parent_node = find_parent_node(base_nodes, patch)
        if parent_node is None:
            result_rows.append({"patch_id": patch.get("patch_id"), "error": "parent_topic_not_found"})
            continue
        parent_base_id = str(parent_node["topic_id"])
        parent_candidate_id = base_to_candidate[parent_base_id]
        parent_slug = slug_for_node(parent_node)
        children = [
            child for child in ((patch.get("change") or {}).get("candidate_child_topics") or []) if isinstance(child, dict)
        ]
        parent_assignments = assignments_by_topic.get(parent_base_id) or []
        parent_atom_ids = sorted({str(item.get("item_id") or "") for item in parent_assignments if str(item.get("item_id") or "")})
        child_results: list[dict[str, Any]] = []
        for index, child in enumerate(children, start=1):
            name = str(child.get("name") or f"{parent_node.get('name')} Child {index}").strip()
            slug = child_slug(parent_slug, name, index)
            child_topic_id = topic_node_id(candidate_view_id, slug)
            route_keywords = merge_unique(string_list(child.get("route_keywords")), limit=32)
            profile_terms = merge_unique(string_list(child.get("profile_terms")), limit=32)
            keywords = merge_unique(route_keywords + profile_terms, limit=48)
            store.upsert_topic_node(
                topic_id=child_topic_id,
                view_id=candidate_view_id,
                parent_id=parent_candidate_id,
                name=name,
                description=str(child.get("description") or ""),
                level=int(parent_node.get("level") or 0) + 1,
                keywords=keywords,
                exemplar_ids=[],
                stats={"shadow_split_child": True},
                embedding=None,
                metadata={
                    "topic_slug": slug,
                    "evolved_slug": slug,
                    "topic_role": "shadow_split_child",
                    "route_keywords": route_keywords,
                    "profile_terms": profile_terms,
                    "route_exposure": "active",
                    "primary_assignment_exposure": "inactive",
                    "answer_exposure": "supplemental",
                    "parent_base_topic_id": parent_base_id,
                    "parent_candidate_topic_id": parent_candidate_id,
                    "source_patch_id": patch.get("patch_id"),
                    "source_proposal_patch_type": "topic_split",
                },
            )
            scored: list[tuple[float, str, list[str]]] = []
            for atom_id in parent_atom_ids:
                atom = atoms_by_id.get(atom_id)
                if atom is None:
                    continue
                score, matched = score_atom_for_child(atom, child)
                if score >= float(min_child_score):
                    scored.append((score, atom_id, matched))
            scored.sort(key=lambda item: (-item[0], item[1]))
            selected = scored[: max(0, int(max_child_assignments_per_topic))]
            for score, atom_id, matched in selected:
                store.upsert_topic_assignment(
                    assignment_id=f"assign_{stable_hash(candidate_view_id, child_assignment_kind, atom_id, child_topic_id, length=24)}",
                    view_id=candidate_view_id,
                    corpus_id=corpus_id,
                    item_kind=child_assignment_kind,
                    item_id=atom_id,
                    topic_id=child_topic_id,
                    confidence=min(0.95, 0.35 + 0.08 * float(score)),
                    reason={
                        "strategy": "evolvemem_shadow_split_overlap",
                        "source_patch_id": patch.get("patch_id"),
                        "parent_base_topic_id": parent_base_id,
                        "parent_candidate_topic_id": parent_candidate_id,
                        "score": round(float(score), 4),
                        "matched_terms": matched[:12],
                    },
                )
            child_results.append(
                {
                    "child_topic_id": child_topic_id,
                    "slug": slug,
                    "name": name,
                    "route_keywords": route_keywords,
                    "profile_terms": profile_terms,
                    "candidate_parent_atom_count": len(parent_atom_ids),
                    "assigned_atom_count": len(selected),
                    "top_matches": [
                        {"atom_id": atom_id, "score": round(float(score), 4), "matched_terms": matched[:8]}
                        for score, atom_id, matched in selected[:8]
                    ],
                }
            )
        result_rows.append(
            {
                "patch_id": patch.get("patch_id"),
                "parent_base_topic_id": parent_base_id,
                "parent_candidate_topic_id": parent_candidate_id,
                "parent_slug": parent_slug,
                "child_count": len(child_results),
                "children": child_results,
            }
        )
    return result_rows


def main() -> None:
    args = parse_args()
    proposal = load_json(args.proposal)
    requested_corpus_ids = {str(item).strip() for item in args.corpus_id if str(item).strip()}
    requested_patch_ids = {str(item).strip() for item in args.patch_id if str(item).strip()}
    patches = selected_split_patches(proposal, corpus_ids=requested_corpus_ids, patch_ids=requested_patch_ids)
    if not patches:
        raise SystemExit("No executable topic_split patches selected.")
    corpus_ids = sorted({scope_corpus(patch) for patch in patches if scope_corpus(patch)})
    store = SQLiteMemoryStore(args.db)
    try:
        results: list[dict[str, Any]] = []
        for corpus_id in corpus_ids:
            corpus_patches = [patch for patch in patches if scope_corpus(patch) == corpus_id]
            active_view = store.get_active_memory_view(corpus_id)
            if active_view is None:
                results.append({"corpus_id": corpus_id, "error": "no_active_memory_view"})
                continue
            candidate_view_id = f"view_{stable_hash(corpus_id, args.name, uuid.uuid4().hex, length=18)}"
            name = f"{args.name}-{corpus_id}-{utc_now_iso()}"
            base_to_candidate, base_nodes, copied_nodes, copied_objects = clone_base_view(
                store,
                corpus_id=corpus_id,
                active_view=active_view,
                candidate_view_id=candidate_view_id,
                name=name,
                proposal_path=str(args.proposal),
                patches=corpus_patches,
                activate=bool(args.activate),
            )
            split_results = apply_split_children(
                store,
                corpus_id=corpus_id,
                candidate_view_id=candidate_view_id,
                base_to_candidate=base_to_candidate,
                base_nodes=base_nodes,
                patches=corpus_patches,
                min_child_score=float(args.min_child_score),
                max_child_assignments_per_topic=int(args.max_child_assignments_per_topic),
                child_assignment_kind=str(args.child_assignment_kind),
            )
            patched_topic_count = sum(int(row.get("child_count") or 0) for row in split_results)
            assigned_count = sum(
                int(child.get("assigned_atom_count") or 0)
                for row in split_results
                for child in row.get("children") or []
            )
            store.update_memory_view_metrics(
                candidate_view_id,
                metrics={
                    "copied_topic_count": copied_nodes,
                    "copied_assignment_edge_count": copied_objects,
                    "split_patch_count": len(corpus_patches),
                    "shadow_child_topic_count": patched_topic_count,
                    "shadow_child_assignment_count": assigned_count,
                    "child_assignment_kind": str(args.child_assignment_kind),
                    "min_child_score": float(args.min_child_score),
                },
            )
            if args.activate:
                store.promote_memory_view(candidate_view_id, promoted_at=utc_now_iso())
            run_id = f"run_{stable_hash(corpus_id, candidate_view_id, 'evolvemem_topic_split_shadow', length=20)}"
            store.add_evolution_run(
                run_id=run_id,
                corpus_id=corpus_id,
                base_view_id=str(active_view["view_id"]),
                candidate_view_id=candidate_view_id,
                trigger={
                    "kind": "evolvemem_topic_split_shadow",
                    "proposal": str(args.proposal),
                    "patch_ids": [patch.get("patch_id") for patch in corpus_patches],
                },
                status="candidate_created",
                result={
                    "candidate_view_id": candidate_view_id,
                    "split_results": split_results,
                },
                created_at=utc_now_iso(),
                completed_at=utc_now_iso(),
            )
            results.append(
                {
                    "corpus_id": corpus_id,
                    "base_view_id": active_view["view_id"],
                    "candidate_view_id": candidate_view_id,
                    "run_id": run_id,
                    "activated": bool(args.activate),
                    "copied_topic_count": copied_nodes,
                    "copied_assignment_edge_count": copied_objects,
                    "split_patch_count": len(corpus_patches),
                    "shadow_child_topic_count": patched_topic_count,
                    "shadow_child_assignment_count": assigned_count,
                    "split_results": split_results,
                }
            )
        store.commit()
    finally:
        store.close()

    summary = {
        "created_at": utc_now_iso(),
        "db": str(args.db),
        "proposal": str(args.proposal),
        "output": str(args.output),
        "selected_patch_ids": [patch.get("patch_id") for patch in patches],
        "min_child_score": float(args.min_child_score),
        "max_child_assignments_per_topic": int(args.max_child_assignments_per_topic),
        "child_assignment_kind": str(args.child_assignment_kind),
        "activate": bool(args.activate),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
