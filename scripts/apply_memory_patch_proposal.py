from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import normalize_topic_slug, stable_hash, topic_node_id, topic_slug, utc_now_iso
from leaf.store import SQLiteMemoryStore


SUPPORTED_PATCH_TYPES = {
    "topic_route_keywords",
    "topic_profile_terms",
    "topic_exposure",
    "answer_exposure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone active memory views and apply low-risk metadata-only patches from a "
            "memory patch proposal. Topic split/merge and retrieval-policy patches are "
            "recorded but not executed."
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="evolvemem-metadata-patch")
    parser.add_argument("--corpus-id", action="append", default=[])
    parser.add_argument("--patch-id", action="append", default=[])
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


def scope_matches(patch: dict[str, Any], *, corpus_id: str, node: dict[str, Any] | None = None) -> bool:
    scope = dict(patch.get("scope") or {})
    scope_corpus = str(scope.get("corpus_id") or "*").strip() or "*"
    if scope_corpus not in {"*", corpus_id}:
        return False
    if node is None:
        return True
    topic_id = str(scope.get("topic_id") or "").strip()
    topic_slug_value = normalize_topic_slug(str(scope.get("topic_slug") or "").strip())
    node_metadata = dict(node.get("metadata") or {})
    aliases = {
        str(node.get("topic_id") or "").strip(),
        str(node_metadata.get("base_topic_id") or "").strip(),
        str(node_metadata.get("source_topic_id") or "").strip(),
        slug_for_node(node),
    }
    aliases.discard("")
    if topic_id and topic_id in aliases:
        return True
    if topic_slug_value and topic_slug_value in aliases:
        return True
    return not topic_id and not topic_slug_value


def selected_patches(proposal: dict[str, Any], *, corpus_ids: set[str], patch_ids: set[str]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    for patch in proposal.get("patches") or []:
        if not isinstance(patch, dict):
            continue
        patch_id = str(patch.get("patch_id") or "").strip()
        if patch_ids and patch_id not in patch_ids:
            continue
        patch_type = str(patch.get("patch_type") or "").strip()
        if patch_type not in SUPPORTED_PATCH_TYPES:
            continue
        scope = dict(patch.get("scope") or {})
        scope_corpus = str(scope.get("corpus_id") or "*").strip() or "*"
        if corpus_ids and scope_corpus not in {"*", *corpus_ids}:
            continue
        if patch.get("invalid_evidence_failure_ids"):
            # The proposal can still be inspected, but this script refuses to execute patches
            # with hallucinated evidence links.
            continue
        patches.append(patch)
    return patches


def patch_node_metadata(node: dict[str, Any], patches: list[dict[str, Any]], *, corpus_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = dict(node.get("metadata") or {})
    applied: list[dict[str, Any]] = []
    for patch in patches:
        if not scope_matches(patch, corpus_id=corpus_id, node=node):
            continue
        change = dict(patch.get("change") or {})
        before = {
            "route_keywords": string_list(metadata.get("route_keywords")),
            "profile_terms": string_list(metadata.get("profile_terms")),
            "route_exposure": metadata.get("route_exposure"),
            "answer_exposure": metadata.get("answer_exposure"),
        }
        add_route_keywords = string_list(change.get("add_route_keywords"))
        remove_route_keywords = {item.lower() for item in string_list(change.get("remove_route_keywords"))}
        if add_route_keywords or remove_route_keywords:
            current = string_list(metadata.get("route_keywords")) or string_list(node.get("keywords"))
            current = [item for item in current if item.lower() not in remove_route_keywords]
            metadata["route_keywords"] = merge_unique(current + add_route_keywords)
        add_profile_terms = string_list(change.get("add_profile_terms"))
        remove_profile_terms = {item.lower() for item in string_list(change.get("remove_profile_terms"))}
        if add_profile_terms or remove_profile_terms:
            current = [item for item in string_list(metadata.get("profile_terms")) if item.lower() not in remove_profile_terms]
            metadata["profile_terms"] = merge_unique(current + add_profile_terms)
        route_exposure = str(change.get("route_exposure") or "").strip()
        if route_exposure:
            metadata["route_exposure"] = route_exposure
        answer_exposure = str(change.get("answer_exposure") or "").strip()
        if answer_exposure:
            metadata["answer_exposure"] = answer_exposure
        after = {
            "route_keywords": string_list(metadata.get("route_keywords")),
            "profile_terms": string_list(metadata.get("profile_terms")),
            "route_exposure": metadata.get("route_exposure"),
            "answer_exposure": metadata.get("answer_exposure"),
        }
        if after != before:
            applied.append(
                {
                    "patch_id": patch.get("patch_id"),
                    "patch_type": patch.get("patch_type"),
                    "before": before,
                    "after": after,
                }
            )
    return metadata, applied


def clone_view_with_patches(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    active_view: dict[str, Any],
    patches: list[dict[str, Any]],
    name: str,
    proposal_path: str,
    activate: bool,
) -> dict[str, Any]:
    now = utc_now_iso()
    base_view_id = str(active_view["view_id"])
    candidate_view_id = f"view_{stable_hash(corpus_id, name, uuid.uuid4().hex, length=18)}"
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

    applied_by_topic: list[dict[str, Any]] = []
    for node in sorted(base_nodes, key=lambda item: (int(item.get("level") or 0), str(item.get("topic_id") or ""))):
        base_topic_id = str(node["topic_id"])
        candidate_topic_id = base_to_candidate[base_topic_id]
        parent_id = base_to_candidate.get(str(node.get("parent_id"))) if node.get("parent_id") else None
        metadata = dict(node.get("metadata") or {})
        metadata.update(
            {
                "base_topic_id": base_topic_id,
                "base_view_id": base_view_id,
                "evolvemem_clone_of_topic_id": base_topic_id,
                "evolvemem_candidate_view_id": candidate_view_id,
            }
        )
        if metadata.get("seed_role") != "root":
            metadata.setdefault("topic_slug", slug_for_node(node))
        patched_metadata, applied = patch_node_metadata(node, patches, corpus_id=corpus_id)
        metadata.update(patched_metadata)
        if applied:
            metadata["evolvemem_applied_patches"] = [
                {"patch_id": item["patch_id"], "patch_type": item["patch_type"]} for item in applied
            ]
            applied_by_topic.append(
                {
                    "base_topic_id": base_topic_id,
                    "candidate_topic_id": candidate_topic_id,
                    "topic_slug": slug_for_node(node),
                    "name": node.get("name"),
                    "applied": applied,
                }
            )
        store.upsert_topic_node(
            topic_id=candidate_topic_id,
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

    copied_assignments = 0
    for assignment in store.list_topic_assignments(base_view_id):
        base_topic_id = str(assignment.get("topic_id") or "")
        candidate_topic_id = base_to_candidate.get(base_topic_id)
        if not candidate_topic_id:
            continue
        store.upsert_topic_assignment(
            assignment_id=f"assign_{stable_hash(candidate_view_id, assignment.get('item_kind'), assignment.get('item_id'), candidate_topic_id, length=20)}",
            view_id=candidate_view_id,
            corpus_id=corpus_id,
            item_kind=str(assignment.get("item_kind") or ""),
            item_id=str(assignment.get("item_id") or ""),
            topic_id=candidate_topic_id,
            confidence=float(assignment.get("confidence") or 0.0),
            reason={
                **dict(assignment.get("reason") or {}),
                "copied_from_view_id": base_view_id,
                "copied_from_assignment_id": assignment.get("assignment_id"),
            },
        )
        copied_assignments += 1

    store.upsert_memory_view(
        view_id=candidate_view_id,
        corpus_id=corpus_id,
        parent_view_id=base_view_id,
        name=name,
        status="active" if activate else "candidate",
        active=False,
        created_at=now,
        metadata={
            "view_type": "evolvemem_metadata_patch",
            "base_view_id": base_view_id,
            "source_proposal": str(proposal_path),
            "supported_patch_types": sorted(SUPPORTED_PATCH_TYPES),
            "applied_patch_ids": sorted({str(item["patch_id"]) for topic in applied_by_topic for item in topic["applied"]}),
        },
        metrics={
            "patched_topic_count": len(applied_by_topic),
            "copied_topic_count": len(base_nodes),
            "copied_assignment_count": copied_assignments,
        },
    )
    if activate:
        store.promote_memory_view(candidate_view_id, promoted_at=utc_now_iso())
    return {
        "corpus_id": corpus_id,
        "base_view_id": base_view_id,
        "candidate_view_id": candidate_view_id,
        "activated": bool(activate),
        "patched_topic_count": len(applied_by_topic),
        "copied_topic_count": len(base_nodes),
        "copied_assignment_count": copied_assignments,
        "applied_by_topic": applied_by_topic,
    }


def main() -> None:
    args = parse_args()
    proposal = load_json(args.proposal)
    requested_corpus_ids = {str(item).strip() for item in args.corpus_id if str(item).strip()}
    requested_patch_ids = {str(item).strip() for item in args.patch_id if str(item).strip()}
    patches = selected_patches(proposal, corpus_ids=requested_corpus_ids, patch_ids=requested_patch_ids)
    corpus_ids = sorted(requested_corpus_ids or {
        str((patch.get("scope") or {}).get("corpus_id") or "").strip()
        for patch in patches
        if str((patch.get("scope") or {}).get("corpus_id") or "").strip() not in {"", "*"}
    })
    if not corpus_ids:
        metadata_corpus_ids = string_list((proposal.get("metadata") or {}).get("corpus_ids"))
        corpus_ids = sorted(metadata_corpus_ids)
    if not corpus_ids:
        raise SystemExit("No corpus ids to patch. Pass --corpus-id or use a proposal with metadata.corpus_ids.")

    store = SQLiteMemoryStore(args.db)
    try:
        results: list[dict[str, Any]] = []
        for corpus_id in corpus_ids:
            active_view = store.get_active_memory_view(corpus_id)
            if active_view is None:
                results.append({"corpus_id": corpus_id, "error": "no_active_memory_view"})
                continue
            corpus_patches = [patch for patch in patches if scope_matches(patch, corpus_id=corpus_id, node=None)]
            result = clone_view_with_patches(
                store,
                corpus_id=corpus_id,
                active_view=active_view,
                patches=corpus_patches,
                name=f"{args.name}-{corpus_id}-{utc_now_iso()}",
                proposal_path=str(args.proposal),
                activate=bool(args.activate),
            )
            results.append(result)
        store.commit()
    finally:
        store.close()

    summary = {
        "created_at": utc_now_iso(),
        "db": str(args.db),
        "proposal": str(args.proposal),
        "output": str(args.output),
        "selected_patch_ids": [patch.get("patch_id") for patch in patches],
        "selected_patch_count": len(patches),
        "supported_patch_types": sorted(SUPPORTED_PATCH_TYPES),
        "activate": bool(args.activate),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
