from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import stable_hash, utc_now_iso
from leaf.memory_overlay import (
    build_event_overlay,
    build_entity_profile_overlay,
    build_temporal_overlay,
    default_retrieval_policy_overlay,
    group_atoms_by_event,
    infer_atom_facets,
    merge_overlay_terms,
    score_atom_utility,
    top_profile_terms_for_topic,
)
from leaf.store import SQLiteMemoryStore
from leaf.topic_soft import topic_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a candidate LEAF memory view with non-destructive sidecar overlays "
            "for profile, temporal, facet, utility, retrieval policy, and answer context."
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", action="append", default=[])
    parser.add_argument("--base-view-id", default="", help="Optional single base view id.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="evolved-overlay-v1")
    parser.add_argument("--max-topic-profile-terms", type=int, default=40)
    parser.add_argument("--max-topic-assignments", type=int, default=0)
    parser.add_argument(
        "--profile-term-target",
        choices=["overlay_profile_terms", "profile_terms"],
        default="overlay_profile_terms",
        help=(
            "Where atom-derived topic terms are stored. overlay_profile_terms is consumed only by "
            "overlay-aware routers; profile_terms reproduces the broader v1 routing behavior."
        ),
    )
    parser.add_argument("--activate", action="store_true")
    parser.add_argument(
        "--enable",
        nargs="+",
        default=["profile", "temporal", "facet", "utility", "policy", "context"],
        choices=["profile", "temporal", "facet", "utility", "policy", "context"],
    )
    return parser.parse_args()


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def clone_view(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    base_view: dict[str, Any],
    candidate_view_id: str,
    name: str,
    enabled: set[str],
    activate: bool,
) -> tuple[dict[str, str], int, int, int]:
    base_view_id = str(base_view["view_id"])
    base_nodes = store.list_topic_nodes(base_view_id)
    base_to_candidate: dict[str, str] = {}
    for node in base_nodes:
        base_topic_id = str(node["topic_id"])
        base_to_candidate[base_topic_id] = f"topic_{stable_hash(candidate_view_id, base_topic_id, length=20)}"

    copied_nodes = 0
    for node in base_nodes:
        base_topic_id = str(node["topic_id"])
        parent_id = base_to_candidate.get(str(node.get("parent_id") or ""))
        metadata = dict(node.get("metadata") or {})
        metadata.update(
            {
                "base_topic_id": base_topic_id,
                "base_view_id": base_view_id,
                "overlay_clone": True,
            }
        )
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
        topic_id = base_to_candidate.get(str(assignment.get("topic_id") or ""))
        if not topic_id:
            continue
        item_kind = str(assignment.get("item_kind") or "")
        item_id = str(assignment.get("item_id") or "")
        reason = dict(assignment.get("reason") or {})
        reason.update({"base_assignment_id": assignment.get("assignment_id"), "base_view_id": base_view_id})
        store.upsert_topic_assignment(
            assignment_id=f"assign_{stable_hash(candidate_view_id, item_kind, item_id, topic_id, length=24)}",
            view_id=candidate_view_id,
            corpus_id=corpus_id,
            item_kind=item_kind,
            item_id=item_id,
            topic_id=topic_id,
            confidence=float(assignment.get("confidence") or 0.0),
            reason=reason,
        )
        copied_assignments += 1

    copied_edges = 0
    for edge in store.conn.execute("select * from leaf_topic_edges where view_id = ?", (base_view_id,)).fetchall():
        src_topic_id = base_to_candidate.get(str(edge["src_topic_id"]))
        dst_topic_id = base_to_candidate.get(str(edge["dst_topic_id"]))
        if not src_topic_id or not dst_topic_id:
            continue
        edge_type = str(edge["edge_type"])
        metadata = json.loads(edge["metadata_json"] or "{}")
        metadata.update({"base_edge_id": edge["edge_id"], "base_view_id": base_view_id})
        store.conn.execute(
            "insert or replace into leaf_topic_edges values (?, ?, ?, ?, ?, ?, ?, ?)",
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

    base_metadata = dict(base_view.get("metadata") or {})
    metadata = {
        **base_metadata,
        "view_type": "evolved_memory_overlay_v1",
        "base_view_id": base_view_id,
        "enabled_overlays": sorted(enabled),
        "overlay_policy": {
            "version": "overlay_v1",
            "search_event_bonus_enabled": "utility" in enabled or "facet" in enabled or "temporal" in enabled,
            "topic_soft_utility_enabled": "utility" in enabled or "facet" in enabled,
            "entity_profile_enabled": "profile" in enabled,
            "temporal_overlay_enabled": "temporal" in enabled,
            "answer_context_enabled": "context" in enabled,
            "max_event_bonus": 0.18,
            "max_topic_soft_atom_bonus": 0.6,
        },
        "retrieval_policy_overlay": default_retrieval_policy_overlay() if "policy" in enabled else {},
        "topic_soft_policy": {
            **dict(base_metadata.get("topic_soft_policy") or {}),
            "overlay_policy": "utility_facet_temporal_v1",
        },
    }
    store.upsert_memory_view(
        view_id=candidate_view_id,
        corpus_id=corpus_id,
        parent_view_id=base_view_id,
        name=name,
        status="active" if activate else "candidate",
        active=False,
        created_at=utc_now_iso(),
        metadata=metadata,
        metrics={},
    )
    return base_to_candidate, copied_nodes, copied_assignments, copied_edges


def build_overlay_for_corpus(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    base_view: dict[str, Any],
    candidate_view_id: str,
    base_to_candidate: dict[str, str],
    enabled: set[str],
    max_topic_profile_terms: int,
    profile_term_target: str,
) -> dict[str, Any]:
    base_view_id = str(base_view["view_id"])
    events = store.get_events(corpus_id=corpus_id)
    atoms = store.list_atoms(corpus_id)
    atoms_by_id = {str(atom.atom_id): atom for atom in atoms}
    atoms_by_event = group_atoms_by_event(atoms)
    base_nodes = store.list_topic_nodes(base_view_id)
    base_node_by_id = {str(node["topic_id"]): node for node in base_nodes}
    base_slug_by_id = {topic_id: topic_slug(base_node_by_id, topic_id) for topic_id in base_node_by_id}

    base_assignments = store.list_topic_assignments(base_view_id)
    assignment_slugs_by_atom: dict[str, list[str]] = defaultdict(list)
    atoms_by_base_topic: dict[str, list[Any]] = defaultdict(list)
    for assignment in base_assignments:
        atom_id = str(assignment.get("item_id") or "")
        topic_id = str(assignment.get("topic_id") or "")
        if not atom_id or not topic_id:
            continue
        slug = base_slug_by_id.get(topic_id)
        if slug:
            assignment_slugs_by_atom[atom_id].append(slug)
        atom = atoms_by_id.get(atom_id)
        if atom is not None:
            atoms_by_base_topic[topic_id].append(atom)

    event_overlays: dict[str, dict[str, Any]] = {}
    facet_counts: Counter[str] = Counter()
    utility_answerability_values: list[float] = []
    for event in events:
        event_atoms = atoms_by_event.get(str(event.event_id), [])
        slugs: list[str] = []
        for atom in event_atoms:
            slugs.extend(assignment_slugs_by_atom.get(str(atom.atom_id), []))
        overlay = build_event_overlay(event=event, atoms=event_atoms, assignment_slugs=slugs)
        event_overlays[str(event.event_id)] = overlay
        facet_counts.update(overlay.get("facets") or [])
        utility_answerability_values.append(float((overlay.get("utility") or {}).get("answerability") or 0.0))

    atom_overlay: dict[str, dict[str, Any]] = {}
    for atom in atoms:
        atom_overlay[str(atom.atom_id)] = {
            "facets": infer_atom_facets(atom),
            "utility": score_atom_utility(atom),
        }

    updated_topics = 0
    if "profile" in enabled or "facet" in enabled:
        candidate_node_by_base_id = {base_id: candidate_id for base_id, candidate_id in base_to_candidate.items()}
        for base_topic_id, atom_rows in atoms_by_base_topic.items():
            candidate_topic_id = candidate_node_by_base_id.get(base_topic_id)
            if not candidate_topic_id:
                continue
            candidate_node = next(
                (node for node in store.list_topic_nodes(candidate_view_id) if str(node["topic_id"]) == candidate_topic_id),
                None,
            )
            if candidate_node is None:
                continue
            profile_terms = top_profile_terms_for_topic(atom_rows, limit=max_topic_profile_terms)
            topic_facets = Counter(facet for atom in atom_rows for facet in infer_atom_facets(atom))
            metadata = dict(candidate_node.get("metadata") or {})
            existing_profile_terms = string_list(metadata.get("profile_terms"))
            if profile_term_target == "profile_terms":
                metadata["profile_terms"] = merge_overlay_terms(
                    existing_profile_terms,
                    profile_terms,
                    limit=max_topic_profile_terms,
                )
            else:
                metadata["profile_terms"] = existing_profile_terms
                metadata["overlay_profile_terms"] = merge_overlay_terms(
                    string_list(metadata.get("overlay_profile_terms")),
                    profile_terms,
                    limit=max_topic_profile_terms,
                )
            metadata["overlay_facets"] = dict(sorted(topic_facets.items()))
            metadata["overlay_profile_source"] = "atom_sidecar_v1"
            stats = dict(candidate_node.get("stats") or {})
            stats["overlay_atom_count"] = len(atom_rows)
            stats["overlay_top_facets"] = dict(topic_facets.most_common(8))
            store.upsert_topic_node(
                topic_id=str(candidate_node["topic_id"]),
                view_id=candidate_view_id,
                parent_id=candidate_node.get("parent_id"),
                name=str(candidate_node.get("name") or ""),
                description=str(candidate_node.get("description") or ""),
                level=int(candidate_node.get("level") or 0),
                keywords=string_list(candidate_node.get("keywords")),
                exemplar_ids=string_list(candidate_node.get("exemplar_ids")),
                stats=stats,
                embedding=candidate_node.get("embedding"),
                metadata=metadata,
            )
            updated_topics += 1

    return {
        "event_overlay_count": len(event_overlays),
        "atom_overlay_count": len(atom_overlay),
        "updated_topic_profiles": updated_topics,
        "facet_counts": dict(sorted(facet_counts.items())),
        "avg_answerability": round(
            sum(utility_answerability_values) / max(1, len(utility_answerability_values)),
            4,
        ),
        "event_overlays": event_overlays,
        "atom_overlay": atom_overlay,
        "entity_profile_overlay": build_entity_profile_overlay(events, atoms) if "profile" in enabled else {},
        "temporal_overlay": build_temporal_overlay(events, atoms) if "temporal" in enabled else {},
    }


def select_base_views(store: SQLiteMemoryStore, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.base_view_id:
        view = store.get_memory_view(args.base_view_id)
        if view is None:
            raise SystemExit(f"Unknown base view id: {args.base_view_id}")
        return [view]
    corpus_ids = [str(item).strip() for item in args.corpus_id if str(item).strip()]
    if not corpus_ids:
        rows = store.conn.execute(
            "select distinct corpus_id from leaf_memory_views where active = 1 order by corpus_id"
        ).fetchall()
        corpus_ids = [str(row["corpus_id"]) for row in rows]
    views: list[dict[str, Any]] = []
    for corpus_id in corpus_ids:
        view = store.get_active_memory_view(corpus_id)
        if view is None:
            raise SystemExit(f"No active memory view for corpus_id={corpus_id}")
        views.append(view)
    return views


def main() -> None:
    args = parse_args()
    enabled = {str(item) for item in args.enable}
    store = SQLiteMemoryStore(args.db)
    try:
        results: list[dict[str, Any]] = []
        for base_view in select_base_views(store, args):
            corpus_id = str(base_view["corpus_id"])
            candidate_view_id = f"view_{stable_hash(corpus_id, args.name, uuid.uuid4().hex, length=18)}"
            base_to_candidate, copied_nodes, copied_assignments, copied_edges = clone_view(
                store,
                corpus_id=corpus_id,
                base_view=base_view,
                candidate_view_id=candidate_view_id,
                name=f"{args.name}-{corpus_id}-{utc_now_iso()}",
                enabled=enabled,
                activate=bool(args.activate),
            )
            overlay = build_overlay_for_corpus(
                store,
                corpus_id=corpus_id,
                base_view=base_view,
                candidate_view_id=candidate_view_id,
                base_to_candidate=base_to_candidate,
                enabled=enabled,
                max_topic_profile_terms=max(0, int(args.max_topic_profile_terms)),
                profile_term_target=str(args.profile_term_target),
            )
            metrics = {
                "copied_nodes": copied_nodes,
                "copied_assignments": copied_assignments,
                "copied_edges": copied_edges,
                "enabled_overlays": sorted(enabled),
                "profile_term_target": str(args.profile_term_target),
                **{
                    k: v
                    for k, v in overlay.items()
                    if k not in {"event_overlays", "atom_overlay", "entity_profile_overlay", "temporal_overlay"}
                },
                "entity_profile_count": len(overlay.get("entity_profile_overlay") or {}),
                "temporal_event_count": int((overlay.get("temporal_overlay") or {}).get("event_count") or 0),
            }
            view = store.get_memory_view(candidate_view_id) or {}
            metadata = dict(view.get("metadata") or {})
            metadata["event_overlay"] = overlay["event_overlays"]
            metadata["atom_overlay"] = overlay["atom_overlay"]
            metadata["entity_profile_overlay"] = overlay["entity_profile_overlay"]
            metadata["temporal_overlay"] = overlay["temporal_overlay"]
            metadata["overlay_artifact_note"] = "Sidecar overlays are stored in view metadata; base LEAF events/atoms are unchanged."
            store.update_memory_view_metadata(candidate_view_id, metadata=metadata)
            store.update_memory_view_metrics(candidate_view_id, metrics=metrics)
            if args.activate:
                store.promote_memory_view(candidate_view_id, promoted_at=utc_now_iso())
            store.add_evolution_run(
                run_id=f"run_{stable_hash(corpus_id, candidate_view_id, 'memory_overlay_v1', length=20)}",
                corpus_id=corpus_id,
                base_view_id=str(base_view["view_id"]),
                candidate_view_id=candidate_view_id,
                trigger={"kind": "memory_overlay_v1", "enabled": sorted(enabled)},
                status="candidate_created",
                result=metrics,
                created_at=utc_now_iso(),
                completed_at=utc_now_iso(),
            )
            results.append(
                {
                    "corpus_id": corpus_id,
                    "base_view_id": str(base_view["view_id"]),
                    "candidate_view_id": candidate_view_id,
                    "metrics": metrics,
                    "activated": bool(args.activate),
                }
            )
        store.commit()
    finally:
        store.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": utc_now_iso(), "db": str(args.db), "results": results}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
