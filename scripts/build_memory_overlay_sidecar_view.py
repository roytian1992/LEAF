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

from leaf.agentic_memory import stable_hash, utc_now_iso
from leaf.memory_overlay import (
    build_entity_profile_overlay,
    build_event_overlay,
    build_temporal_overlay,
    default_retrieval_policy_overlay,
    group_atoms_by_event,
    infer_atom_facets,
    score_atom_utility,
)
from leaf.store import SQLiteMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a metadata-only memory overlay sidecar view for a corpus that may not have a topic tree. "
            "The sidecar leaves LEAF events/atoms unchanged and can be used by retrieval-mode=overlay_selective."
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="memory-overlay-sidecar")
    parser.add_argument("--activate", action="store_true")
    return parser.parse_args()


def build_sidecar(store: SQLiteMemoryStore, corpus_id: str) -> dict[str, Any]:
    events = store.get_events(corpus_id=corpus_id)
    atoms = store.list_atoms(corpus_id)
    atoms_by_event = group_atoms_by_event(atoms)
    event_overlay: dict[str, dict[str, Any]] = {}
    answerability_values: list[float] = []
    facet_counts: dict[str, int] = {}
    for event in events:
        overlay = build_event_overlay(event=event, atoms=atoms_by_event.get(str(event.event_id), []), assignment_slugs=[])
        event_overlay[str(event.event_id)] = overlay
        answerability_values.append(float((overlay.get("utility") or {}).get("answerability") or 0.0))
        for facet in overlay.get("facets") or []:
            facet_counts[str(facet)] = facet_counts.get(str(facet), 0) + 1
    atom_overlay = {
        str(atom.atom_id): {
            "facets": infer_atom_facets(atom),
            "utility": score_atom_utility(atom),
        }
        for atom in atoms
    }
    entity_profile_overlay = build_entity_profile_overlay(events, atoms)
    return {
        "event_overlay": event_overlay,
        "atom_overlay": atom_overlay,
        "entity_profile_overlay": entity_profile_overlay,
        "temporal_overlay": build_temporal_overlay(events, atoms),
        "metrics": {
            "event_overlay_count": len(event_overlay),
            "atom_overlay_count": len(atom_overlay),
            "entity_profile_count": len(entity_profile_overlay),
            "temporal_event_count": len(events),
            "facet_counts": dict(sorted(facet_counts.items())),
            "avg_answerability": round(sum(answerability_values) / max(1, len(answerability_values)), 4),
        },
    }


def main() -> None:
    args = parse_args()
    store = SQLiteMemoryStore(args.db)
    results: list[dict[str, Any]] = []
    try:
        for corpus_id in [str(item).strip() for item in args.corpus_id if str(item).strip()]:
            sidecar = build_sidecar(store, corpus_id)
            view_id = f"view_{stable_hash(corpus_id, args.name, uuid.uuid4().hex, length=18)}"
            metadata: dict[str, Any] = {
                "view_type": "memory_overlay_sidecar_v1",
                "base_view_id": None,
                "overlay_policy": {
                    "version": "overlay_sidecar_v1",
                    "answer_context_enabled": True,
                    "entity_profile_enabled": True,
                    "temporal_overlay_enabled": True,
                    "utility_enabled": True,
                    "facet_enabled": True,
                },
                "retrieval_policy_overlay": default_retrieval_policy_overlay(),
                "event_overlay": sidecar["event_overlay"],
                "atom_overlay": sidecar["atom_overlay"],
                "entity_profile_overlay": sidecar["entity_profile_overlay"],
                "temporal_overlay": sidecar["temporal_overlay"],
                "overlay_artifact_note": "Metadata-only sidecar over existing LEAF events/atoms; no topic tree required.",
            }
            metrics = dict(sidecar["metrics"])
            store.upsert_memory_view(
                view_id=view_id,
                corpus_id=corpus_id,
                parent_view_id=None,
                name=f"{args.name}-{corpus_id}-{utc_now_iso()}",
                status="active" if args.activate else "candidate",
                active=False,
                created_at=utc_now_iso(),
                metadata=metadata,
                metrics=metrics,
            )
            if args.activate:
                store.promote_memory_view(view_id, promoted_at=utc_now_iso())
            results.append(
                {
                    "corpus_id": corpus_id,
                    "candidate_view_id": view_id,
                    "activated": bool(args.activate),
                    "metrics": metrics,
                }
            )
        store.commit()
    finally:
        store.close()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": utc_now_iso(), "db": str(args.db), "results": results}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
