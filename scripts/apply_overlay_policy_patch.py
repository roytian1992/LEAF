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
from leaf.store import SQLiteMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone a memory view and apply an overlay retrieval/context policy proposal to the clone metadata. "
            "This is sidecar-only: events, atoms, topic nodes, and assignments are not changed."
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--corpus-id", action="append", default=[])
    parser.add_argument("--base-view-id", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="evolved-overlay-policy")
    parser.add_argument("--activate", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value or {}, ensure_ascii=False))


def select_views(store: SQLiteMemoryStore, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.base_view_id:
        view = store.get_memory_view(args.base_view_id)
        if view is None:
            raise SystemExit(f"Unknown base view id: {args.base_view_id}")
        return [view]
    corpus_ids = [str(item).strip() for item in args.corpus_id if str(item).strip()]
    if not corpus_ids:
        raise SystemExit("Provide --base-view-id or at least one --corpus-id")
    views: list[dict[str, Any]] = []
    for corpus_id in corpus_ids:
        view = store.get_active_memory_view(corpus_id)
        if view is None:
            raise SystemExit(f"No active memory view for corpus_id={corpus_id}")
        views.append(view)
    return views


def clone_view_shell(
    store: SQLiteMemoryStore,
    *,
    base_view: dict[str, Any],
    candidate_view_id: str,
    name: str,
    metadata: dict[str, Any],
    activate: bool,
) -> None:
    store.upsert_memory_view(
        view_id=candidate_view_id,
        corpus_id=str(base_view["corpus_id"]),
        parent_view_id=str(base_view["view_id"]),
        name=name,
        status="active" if activate else "candidate",
        active=False,
        created_at=utc_now_iso(),
        metadata=metadata,
        metrics=json_clone(base_view.get("metrics")),
    )
    if activate:
        store.promote_memory_view(candidate_view_id, promoted_at=utc_now_iso())


def main() -> None:
    args = parse_args()
    proposal = load_json(args.proposal)
    policy_patch = proposal.get("policy_patch")
    if not isinstance(policy_patch, dict):
        raise SystemExit("Proposal must contain object field policy_patch")

    store = SQLiteMemoryStore(args.db)
    try:
        results: list[dict[str, Any]] = []
        for base_view in select_views(store, args):
            corpus_id = str(base_view["corpus_id"])
            base_view_id = str(base_view["view_id"])
            candidate_view_id = f"view_{stable_hash(corpus_id, args.name, base_view_id, uuid.uuid4().hex, length=18)}"
            metadata = json_clone(base_view.get("metadata"))
            prior_policy = json_clone(metadata.get("retrieval_policy_overlay"))
            metadata["retrieval_policy_overlay"] = policy_patch
            metadata["overlay_policy_patch"] = {
                "proposal": str(args.proposal),
                "applied_at": utc_now_iso(),
                "base_view_id": base_view_id,
                "prior_policy_version": prior_policy.get("version"),
                "new_policy_version": policy_patch.get("version"),
            }
            metadata["view_type"] = "evolved_memory_overlay_policy_v1"
            clone_view_shell(
                store,
                base_view=base_view,
                candidate_view_id=candidate_view_id,
                name=f"{args.name}-{corpus_id}-{utc_now_iso()}",
                metadata=metadata,
                activate=bool(args.activate),
            )
            run_id = f"run_{stable_hash(corpus_id, candidate_view_id, args.proposal, length=20)}"
            store.add_evolution_run(
                run_id=run_id,
                corpus_id=corpus_id,
                base_view_id=base_view_id,
                candidate_view_id=candidate_view_id,
                trigger={
                    "kind": "memory_overlay_policy_patch",
                    "proposal": str(args.proposal),
                    "activated": bool(args.activate),
                },
                status="promoted" if args.activate else "candidate",
                result={
                    "policy_version": policy_patch.get("version"),
                    "prior_policy_version": prior_policy.get("version"),
                    "sidecar_only": True,
                },
                created_at=utc_now_iso(),
                completed_at=utc_now_iso(),
            )
            results.append(
                {
                    "corpus_id": corpus_id,
                    "base_view_id": base_view_id,
                    "candidate_view_id": candidate_view_id,
                    "run_id": run_id,
                    "status": "active" if args.activate else "candidate",
                    "policy_version": policy_patch.get("version"),
                }
            )
        store.commit()
    finally:
        store.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": utc_now_iso(),
        "db": str(args.db),
        "proposal": str(args.proposal),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
