from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_locomo import load_locomo_samples, locomo_sample_to_turns, sanitize_sample_id
from leaf.service import LEAFService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoCoMo as incremental online ingest and record active topic-tree growth.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--max-turns-per-sample", type=int, default=0)
    parser.add_argument("--chunk-turns", type=int, default=25)
    parser.add_argument("--chunk-limit", type=int, default=0)
    parser.add_argument("--turns-threshold", type=int, default=40)
    parser.add_argument("--atoms-threshold", type=int, default=24)
    parser.add_argument("--trigger-policy", choices=["any", "all", "turns"], default="any")
    parser.add_argument("--growth-strategy", choices=["global_terms", "node_local"], default="global_terms")
    parser.add_argument("--min-cluster-atoms", type=int, default=2)
    parser.add_argument("--max-new-topics", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--window-atom-limit", type=int, default=80)
    parser.add_argument("--disable-secondary-assignment", action="store_true")
    parser.add_argument("--secondary-max-assignments", type=int, default=50)
    parser.add_argument("--secondary-min-score", type=float, default=3.0)
    parser.add_argument("--secondary-min-term-overlap", type=int, default=2)
    parser.add_argument("--secondary-min-embedding-score", type=float, default=0.0)
    parser.add_argument("--secondary-text-mode", choices=["content_entities", "content_only"], default="content_entities")
    parser.add_argument("--secondary-max-profile-terms", type=int, default=0)
    parser.add_argument("--secondary-min-score-margin", type=float, default=0.0)
    parser.add_argument("--secondary-min-score-ratio", type=float, default=0.0)
    parser.add_argument("--disable-evolved-primary-assignment", action="store_true")
    parser.add_argument(
        "--evolved-primary-assignment-mode",
        choices=["all", "none", "quality_v0", "quality_v1"],
        default="all",
        help="Controls whether evolved topics can become primary atom assignments.",
    )
    parser.add_argument("--assign-existing-atoms", action="store_true")
    parser.add_argument("--view-name", default="online-topic-growth-seed-v0")
    parser.add_argument("--print-tree", action="store_true")
    return parser.parse_args()


def _view_summary(service: LEAFService, corpus_id: str) -> dict[str, Any]:
    view = service.get_active_agentic_memory_view(corpus_id)
    if view is None:
        return {"active_view_id": None, "node_count": 0, "evolved_node_count": 0}
    nodes = service.store.list_topic_nodes(str(view["view_id"]))
    evolved_nodes = [
        node
        for node in nodes
        if (node.get("metadata") or {}).get("evolved_slug")
        or (node.get("metadata") or {}).get("evolution_source")
    ]
    return {
        "active_view_id": str(view["view_id"]),
        "view_name": str(view.get("name") or ""),
        "node_count": len(nodes),
        "evolved_node_count": len(evolved_nodes),
        "metadata": dict(view.get("metadata") or {}),
    }


def _topic_tree(service: LEAFService, corpus_id: str, *, enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    return service.get_agentic_topic_tree(corpus_id)


def _chunk_turns(turns: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    return [turns[index : index + size] for index in range(0, len(turns), size)]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = load_locomo_samples(args.input)
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]

    service = LEAFService(config_path=args.config, db_path=args.db)
    evolved_primary_assignment_mode = str(args.evolved_primary_assignment_mode)
    if args.disable_evolved_primary_assignment:
        evolved_primary_assignment_mode = "none"
    if service.config.ingest is not None:
        service.config.ingest.online_evolution_enabled = True
        service.config.ingest.online_evolution_turns_threshold = max(1, int(args.turns_threshold))
        service.config.ingest.online_evolution_atoms_threshold = max(1, int(args.atoms_threshold))
        service.config.ingest.online_evolution_trigger_policy = str(args.trigger_policy)
        service.config.ingest.online_evolution_growth_strategy = str(args.growth_strategy)
        service.config.ingest.online_evolution_min_cluster_atoms = max(1, int(args.min_cluster_atoms))
        service.config.ingest.online_evolution_max_new_topics = max(0, int(args.max_new_topics))
        service.config.ingest.online_evolution_max_depth = max(1, int(args.max_depth))
        service.config.ingest.online_evolution_window_atom_limit = max(1, int(args.window_atom_limit))
        service.config.ingest.online_evolution_secondary_assignment_enabled = not bool(
            args.disable_secondary_assignment
        )
        service.config.ingest.online_evolution_secondary_max_assignments = max(0, int(args.secondary_max_assignments))
        service.config.ingest.online_evolution_secondary_min_score = float(args.secondary_min_score)
        service.config.ingest.online_evolution_secondary_min_term_overlap = max(
            1, int(args.secondary_min_term_overlap)
        )
        service.config.ingest.online_evolution_secondary_min_embedding_score = float(args.secondary_min_embedding_score)
        service.config.ingest.online_evolution_secondary_text_mode = str(args.secondary_text_mode)
        service.config.ingest.online_evolution_secondary_max_profile_terms = max(
            0, int(args.secondary_max_profile_terms)
        )
        service.config.ingest.online_evolution_secondary_min_score_margin = float(args.secondary_min_score_margin)
        service.config.ingest.online_evolution_secondary_min_score_ratio = float(args.secondary_min_score_ratio)
        service.config.ingest.online_evolution_evolved_primary_assignment_enabled = not bool(
            args.disable_evolved_primary_assignment
        )
        service.config.ingest.online_evolution_evolved_primary_assignment_mode = evolved_primary_assignment_mode
    rows: list[dict[str, Any]] = []
    try:
        for sample_index, sample in enumerate(samples, start=1):
            sample_id, turns = locomo_sample_to_turns(sample)
            if args.max_turns_per_sample > 0:
                turns = turns[: args.max_turns_per_sample]
            corpus_id = f"locomo_{sanitize_sample_id(sample_id)}"
            print(
                f"[online-growth] sample={sample_id} corpus={corpus_id} turns={len(turns)} bootstrap_start",
                flush=True,
            )
            bootstrap = service.ensure_seed_agentic_memory_view(
                corpus_id,
                name=args.view_name,
                activate=True,
                assign_existing_atoms=bool(args.assign_existing_atoms),
                metadata={
                    "created_by": "scripts/run_locomo_online_growth.py",
                    "online_growth_runner": True,
                },
            )
            before = _view_summary(service, corpus_id)
            chunks = _chunk_turns(turns, args.chunk_turns)
            if args.chunk_limit > 0:
                chunks = chunks[: args.chunk_limit]
            chunk_rows: list[dict[str, Any]] = []
            for chunk_index, chunk in enumerate(chunks, start=1):
                started = time.perf_counter()
                ingest = service.append_turns(
                    corpus_id=corpus_id,
                    title=f"LoCoMo {sample_id}",
                    turns=chunk,
                    ingest_mode="online",
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
                after = _view_summary(service, corpus_id)
                evolution = dict(ingest.get("agentic_memory_evolution") or {})
                row = {
                    "chunk_index": chunk_index,
                    "turn_start": (chunk_index - 1) * max(1, int(args.chunk_turns)),
                    "turn_count": len(chunk),
                    "elapsed_ms": elapsed_ms,
                    "events_written": int(ingest.get("events_written") or 0),
                    "atoms_written": int(ingest.get("atoms_written") or 0),
                    "assignment": ingest.get("agentic_memory_assignment"),
                    "evolution": evolution,
                    "active_view_before": before,
                    "active_view_after": after,
                    "new_active_view": before.get("active_view_id") != after.get("active_view_id"),
                }
                chunk_rows.append(row)
                before = after
                status = evolution.get("status") or evolution.get("reason")
                print(
                    "[online-growth] "
                    f"sample={sample_id} chunk={chunk_index}/{len(chunks)} "
                    f"turns={len(chunk)} atoms={row['atoms_written']} "
                    f"evo_triggered={bool(evolution.get('triggered'))} "
                    f"status={status} active_view={after.get('active_view_id')} "
                    f"nodes={after.get('node_count')} evolved_nodes={after.get('evolved_node_count')}",
                    flush=True,
                )
            sample_row = {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "corpus_id": corpus_id,
                "turn_count": len(turns),
                "chunk_turns": max(1, int(args.chunk_turns)),
                "chunk_count": len(chunks),
                "bootstrap": bootstrap,
                "final_view": _view_summary(service, corpus_id),
                "tree": _topic_tree(service, corpus_id, enabled=bool(args.print_tree)),
                "chunks": chunk_rows,
            }
            rows.append(sample_row)
            output_path.write_text(
                json.dumps({"completed": False, "rows": rows}, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    finally:
        service.close()

    payload = {
        "completed": True,
        "config": args.config,
        "db": args.db,
        "input": args.input,
        "sample_limit": args.sample_limit,
        "max_turns_per_sample": args.max_turns_per_sample,
        "chunk_turns": args.chunk_turns,
        "chunk_limit": args.chunk_limit,
        "online_evolution": {
            "turns_threshold": args.turns_threshold,
            "atoms_threshold": args.atoms_threshold,
            "trigger_policy": args.trigger_policy,
            "growth_strategy": args.growth_strategy,
            "min_cluster_atoms": args.min_cluster_atoms,
            "max_new_topics": args.max_new_topics,
            "max_depth": args.max_depth,
            "window_atom_limit": args.window_atom_limit,
            "secondary_assignment_enabled": not bool(args.disable_secondary_assignment),
            "secondary_max_assignments": args.secondary_max_assignments,
            "secondary_min_score": args.secondary_min_score,
            "secondary_min_term_overlap": args.secondary_min_term_overlap,
            "secondary_min_embedding_score": args.secondary_min_embedding_score,
            "secondary_text_mode": args.secondary_text_mode,
            "secondary_max_profile_terms": args.secondary_max_profile_terms,
            "secondary_min_score_margin": args.secondary_min_score_margin,
            "secondary_min_score_ratio": args.secondary_min_score_ratio,
            "evolved_primary_assignment_enabled": not bool(args.disable_evolved_primary_assignment),
            "evolved_primary_assignment_mode": evolved_primary_assignment_mode,
        },
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"completed": True, "sample_count": len(rows), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
