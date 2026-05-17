from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .agentic_memory import (
    active_topic_hints_for_text,
    assign_atoms_to_topic_view,
    bootstrap_seed_memory_view,
    grow_topic_tree_from_recent_atoms,
    route_query_to_topics,
    stable_hash,
    topic_tree_outline,
    utc_now_iso,
)
from .clients import ChatClient, EmbeddingClient
from .config import load_config
from .extract import AtomExtractor, set_language_mode as set_extract_language_mode
from .grounding import set_language_mode as set_grounding_language_mode
from .indexer import LEAFIndexer
from .store import SQLiteMemoryStore
from .search import retrieve_leaf_memory


class LEAFService:
    def __init__(self, config_path: str | Path, db_path: str | Path):
        self.config = load_config(config_path)
        language_mode = str((self.config.language.mode if self.config.language is not None else "en") or "en").strip().lower()
        set_extract_language_mode(language_mode)
        set_grounding_language_mode(language_mode)
        self.store = SQLiteMemoryStore(db_path)
        self._search_corpus_cache: dict[str, dict[str, Any]] = {}
        self.llm = ChatClient(self.config.llm) if self.config.llm.base_url else None
        self.memory_llm = (
            ChatClient(self.config.memory_llm)
            if self.config.memory_llm and self.config.memory_llm.base_url
            else self.llm
        )
        self.embedding = EmbeddingClient(self.config.embedding) if self.config.embedding.base_url else None
        self.indexer = LEAFIndexer(
            store=self.store,
            atom_extractor=AtomExtractor(self.memory_llm),
            embedding_client=self.embedding,
            reconciliation_llm=self.memory_llm,
            topic_hint_provider=self._topic_hints_for_extraction,
        )

    def close(self) -> None:
        self.store.close()

    def append_json(
        self,
        corpus_id: str,
        title: str,
        path: str | Path,
        *,
        ingest_mode: str | None = None,
    ) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        turns = self._normalize_turns(payload)
        return self.append_turns(corpus_id=corpus_id, title=title, turns=turns, ingest_mode=ingest_mode)

    def append_turns(
        self,
        corpus_id: str,
        title: str,
        turns: list[dict[str, Any]],
        *,
        ingest_mode: str | None = None,
    ) -> dict[str, Any]:
        self._search_corpus_cache.pop(str(corpus_id), None)
        resolved_mode = self._resolve_ingest_mode(ingest_mode)
        result = self.indexer.append_turns(
            corpus_id=corpus_id,
            title=title,
            turns=turns,
            refresh_snapshots=(resolved_mode != "migration"),
        )
        if resolved_mode == "migration":
            migration = self.migrate_corpus(corpus_id=corpus_id, title=title)
            result = self._merge_ingest_and_migration_results(result, migration)
        result["agentic_memory_assignment"] = self.assign_new_atoms_to_active_memory_view(
            corpus_id=corpus_id,
            atom_ids=[str(atom_id) for atom_id in (result.get("written_atom_ids") or [])],
        )
        if resolved_mode == "migration":
            result["agentic_memory_evolution"] = {
                "triggered": False,
                "reason": "migration_mode_batch_import",
            }
        else:
            result["agentic_memory_evolution"] = self.maybe_evolve_agentic_memory_after_ingest(
                corpus_id=corpus_id,
                ingest_result=result,
                use_config=True,
            )
        result["ingest_mode"] = resolved_mode
        return result

    def migrate_corpus(
        self,
        corpus_id: str,
        title: str | None = None,
        *,
        refresh_derived: bool | None = None,
        build_entity_facets: bool | None = None,
        build_entity_bridges: bool | None = None,
        bridge_mode: str | None = None,
    ) -> dict[str, Any]:
        resolved_title = self._resolve_corpus_title(corpus_id=corpus_id, title=title)
        ingest_cfg = self.config.ingest
        refresh_flag = ingest_cfg.migration_refresh_derived if refresh_derived is None else bool(refresh_derived)
        facet_flag = (
            ingest_cfg.migration_build_entity_facets if build_entity_facets is None else bool(build_entity_facets)
        )
        bridge_flag = (
            ingest_cfg.migration_build_entity_bridges if build_entity_bridges is None else bool(build_entity_bridges)
        )
        bridge_mode_value = str(bridge_mode or ingest_cfg.migration_bridge_mode or "hybrid").strip().lower()

        self._search_corpus_cache.pop(str(corpus_id), None)
        started_at = time.perf_counter()
        steps: dict[str, Any] = {}
        if refresh_flag:
            steps["derived"] = self.indexer.backfill_derived_snapshots(
                corpus_id=corpus_id,
                title=resolved_title,
                refresh=True,
            )
        if facet_flag:
            steps["entity_facets"] = self.indexer.backfill_entity_facets(
                corpus_id=corpus_id,
                title=resolved_title,
            )
        if bridge_flag:
            steps["entity_bridges"] = self.indexer.backfill_entity_bridges(
                corpus_id=corpus_id,
                title=resolved_title,
                refresh=True,
                mode=bridge_mode_value,
            )
        return {
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
            "refresh_derived": refresh_flag,
            "build_entity_facets": facet_flag,
            "build_entity_bridges": bridge_flag,
            "bridge_mode": bridge_mode_value,
            "steps": steps,
        }

    def backfill_entity_bridges(
        self,
        corpus_id: str,
        title: str | None = None,
        *,
        refresh: bool = True,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        resolved_title = self._resolve_corpus_title(corpus_id=corpus_id, title=title)
        self._search_corpus_cache.pop(str(corpus_id), None)
        return self.indexer.backfill_entity_bridges(
            corpus_id=corpus_id,
            title=resolved_title,
            refresh=refresh,
            mode=mode,
        )

    def backfill_derived_snapshots(
        self,
        corpus_id: str,
        title: str | None = None,
        *,
        refresh: bool = True,
    ) -> dict[str, Any]:
        resolved_title = self._resolve_corpus_title(corpus_id=corpus_id, title=title)
        self._search_corpus_cache.pop(str(corpus_id), None)
        return self.indexer.backfill_derived_snapshots(
            corpus_id=corpus_id,
            title=resolved_title,
            refresh=refresh,
        )

    def backfill_entity_facets(
        self,
        corpus_id: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        resolved_title = self._resolve_corpus_title(corpus_id=corpus_id, title=title)
        self._search_corpus_cache.pop(str(corpus_id), None)
        return self.indexer.backfill_entity_facets(
            corpus_id=corpus_id,
            title=resolved_title,
        )

    def search(
        self,
        corpus_id: str,
        question: str,
        snapshot_limit: int = 6,
        raw_span_limit: int = 8,
        *,
        local_support_mode: str = "off",
        trace_memory: bool = False,
    ) -> dict[str, Any]:
        if self.embedding is None:
            raise RuntimeError("Embedding model is not configured.")
        corpus_cache = self._get_search_corpus_cache(corpus_id)
        result = retrieve_leaf_memory(
            store=self.store,
            corpus_id=corpus_id,
            question=question,
            embedding=self.embedding,
            snapshot_limit=snapshot_limit,
            raw_span_limit=raw_span_limit,
            corpus_cache=corpus_cache,
            local_support_mode=local_support_mode,
        )
        active_view = self.store.get_active_memory_view(corpus_id)
        if active_view is not None:
            result["agentic_memory"] = {
                "active_view_id": active_view["view_id"],
                "view_name": active_view["name"],
            }
        if trace_memory:
            self._record_search_trace(
                corpus_id=corpus_id,
                question=question,
                result=result,
                active_view=active_view,
            )
        return result

    def bootstrap_agentic_memory_view(
        self,
        corpus_id: str,
        *,
        name: str = "seed-topic-tree-v0",
        parent_view_id: str | None = None,
        activate: bool = False,
        assign_atoms: bool = True,
        assignment_limit: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = bootstrap_seed_memory_view(
            self.store,
            corpus_id=corpus_id,
            name=name,
            parent_view_id=parent_view_id,
            activate=activate,
            assign_atoms=assign_atoms,
            assignment_limit=assignment_limit,
            metadata=metadata,
        )
        self._search_corpus_cache.pop(str(corpus_id), None)
        return result

    def ensure_seed_agentic_memory_view(
        self,
        corpus_id: str,
        *,
        name: str = "seed-topic-tree-v0",
        activate: bool = True,
        assign_existing_atoms: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_view = self.store.get_active_memory_view(corpus_id)
        if active_view is not None:
            return {
                "created": False,
                "view": active_view,
                "assignment": {"assignments_written": 0, "reason": "active_view_exists"},
            }
        return {
            "created": True,
            "view": self.bootstrap_agentic_memory_view(
                corpus_id=corpus_id,
                name=name,
                activate=activate,
                assign_atoms=assign_existing_atoms,
                metadata={
                    "created_by": "LEAFService.ensure_seed_agentic_memory_view",
                    **(metadata or {}),
                },
            ),
        }

    def get_active_agentic_memory_view(self, corpus_id: str) -> dict[str, Any] | None:
        return self.store.get_active_memory_view(corpus_id)

    def list_agentic_memory_views(self, corpus_id: str) -> list[dict[str, Any]]:
        return self.store.list_memory_views(corpus_id)

    def assign_new_atoms_to_active_memory_view(
        self,
        corpus_id: str,
        atom_ids: list[str],
    ) -> dict[str, Any]:
        active_view = self.store.get_active_memory_view(corpus_id)
        normalized_atom_ids = list(dict.fromkeys(str(atom_id).strip() for atom_id in atom_ids if str(atom_id).strip()))
        if active_view is None:
            return {
                "assigned": False,
                "reason": "no_active_view",
                "atom_count": len(normalized_atom_ids),
                "assignments_written": 0,
            }
        if not normalized_atom_ids:
            return {
                "assigned": False,
                "reason": "no_new_atoms",
                "active_view_id": active_view["view_id"],
                "atom_count": 0,
                "assignments_written": 0,
            }
        assignment = assign_atoms_to_topic_view(
            self.store,
            corpus_id=corpus_id,
            view_id=str(active_view["view_id"]),
            atom_ids=normalized_atom_ids,
            commit=True,
        )
        self._search_corpus_cache.pop(str(corpus_id), None)
        return {
            "assigned": True,
            "active_view_id": active_view["view_id"],
            **assignment,
        }

    def backfill_active_memory_view_assignments(
        self,
        corpus_id: str,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        active_view = self.store.get_active_memory_view(corpus_id)
        if active_view is None:
            return {
                "assigned": False,
                "reason": "no_active_view",
                "assignments_written": 0,
            }
        assignment = assign_atoms_to_topic_view(
            self.store,
            corpus_id=corpus_id,
            view_id=str(active_view["view_id"]),
            limit=limit,
            commit=True,
        )
        self._search_corpus_cache.pop(str(corpus_id), None)
        return {
            "assigned": True,
            "active_view_id": active_view["view_id"],
            **assignment,
        }

    def route_query_topics(self, corpus_id: str, question: str, *, top_k: int = 3) -> dict[str, Any] | None:
        active_view = self.store.get_active_memory_view(corpus_id)
        if active_view is None:
            return None
        routes = route_query_to_topics(
            self.store,
            view_id=str(active_view["view_id"]),
            query=question,
            top_k=top_k,
        )
        return {
            "active_view_id": active_view["view_id"],
            "view_name": active_view["name"],
            "router": "keyword_shadow_v0",
            "top_k": top_k,
            "routes": routes,
        }

    def get_agentic_topic_tree(self, corpus_id: str, *, view_id: str | None = None) -> dict[str, Any] | None:
        view = self.store.get_memory_view(view_id) if view_id else self.store.get_active_memory_view(corpus_id)
        if view is None:
            return None
        return {
            "view": view,
            "tree": topic_tree_outline(self.store, view_id=str(view["view_id"])),
        }

    def maybe_evolve_agentic_memory_after_ingest(
        self,
        *,
        corpus_id: str,
        ingest_result: dict[str, Any],
        force: bool = False,
        turns_threshold: int = 50,
        atoms_threshold: int = 40,
        min_cluster_atoms: int = 3,
        max_new_topics: int = 3,
        max_depth: int = 4,
        window_atom_limit: int = 80,
        trigger_policy: str = "any",
        growth_strategy: str = "global_terms",
        evolved_primary_assignment_enabled: bool = True,
        evolved_primary_assignment_mode: str = "all",
        use_config: bool = False,
    ) -> dict[str, Any]:
        ingest_cfg = self.config.ingest
        if not force and ingest_cfg is not None and not bool(ingest_cfg.online_evolution_enabled):
            return {"triggered": False, "reason": "online_evolution_disabled"}
        if use_config and ingest_cfg is not None:
            turns_threshold = int(ingest_cfg.online_evolution_turns_threshold)
            atoms_threshold = int(ingest_cfg.online_evolution_atoms_threshold)
            min_cluster_atoms = int(ingest_cfg.online_evolution_min_cluster_atoms)
            max_new_topics = int(ingest_cfg.online_evolution_max_new_topics)
            max_depth = int(ingest_cfg.online_evolution_max_depth)
            window_atom_limit = int(ingest_cfg.online_evolution_window_atom_limit)
            trigger_policy = str(ingest_cfg.online_evolution_trigger_policy)
            growth_strategy = str(ingest_cfg.online_evolution_growth_strategy)
            evolved_primary_assignment_enabled = bool(
                ingest_cfg.online_evolution_evolved_primary_assignment_enabled
            )
            evolved_primary_assignment_mode = str(
                ingest_cfg.online_evolution_evolved_primary_assignment_mode
            )
        evolved_primary_assignment_mode = str(evolved_primary_assignment_mode or "all").strip().lower()
        if not evolved_primary_assignment_enabled and evolved_primary_assignment_mode == "all":
            evolved_primary_assignment_mode = "none"
        secondary_kwargs: dict[str, Any] = {}
        if use_config and ingest_cfg is not None:
            secondary_kwargs = {
                "secondary_assignment_enabled": bool(ingest_cfg.online_evolution_secondary_assignment_enabled),
                "secondary_max_assignments": int(ingest_cfg.online_evolution_secondary_max_assignments),
                "secondary_min_score": float(ingest_cfg.online_evolution_secondary_min_score),
                "secondary_min_term_overlap": int(ingest_cfg.online_evolution_secondary_min_term_overlap),
                "secondary_min_embedding_score": float(ingest_cfg.online_evolution_secondary_min_embedding_score),
                "secondary_text_mode": str(ingest_cfg.online_evolution_secondary_text_mode),
                "secondary_max_profile_terms": int(ingest_cfg.online_evolution_secondary_max_profile_terms),
                "secondary_min_score_margin": float(ingest_cfg.online_evolution_secondary_min_score_margin),
                "secondary_min_score_ratio": float(ingest_cfg.online_evolution_secondary_min_score_ratio),
            }
        active_view = self.store.get_active_memory_view(corpus_id)
        if active_view is None:
            return {"triggered": False, "reason": "no_active_view"}
        written_atom_ids = [str(atom_id) for atom_id in (ingest_result.get("written_atom_ids") or []) if str(atom_id)]
        turn_count = int(ingest_result.get("events_written") or ingest_result.get("turn_count") or 0)
        atom_count = len(written_atom_ids)
        view_metadata = dict(active_view.get("metadata") or {})
        pending = dict(view_metadata.get("online_evolution_pending") or {})
        pending_turns = int(pending.get("turns") or 0) + turn_count
        pending_atoms = int(pending.get("atoms") or 0) + atom_count
        pending_atom_ids = [str(atom_id) for atom_id in (pending.get("atom_ids") or []) if str(atom_id)]
        pending_atom_ids.extend(written_atom_ids)
        pending_atom_ids = list(dict.fromkeys(pending_atom_ids))[-max(1, int(window_atom_limit)) :]

        policy = str(trigger_policy or "any").strip().lower()
        if policy == "all":
            threshold_met = pending_turns >= int(turns_threshold) and pending_atoms >= int(atoms_threshold)
        elif policy == "turns":
            threshold_met = pending_turns >= int(turns_threshold)
        else:
            threshold_met = pending_turns >= int(turns_threshold) or pending_atoms >= int(atoms_threshold)
        should_trigger = force or threshold_met
        if not should_trigger:
            view_metadata["online_evolution_pending"] = {
                "turns": pending_turns,
                "atoms": pending_atoms,
                "atom_ids": pending_atom_ids,
                "updated_at": utc_now_iso(),
            }
            self.store.update_memory_view_metadata(str(active_view["view_id"]), metadata=view_metadata)
            self.store.commit()
            return {
                "triggered": False,
                "reason": "below_threshold",
                "active_view_id": active_view["view_id"],
                "pending_turns": pending_turns,
                "pending_atoms": pending_atoms,
                "turns_threshold": turns_threshold,
                "atoms_threshold": atoms_threshold,
                "trigger_policy": policy,
            }

        run_created_at = utc_now_iso()
        run_id = f"evo_{stable_hash(corpus_id, str(active_view['view_id']), run_created_at, length=24)}"
        trigger = {
            "mode": "online_per_n_conversation",
            "force": bool(force),
            "pending_turns": pending_turns,
            "pending_atoms": pending_atoms,
            "recent_atom_count": len(pending_atom_ids),
            "turns_threshold": turns_threshold,
            "atoms_threshold": atoms_threshold,
            "trigger_policy": policy,
            "min_cluster_atoms": min_cluster_atoms,
            "max_new_topics": max_new_topics,
            "max_depth": max_depth,
            "growth_strategy": growth_strategy,
            "evolved_primary_assignment_enabled": bool(evolved_primary_assignment_enabled),
            "evolved_primary_assignment_mode": evolved_primary_assignment_mode,
        }
        self.store.add_evolution_run(
            run_id=run_id,
            corpus_id=corpus_id,
            base_view_id=str(active_view["view_id"]),
            status="running",
            trigger=trigger,
            result={},
            created_at=run_created_at,
        )
        self.store.commit()
        try:
            growth = grow_topic_tree_from_recent_atoms(
                self.store,
                corpus_id=corpus_id,
                base_view=active_view,
                recent_atom_ids=pending_atom_ids,
                name=f"online-topic-growth-{run_created_at}",
                max_new_topics=max_new_topics,
                min_cluster_atoms=min_cluster_atoms,
                window_atom_limit=window_atom_limit,
                max_depth=max_depth,
                growth_strategy=growth_strategy,
                evolved_primary_assignment_enabled=bool(evolved_primary_assignment_enabled),
                evolved_primary_assignment_mode=evolved_primary_assignment_mode,
                activate=True,
                trigger=trigger,
                **secondary_kwargs,
            )
            completed_at = utc_now_iso()
            status = "promoted" if growth.get("status") == "promoted" else "skipped"
            self.store.add_evolution_run(
                run_id=run_id,
                corpus_id=corpus_id,
                base_view_id=str(active_view["view_id"]),
                candidate_view_id=str(growth.get("view_id") or "") or None,
                status=status,
                trigger=trigger,
                result=growth,
                created_at=run_created_at,
                completed_at=completed_at,
            )
            if growth.get("status") == "promoted":
                new_view = self.store.get_active_memory_view(corpus_id)
                if new_view is not None:
                    metadata = dict(new_view.get("metadata") or {})
                    metadata["online_evolution_pending"] = {
                        "turns": 0,
                        "atoms": 0,
                        "atom_ids": [],
                        "updated_at": completed_at,
                    }
                    self.store.update_memory_view_metadata(str(new_view["view_id"]), metadata=metadata)
            else:
                view_metadata["online_evolution_pending"] = {
                    "turns": 0,
                    "atoms": 0,
                    "atom_ids": [],
                    "updated_at": completed_at,
                    "last_skipped_reason": growth.get("status"),
                }
                self.store.update_memory_view_metadata(str(active_view["view_id"]), metadata=view_metadata)
            self.store.commit()
            self._search_corpus_cache.pop(str(corpus_id), None)
            return {"triggered": True, "run_id": run_id, **growth}
        except Exception as exc:
            completed_at = utc_now_iso()
            self.store.add_evolution_run(
                run_id=run_id,
                corpus_id=corpus_id,
                base_view_id=str(active_view["view_id"]),
                status="failed",
                trigger=trigger,
                result={"error": str(exc)},
                created_at=run_created_at,
                completed_at=completed_at,
            )
            self.store.commit()
            raise

    def _topic_hints_for_extraction(self, corpus_id: str, text: str) -> dict[str, Any] | None:
        return active_topic_hints_for_text(
            self.store,
            corpus_id=corpus_id,
            text=text,
            limit=5,
        )

    def get_root_snapshot(self, corpus_id: str) -> dict[str, Any] | None:
        snapshot = self.store.get_snapshot(corpus_id=corpus_id, snapshot_kind="root", scope_id=corpus_id)
        return snapshot.to_dict() if snapshot else None

    def get_state_snapshot(
        self,
        corpus_id: str,
        entity: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        if entity is not None:
            return self.get_entity_snapshot(corpus_id=corpus_id, entity=entity)
        if session_id is not None:
            return self.get_session_snapshot(corpus_id=corpus_id, session_id=session_id)
        return self.get_root_snapshot(corpus_id=corpus_id)

    def get_session_snapshot(self, corpus_id: str, session_id: str) -> dict[str, Any] | None:
        snapshot = self.store.get_snapshot(corpus_id=corpus_id, snapshot_kind="session", scope_id=session_id)
        if snapshot is None:
            return None
        objects = self.store.get_objects_for_session(corpus_id=corpus_id, session_id=session_id)
        versions = []
        for obj in objects:
            version = self.store.get_latest_version(obj.object_id)
            if version is not None:
                versions.append(version.to_dict())
        payload = snapshot.to_dict()
        payload["active_versions"] = versions
        return payload

    def get_entity_snapshot(self, corpus_id: str, entity: str) -> dict[str, Any] | None:
        snapshot = self.store.get_snapshot(corpus_id=corpus_id, snapshot_kind="entity", scope_id=entity)
        if snapshot is None:
            return None
        objects = self.store.get_objects_for_subject(corpus_id=corpus_id, subject=entity)
        versions = []
        for obj in objects:
            version = self.store.get_latest_version(obj.object_id)
            if version is not None:
                versions.append(version.to_dict())
        payload = snapshot.to_dict()
        payload["active_versions"] = versions
        return payload

    def get_entity_timeline(self, corpus_id: str, entity: str) -> dict[str, Any]:
        objects = self.store.get_objects_for_subject(corpus_id=corpus_id, subject=entity)
        timeline = []
        for obj in objects:
            timeline.append(
                {
                    "object": obj.to_dict(),
                    "versions": [version.to_dict() for version in self.store.get_object_versions(obj.object_id)],
                }
            )
        events = [event.to_dict() for event in self.store.get_events_for_entity(corpus_id=corpus_id, entity=entity, limit=32)]
        return {"entity": entity, "timeline": timeline, "events": events}

    def list_sessions(self, corpus_id: str) -> list[str]:
        return self.store.list_session_ids(corpus_id)

    def list_entities(self, corpus_id: str) -> list[str]:
        return self.store.list_subjects(corpus_id)

    def list_corpora(self) -> list[str]:
        return self.store.list_corpora()

    def _record_search_trace(
        self,
        *,
        corpus_id: str,
        question: str,
        result: dict[str, Any],
        active_view: dict[str, Any] | None,
    ) -> None:
        created_at = utc_now_iso()
        view_id = str(active_view["view_id"]) if active_view is not None else None
        retrieved_ids: list[str] = []
        for page in result.get("pages") or []:
            page_id = str(page.get("page_id") or page.get("snapshot_id") or "").strip()
            if page_id:
                retrieved_ids.append(f"page:{page_id}")
        for raw_span in result.get("raw_spans") or []:
            span_id = str(raw_span.get("event_id") or raw_span.get("span_id") or "").strip()
            if span_id:
                retrieved_ids.append(f"event:{span_id}")
        trace_id = f"trace_{stable_hash(corpus_id, view_id or '', question, created_at, length=24)}"
        self.store.add_search_trace(
            trace_id=trace_id,
            corpus_id=corpus_id,
            view_id=view_id,
            question=question,
            query_schema={
                "snapshot_limit": result.get("snapshot_limit"),
                "raw_span_limit": result.get("raw_span_limit"),
            },
            retrieved_ids=retrieved_ids,
            metrics={
                "search_elapsed_ms": (result.get("timing") or {}).get("search_elapsed_ms"),
                "page_count": len(result.get("pages") or []),
                "raw_span_count": len(result.get("raw_spans") or []),
            },
            created_at=created_at,
        )
        self.store.commit()

    def _get_search_corpus_cache(self, corpus_id: str) -> dict[str, Any]:
        key = str(corpus_id)
        cached = self._search_corpus_cache.get(key)
        if cached is not None:
            return cached
        root_snapshot = self.store.get_snapshot(corpus_id=key, snapshot_kind="root", scope_id=key)
        entity_snapshots = self.store.list_snapshots(corpus_id=key, snapshot_kind="entity")
        entity_facet_snapshots = self.store.list_snapshots(corpus_id=key, snapshot_kind="entity_facet")
        entity_aspect_snapshots = self.store.list_snapshots(corpus_id=key, snapshot_kind="entity_aspect")
        entity_slot_snapshots = self.store.list_snapshots(corpus_id=key, snapshot_kind="entity_slot")
        bridge_snapshots = self.store.list_snapshots(corpus_id=key, snapshot_kind="bridge")
        session_snapshots = self.store.list_snapshots(corpus_id=key, snapshot_kind="session")
        session_pages = self.store.list_snapshots(corpus_id=key, snapshot_kind="session_page")
        session_blocks = self.store.list_snapshots(corpus_id=key, snapshot_kind="session_block")
        all_events = self.store.get_events(corpus_id=key)
        event_lookup = {event.event_id: event for event in all_events}
        session_turn_lookup: dict[str, dict[int, Any]] = {}
        ordered_session_ids: list[str] = []
        entity_event_ids: dict[str, list[str]] = {}
        for event in all_events:
            for entity in event.canonical_entity_refs or []:
                entity_key = str(entity or "").strip().lower()
                if not entity_key:
                    continue
                entity_event_ids.setdefault(entity_key, []).append(str(event.event_id))
        for event in all_events:
            session_id = str(event.session_id)
            if session_id not in session_turn_lookup:
                session_turn_lookup[session_id] = {}
                ordered_session_ids.append(session_id)
            session_turn_lookup[session_id][int(event.turn_index)] = event
        for subject, event_id in self.store.list_subject_event_ids(corpus_id=key):
            subject_key = str(subject or "").strip().lower()
            if not subject_key:
                continue
            entity_event_ids.setdefault(subject_key, []).append(str(event_id))
        entity_event_ids = {
            entity: list(dict.fromkeys(event_ids))
            for entity, event_ids in entity_event_ids.items()
        }
        cached = {
            "root_snapshot": root_snapshot,
            "entity_snapshots": entity_snapshots,
            "entity_facet_snapshots": entity_facet_snapshots,
            "entity_aspect_snapshots": entity_aspect_snapshots,
            "entity_slot_snapshots": entity_slot_snapshots,
            "bridge_snapshots": bridge_snapshots,
            "session_snapshots": session_snapshots,
            "session_pages": session_pages,
            "session_blocks": session_blocks,
            "all_events": all_events,
            "event_lookup": event_lookup,
            "session_turn_lookup": session_turn_lookup,
            "ordered_session_ids": ordered_session_ids,
            "entity_event_ids": entity_event_ids,
            "entity_events": {},
            "objects_by_subject": {},
            "object_by_id": {},
            "latest_version_by_object": {},
            "token_to_event_ids": None,
            "session_event_rows": {},
        }
        self._search_corpus_cache[key] = cached
        return cached

    def _resolve_ingest_mode(self, ingest_mode: str | None) -> str:
        mode = str(ingest_mode or self.config.ingest.mode or "online").strip().lower()
        if mode not in {"online", "migration"}:
            raise ValueError(f"Unsupported ingest mode: {mode}")
        return mode

    def _resolve_corpus_title(self, *, corpus_id: str, title: str | None = None) -> str:
        resolved_title = str(title or "").strip()
        if resolved_title:
            return resolved_title
        root_snapshot = self.store.get_snapshot(corpus_id=corpus_id, snapshot_kind="root", scope_id=corpus_id)
        return str(root_snapshot.title).strip() if root_snapshot is not None else str(corpus_id)

    @staticmethod
    def _extract_snapshot_counts(row: dict[str, Any]) -> dict[str, int]:
        snapshot_kinds = {
            "bridge",
            "entity",
            "entity_aspect",
            "entity_facet",
            "entity_slot",
            "root",
            "session",
            "session_block",
            "session_page",
        }
        counts: dict[str, int] = {}
        for key, value in (row or {}).items():
            if key in snapshot_kinds and isinstance(value, (int, float)):
                counts[str(key)] = int(value)
        return counts

    @staticmethod
    def _sum_numeric_maps(rows: list[dict[str, int]]) -> dict[str, int]:
        total: dict[str, int] = {}
        for row in rows:
            for key, value in row.items():
                total[str(key)] = int(total.get(str(key), 0)) + int(value)
        return dict(sorted(total.items()))

    def _merge_ingest_and_migration_results(
        self,
        ingest_result: dict[str, Any],
        migration_result: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(ingest_result or {})
        migration_payload = dict(migration_result or {})
        merged["migration"] = migration_payload
        base_elapsed = float(merged.get("ingest_elapsed_ms") or 0.0)
        merged["ingest_elapsed_ms"] = round(base_elapsed + float(migration_payload.get("elapsed_ms") or 0.0), 2)

        snapshot_rows = [dict(merged.get("snapshot_upserts_by_kind") or {})]
        for step_result in (migration_payload.get("steps") or {}).values():
            if isinstance(step_result, dict):
                snapshot_rows.append(self._extract_snapshot_counts(step_result))
        snapshot_counts = self._sum_numeric_maps(snapshot_rows)
        merged["snapshot_upserts_by_kind"] = snapshot_counts
        merged["snapshot_upserts_total"] = int(sum(snapshot_counts.values()))
        return merged

    @staticmethod
    def _normalize_turns(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get("turns"), list):
                return [item for item in payload["turns"] if isinstance(item, dict)]
            if isinstance(payload.get("conversation"), list):
                return [item for item in payload["conversation"] if isinstance(item, dict)]
            if isinstance(payload.get("messages"), list):
                return [item for item in payload["messages"] if isinstance(item, dict)]
        raise ValueError("Unsupported conversation JSON format")
