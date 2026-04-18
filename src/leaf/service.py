from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
            ChatClient(self.config.additional_llm)
            if self.config.additional_llm and self.config.additional_llm.base_url
            else self.llm
        )
        self.embedding = EmbeddingClient(self.config.embedding) if self.config.embedding.base_url else None
        self.indexer = LEAFIndexer(
            store=self.store,
            atom_extractor=AtomExtractor(self.memory_llm),
            embedding_client=self.embedding,
            reconciliation_llm=self.memory_llm,
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
    ) -> dict[str, Any]:
        if self.embedding is None:
            raise RuntimeError("Embedding model is not configured.")
        corpus_cache = self._get_search_corpus_cache(corpus_id)
        return retrieve_leaf_memory(
            store=self.store,
            corpus_id=corpus_id,
            question=question,
            embedding=self.embedding,
            snapshot_limit=snapshot_limit,
            raw_span_limit=raw_span_limit,
            corpus_cache=corpus_cache,
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
