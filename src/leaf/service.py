from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clients import ChatClient, EmbeddingClient
from .config import load_config
from .extract import AtomExtractor
from .indexer import LEAFIndexer
from .store import SQLiteMemoryStore
from .search import retrieve_leaf_memory


class LEAFService:
    def __init__(self, config_path: str | Path, db_path: str | Path):
        self.config = load_config(config_path)
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

    def append_json(self, corpus_id: str, title: str, path: str | Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        turns = self._normalize_turns(payload)
        return self.append_turns(corpus_id=corpus_id, title=title, turns=turns)

    def append_turns(self, corpus_id: str, title: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
        self._search_corpus_cache.pop(str(corpus_id), None)
        return self.indexer.append_turns(corpus_id=corpus_id, title=title, turns=turns)

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
        session_snapshots = self.store.list_snapshots(corpus_id=key, snapshot_kind="session")
        session_pages = self.store.list_snapshots(corpus_id=key, snapshot_kind="session_page")
        session_blocks = self.store.list_snapshots(corpus_id=key, snapshot_kind="session_block")
        all_events = self.store.get_events(corpus_id=key)
        event_lookup = {event.event_id: event for event in all_events}
        session_turn_lookup: dict[str, dict[int, Any]] = {}
        ordered_session_ids: list[str] = []
        for event in all_events:
            session_id = str(event.session_id)
            if session_id not in session_turn_lookup:
                session_turn_lookup[session_id] = {}
                ordered_session_ids.append(session_id)
            session_turn_lookup[session_id][int(event.turn_index)] = event
        cached = {
            "root_snapshot": root_snapshot,
            "entity_snapshots": entity_snapshots,
            "session_snapshots": session_snapshots,
            "session_pages": session_pages,
            "session_blocks": session_blocks,
            "all_events": all_events,
            "event_lookup": event_lookup,
            "session_turn_lookup": session_turn_lookup,
            "ordered_session_ids": ordered_session_ids,
            "entity_events": {},
            "objects_by_subject": {},
            "object_by_id": {},
            "latest_version_by_object": {},
            "token_to_event_ids": None,
            "session_event_rows": {},
        }
        self._search_corpus_cache[key] = cached
        return cached

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
