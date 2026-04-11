from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Any

from .clients import ChatClient, EmbeddingClient, OpenAICompatError, extract_json_object
from .extract import (
    AtomExtractor,
    canonicalize_entities,
    extract_entities,
    extract_semantic_references,
    make_synopsis,
    merge_memory_refs,
    stable_id,
    summarize_texts,
)
from .grounding import derive_temporal_grounding, span_surface_text
from .records import (
    MemoryAtomRecord,
    MemoryEventRecord,
    MemoryEvidenceLinkRecord,
    MemoryObjectRecord,
    MemoryObjectVersionRecord,
    MemorySnapshotRecord,
    StateCandidate,
)
from .store import SQLiteMemoryStore
from .schemas import RawSpan

SINGLETON_SLOTS = {"location", "employer", "occupation", "relationship_status", "home_city"}
AMBIGUOUS_ACTIONS = {"PATCH", "SUPERSEDE", "TENTATIVE"}
INGEST_MODE_ONLINE = "online"
INGEST_MODE_MIGRATION = "migration"
INGEST_MODES = {INGEST_MODE_ONLINE, INGEST_MODE_MIGRATION}


def _normalize_value(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]+", " ", str(text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _token_set(text: str) -> set[str]:
    return {token for token in _normalize_value(text).split() if len(token) > 1}


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens.union(right_tokens))


class LEAFIndexer:
    def __init__(
        self,
        store: SQLiteMemoryStore,
        atom_extractor: AtomExtractor,
        embedding_client: EmbeddingClient | None = None,
        reconciliation_llm: ChatClient | None = None,
    ):
        self.store = store
        self.atom_extractor = atom_extractor
        self.embedding_client = embedding_client
        self.reconciliation_llm = reconciliation_llm

    def append_turns(
        self,
        corpus_id: str,
        title: str,
        turns: list[dict[str, Any]],
        ingest_mode: str = INGEST_MODE_ONLINE,
    ) -> dict[str, Any]:
        ingest_mode = self._normalize_ingest_mode(ingest_mode)
        started_at = time.perf_counter()
        prepare_started_at = time.perf_counter()
        prepared_turns = self.prepare_turns(corpus_id=corpus_id, turns=turns, ingest_mode=ingest_mode)
        prepare_elapsed_ms = round((time.perf_counter() - prepare_started_at) * 1000.0, 2)
        result = self.append_prepared_turns(
            corpus_id=corpus_id,
            title=title,
            prepared_turns=prepared_turns,
            ingest_mode=ingest_mode,
        )
        stage_timings_ms = dict(result.get("stage_timings_ms") or {})
        stage_timings_ms["prepare_total_ms"] = prepare_elapsed_ms
        result["stage_timings_ms"] = dict(sorted(stage_timings_ms.items()))
        result["prepared_turn_count"] = len(prepared_turns)
        result["ingest_elapsed_ms"] = round((time.perf_counter() - started_at) * 1000.0, 2)
        return result

    def prepare_turns(
        self,
        corpus_id: str,
        turns: list[dict[str, Any]],
        ingest_mode: str = INGEST_MODE_ONLINE,
    ) -> list[dict[str, Any]]:
        ingest_mode = self._normalize_ingest_mode(ingest_mode)
        prepared_inputs = self._build_prepared_inputs(corpus_id=corpus_id, turns=turns)
        return self._prepare_turns_parallel(
            prepared_inputs,
            max_workers=self._prepare_worker_count_for_mode(ingest_mode),
        )

    def append_prepared_turns(
        self,
        corpus_id: str,
        title: str,
        prepared_turns: list[dict[str, Any]],
        ingest_mode: str = INGEST_MODE_ONLINE,
    ) -> dict[str, Any]:
        ingest_mode = self._normalize_ingest_mode(ingest_mode)
        started_at = time.perf_counter()
        apply_started_at = time.perf_counter()
        touched_sessions: set[str] = set()
        touched_subjects: set[str] = set()
        events_written = 0
        atoms_written = 0
        objects_written = 0
        state_candidates = 0
        evidence_links_written = 0
        state_action_counts: dict[str, int] = defaultdict(int)
        apply_metrics: dict[str, Any] = {}
        snapshot_elapsed_ms = 0.0
        try:
            apply_metrics = self._apply_prepared_turns(prepared_turns, ingest_mode=ingest_mode)
            apply_elapsed_ms = round((time.perf_counter() - apply_started_at) * 1000.0, 2)
            events_written = int(apply_metrics["events_written"])
            atoms_written = int(apply_metrics["atoms_written"])
            objects_written = int(apply_metrics["objects_written"])
            state_candidates = int(apply_metrics["state_candidates"])
            evidence_links_written = int(apply_metrics["evidence_links_written"])
            for action, count in dict(apply_metrics["state_action_counts"]).items():
                state_action_counts[str(action)] += int(count)
            touched_sessions.update(apply_metrics["touched_sessions"])
            touched_subjects.update(apply_metrics["touched_subjects"])
            snapshot_started_at = time.perf_counter()
            if ingest_mode == INGEST_MODE_MIGRATION:
                session_snapshots, entity_snapshots, root_snapshot = self._build_snapshots_migration(
                    corpus_id=corpus_id,
                    title=title,
                    touched_sessions=sorted(touched_sessions),
                    touched_subjects=sorted(touched_subjects),
                )
            else:
                session_snapshots = [
                    self._build_session_snapshot(corpus_id=corpus_id, title=title, session_id=session_id)
                    for session_id in sorted(touched_sessions)
                ]
                entity_snapshots = [
                    self._build_entity_snapshot(corpus_id=corpus_id, title=title, subject=subject)
                    for subject in sorted(touched_subjects)
                ]
                root_snapshot = self._build_root_snapshot(
                    corpus_id=corpus_id,
                    title=title,
                    session_snapshots=session_snapshots,
                    entity_snapshots=entity_snapshots,
                )
            self._upsert_snapshots_with_embeddings(session_snapshots + entity_snapshots)
            self._upsert_snapshots_with_embeddings([root_snapshot])
            snapshot_elapsed_ms = round((time.perf_counter() - snapshot_started_at) * 1000.0, 2)
            self.store.commit()
        except Exception:
            self.store.rollback()
            raise
        return {
            "ingest_elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
            "ingest_mode": ingest_mode,
            "apply_strategy": str(apply_metrics.get("apply_strategy") or ingest_mode),
            "events_written": events_written,
            "atoms_written": atoms_written,
            "objects_written": objects_written,
            "state_candidates": state_candidates,
            "evidence_links_written": evidence_links_written,
            "state_action_counts": dict(sorted(state_action_counts.items())),
            "state_cache_metrics": dict(apply_metrics.get("state_cache_metrics") or {}),
            "touched_sessions": sorted(touched_sessions),
            "touched_subjects": sorted(touched_subjects),
            "prepared_turn_count": len(prepared_turns),
            "stage_timings_ms": {
                "apply_total_ms": apply_elapsed_ms,
                "snapshot_total_ms": snapshot_elapsed_ms,
            },
        }

    def _build_prepared_inputs(
        self,
        corpus_id: str,
        turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        per_session_index: dict[str, int] = {}
        prepared_inputs: list[dict[str, Any]] = []
        for turn in turns:
            session_id = str(turn.get("session_id") or "session-1")
            speaker = str(turn.get("speaker") or turn.get("role") or "unknown")
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            if session_id not in per_session_index:
                per_session_index[session_id] = self.store.get_next_turn_index(corpus_id, session_id)
            turn_index = per_session_index[session_id]
            per_session_index[session_id] += 1
            timestamp = str(turn.get("timestamp")) if turn.get("timestamp") else None
            metadata = {
                key: value
                for key, value in turn.items()
                if key not in {"session_id", "speaker", "role", "text", "content", "timestamp"}
            }
            prepared_inputs.append(
                {
                    "corpus_id": corpus_id,
                    "session_id": session_id,
                    "speaker": speaker,
                    "text": text,
                    "turn_index": turn_index,
                    "timestamp": timestamp,
                    "metadata": metadata,
                }
            )
        return prepared_inputs

    @staticmethod
    def _normalize_ingest_mode(ingest_mode: str | None) -> str:
        normalized = str(ingest_mode or INGEST_MODE_ONLINE).strip().lower()
        if normalized not in INGEST_MODES:
            allowed = ", ".join(sorted(INGEST_MODES))
            raise ValueError(f"Unsupported ingest mode: {ingest_mode!r}. Expected one of: {allowed}")
        return normalized

    def _ingest_prepare_worker_count(self) -> int:
        raw_value = str(os.environ.get("LEAF_INGEST_PREPARE_WORKERS", "")).strip()
        if raw_value:
            try:
                return max(1, int(raw_value))
            except ValueError:
                return 4
        return 4

    def _migration_prepare_worker_count(self) -> int:
        raw_value = str(os.environ.get("LEAF_MIGRATION_PREPARE_WORKERS", "")).strip()
        if raw_value:
            try:
                return max(1, int(raw_value))
            except ValueError:
                return 16
        return 16

    def _prepare_worker_count_for_mode(self, ingest_mode: str) -> int:
        if ingest_mode == INGEST_MODE_MIGRATION:
            return self._migration_prepare_worker_count()
        return self._ingest_prepare_worker_count()

    def _embedding_worker_count(self) -> int:
        raw_value = str(os.environ.get("LEAF_EMBED_WORKERS", "")).strip()
        if raw_value:
            try:
                return max(1, int(raw_value))
            except ValueError:
                return 8
        return 8

    def _embedding_batch_size(self) -> int:
        raw_value = str(os.environ.get("LEAF_EMBED_BATCH_SIZE", "")).strip()
        if raw_value:
            try:
                return max(1, int(raw_value))
            except ValueError:
                return 32
        return 32

    def _prepare_turns_parallel(
        self,
        prepared_inputs: list[dict[str, Any]],
        *,
        max_workers: int,
    ) -> list[dict[str, Any]]:
        if not prepared_inputs:
            return []
        if max_workers <= 1 or len(prepared_inputs) == 1:
            prepared_turns = [self._prepare_single_turn(**payload) for payload in prepared_inputs]
            self._assign_prepared_turn_embeddings(prepared_turns)
            return prepared_turns
        ordered_results: list[dict[str, Any] | None] = [None] * len(prepared_inputs)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(prepared_inputs))) as executor:
            futures = [
                executor.submit(self._prepare_single_turn, **payload)
                for payload in prepared_inputs
            ]
            for index, future in enumerate(futures):
                ordered_results[index] = future.result()
        prepared_turns = [result for result in ordered_results if result is not None]
        self._assign_prepared_turn_embeddings(prepared_turns)
        return prepared_turns

    @staticmethod
    def serialize_prepared_turns(prepared_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for prepared_turn in prepared_turns:
            payload.append(
                {
                    "event": prepared_turn["event"].to_dict(),
                    "atoms": [atom.to_dict() for atom in prepared_turn.get("atoms") or []],
                    "state_candidates": [
                        candidate.to_dict() for candidate in prepared_turn.get("state_candidates") or []
                    ],
                }
            )
        return payload

    @staticmethod
    def deserialize_prepared_turns(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared_turns: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            event_payload = item.get("event") or {}
            if not isinstance(event_payload, dict) or not event_payload:
                continue
            prepared_turns.append(
                {
                    "event": MemoryEventRecord(**event_payload),
                    "atoms": [
                        MemoryAtomRecord(**atom_payload)
                        for atom_payload in item.get("atoms") or []
                        if isinstance(atom_payload, dict)
                    ],
                    "state_candidates": [
                        StateCandidate(**candidate_payload)
                        for candidate_payload in item.get("state_candidates") or []
                        if isinstance(candidate_payload, dict)
                    ],
                }
            )
        return prepared_turns

    def _apply_prepared_turns(
        self,
        prepared_turns: list[dict[str, Any]],
        *,
        ingest_mode: str,
    ) -> dict[str, Any]:
        if ingest_mode == INGEST_MODE_MIGRATION:
            return self._apply_prepared_turns_migration(prepared_turns)
        return self._apply_prepared_turns_online(prepared_turns)

    def _apply_prepared_turns_online(self, prepared_turns: list[dict[str, Any]]) -> dict[str, Any]:
        touched_sessions: set[str] = set()
        touched_subjects: set[str] = set()
        state_action_counts: dict[str, int] = defaultdict(int)
        metrics = {
            "apply_strategy": "store_serial",
            "events_written": 0,
            "atoms_written": 0,
            "objects_written": 0,
            "state_candidates": 0,
            "evidence_links_written": 0,
            "state_action_counts": state_action_counts,
            "touched_sessions": touched_sessions,
            "touched_subjects": touched_subjects,
            "state_cache_metrics": {
                "enabled": False,
                "object_hits": 0,
                "object_misses": 0,
            },
        }
        for prepared_turn in prepared_turns:
            event, atoms, touched_object_count, state_subjects, turn_metrics = self._apply_prepared_turn(prepared_turn)
            metrics["events_written"] += 1
            metrics["atoms_written"] += len(atoms)
            metrics["objects_written"] += touched_object_count
            metrics["state_candidates"] += int(turn_metrics["state_candidates"])
            metrics["evidence_links_written"] += int(turn_metrics["evidence_links_written"])
            for action, count in dict(turn_metrics["state_action_counts"]).items():
                state_action_counts[str(action)] += int(count)
            touched_sessions.add(event.session_id)
            touched_subjects.update(state_subjects)
        return metrics

    def _apply_prepared_turns_migration(self, prepared_turns: list[dict[str, Any]]) -> dict[str, Any]:
        touched_sessions: set[str] = set()
        touched_subjects: set[str] = set()
        state_action_counts: dict[str, int] = defaultdict(int)
        object_cache: dict[str, MemoryObjectRecord | None] = {}
        latest_version_cache: dict[str, MemoryObjectVersionRecord | None] = {}
        state_cache_metrics = {
            "enabled": True,
            "object_hits": 0,
            "object_misses": 0,
            "cached_objects": 0,
        }
        metrics = {
            "apply_strategy": "state_cache_serial",
            "events_written": 0,
            "atoms_written": 0,
            "objects_written": 0,
            "state_candidates": 0,
            "evidence_links_written": 0,
            "state_action_counts": state_action_counts,
            "touched_sessions": touched_sessions,
            "touched_subjects": touched_subjects,
            "state_cache_metrics": state_cache_metrics,
        }
        for prepared_turn in prepared_turns:
            event, atoms, touched_object_count, state_subjects, turn_metrics = self._apply_prepared_turn_with_state_cache(
                prepared_turn,
                object_cache=object_cache,
                latest_version_cache=latest_version_cache,
                state_cache_metrics=state_cache_metrics,
            )
            metrics["events_written"] += 1
            metrics["atoms_written"] += len(atoms)
            metrics["objects_written"] += touched_object_count
            metrics["state_candidates"] += int(turn_metrics["state_candidates"])
            metrics["evidence_links_written"] += int(turn_metrics["evidence_links_written"])
            for action, count in dict(turn_metrics["state_action_counts"]).items():
                state_action_counts[str(action)] += int(count)
            touched_sessions.add(event.session_id)
            touched_subjects.update(state_subjects)
        state_cache_metrics["cached_objects"] = len(object_cache)
        return metrics

    def _prepare_single_turn(
        self,
        corpus_id: str,
        session_id: str,
        speaker: str,
        text: str,
        turn_index: int,
        timestamp: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        surface = span_surface_text(speaker, text, metadata)
        semantic_refs = extract_semantic_references(surface)
        effective_metadata = dict(metadata)
        if semantic_refs:
            effective_metadata["semantic_refs"] = semantic_refs
        effective_metadata["temporal_grounding"] = derive_temporal_grounding(text=surface, timestamp=timestamp)
        span = RawSpan(
            span_id=stable_id("leaf_raw", corpus_id, session_id, str(turn_index), speaker, text),
            corpus_id=corpus_id,
            session_id=session_id,
            speaker=speaker,
            text=text,
            turn_index=turn_index,
            timestamp=timestamp,
            metadata=effective_metadata,
            embedding=None,
        )
        raw_refs = extract_entities(text) + semantic_refs
        canonical_refs = canonicalize_entities(raw_refs)
        event = MemoryEventRecord(
            event_id=stable_id("leaf_evt", corpus_id, session_id, str(turn_index), speaker, text),
            corpus_id=corpus_id,
            session_id=session_id,
            speaker=speaker,
            text=text,
            turn_index=turn_index,
            timestamp=timestamp,
            raw_span_id=span.span_id,
            entity_refs=list(dict.fromkeys(raw_refs))[:12],
            canonical_entity_refs=canonical_refs[:12],
            metadata=effective_metadata,
            embedding=None,
        )
        extracted_atoms = self.atom_extractor.extract_atoms(span)
        atoms = [
            MemoryAtomRecord(
                atom_id=atom.atom_id,
                event_id=event.event_id,
                corpus_id=atom.corpus_id,
                span_id=atom.span_id,
                atom_type=atom.atom_type,
                content=atom.content,
                entities=atom.entities,
                canonical_entities=atom.canonical_entities,
                support_span_ids=atom.support_span_ids,
                derived_from_atom_ids=atom.derived_from_atom_ids,
                memory_kind=atom.memory_kind,
                status=atom.status,
                time_range=atom.time_range,
                confidence=atom.confidence,
                metadata=atom.metadata,
            )
            for atom in extracted_atoms
        ]
        event.atom_ids = [atom.atom_id for atom in atoms]
        return {
            "event": event,
            "atoms": atoms,
            "state_candidates": self._derive_state_candidates(event=event, atoms=atoms),
            "embedding_text": surface,
        }

    def _apply_prepared_turn(
        self,
        prepared_turn: dict[str, Any],
    ) -> tuple[MemoryEventRecord, list[MemoryAtomRecord], int, set[str], dict[str, Any]]:
        event = prepared_turn["event"]
        atoms = list(prepared_turn.get("atoms") or [])
        state_candidates = list(prepared_turn.get("state_candidates") or [])
        self.store.add_event(event)
        for atom in atoms:
            self.store.add_atom(atom)
        touched_subjects: set[str] = set()
        touched_object_count = 0
        evidence_links_written = 0
        state_action_counts: dict[str, int] = defaultdict(int)
        for candidate in state_candidates:
            action, target_object, target_version, reason = self._decide_state_action(candidate)
            state_action_counts[action] += 1
            if action == "NONE":
                if target_object is not None and target_version is not None:
                    self._add_evidence_link(
                        candidate=candidate,
                        object_id=target_object.object_id,
                        version_id=target_version.version_id,
                        role="support",
                        reason=reason,
                    )
                    evidence_links_written += 1
                    touched_subjects.add(target_object.subject)
                continue
            if action == "ADD":
                obj, version = self._create_object_and_version(candidate, operation="ADD", status="active")
                self.store.upsert_object(obj)
                self.store.add_version(version)
                self._add_evidence_link(candidate, obj.object_id, version.version_id, "origin", reason)
                evidence_links_written += 1
                touched_subjects.add(obj.subject)
                touched_object_count += 1
                continue
            if target_object is None:
                obj, version = self._create_object_and_version(candidate, operation=action, status="active")
                self.store.upsert_object(obj)
                self.store.add_version(version)
                self._add_evidence_link(candidate, obj.object_id, version.version_id, "origin", reason)
                evidence_links_written += 1
                touched_subjects.add(obj.subject)
                touched_object_count += 1
                continue
            if action == "TENTATIVE":
                target_object.status = "active"
                target_object.updated_at_event_id = candidate.event_id
                self.store.upsert_object(target_object)
                version = self._new_version(candidate, target_object.object_id, operation="TENTATIVE", status="tentative")
                self.store.add_version(version)
                self._add_evidence_link(candidate, target_object.object_id, version.version_id, "tentative", reason)
                evidence_links_written += 1
                touched_subjects.add(target_object.subject)
                touched_object_count += 1
                continue
            if target_version is not None:
                self.store.update_version_window(
                    version_id=target_version.version_id,
                    status="superseded",
                    valid_to=candidate.valid_from,
                )
            version = self._new_version(
                candidate,
                target_object.object_id,
                operation=action,
                status="active",
                metadata={"previous_version_id": target_version.version_id if target_version else None, "reason": reason},
            )
            target_object.latest_version_id = version.version_id
            target_object.updated_at_event_id = candidate.event_id
            target_object.status = "active"
            target_object.canonical_entities = list(
                dict.fromkeys(target_object.canonical_entities + [candidate.subject])
            )[:8]
            self.store.upsert_object(target_object)
            self.store.add_version(version)
            self._add_evidence_link(candidate, target_object.object_id, version.version_id, "update", reason)
            evidence_links_written += 1
            touched_subjects.add(target_object.subject)
            touched_object_count += 1
        return event, atoms, touched_object_count, touched_subjects, {
            "state_candidates": len(state_candidates),
            "state_action_counts": dict(sorted(state_action_counts.items())),
            "evidence_links_written": evidence_links_written,
        }

    def _apply_prepared_turn_with_state_cache(
        self,
        prepared_turn: dict[str, Any],
        *,
        object_cache: dict[str, MemoryObjectRecord | None],
        latest_version_cache: dict[str, MemoryObjectVersionRecord | None],
        state_cache_metrics: dict[str, int | bool],
    ) -> tuple[MemoryEventRecord, list[MemoryAtomRecord], int, set[str], dict[str, Any]]:
        event = prepared_turn["event"]
        atoms = list(prepared_turn.get("atoms") or [])
        state_candidates = list(prepared_turn.get("state_candidates") or [])
        self.store.add_event(event)
        for atom in atoms:
            self.store.add_atom(atom)
        touched_subjects: set[str] = set()
        touched_object_count = 0
        evidence_links_written = 0
        state_action_counts: dict[str, int] = defaultdict(int)
        for candidate in state_candidates:
            action, target_object, target_version, reason = self._decide_state_action_with_cache(
                candidate,
                object_cache=object_cache,
                latest_version_cache=latest_version_cache,
                state_cache_metrics=state_cache_metrics,
            )
            state_action_counts[action] += 1
            if action == "NONE":
                if target_object is not None and target_version is not None:
                    self._add_evidence_link(
                        candidate=candidate,
                        object_id=target_object.object_id,
                        version_id=target_version.version_id,
                        role="support",
                        reason=reason,
                    )
                    evidence_links_written += 1
                    touched_subjects.add(target_object.subject)
                continue
            if action == "ADD":
                obj, version = self._create_object_and_version(candidate, operation="ADD", status="active")
                self.store.upsert_object(obj)
                self.store.add_version(version)
                object_cache[obj.object_id] = obj
                latest_version_cache[obj.object_id] = version
                self._add_evidence_link(candidate, obj.object_id, version.version_id, "origin", reason)
                evidence_links_written += 1
                touched_subjects.add(obj.subject)
                touched_object_count += 1
                continue
            if target_object is None:
                obj, version = self._create_object_and_version(candidate, operation=action, status="active")
                self.store.upsert_object(obj)
                self.store.add_version(version)
                object_cache[obj.object_id] = obj
                latest_version_cache[obj.object_id] = version
                self._add_evidence_link(candidate, obj.object_id, version.version_id, "origin", reason)
                evidence_links_written += 1
                touched_subjects.add(obj.subject)
                touched_object_count += 1
                continue
            if action == "TENTATIVE":
                target_object.status = "active"
                target_object.updated_at_event_id = candidate.event_id
                self.store.upsert_object(target_object)
                object_cache[target_object.object_id] = target_object
                version = self._new_version(candidate, target_object.object_id, operation="TENTATIVE", status="tentative")
                self.store.add_version(version)
                self._add_evidence_link(candidate, target_object.object_id, version.version_id, "tentative", reason)
                evidence_links_written += 1
                touched_subjects.add(target_object.subject)
                touched_object_count += 1
                continue
            if target_version is not None:
                self.store.update_version_window(
                    version_id=target_version.version_id,
                    status="superseded",
                    valid_to=candidate.valid_from,
                )
                target_version.status = "superseded"
                target_version.valid_to = candidate.valid_from
            version = self._new_version(
                candidate,
                target_object.object_id,
                operation=action,
                status="active",
                metadata={"previous_version_id": target_version.version_id if target_version else None, "reason": reason},
            )
            target_object.latest_version_id = version.version_id
            target_object.updated_at_event_id = candidate.event_id
            target_object.status = "active"
            target_object.canonical_entities = list(
                dict.fromkeys(target_object.canonical_entities + [candidate.subject])
            )[:8]
            self.store.upsert_object(target_object)
            self.store.add_version(version)
            object_cache[target_object.object_id] = target_object
            latest_version_cache[target_object.object_id] = version
            self._add_evidence_link(candidate, target_object.object_id, version.version_id, "update", reason)
            evidence_links_written += 1
            touched_subjects.add(target_object.subject)
            touched_object_count += 1
        return event, atoms, touched_object_count, touched_subjects, {
            "state_candidates": len(state_candidates),
            "state_action_counts": dict(sorted(state_action_counts.items())),
            "evidence_links_written": evidence_links_written,
        }

    def _derive_state_candidates(
        self,
        event: MemoryEventRecord,
        atoms: list[MemoryAtomRecord],
    ) -> list[StateCandidate]:
        candidates: list[StateCandidate] = []
        for atom in atoms:
            if atom.memory_kind not in {"state", "preference", "plan", "relation"}:
                continue
            subject = self._resolve_subject(event=event, atom=atom)
            value = atom.content.strip()
            normalized_value = _normalize_value(value)
            if not normalized_value:
                continue
            for slot in self._infer_candidate_slots(atom):
                policy = "singleton" if slot in SINGLETON_SLOTS else "multi"
                candidate = StateCandidate(
                    candidate_id=stable_id("leaf_cand", event.event_id, atom.atom_id, subject, slot, normalized_value),
                    corpus_id=event.corpus_id,
                    event_id=event.event_id,
                    span_id=atom.span_id,
                    atom_id=atom.atom_id,
                    subject=subject,
                    slot=slot,
                    value=value,
                    normalized_value=normalized_value,
                    memory_kind=atom.memory_kind,
                    policy=policy,
                    status=atom.status,
                    confidence=atom.confidence,
                    valid_from=event.timestamp,
                    metadata={
                        "speaker": event.speaker,
                        "session_id": event.session_id,
                        "canonical_entities": atom.canonical_entities,
                        "entities": atom.entities,
                    },
                )
                candidates.append(candidate)
        return candidates

    def _decide_state_action(
        self,
        candidate: StateCandidate,
    ) -> tuple[str, MemoryObjectRecord | None, MemoryObjectVersionRecord | None, str]:
        object_id = self._object_id_for_candidate(candidate)
        obj = self.store.get_object(object_id)
        version = self.store.get_latest_version(object_id) if obj else None
        return self._decide_state_action_from_current(candidate=candidate, obj=obj, version=version)

    def _decide_state_action_with_cache(
        self,
        candidate: StateCandidate,
        *,
        object_cache: dict[str, MemoryObjectRecord | None],
        latest_version_cache: dict[str, MemoryObjectVersionRecord | None],
        state_cache_metrics: dict[str, int | bool],
    ) -> tuple[str, MemoryObjectRecord | None, MemoryObjectVersionRecord | None, str]:
        object_id = self._object_id_for_candidate(candidate)
        if object_id in object_cache:
            state_cache_metrics["object_hits"] = int(state_cache_metrics.get("object_hits", 0)) + 1
            obj = object_cache[object_id]
            version = latest_version_cache.get(object_id)
        else:
            state_cache_metrics["object_misses"] = int(state_cache_metrics.get("object_misses", 0)) + 1
            loaded_object = self.store.get_object(object_id)
            obj = self._copy_object_record(loaded_object) if loaded_object is not None else None
            loaded_version = self.store.get_latest_version(object_id) if loaded_object else None
            version = self._copy_version_record(loaded_version) if loaded_version is not None else None
            object_cache[object_id] = obj
            latest_version_cache[object_id] = version
        return self._decide_state_action_from_current(candidate=candidate, obj=obj, version=version)

    def _decide_state_action_from_current(
        self,
        *,
        candidate: StateCandidate,
        obj: MemoryObjectRecord | None,
        version: MemoryObjectVersionRecord | None,
    ) -> tuple[str, MemoryObjectRecord | None, MemoryObjectVersionRecord | None, str]:
        if obj is None or version is None:
            return "ADD", obj, version, "new_object"
        if version.normalized_value == candidate.normalized_value:
            return "NONE", obj, version, "same_normalized_value"
        if candidate.status != "active":
            return "TENTATIVE", obj, version, "candidate_not_active"
        if candidate.policy == "multi":
            if object_id != obj.object_id:
                return "ADD", obj, version, "multi_valued_slot_new_value"
            return "NONE", obj, version, "multi_valued_duplicate"
        overlap = _token_overlap(version.normalized_value, candidate.normalized_value)
        if candidate.normalized_value in version.normalized_value or version.normalized_value in candidate.normalized_value:
            if len(candidate.normalized_value) > len(version.normalized_value):
                return "PATCH", obj, version, "candidate_more_specific"
            return "NONE", obj, version, "existing_more_specific"
        if overlap <= 0.2:
            return "SUPERSEDE", obj, version, "singleton_slot_low_overlap"
        llm_action = self._llm_reconcile(candidate=candidate, current=obj, current_version=version)
        if llm_action in AMBIGUOUS_ACTIONS:
            return llm_action, obj, version, "llm_reconciled"
        if llm_action == "NONE":
            return "NONE", obj, version, "llm_duplicate"
        return "SUPERSEDE", obj, version, "default_supersede"

    @staticmethod
    def _copy_object_record(record: MemoryObjectRecord | None) -> MemoryObjectRecord | None:
        if record is None:
            return None
        return MemoryObjectRecord(**record.to_dict())

    @staticmethod
    def _copy_version_record(record: MemoryObjectVersionRecord | None) -> MemoryObjectVersionRecord | None:
        if record is None:
            return None
        return MemoryObjectVersionRecord(**record.to_dict())

    @staticmethod
    def _snapshot_embedding_payload(title: str, summary: str) -> str:
        del title
        return str(summary or "").strip()

    def _embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        if self.embedding_client is None:
            return [None] * len(texts)
        try:
            return self.embedding_client.embed_many(texts)
        except OpenAICompatError:
            return [self._embed_text(text) for text in texts]

    def _embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        if self.embedding_client is None:
            return [None] * len(texts)
        results: list[list[float] | None] = [None] * len(texts)
        non_empty_positions = [index for index, text in enumerate(texts) if str(text or '').strip()]
        if not non_empty_positions:
            return results
        batch_size = self._embedding_batch_size()
        batched_positions = [
            non_empty_positions[start : start + batch_size]
            for start in range(0, len(non_empty_positions), batch_size)
        ]
        if len(batched_positions) == 1 or self._embedding_worker_count() <= 1:
            for positions in batched_positions:
                batch_results = self._embed_batch([texts[index] for index in positions])
                for index, embedding in zip(positions, batch_results):
                    results[index] = embedding
            return results
        with ThreadPoolExecutor(max_workers=min(self._embedding_worker_count(), len(batched_positions))) as executor:
            future_map = {
                executor.submit(self._embed_batch, [texts[index] for index in positions]): positions
                for positions in batched_positions
            }
            for future, positions in future_map.items():
                batch_results = future.result()
                for index, embedding in zip(positions, batch_results):
                    results[index] = embedding
        return results

    def _assign_prepared_turn_embeddings(self, prepared_turns: list[dict[str, Any]]) -> None:
        if not prepared_turns:
            return
        embedding_inputs = [str(item.get("embedding_text") or "").strip() for item in prepared_turns]
        embeddings = self._embed_texts(embedding_inputs)
        for prepared_turn, embedding in zip(prepared_turns, embeddings):
            prepared_turn["event"].embedding = embedding
            prepared_turn.pop("embedding_text", None)

    def _upsert_snapshots_with_embeddings(self, snapshots: list[MemorySnapshotRecord]) -> None:
        if not snapshots:
            return
        embedding_inputs = [
            self._snapshot_embedding_payload(snapshot.title, snapshot.summary)
            for snapshot in snapshots
        ]
        embeddings = self._embed_texts(embedding_inputs)
        for snapshot, embedding in zip(snapshots, embeddings):
            snapshot.embedding = embedding
            self.store.upsert_snapshot(snapshot)

    def _build_snapshots_migration(
        self,
        *,
        corpus_id: str,
        title: str,
        touched_sessions: list[str],
        touched_subjects: list[str],
    ) -> tuple[list[MemorySnapshotRecord], list[MemorySnapshotRecord], MemorySnapshotRecord]:
        events = self.store.get_events(corpus_id=corpus_id)
        objects = self.store.get_objects(corpus_id=corpus_id)
        versions = self.store.get_object_versions_for_corpus(corpus_id=corpus_id)
        links = self.store.get_evidence_links(corpus_id=corpus_id)

        events_by_id = {event.event_id: event for event in events}
        events_by_session: dict[str, list[MemoryEventRecord]] = defaultdict(list)
        for event in events:
            events_by_session[event.session_id].append(event)

        objects_by_id = {obj.object_id: obj for obj in objects}
        objects_by_subject: dict[str, list[MemoryObjectRecord]] = defaultdict(list)
        for obj in objects:
            objects_by_subject[obj.subject].append(obj)

        versions_by_id = {version.version_id: version for version in versions}
        versions_by_object: dict[str, list[MemoryObjectVersionRecord]] = defaultdict(list)
        for version in versions:
            versions_by_object[version.object_id].append(version)

        latest_versions_by_object: dict[str, MemoryObjectVersionRecord] = {}
        for obj in objects:
            if obj.latest_version_id and obj.latest_version_id in versions_by_id:
                latest_versions_by_object[obj.object_id] = versions_by_id[obj.latest_version_id]

        touched_session_set = set(touched_sessions)
        touched_subject_set = set(touched_subjects)
        session_object_ids: dict[str, set[str]] = defaultdict(set)
        linked_subjects_by_event: dict[str, set[str]] = defaultdict(set)
        for link in links:
            event = events_by_id.get(link.event_id)
            obj = objects_by_id.get(link.object_id)
            if event is None or obj is None:
                continue
            if event.session_id in touched_session_set:
                session_object_ids[event.session_id].add(obj.object_id)
            if obj.subject in touched_subject_set:
                linked_subjects_by_event[event.event_id].add(obj.subject)

        subject_events: dict[str, list[MemoryEventRecord]] = {subject: [] for subject in touched_subjects}
        subject_seen_event_ids: dict[str, set[str]] = {subject: set() for subject in touched_subjects}
        for event in events:
            event_subjects = set(linked_subjects_by_event.get(event.event_id) or set())
            event_subjects.update(subject for subject in event.canonical_entity_refs if subject in touched_subject_set)
            for subject in event_subjects:
                seen_event_ids = subject_seen_event_ids[subject]
                if event.event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event.event_id)
                subject_events[subject].append(event)

        session_snapshots = [
            self._build_session_snapshot_from_state(
                corpus_id=corpus_id,
                title=title,
                session_id=session_id,
                events=events_by_session.get(session_id) or [],
                object_ids=sorted(
                    session_object_ids.get(session_id) or set(),
                    key=lambda object_id: (
                        objects_by_id[object_id].subject,
                        objects_by_id[object_id].slot,
                        objects_by_id[object_id].object_id,
                    ),
                ),
                latest_versions_by_object=latest_versions_by_object,
            )
            for session_id in touched_sessions
        ]
        entity_snapshots = []
        for subject in touched_subjects:
            subject_recent_events_asc = subject_events.get(subject) or []
            subject_events_by_session: dict[str, list[MemoryEventRecord]] = defaultdict(list)
            for event in subject_recent_events_asc:
                subject_events_by_session[event.session_id].append(event)
            limited_events_desc: list[MemoryEventRecord] = []
            for session_id in sorted(subject_events_by_session):
                for event in reversed(subject_events_by_session[session_id]):
                    limited_events_desc.append(event)
                    if len(limited_events_desc) >= 12:
                        break
                if len(limited_events_desc) >= 12:
                    break
            entity_snapshots.append(
                self._build_entity_snapshot_from_state(
                    corpus_id=corpus_id,
                    title=title,
                    subject=subject,
                    objects=objects_by_subject.get(subject) or [],
                    versions_by_object=versions_by_object,
                    recent_events=list(reversed(limited_events_desc)),
                )
            )
        root_snapshot = self._build_root_snapshot(
            corpus_id=corpus_id,
            title=title,
            session_snapshots=session_snapshots,
            entity_snapshots=entity_snapshots,
        )
        return session_snapshots, entity_snapshots, root_snapshot

    def _create_object_and_version(
        self,
        candidate: StateCandidate,
        operation: str,
        status: str,
    ) -> tuple[MemoryObjectRecord, MemoryObjectVersionRecord]:
        object_id = self._object_id_for_candidate(candidate)
        version = self._new_version(candidate, object_id, operation=operation, status=status)
        obj = MemoryObjectRecord(
            object_id=object_id,
            corpus_id=candidate.corpus_id,
            subject=candidate.subject,
            slot=candidate.slot,
            memory_kind=candidate.memory_kind,
            policy=candidate.policy,
            latest_version_id=version.version_id if status == "active" else None,
            status="active",
            aliases=[],
            canonical_entities=[candidate.subject],
            created_at_event_id=candidate.event_id,
            updated_at_event_id=candidate.event_id,
            metadata={"source": "leaf", "slot_policy": candidate.policy},
        )
        if status == "tentative":
            obj.latest_version_id = version.version_id
            obj.status = "tentative"
        return obj, version

    def _new_version(
        self,
        candidate: StateCandidate,
        object_id: str,
        operation: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryObjectVersionRecord:
        version_metadata = dict(candidate.metadata)
        if metadata:
            version_metadata.update(metadata)
        return MemoryObjectVersionRecord(
            version_id=stable_id("leaf_ver", object_id, candidate.event_id, operation, candidate.normalized_value),
            object_id=object_id,
            corpus_id=candidate.corpus_id,
            value=candidate.value,
            normalized_value=candidate.normalized_value,
            summary=f"{candidate.subject} [{candidate.slot}]: {candidate.value}",
            operation=operation,
            status=status,
            confidence=candidate.confidence,
            valid_from=candidate.valid_from,
            valid_to=None,
            event_id=candidate.event_id,
            atom_id=candidate.atom_id,
            metadata=version_metadata,
        )

    def _add_evidence_link(
        self,
        candidate: StateCandidate,
        object_id: str,
        version_id: str | None,
        role: str,
        reason: str,
    ) -> None:
        link = MemoryEvidenceLinkRecord(
            link_id=stable_id("leaf_link", object_id, candidate.event_id, candidate.atom_id, role),
            corpus_id=candidate.corpus_id,
            object_id=object_id,
            version_id=version_id,
            event_id=candidate.event_id,
            span_id=candidate.span_id,
            atom_id=candidate.atom_id,
            role=role,
            metadata={"reason": reason},
        )
        self.store.add_evidence_link(link)

    def _build_session_snapshot(self, corpus_id: str, title: str, session_id: str) -> MemorySnapshotRecord:
        events = self.store.get_events(corpus_id=corpus_id, session_id=session_id)
        objects = self.store.get_objects_for_session(corpus_id=corpus_id, session_id=session_id)
        latest_versions_by_object: dict[str, MemoryObjectVersionRecord] = {}
        object_ids: list[str] = []
        for obj in objects:
            version = self.store.get_latest_version(obj.object_id)
            if version is None:
                continue
            object_ids.append(obj.object_id)
            latest_versions_by_object[obj.object_id] = version
        return self._build_session_snapshot_from_state(
            corpus_id=corpus_id,
            title=title,
            session_id=session_id,
            events=events,
            object_ids=object_ids,
            latest_versions_by_object=latest_versions_by_object,
        )

    def _build_session_snapshot_from_state(
        self,
        *,
        corpus_id: str,
        title: str,
        session_id: str,
        events: list[MemoryEventRecord],
        object_ids: list[str],
        latest_versions_by_object: dict[str, MemoryObjectVersionRecord],
    ) -> MemorySnapshotRecord:
        active_summaries = [
            latest_versions_by_object[object_id].summary
            for object_id in object_ids
            if object_id in latest_versions_by_object and latest_versions_by_object[object_id].status == "active"
        ]
        event_texts = [f"{event.speaker}: {event.text}" for event in events]
        summary = summarize_texts(active_summaries + event_texts, max_chars=760)
        synopsis = make_synopsis(active_summaries[:4] + event_texts[:4], max_chars=180)
        refs = merge_memory_refs(active_summaries + event_texts, limit=16)
        child_ids = [event.event_id for event in events[-12:]]
        return MemorySnapshotRecord(
            snapshot_id=stable_id("leaf_snap", corpus_id, "session", session_id),
            corpus_id=corpus_id,
            parent_id=stable_id("leaf_snap", corpus_id, "root"),
            snapshot_kind="session",
            scope_id=session_id,
            title=f"{title}:{session_id}",
            synopsis=synopsis,
            summary=summary,
            object_ids=object_ids[:24],
            event_ids=[event.event_id for event in events[-24:]],
            raw_refs=[event.raw_span_id for event in events[-24:] if event.raw_span_id],
            child_ids=child_ids,
            entity_refs=refs,
            canonical_entity_refs=canonicalize_entities(refs),
            time_range=self._time_range(events),
            metadata={
                "num_events": len(events),
                "num_objects": len(object_ids),
                "semantic_role": "session_snapshot",
            },
            embedding=None,
        )

    def _build_entity_snapshot(self, corpus_id: str, title: str, subject: str) -> MemorySnapshotRecord:
        objects = self.store.get_objects_for_subject(corpus_id=corpus_id, subject=subject)
        versions_by_object: dict[str, list[MemoryObjectVersionRecord]] = {}
        for obj in objects:
            versions_by_object[obj.object_id] = self.store.get_object_versions(obj.object_id)
        recent_events = self.store.get_events_for_entity(corpus_id=corpus_id, entity=subject, limit=12)
        return self._build_entity_snapshot_from_state(
            corpus_id=corpus_id,
            title=title,
            subject=subject,
            objects=objects,
            versions_by_object=versions_by_object,
            recent_events=recent_events,
        )

    def _build_entity_snapshot_from_state(
        self,
        *,
        corpus_id: str,
        title: str,
        subject: str,
        objects: list[MemoryObjectRecord],
        versions_by_object: dict[str, list[MemoryObjectVersionRecord]],
        recent_events: list[MemoryEventRecord],
    ) -> MemorySnapshotRecord:
        versions: list[MemoryObjectVersionRecord] = []
        object_ids: list[str] = []
        for obj in objects:
            object_ids.append(obj.object_id)
            versions.extend(versions_by_object.get(obj.object_id) or [])
        active_versions = [version.summary for version in versions if version.status == "active"]
        event_texts = [f"{event.speaker}: {event.text}" for event in recent_events]
        summary = summarize_texts(active_versions + event_texts, max_chars=760)
        synopsis = make_synopsis(active_versions[:4] + event_texts[:4], max_chars=180)
        refs = merge_memory_refs([subject] + active_versions + event_texts, limit=16)
        return MemorySnapshotRecord(
            snapshot_id=stable_id("leaf_snap", corpus_id, "entity", subject),
            corpus_id=corpus_id,
            parent_id=stable_id("leaf_snap", corpus_id, "root"),
            snapshot_kind="entity",
            scope_id=subject,
            title=f"{title}:{subject}",
            synopsis=synopsis,
            summary=summary,
            object_ids=object_ids[:32],
            event_ids=[event.event_id for event in recent_events],
            raw_refs=[event.raw_span_id for event in recent_events if event.raw_span_id],
            child_ids=object_ids[:24],
            entity_refs=refs,
            canonical_entity_refs=canonicalize_entities(refs),
            time_range=self._time_range(recent_events),
            metadata={
                "num_objects": len(object_ids),
                "num_events": len(recent_events),
                "semantic_role": "entity_snapshot",
            },
            embedding=None,
        )

    def _build_root_snapshot(
        self,
        corpus_id: str,
        title: str,
        session_snapshots: list[MemorySnapshotRecord] | None = None,
        entity_snapshots: list[MemorySnapshotRecord] | None = None,
    ) -> MemorySnapshotRecord:
        if session_snapshots is None:
            session_snapshots = self.store.list_snapshots(corpus_id=corpus_id, snapshot_kind="session")
        if entity_snapshots is None:
            entity_snapshots = self.store.list_snapshots(corpus_id=corpus_id, snapshot_kind="entity")
        events = self.store.get_events(corpus_id=corpus_id)
        summary_inputs = [snapshot.summary for snapshot in session_snapshots[-6:]] + [snapshot.summary for snapshot in entity_snapshots[:6]]
        if not summary_inputs:
            summary_inputs = [f"{event.speaker}: {event.text}" for event in events[-12:]]
        refs = merge_memory_refs(summary_inputs, limit=20)
        summary = summarize_texts(summary_inputs, max_chars=900)
        return MemorySnapshotRecord(
            snapshot_id=stable_id("leaf_snap", corpus_id, "root"),
            corpus_id=corpus_id,
            parent_id=None,
            snapshot_kind="root",
            scope_id=corpus_id,
            title=title,
            synopsis=make_synopsis(summary_inputs[:6], max_chars=220),
            summary=summary,
            object_ids=[],
            event_ids=[event.event_id for event in events[-24:]],
            raw_refs=[event.raw_span_id for event in events[-24:] if event.raw_span_id],
            child_ids=[snapshot.snapshot_id for snapshot in session_snapshots] + [snapshot.snapshot_id for snapshot in entity_snapshots[:24]],
            entity_refs=refs,
            canonical_entity_refs=canonicalize_entities(refs, limit=20),
            time_range=self._time_range(events),
            metadata={
                "num_sessions": len(session_snapshots),
                "num_entities": len(entity_snapshots),
                "num_events": len(events),
                "semantic_role": "corpus_snapshot",
            },
            embedding=None,
        )

    def _resolve_subject(self, event: MemoryEventRecord, atom: MemoryAtomRecord) -> str:
        lowered = atom.content.lower().strip()
        if (
            lowered.startswith("i ")
            or lowered.startswith("i'")
            or lowered.startswith("my ")
            or re.search(r"\b(i|my|me|i'm)\b", lowered) is not None
        ):
            canonical_speaker = canonicalize_entities([event.speaker], limit=1)
            return canonical_speaker[0] if canonical_speaker else event.speaker
        if atom.canonical_entities:
            return atom.canonical_entities[0]
        if event.canonical_entity_refs:
            return event.canonical_entity_refs[0]
        canonical_speaker = canonicalize_entities([event.speaker], limit=1)
        return canonical_speaker[0] if canonical_speaker else event.speaker

    @staticmethod
    def _infer_candidate_slots(atom: MemoryAtomRecord) -> list[str]:
        lowered = atom.content.lower()
        slots: list[str] = []
        if any(marker in lowered for marker in ["live in", "lives in", "moved to", "located in", "based in", "from "]):
            slots.append("location")
        if any(marker in lowered for marker in ["works at", "work at", "employed by", "joined ", "company"]):
            slots.append("employer")
        if any(marker in lowered for marker in ["work as", "works as", "job", "occupation", "profession"]):
            slots.append("occupation")
        if any(marker in lowered for marker in ["married", "dating", "partner", "boyfriend", "girlfriend", "spouse"]):
            slots.append("relationship_status")
        if atom.memory_kind == "preference":
            slots.append("preference")
        if atom.memory_kind == "plan":
            slots.append("plan")
        if atom.memory_kind == "relation":
            slots.append("relationship")
        if not slots:
            slots.append(atom.atom_type or atom.memory_kind)
        deduped: list[str] = []
        seen: set[str] = set()
        for slot in slots:
            if slot in seen:
                continue
            seen.add(slot)
            deduped.append(slot)
        return deduped

    @staticmethod
    def _object_id_for_candidate(candidate: StateCandidate) -> str:
        if candidate.policy == "singleton":
            return stable_id("leaf_obj", candidate.corpus_id, candidate.subject, candidate.slot)
        return stable_id("leaf_obj", candidate.corpus_id, candidate.subject, candidate.slot, candidate.normalized_value)

    def _llm_reconcile(
        self,
        candidate: StateCandidate,
        current: MemoryObjectRecord,
        current_version: MemoryObjectVersionRecord,
    ) -> str | None:
        if self.reconciliation_llm is None:
            return None
        messages = [
            {
                "role": "system",
                "content": (
                    "Decide how a new state memory should affect an existing state object. "
                    "Return JSON with key 'action'. Allowed values: NONE, PATCH, SUPERSEDE, TENTATIVE."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"subject={current.subject}\n"
                    f"slot={current.slot}\n"
                    f"current_value={current_version.value}\n"
                    f"current_status={current_version.status}\n"
                    f"new_value={candidate.value}\n"
                    f"new_status={candidate.status}\n"
                    f"memory_kind={candidate.memory_kind}\n"
                    "Respond with a JSON object only."
                ),
            },
        ]
        try:
            payload = extract_json_object(
                self.reconciliation_llm.text(messages, max_tokens=120, temperature=0.0)
            )
        except (OpenAICompatError, ValueError):
            return None
        action = str(payload.get("action") or "").strip().upper()
        if action in {"NONE", "PATCH", "SUPERSEDE", "TENTATIVE"}:
            return action
        return None

    def _embed_text(self, text: str) -> list[float] | None:
        if self.embedding_client is None:
            return None
        try:
            return self.embedding_client.embed(text)
        except OpenAICompatError:
            return None

    @staticmethod
    def _time_range(events: list[MemoryEventRecord]) -> str | None:
        timestamps = [event.timestamp for event in events if event.timestamp]
        if not timestamps:
            return None
        return f"{timestamps[0]}..{timestamps[-1]}"
