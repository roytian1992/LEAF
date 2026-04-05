from __future__ import annotations

import re
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
    ) -> dict[str, Any]:
        per_session_index: dict[str, int] = {}
        touched_sessions: set[str] = set()
        touched_subjects: set[str] = set()
        events_written = 0
        atoms_written = 0
        objects_written = 0
        try:
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
                event, atoms, touched_object_count, state_subjects = self._append_single_turn(
                    corpus_id=corpus_id,
                    session_id=session_id,
                    speaker=speaker,
                    text=text,
                    turn_index=turn_index,
                    timestamp=timestamp,
                    metadata=metadata,
                )
                events_written += 1
                atoms_written += len(atoms)
                objects_written += touched_object_count
                touched_sessions.add(event.session_id)
                touched_subjects.update(state_subjects)
            for session_id in sorted(touched_sessions):
                self._refresh_session_snapshot(corpus_id=corpus_id, title=title, session_id=session_id)
            for subject in sorted(touched_subjects):
                self._refresh_entity_snapshot(corpus_id=corpus_id, title=title, subject=subject)
            self._refresh_root_snapshot(corpus_id=corpus_id, title=title)
            self.store.commit()
        except Exception:
            self.store.rollback()
            raise
        return {
            "events_written": events_written,
            "atoms_written": atoms_written,
            "objects_written": objects_written,
            "touched_sessions": sorted(touched_sessions),
            "touched_subjects": sorted(touched_subjects),
        }

    def _append_single_turn(
        self,
        corpus_id: str,
        session_id: str,
        speaker: str,
        text: str,
        turn_index: int,
        timestamp: str | None,
        metadata: dict[str, Any],
    ) -> tuple[MemoryEventRecord, list[MemoryAtomRecord], int, set[str]]:
        surface = span_surface_text(speaker, text, metadata)
        semantic_refs = extract_semantic_references(surface)
        if semantic_refs:
            metadata["semantic_refs"] = semantic_refs
        metadata["temporal_grounding"] = derive_temporal_grounding(text=surface, timestamp=timestamp)
        span = RawSpan(
            span_id=stable_id("leaf_raw", corpus_id, session_id, str(turn_index), speaker, text),
            corpus_id=corpus_id,
            session_id=session_id,
            speaker=speaker,
            text=text,
            turn_index=turn_index,
            timestamp=timestamp,
            metadata=metadata,
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
            metadata=metadata,
            embedding=self._embed_text(surface),
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
        self.store.add_event(event)
        for atom in atoms:
            self.store.add_atom(atom)
        touched_subjects: set[str] = set()
        touched_object_count = 0
        for candidate in self._derive_state_candidates(event=event, atoms=atoms):
            action, target_object, target_version, reason = self._decide_state_action(candidate)
            if action == "NONE":
                if target_object is not None and target_version is not None:
                    self._add_evidence_link(
                        candidate=candidate,
                        object_id=target_object.object_id,
                        version_id=target_version.version_id,
                        role="support",
                        reason=reason,
                    )
                    touched_subjects.add(target_object.subject)
                continue
            if action == "ADD":
                obj, version = self._create_object_and_version(candidate, operation="ADD", status="active")
                self.store.upsert_object(obj)
                self.store.add_version(version)
                self._add_evidence_link(candidate, obj.object_id, version.version_id, "origin", reason)
                touched_subjects.add(obj.subject)
                touched_object_count += 1
                continue
            if target_object is None:
                obj, version = self._create_object_and_version(candidate, operation=action, status="active")
                self.store.upsert_object(obj)
                self.store.add_version(version)
                self._add_evidence_link(candidate, obj.object_id, version.version_id, "origin", reason)
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
            touched_subjects.add(target_object.subject)
            touched_object_count += 1
        return event, atoms, touched_object_count, touched_subjects

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

    def _refresh_session_snapshot(self, corpus_id: str, title: str, session_id: str) -> None:
        events = self.store.get_events(corpus_id=corpus_id, session_id=session_id)
        objects = self.store.get_objects_for_session(corpus_id=corpus_id, session_id=session_id)
        active_summaries: list[str] = []
        object_ids: list[str] = []
        for obj in objects:
            version = self.store.get_latest_version(obj.object_id)
            if version is None:
                continue
            object_ids.append(obj.object_id)
            if version.status == "active":
                active_summaries.append(version.summary)
        event_texts = [f"{event.speaker}: {event.text}" for event in events]
        summary = summarize_texts(active_summaries + event_texts, max_chars=760)
        synopsis = make_synopsis(active_summaries[:4] + event_texts[:4], max_chars=180)
        refs = merge_memory_refs(active_summaries + event_texts, limit=16)
        child_ids = [event.event_id for event in events[-12:]]
        snapshot = MemorySnapshotRecord(
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
            embedding=self._embed_text(summary),
        )
        self.store.upsert_snapshot(snapshot)

    def _refresh_entity_snapshot(self, corpus_id: str, title: str, subject: str) -> None:
        objects = self.store.get_objects_for_subject(corpus_id=corpus_id, subject=subject)
        versions: list[MemoryObjectVersionRecord] = []
        object_ids: list[str] = []
        for obj in objects:
            object_ids.append(obj.object_id)
            versions.extend(self.store.get_object_versions(obj.object_id))
        active_versions = [version.summary for version in versions if version.status == "active"]
        recent_events = self.store.get_events_for_entity(corpus_id=corpus_id, entity=subject, limit=12)
        event_texts = [f"{event.speaker}: {event.text}" for event in recent_events]
        summary = summarize_texts(active_versions + event_texts, max_chars=760)
        synopsis = make_synopsis(active_versions[:4] + event_texts[:4], max_chars=180)
        refs = merge_memory_refs([subject] + active_versions + event_texts, limit=16)
        snapshot = MemorySnapshotRecord(
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
            embedding=self._embed_text(summary),
        )
        self.store.upsert_snapshot(snapshot)

    def _refresh_root_snapshot(self, corpus_id: str, title: str) -> None:
        session_snapshots = self.store.list_snapshots(corpus_id=corpus_id, snapshot_kind="session")
        entity_snapshots = self.store.list_snapshots(corpus_id=corpus_id, snapshot_kind="entity")
        events = self.store.get_events(corpus_id=corpus_id)
        summary_inputs = [snapshot.summary for snapshot in session_snapshots[-6:]] + [snapshot.summary for snapshot in entity_snapshots[:6]]
        if not summary_inputs:
            summary_inputs = [f"{event.speaker}: {event.text}" for event in events[-12:]]
        refs = merge_memory_refs(summary_inputs, limit=20)
        snapshot = MemorySnapshotRecord(
            snapshot_id=stable_id("leaf_snap", corpus_id, "root"),
            corpus_id=corpus_id,
            parent_id=None,
            snapshot_kind="root",
            scope_id=corpus_id,
            title=title,
            synopsis=make_synopsis(summary_inputs[:6], max_chars=220),
            summary=summarize_texts(summary_inputs, max_chars=900),
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
            embedding=self._embed_text(title + "\n" + summarize_texts(summary_inputs, max_chars=900)),
        )
        self.store.upsert_snapshot(snapshot)

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
