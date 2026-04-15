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
    build_text_tags,
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
SESSION_PAGE_SIZE = 8
SESSION_BLOCK_LEAF_EVENTS = 6
SESSION_BLOCK_TOP_EVENTS = 24
MERGED_ATOM_MAX_UNITS = 220
MERGED_ATOM_MIN_TURNS = 2
MERGED_ATOM_MAX_TURNS = 6


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
        started_at = time.perf_counter()
        before_stats = self.store.get_corpus_stats(corpus_id)
        per_session_index: dict[str, int] = {}
        touched_sessions: set[str] = set()
        touched_subjects: set[str] = set()
        events_written = 0
        atoms_written = 0
        objects_written = 0
        evidence_links_written = 0
        state_candidates = 0
        state_action_counts: dict[str, int] = defaultdict(int)
        snapshot_upserts_by_kind: dict[str, int] = defaultdict(int)
        input_text_chars = 0
        input_text_tokens_est = 0
        self._ingest_runtime_metrics = {"reconcile_llm_calls": 0}
        try:
            prepared_inputs: list[dict[str, Any]] = []
            for turn in turns:
                session_id = str(turn.get("session_id") or "session-1")
                speaker = str(turn.get("speaker") or turn.get("role") or "unknown")
                text = str(turn.get("text") or turn.get("content") or "").strip()
                if not text:
                    continue
                input_text_chars += len(text)
                input_text_tokens_est += self._estimate_text_tokens(text)
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
            prepare_worker_count = self._ingest_prepare_worker_count()
            prepared_turns = self._prepare_turns_parallel(prepared_inputs, max_workers=prepare_worker_count)
            prepared_turns = self._attach_chunk_extractions(prepared_turns)
            extraction_chunk_count = sum(1 for prepared_turn in prepared_turns if prepared_turn.get("atoms"))
            for prepared_turn in prepared_turns:
                event, atoms, touched_object_count, state_subjects, turn_metrics = self._apply_prepared_turn(
                    prepared_turn
                )
                events_written += 1
                atoms_written += len(atoms)
                objects_written += touched_object_count
                evidence_links_written += int(turn_metrics["evidence_links_written"])
                state_candidates += int(turn_metrics["state_candidates"])
                for action, count in dict(turn_metrics["state_action_counts"]).items():
                    state_action_counts[str(action)] += int(count)
                touched_sessions.add(event.session_id)
                touched_subjects.update(state_subjects)
            for session_id in sorted(touched_sessions):
                refresh_counts = self._refresh_session_snapshot(corpus_id=corpus_id, title=title, session_id=session_id)
                for kind, count in refresh_counts.items():
                    snapshot_upserts_by_kind[kind] += int(count)
            for subject in sorted(touched_subjects):
                refresh_counts = self._refresh_entity_snapshot(corpus_id=corpus_id, title=title, subject=subject)
                for kind, count in refresh_counts.items():
                    snapshot_upserts_by_kind[kind] += int(count)
            if events_written > 0:
                refresh_counts = self._refresh_root_snapshot(corpus_id=corpus_id, title=title)
                for kind, count in refresh_counts.items():
                    snapshot_upserts_by_kind[kind] += int(count)
            self.store.commit()
        except Exception:
            self.store.rollback()
            raise
        finally:
            runtime_metrics = dict(getattr(self, "_ingest_runtime_metrics", {}) or {})
            self._ingest_runtime_metrics = None
        after_stats = self.store.get_corpus_stats(corpus_id)
        return {
            "ingest_elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
            "turn_count": len(turns),
            "input_text_chars": input_text_chars,
            "input_text_tokens_est": input_text_tokens_est,
            "prepare_workers": prepare_worker_count,
            "events_written": events_written,
            "atoms_written": atoms_written,
            "objects_written": objects_written,
            "evidence_links_written": evidence_links_written,
            "state_candidates": state_candidates,
            "state_action_counts": dict(sorted(state_action_counts.items())),
                "memory_llm_calls_est": {
                "atom_extraction": extraction_chunk_count if self.atom_extractor.llm is not None else 0,
                "reconciliation": int(runtime_metrics.get("reconcile_llm_calls", 0)),
                "total": (
                    (extraction_chunk_count if self.atom_extractor.llm is not None else 0)
                    + int(runtime_metrics.get("reconcile_llm_calls", 0))
                ),
            },
            "extraction_chunk_count": extraction_chunk_count,
            "snapshot_upserts_by_kind": dict(sorted(snapshot_upserts_by_kind.items())),
            "snapshot_upserts_total": int(sum(snapshot_upserts_by_kind.values())),
            "touched_sessions": sorted(touched_sessions),
            "touched_subjects": sorted(touched_subjects),
            "corpus_stats_before": before_stats,
            "corpus_stats_after": after_stats,
            "corpus_stats_delta": self._diff_stats(before_stats, after_stats),
        }

    def _ingest_prepare_worker_count(self) -> int:
        raw_value = str(os.environ.get("LEAF_INGEST_PREPARE_WORKERS", "")).strip()
        if raw_value:
            try:
                return max(1, int(raw_value))
            except ValueError:
                return 4
        return 4

    def _prepare_turns_parallel(
        self,
        prepared_inputs: list[dict[str, Any]],
        *,
        max_workers: int,
    ) -> list[dict[str, Any]]:
        if not prepared_inputs:
            return []
        if max_workers <= 1 or len(prepared_inputs) == 1:
            return [self._prepare_single_turn(**payload) for payload in prepared_inputs]
        ordered_results: list[dict[str, Any] | None] = [None] * len(prepared_inputs)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(prepared_inputs))) as executor:
            futures = [
                executor.submit(self._prepare_single_turn, **payload)
                for payload in prepared_inputs
            ]
            for index, future in enumerate(futures):
                ordered_results[index] = future.result()
        return [result for result in ordered_results if result is not None]

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
            embedding=self._embed_text(surface),
        )
        return {
            "event": event,
            "span": span,
            "atoms": [],
            "state_candidates": [],
        }

    def _attach_chunk_extractions(self, prepared_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not prepared_turns:
            return prepared_turns
        grouped_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
        session_order: list[str] = []
        for prepared_turn in prepared_turns:
            session_id = str(prepared_turn["event"].session_id)
            if session_id not in grouped_by_session:
                session_order.append(session_id)
            grouped_by_session[session_id].append(prepared_turn)
            prepared_turn["atoms"] = []
            prepared_turn["state_candidates"] = []
        for session_id in session_order:
            session_turns = grouped_by_session[session_id]
            chunks = self._build_extraction_chunks(session_turns)
            previous_chunk: list[dict[str, Any]] | None = None
            for chunk in chunks:
                anchor_turn = chunk[-1]
                anchor_event = anchor_turn["event"]
                extraction_span = self._build_chunk_extraction_span(chunk, previous_chunk)
                extracted_atoms = self.atom_extractor.extract_atoms(extraction_span)
                support_span_ids = [str(item["span"].span_id) for item in chunk if item.get("span") is not None]
                anchor_span_id = str(anchor_turn["span"].span_id)
                atoms = [
                    MemoryAtomRecord(
                        atom_id=atom.atom_id,
                        event_id=anchor_event.event_id,
                        corpus_id=atom.corpus_id,
                        span_id=anchor_span_id,
                        atom_type=atom.atom_type,
                        content=atom.content,
                        entities=atom.entities,
                        canonical_entities=atom.canonical_entities,
                        support_span_ids=support_span_ids,
                        derived_from_atom_ids=atom.derived_from_atom_ids,
                        memory_kind=atom.memory_kind,
                        status=atom.status,
                        time_range=atom.time_range,
                        confidence=atom.confidence,
                        metadata={
                            **dict(atom.metadata or {}),
                            "source_speaker": anchor_event.speaker,
                            "merged_chunk_span_id": extraction_span.span_id,
                            "merged_turn_count": len(chunk),
                            "support_span_ids": support_span_ids,
                            "merged_turn_indexes": [int(item["event"].turn_index) for item in chunk],
                            "chunk_session_id": anchor_event.session_id,
                        },
                    )
                    for atom in extracted_atoms
                ]
                anchor_turn["atoms"] = atoms
                anchor_turn["state_candidates"] = self._derive_state_candidates(event=anchor_event, atoms=atoms)
                previous_chunk = chunk
        return prepared_turns

    def _apply_prepared_turn(
        self,
        prepared_turn: dict[str, Any],
    ) -> tuple[MemoryEventRecord, list[MemoryAtomRecord], int, set[str], dict[str, Any]]:
        event = prepared_turn["event"]
        atoms = list(prepared_turn.get("atoms") or [])
        state_candidates = list(prepared_turn.get("state_candidates") or [])
        event.atom_ids = [atom.atom_id for atom in atoms]
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

    def _embed_snapshot(self, title: str, summary: str, tags: list[str] | None = None) -> list[float] | None:
        parts = [str(title or "").strip(), str(summary or "").strip()]
        if tags:
            parts.append("Tags: " + ", ".join(str(tag).strip() for tag in tags if str(tag).strip()))
        payload = "\n".join(part for part in parts if part)
        return self._embed_text(payload)

    @staticmethod
    def _text_unit_count(text: str) -> int:
        lowered = str(text or "").strip()
        if not lowered:
            return 0
        return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[\u3400-\u9fff]", lowered))

    @staticmethod
    def _turn_signal_terms(prepared_turn: dict[str, Any]) -> set[str]:
        event = prepared_turn["event"]
        metadata = dict(event.metadata or {})
        refs = [str(item).strip().lower() for item in (event.canonical_entity_refs or []) if str(item).strip()]
        refs.extend(str(item).strip().lower() for item in (metadata.get("semantic_refs") or []) if str(item).strip())
        return set(refs)

    def _should_merge_turn(self, chunk: list[dict[str, Any]], next_turn: dict[str, Any], chunk_units: int) -> bool:
        if not chunk:
            return True
        next_units = self._text_unit_count(next_turn["event"].text)
        if len(chunk) >= MERGED_ATOM_MAX_TURNS:
            return False
        if chunk_units + next_units > MERGED_ATOM_MAX_UNITS:
            return False
        if len(chunk) < MERGED_ATOM_MIN_TURNS:
            return True
        last_turn = chunk[-1]
        last_event = last_turn["event"]
        next_event = next_turn["event"]
        overlap = self._turn_signal_terms(last_turn).intersection(self._turn_signal_terms(next_turn))
        if overlap:
            return True
        if str(last_event.speaker) == str(next_event.speaker):
            return True
        if next_units <= 36:
            return True
        if chunk_units <= 120:
            return True
        return False

    def _build_extraction_chunks(self, session_turns: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not session_turns:
            return []
        chunks: list[list[dict[str, Any]]] = []
        current_chunk: list[dict[str, Any]] = []
        current_units = 0
        for prepared_turn in session_turns:
            turn_units = self._text_unit_count(prepared_turn["event"].text)
            if current_chunk and not self._should_merge_turn(current_chunk, prepared_turn, current_units):
                chunks.append(current_chunk)
                current_chunk = []
                current_units = 0
            current_chunk.append(prepared_turn)
            current_units += turn_units
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    @staticmethod
    def _last_sentence(text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return ""
        pieces = [piece.strip() for piece in re.split(r"(?<=[\.\!\?。！？])\s+|\n+", stripped) if piece.strip()]
        return pieces[-1] if pieces else stripped

    def _build_chunk_extraction_span(
        self,
        chunk: list[dict[str, Any]],
        previous_chunk: list[dict[str, Any]] | None,
    ) -> RawSpan:
        anchor_turn = chunk[-1]
        anchor_event = anchor_turn["event"]
        support_span_ids = [str(item["span"].span_id) for item in chunk if item.get("span") is not None]
        overlap_sentence = ""
        if previous_chunk:
            overlap_sentence = self._last_sentence(previous_chunk[-1]["event"].text)
        lines: list[str] = []
        if overlap_sentence:
            lines.append(f"Context overlap: {overlap_sentence}")
        for prepared_turn in chunk:
            event = prepared_turn["event"]
            lines.append(f"{event.speaker}: {event.text}")
        merged_text = "\n".join(line for line in lines if line.strip())
        semantic_refs = merge_memory_refs([prepared_turn["event"].text for prepared_turn in chunk], limit=16)
        metadata = {
            "source": "merged_chunk",
            "support_span_ids": support_span_ids,
            "source_speakers": list(dict.fromkeys(str(item["event"].speaker) for item in chunk)),
            "merged_turn_count": len(chunk),
            "merged_turn_indexes": [int(item["event"].turn_index) for item in chunk],
            "semantic_refs": semantic_refs,
        }
        if overlap_sentence:
            metadata["overlap_sentence"] = overlap_sentence
        return RawSpan(
            span_id=stable_id(
                "leaf_chunk",
                anchor_event.corpus_id,
                anchor_event.session_id,
                str(chunk[0]["event"].turn_index),
                str(chunk[-1]["event"].turn_index),
            ),
            corpus_id=anchor_event.corpus_id,
            session_id=anchor_event.session_id,
            speaker=anchor_event.speaker,
            text=merged_text,
            turn_index=anchor_event.turn_index,
            timestamp=anchor_event.timestamp,
            metadata=metadata,
            embedding=None,
        )

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        stripped = str(text or "")
        if not stripped:
            return 0
        return max(1, (len(stripped) + 3) // 4)

    @staticmethod
    def _diff_stats(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        diff: dict[str, Any] = {}
        for key, value in after.items():
            previous = before.get(key)
            if isinstance(value, dict):
                nested_keys = sorted(set(previous.keys() if isinstance(previous, dict) else ()).union(value.keys()))
                diff[key] = {nested_key: int(value.get(nested_key, 0)) - int((previous or {}).get(nested_key, 0)) for nested_key in nested_keys}
            elif isinstance(value, int):
                diff[key] = value - int(previous or 0)
            else:
                diff[key] = value
        return diff

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

    def _refresh_session_snapshot(self, corpus_id: str, title: str, session_id: str) -> dict[str, int]:
        events = self.store.get_events(corpus_id=corpus_id, session_id=session_id)
        objects = self.store.get_objects_for_session(corpus_id=corpus_id, session_id=session_id)
        session_snapshot_id = stable_id("leaf_snap", corpus_id, "session", session_id)
        session_blocks, block_count = self._build_session_blocks(
            corpus_id=corpus_id,
            title=title,
            session_id=session_id,
            events=events,
            parent_snapshot_id=session_snapshot_id,
        )
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
        tags = build_text_tags(active_summaries + event_texts, max_tags=5)
        snapshot = MemorySnapshotRecord(
            snapshot_id=session_snapshot_id,
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
            child_ids=[block.snapshot_id for block in session_blocks],
            entity_refs=refs,
            canonical_entity_refs=canonicalize_entities(refs),
            time_range=self._time_range(events),
            metadata={
                "num_events": len(events),
                "num_objects": len(object_ids),
                "num_blocks": len(session_blocks),
                "semantic_role": "session_snapshot",
                "tags": tags,
            },
            embedding=self._embed_snapshot(f"{title}:{session_id}", summary, tags),
        )
        self.store.upsert_snapshot(snapshot)
        return {"session": 1, "session_block": block_count}

    def _build_session_blocks(
        self,
        corpus_id: str,
        title: str,
        session_id: str,
        events: list[MemoryEventRecord],
        parent_snapshot_id: str,
    ) -> tuple[list[MemorySnapshotRecord], int]:
        if not events:
            return [], 0
        top_chunks = self._partition_event_ranges(events, max_chunk_size=SESSION_BLOCK_TOP_EVENTS)
        built_blocks: list[MemorySnapshotRecord] = []
        total_blocks = 0
        for chunk_index, chunk in enumerate(top_chunks, start=1):
            block, block_count = self._build_session_block_node(
                corpus_id=corpus_id,
                title=title,
                session_id=session_id,
                events=chunk,
                parent_snapshot_id=parent_snapshot_id,
                depth=0,
                branch_label=str(chunk_index),
            )
            built_blocks.append(block)
            total_blocks += block_count
        return built_blocks, total_blocks

    def _build_session_pages(
        self,
        corpus_id: str,
        title: str,
        session_id: str,
        events: list[MemoryEventRecord],
        parent_snapshot_id: str,
    ) -> list[MemorySnapshotRecord]:
        pages: list[MemorySnapshotRecord] = []
        if not events:
            return pages
        num_pages = (len(events) + SESSION_PAGE_SIZE - 1) // SESSION_PAGE_SIZE
        for page_index, start in enumerate(range(0, len(events), SESSION_PAGE_SIZE), start=1):
            chunk = events[start : start + SESSION_PAGE_SIZE]
            chunk_texts = [f"{event.speaker}: {event.text}" for event in chunk]
            refs = merge_memory_refs(chunk_texts, limit=12)
            page_title = f"{title}:{session_id}:page-{page_index}"
            page_summary = summarize_texts(chunk_texts, max_chars=480)
            tags = build_text_tags(chunk_texts, max_tags=5)
            page = MemorySnapshotRecord(
                snapshot_id=stable_id("leaf_snap", corpus_id, "session_page", session_id, str(page_index)),
                corpus_id=corpus_id,
                parent_id=parent_snapshot_id,
                snapshot_kind="session_page",
                scope_id=f"{session_id}:page:{page_index}",
                title=page_title,
                synopsis=make_synopsis(chunk_texts[:4], max_chars=160),
                summary=page_summary,
                object_ids=[],
                event_ids=[event.event_id for event in chunk],
                raw_refs=[event.raw_span_id for event in chunk if event.raw_span_id],
                child_ids=[event.event_id for event in chunk],
                entity_refs=refs,
                canonical_entity_refs=canonicalize_entities(refs),
                time_range=self._time_range(chunk),
                metadata={
                    "semantic_role": "session_page",
                    "session_id": session_id,
                    "page_index": page_index,
                    "num_pages": num_pages,
                    "start_turn_index": chunk[0].turn_index,
                    "end_turn_index": chunk[-1].turn_index,
                    "num_events": len(chunk),
                    "tags": tags,
                },
                embedding=self._embed_snapshot(page_title, page_summary, tags),
            )
            self.store.upsert_snapshot(page)
            pages.append(page)
        return pages

    def _build_session_block_node(
        self,
        corpus_id: str,
        title: str,
        session_id: str,
        events: list[MemoryEventRecord],
        parent_snapshot_id: str,
        depth: int,
        branch_label: str,
    ) -> tuple[MemorySnapshotRecord, int]:
        start_turn_index = events[0].turn_index
        end_turn_index = events[-1].turn_index
        snapshot_id = stable_id(
            "leaf_snap",
            corpus_id,
            "session_block",
            session_id,
            str(start_turn_index),
            str(end_turn_index),
        )

        child_blocks: list[MemorySnapshotRecord] = []
        child_block_count = 0
        is_leaf = len(events) <= SESSION_BLOCK_LEAF_EVENTS
        if not is_leaf:
            midpoint = max(1, len(events) // 2)
            left_chunk = events[:midpoint]
            right_chunk = events[midpoint:]
            child_chunks = [chunk for chunk in (left_chunk, right_chunk) if chunk]
            for child_index, child_chunk in enumerate(child_chunks, start=1):
                child_block, nested_count = self._build_session_block_node(
                    corpus_id=corpus_id,
                    title=title,
                    session_id=session_id,
                    events=child_chunk,
                    parent_snapshot_id=snapshot_id,
                    depth=depth + 1,
                    branch_label=f"{branch_label}.{child_index}",
                )
                child_blocks.append(child_block)
                child_block_count += nested_count

        if child_blocks:
            summary_inputs = [block.summary for block in child_blocks]
            synopsis_inputs = [block.synopsis for block in child_blocks]
            refs = merge_memory_refs(summary_inputs + synopsis_inputs, limit=12)
            event_ids = []
            raw_refs = []
        else:
            chunk_texts = [f"{event.speaker}: {event.text}" for event in events]
            summary_inputs = chunk_texts
            synopsis_inputs = chunk_texts[:4]
            refs = merge_memory_refs(chunk_texts, limit=12)
            event_ids = [event.event_id for event in events]
            raw_refs = [event.raw_span_id for event in events if event.raw_span_id]

        summary = summarize_texts(summary_inputs, max_chars=480 if is_leaf else 560)
        synopsis = make_synopsis(synopsis_inputs[:4], max_chars=160)
        tags = build_text_tags(summary_inputs, max_tags=5)
        block = MemorySnapshotRecord(
            snapshot_id=snapshot_id,
            corpus_id=corpus_id,
            parent_id=parent_snapshot_id,
            snapshot_kind="session_block",
            scope_id=f"{session_id}:block:{start_turn_index}-{end_turn_index}",
            title=f"{title}:{session_id}:block-{branch_label}",
            synopsis=synopsis,
            summary=summary,
            object_ids=[],
            event_ids=event_ids,
            raw_refs=raw_refs,
            child_ids=[child.snapshot_id for child in child_blocks],
            entity_refs=refs,
            canonical_entity_refs=canonicalize_entities(refs),
            time_range=self._time_range(events),
            metadata={
                "semantic_role": "session_block",
                "session_id": session_id,
                "depth": depth,
                "branch_label": branch_label,
                "is_leaf": is_leaf,
                "start_turn_index": start_turn_index,
                "end_turn_index": end_turn_index,
                "num_events": len(events),
                "num_children": len(child_blocks),
                "tags": tags,
            },
            embedding=self._embed_snapshot(f"{title}:{session_id}:block-{branch_label}", summary, tags),
        )
        self.store.upsert_snapshot(block)
        return block, 1 + child_block_count

    @staticmethod
    def _partition_event_ranges(
        events: list[MemoryEventRecord],
        max_chunk_size: int,
    ) -> list[list[MemoryEventRecord]]:
        if not events:
            return []
        if len(events) <= max_chunk_size:
            return [events]
        chunk_count = (len(events) + max_chunk_size - 1) // max_chunk_size
        chunk_size = (len(events) + chunk_count - 1) // chunk_count
        return [events[start : start + chunk_size] for start in range(0, len(events), chunk_size)]

    def _refresh_entity_snapshot(self, corpus_id: str, title: str, subject: str) -> dict[str, int]:
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
        tags = build_text_tags([subject] + active_versions + event_texts, max_tags=5)
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
                "tags": tags,
            },
            embedding=self._embed_snapshot(f"{title}:{subject}", summary, tags),
        )
        self.store.upsert_snapshot(snapshot)
        return {"entity": 1}

    def _refresh_root_snapshot(self, corpus_id: str, title: str) -> dict[str, int]:
        session_snapshots = self.store.list_snapshots(corpus_id=corpus_id, snapshot_kind="session")
        entity_snapshots = self.store.list_snapshots(corpus_id=corpus_id, snapshot_kind="entity")
        events = self.store.get_events(corpus_id=corpus_id)
        summary_inputs = [snapshot.summary for snapshot in session_snapshots[-6:]] + [snapshot.summary for snapshot in entity_snapshots[:6]]
        if not summary_inputs:
            summary_inputs = [f"{event.speaker}: {event.text}" for event in events[-12:]]
        refs = merge_memory_refs(summary_inputs, limit=20)
        root_summary = summarize_texts(summary_inputs, max_chars=900)
        tags = build_text_tags(summary_inputs, max_tags=5)
        snapshot = MemorySnapshotRecord(
            snapshot_id=stable_id("leaf_snap", corpus_id, "root"),
            corpus_id=corpus_id,
            parent_id=None,
            snapshot_kind="root",
            scope_id=corpus_id,
            title=title,
            synopsis=make_synopsis(summary_inputs[:6], max_chars=220),
            summary=root_summary,
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
                "tags": tags,
            },
            embedding=self._embed_snapshot(title, root_summary, tags),
        )
        self.store.upsert_snapshot(snapshot)
        return {"root": 1}

    def _resolve_subject(self, event: MemoryEventRecord, atom: MemoryAtomRecord) -> str:
        lowered = atom.content.lower().strip()
        if (
            lowered.startswith("i ")
            or lowered.startswith("i'")
            or lowered.startswith("my ")
            or re.search(r"\b(i|my|me|i'm)\b", lowered) is not None
        ):
            source_speaker = str(atom.metadata.get("source_speaker") or event.speaker)
            canonical_speaker = canonicalize_entities([source_speaker], limit=1)
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
        if isinstance(getattr(self, "_ingest_runtime_metrics", None), dict):
            self._ingest_runtime_metrics["reconcile_llm_calls"] = int(
                self._ingest_runtime_metrics.get("reconcile_llm_calls", 0)
            ) + 1
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
