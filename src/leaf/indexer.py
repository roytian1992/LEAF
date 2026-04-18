from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .clients import (
    ChatClient,
    EmbeddingClient,
    OpenAICompatError,
    estimate_message_tokens,
    extract_chat_text,
    extract_json_object,
    extract_prompt_tokens,
)
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
ENTITY_SLOT_EVENT_LIMIT = 16
ENTITY_SLOT_OBJECT_LIMIT = 16
ENTITY_ASPECT_EVENT_LIMIT = 10
ENTITY_FACET_EVENT_LIMIT = 12
ENTITY_FACET_OBJECT_LIMIT = 16
ENTITY_FACET_MAX_PER_SUBJECT = 16
ENABLE_ENTITY_ASPECT_SNAPSHOTS = False
ENABLE_ENTITY_FACET_SNAPSHOTS = False
MERGED_ATOM_MAX_UNITS = 220
MERGED_ATOM_MIN_TURNS = 2
MERGED_ATOM_MAX_TURNS = 6
BRIDGE_MIN_SESSION_COUNT = 2
BRIDGE_MIN_EVENT_COUNT = 2
BRIDGE_MAX_EVENTS = 8
BRIDGE_MAX_PER_ENTITY = 6
BRIDGE_CLUSTER_MODES = {"hybrid", "embedding_cluster", "graph_lexical"}
BRIDGE_HYBRID_CLUSTER_THRESHOLD = 0.58
BRIDGE_EMBED_CLUSTER_THRESHOLD = 0.72
BRIDGE_GRAPH_EDGE_SCORE_THRESHOLD = 0.22
BRIDGE_GRAPH_EMBED_THRESHOLD = 0.88
BRIDGE_STRONG_ANCHOR_TOKEN_IDF = 2.25
BRIDGE_STRONG_ANCHOR_PHRASE_IDF = 2.8
BRIDGE_HUB_TOKEN_RATIO = 0.16
BRIDGE_GENERIC_SUPPORT_TOKENS = {
    "amazing",
    "awesome",
    "beautiful",
    "congrats",
    "congratulations",
    "glad",
    "good",
    "great",
    "happy",
    "love",
    "lovely",
    "nice",
    "photo",
    "picture",
    "proud",
    "sorry",
    "support",
    "supportive",
    "supported",
    "supporting",
    "thanks",
    "thank",
    "wonderful",
    "wow",
}
BRIDGE_TOKEN_STOPWORDS = {
    "about",
    "after",
    "again",
    "around",
    "back",
    "because",
    "been",
    "being",
    "came",
    "come",
    "cool",
    "days",
    "definitely",
    "felt",
    "from",
    "glad",
    "going",
    "good",
    "great",
    "just",
    "last",
    "like",
    "made",
    "make",
    "moments",
    "more",
    "really",
    "said",
    "some",
    "something",
    "still",
    "such",
    "than",
    "that",
    "there",
    "they",
    "this",
    "those",
    "through",
    "time",
    "times",
    "took",
    "want",
    "with",
    "would",
    "yeah",
}
FACET_GENERIC_TERMS = {
    "activity",
    "conversation",
    "event",
    "goal",
    "identity",
    "memory",
    "observation",
    "person",
    "plan",
    "preference",
    "profile",
    "relation",
    "relationship",
    "session",
    "state",
    "story",
    "thing",
    "update",
}


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


def _ordered_unique_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _bridge_support_tokens(text: str) -> list[str]:
    return [
        token
        for token in _normalize_value(text).split()
        if len(token) >= 4 and token not in BRIDGE_TOKEN_STOPWORDS
    ]


def _bridge_support_phrases(tokens: list[str]) -> list[str]:
    phrases: list[str] = []
    for index in range(len(tokens) - 1):
        left = tokens[index]
        right = tokens[index + 1]
        if left in BRIDGE_GENERIC_SUPPORT_TOKENS and right in BRIDGE_GENERIC_SUPPORT_TOKENS:
            continue
        phrases.append(f"{left} {right}")
    return phrases


def _slugify_scope_term(text: str) -> str:
    normalized = _normalize_value(text).replace(" ", "_").strip("_")
    return normalized[:64] if normalized else ""


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
        self._reconcile_cache_dir = self._resolve_reconcile_cache_dir()

    def append_turns(
        self,
        corpus_id: str,
        title: str,
        turns: list[dict[str, Any]],
        *,
        refresh_snapshots: bool = True,
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
        self._ingest_runtime_metrics = {
            "reconcile_llm_calls": 0,
            "reconcile_prompt_tokens_total": 0,
            "reconcile_prompt_tokens_provider_usage_calls": 0,
            "reconcile_prompt_tokens_estimated_calls": 0,
        }
        self.atom_extractor.reset_runtime_metrics()
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
            extraction_worker_count = self._ingest_extraction_worker_count(default_workers=prepare_worker_count)
            prepared_turns = self._attach_chunk_extractions(prepared_turns, max_workers=extraction_worker_count)
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
            if refresh_snapshots:
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
            atom_runtime_metrics = self.atom_extractor.consume_runtime_metrics()
        after_stats = self.store.get_corpus_stats(corpus_id)
        atom_prompt_tokens_total = int(atom_runtime_metrics.get("atom_prompt_tokens_total", 0))
        reconcile_prompt_tokens_total = int(runtime_metrics.get("reconcile_prompt_tokens_total", 0))
        provider_usage_calls = int(atom_runtime_metrics.get("atom_prompt_tokens_provider_usage_calls", 0)) + int(
            runtime_metrics.get("reconcile_prompt_tokens_provider_usage_calls", 0)
        )
        estimated_calls = int(atom_runtime_metrics.get("atom_prompt_tokens_estimated_calls", 0)) + int(
            runtime_metrics.get("reconcile_prompt_tokens_estimated_calls", 0)
        )
        return {
            "ingest_elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
            "turn_count": len(turns),
            "input_text_chars": input_text_chars,
            "input_text_tokens_est": input_text_tokens_est,
            "prepare_workers": prepare_worker_count,
            "extraction_workers": extraction_worker_count,
            "snapshot_refresh_skipped": not refresh_snapshots,
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
            "memory_llm_prompt_tokens_est": {
                "atom_extraction": atom_prompt_tokens_total,
                "reconciliation": reconcile_prompt_tokens_total,
                "total": atom_prompt_tokens_total + reconcile_prompt_tokens_total,
            },
            "memory_llm_prompt_token_source": {
                "provider_usage_calls": provider_usage_calls,
                "estimated_calls": estimated_calls,
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

    def _ingest_extraction_worker_count(self, *, default_workers: int) -> int:
        raw_value = str(os.environ.get("LEAF_INGEST_EXTRACTION_WORKERS", "")).strip()
        if raw_value:
            try:
                return max(1, int(raw_value))
            except ValueError:
                return max(1, default_workers)
        return max(1, default_workers)

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

    def _attach_chunk_extractions(
        self,
        prepared_turns: list[dict[str, Any]],
        *,
        max_workers: int,
    ) -> list[dict[str, Any]]:
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
        chunk_jobs: list[dict[str, Any]] = []
        for session_id in session_order:
            session_turns = grouped_by_session[session_id]
            chunks = self._build_extraction_chunks(session_turns)
            previous_chunk: list[dict[str, Any]] | None = None
            for chunk in chunks:
                anchor_turn = chunk[-1]
                anchor_event = anchor_turn["event"]
                extraction_span = self._build_chunk_extraction_span(chunk, previous_chunk)
                support_span_ids = [str(item["span"].span_id) for item in chunk if item.get("span") is not None]
                chunk_jobs.append(
                    {
                        "anchor_turn": anchor_turn,
                        "anchor_event": anchor_event,
                        "extraction_span": extraction_span,
                        "support_span_ids": support_span_ids,
                        "merged_turn_indexes": [int(item["event"].turn_index) for item in chunk],
                    }
                )
                previous_chunk = chunk
        if not chunk_jobs:
            return prepared_turns
        extracted_atoms_by_job: list[list[Any] | None] = [None] * len(chunk_jobs)
        if max_workers <= 1 or len(chunk_jobs) == 1:
            for index, job in enumerate(chunk_jobs):
                extracted_atoms_by_job[index] = self.atom_extractor.extract_atoms(job["extraction_span"])
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(chunk_jobs))) as executor:
                futures = [
                    executor.submit(self.atom_extractor.extract_atoms, job["extraction_span"])
                    for job in chunk_jobs
                ]
                for index, future in enumerate(futures):
                    extracted_atoms_by_job[index] = future.result()
        for job, extracted_atoms in zip(chunk_jobs, extracted_atoms_by_job):
            anchor_turn = job["anchor_turn"]
            anchor_event = job["anchor_event"]
            extraction_span = job["extraction_span"]
            support_span_ids = list(job["support_span_ids"])
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
                        "merged_turn_count": len(support_span_ids),
                        "support_span_ids": support_span_ids,
                        "merged_turn_indexes": list(job["merged_turn_indexes"]),
                        "chunk_session_id": anchor_event.session_id,
                    },
                )
                for atom in (extracted_atoms or [])
            ]
            anchor_turn["atoms"] = atoms
            anchor_turn["state_candidates"] = self._derive_state_candidates(event=anchor_event, atoms=atoms)
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
        latest_versions_by_object = self.store.get_latest_versions([obj.object_id for obj in objects])
        session_snapshot_id = stable_id("leaf_snap", corpus_id, "session", session_id)
        self.store.delete_snapshots_by_scope_prefix(corpus_id=corpus_id, snapshot_kind="session_page", scope_prefix=f"{session_id}:page:")
        self.store.delete_snapshots_by_scope_prefix(corpus_id=corpus_id, snapshot_kind="session_block", scope_prefix=f"{session_id}:block:")
        session_pages = self._build_session_pages(
            corpus_id=corpus_id,
            title=title,
            session_id=session_id,
            events=events,
            parent_snapshot_id=session_snapshot_id,
        )
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
            version = latest_versions_by_object.get(obj.object_id)
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
            child_ids=[page.snapshot_id for page in session_pages] + [block.snapshot_id for block in session_blocks],
            entity_refs=refs,
            canonical_entity_refs=canonicalize_entities(refs),
            time_range=self._time_range(events),
            metadata={
                "num_events": len(events),
                "num_objects": len(object_ids),
                "num_pages": len(session_pages),
                "num_blocks": len(session_blocks),
                "semantic_role": "session_snapshot",
                "tags": tags,
            },
            embedding=self._embed_snapshot(f"{title}:{session_id}", summary, tags),
        )
        self.store.upsert_snapshot(snapshot)
        return {"session": 1, "session_page": len(session_pages), "session_block": block_count}

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

    def _refresh_entity_snapshot(
        self,
        corpus_id: str,
        title: str,
        subject: str,
        *,
        build_entity_aspects: bool = ENABLE_ENTITY_ASPECT_SNAPSHOTS,
        build_entity_facets: bool = ENABLE_ENTITY_FACET_SNAPSHOTS,
    ) -> dict[str, int]:
        objects = self.store.get_objects_for_subject(corpus_id=corpus_id, subject=subject)
        object_ids = [obj.object_id for obj in objects]
        versions_by_object = self.store.get_object_versions_for_objects(object_ids)
        latest_versions_by_object = self.store.get_latest_versions(object_ids)
        self.store.delete_snapshots_by_scope_prefix(corpus_id=corpus_id, snapshot_kind="entity_slot", scope_prefix=f"{subject}:")
        self.store.delete_snapshots_by_scope_prefix(corpus_id=corpus_id, snapshot_kind="entity_aspect", scope_prefix=f"{subject}:")
        self.store.delete_snapshots_by_scope_prefix(corpus_id=corpus_id, snapshot_kind="entity_facet", scope_prefix=f"{subject}:")
        versions: list[MemoryObjectVersionRecord] = []
        for obj in objects:
            versions.extend(versions_by_object.get(obj.object_id, []))
        active_versions = [version.summary for version in versions if version.status == "active"]
        recent_events = self.store.get_events_for_entity(corpus_id=corpus_id, entity=subject, limit=12)
        event_texts = [f"{event.speaker}: {event.text}" for event in recent_events]
        summary = summarize_texts(active_versions + event_texts, max_chars=760)
        synopsis = make_synopsis(active_versions[:4] + event_texts[:4], max_chars=180)
        refs = merge_memory_refs([subject] + active_versions + event_texts, limit=16)
        tags = build_text_tags([subject] + active_versions + event_texts, max_tags=5)
        entity_snapshot_id = stable_id("leaf_snap", corpus_id, "entity", subject)
        slot_snapshots = self._build_entity_slot_snapshots(
            corpus_id=corpus_id,
            title=title,
            subject=subject,
            objects=objects,
            parent_snapshot_id=entity_snapshot_id,
            latest_versions_by_object=latest_versions_by_object,
        )
        aspect_snapshots: list[MemorySnapshotRecord] = []
        if build_entity_aspects:
            aspect_snapshots = self._build_entity_aspect_snapshots(
                corpus_id=corpus_id,
                title=title,
                subject=subject,
                objects=objects,
                parent_snapshot_id=entity_snapshot_id,
                latest_versions_by_object=latest_versions_by_object,
            )
        facet_snapshots: list[MemorySnapshotRecord] = []
        if build_entity_facets:
            facet_snapshots = self._build_entity_facet_snapshots(
                corpus_id=corpus_id,
                title=title,
                subject=subject,
                objects=objects,
                parent_snapshot_id=entity_snapshot_id,
                latest_versions_by_object=latest_versions_by_object,
            )
        snapshot = MemorySnapshotRecord(
            snapshot_id=entity_snapshot_id,
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
            child_ids=[facet_snapshot.snapshot_id for facet_snapshot in facet_snapshots]
            + [aspect_snapshot.snapshot_id for aspect_snapshot in aspect_snapshots]
            + [slot_snapshot.snapshot_id for slot_snapshot in slot_snapshots]
            + object_ids[:24],
            entity_refs=refs,
            canonical_entity_refs=canonicalize_entities(refs),
            time_range=self._time_range(recent_events),
            metadata={
                "num_objects": len(object_ids),
                "num_events": len(recent_events),
                "num_facet_snapshots": len(facet_snapshots),
                "num_aspect_snapshots": len(aspect_snapshots),
                "num_slot_snapshots": len(slot_snapshots),
                "semantic_role": "entity_snapshot",
                "tags": tags,
            },
            embedding=self._embed_snapshot(f"{title}:{subject}", summary, tags),
        )
        self.store.upsert_snapshot(snapshot)
        return {
            "entity": 1,
            "entity_facet": len(facet_snapshots),
            "entity_slot": len(slot_snapshots),
            "entity_aspect": len(aspect_snapshots),
        }

    def _build_entity_slot_snapshots(
        self,
        *,
        corpus_id: str,
        title: str,
        subject: str,
        objects: list[MemoryObjectRecord],
        parent_snapshot_id: str,
        latest_versions_by_object: dict[str, MemoryObjectVersionRecord] | None = None,
    ) -> list[MemorySnapshotRecord]:
        slot_snapshots: list[MemorySnapshotRecord] = []
        objects_by_slot: dict[str, list[MemoryObjectRecord]] = defaultdict(list)
        for obj in objects:
            slot = str(obj.slot or "").strip()
            if not slot:
                continue
            objects_by_slot[slot].append(obj)
        for slot, slot_objects in sorted(objects_by_slot.items()):
            active_versions: list[MemoryObjectVersionRecord] = []
            slot_object_ids: list[str] = []
            for obj in slot_objects:
                slot_object_ids.append(obj.object_id)
                latest_version = (latest_versions_by_object or {}).get(obj.object_id) or self.store.get_latest_version(obj.object_id)
                if latest_version is not None and str(latest_version.status or "") == "active":
                    active_versions.append(latest_version)
            if not active_versions and not slot_object_ids:
                continue
            support_events = self.store.get_events_for_object_ids(
                corpus_id=corpus_id,
                object_ids=slot_object_ids,
                limit=ENTITY_SLOT_EVENT_LIMIT,
            )
            support_atoms = self.store.get_atoms_for_events([event.event_id for event in support_events]) if support_events else []
            support_events = self._expand_support_events_from_atoms(
                corpus_id=corpus_id,
                support_events=support_events,
                support_atoms=support_atoms,
                limit=ENTITY_SLOT_EVENT_LIMIT,
            )
            version_summaries = [version.summary for version in active_versions]
            event_texts = [f"{event.speaker}: {event.text}" for event in support_events]
            summary_inputs = [subject, slot] + version_summaries + event_texts
            refs = merge_memory_refs(summary_inputs, limit=16)
            tags = build_text_tags(summary_inputs, max_tags=5)
            slot_summary = summarize_texts(summary_inputs, max_chars=760)
            slot_snapshot = MemorySnapshotRecord(
                snapshot_id=stable_id("leaf_snap", corpus_id, "entity_slot", subject, slot),
                corpus_id=corpus_id,
                parent_id=parent_snapshot_id,
                snapshot_kind="entity_slot",
                scope_id=f"{subject}:{slot}",
                title=f"{title}:{subject}:{slot}",
                synopsis=make_synopsis(version_summaries[:3] + event_texts[:3], max_chars=180),
                summary=slot_summary,
                object_ids=slot_object_ids[:ENTITY_SLOT_OBJECT_LIMIT],
                event_ids=[event.event_id for event in support_events[:ENTITY_SLOT_EVENT_LIMIT]],
                raw_refs=[event.raw_span_id for event in support_events[:ENTITY_SLOT_EVENT_LIMIT] if event.raw_span_id],
                child_ids=slot_object_ids[:ENTITY_SLOT_OBJECT_LIMIT],
                entity_refs=[subject, slot] + refs,
                canonical_entity_refs=canonicalize_entities([subject] + refs, limit=16),
                time_range=self._time_range(support_events),
                metadata={
                    "semantic_role": "entity_slot_snapshot",
                    "subject": subject,
                    "slot": slot,
                    "memory_kind": slot_objects[0].memory_kind if slot_objects else "",
                    "num_objects": len(slot_object_ids),
                    "num_events": len(support_events),
                    "tags": tags,
                },
                embedding=self._embed_snapshot(f"{title}:{subject}:{slot}", slot_summary, tags),
            )
            self.store.upsert_snapshot(slot_snapshot)
            slot_snapshots.append(slot_snapshot)
        return slot_snapshots

    def _build_entity_aspect_snapshots(
        self,
        *,
        corpus_id: str,
        title: str,
        subject: str,
        objects: list[MemoryObjectRecord],
        parent_snapshot_id: str,
        latest_versions_by_object: dict[str, MemoryObjectVersionRecord] | None = None,
    ) -> list[MemorySnapshotRecord]:
        aspect_snapshots: list[MemorySnapshotRecord] = []
        for obj in objects:
            latest_version = (latest_versions_by_object or {}).get(obj.object_id) or self.store.get_latest_version(obj.object_id)
            if latest_version is None or str(latest_version.status or "") != "active":
                continue
            support_events = self.store.get_events_for_object_ids(
                corpus_id=corpus_id,
                object_ids=[obj.object_id],
                limit=ENTITY_ASPECT_EVENT_LIMIT,
            )
            support_atoms = self.store.get_atoms_for_events([event.event_id for event in support_events]) if support_events else []
            support_events = self._expand_support_events_from_atoms(
                corpus_id=corpus_id,
                support_events=support_events,
                support_atoms=support_atoms,
                limit=ENTITY_ASPECT_EVENT_LIMIT,
            )
            version_summary = str(latest_version.summary or "").strip()
            event_texts = [f"{event.speaker}: {event.text}" for event in support_events]
            summary_inputs = [subject, obj.slot, version_summary] + event_texts
            tags = build_text_tags(summary_inputs, max_tags=6)
            aspect_terms = _ordered_unique_strings(
                [str(obj.slot or "").strip().lower()]
                + [str(tag).strip().lower() for tag in tags if str(tag).strip()]
            )[:4]
            aspect_key = "-".join(term.replace(" ", "_") for term in aspect_terms[:2] if term) or str(obj.slot or "aspect")
            refs = merge_memory_refs(summary_inputs, limit=16)
            aspect_summary = summarize_texts(summary_inputs, max_chars=760)
            aspect_snapshots.append(
                MemorySnapshotRecord(
                    snapshot_id=stable_id("leaf_snap", corpus_id, "entity_aspect", subject, obj.object_id),
                    corpus_id=corpus_id,
                    parent_id=parent_snapshot_id,
                    snapshot_kind="entity_aspect",
                    scope_id=f"{subject}:aspect:{obj.slot}:{obj.object_id}",
                    title=f"{title}:{subject}:aspect-{aspect_key}",
                    synopsis=make_synopsis([version_summary] + event_texts[:3], max_chars=180),
                    summary=aspect_summary,
                    object_ids=[obj.object_id],
                    event_ids=[event.event_id for event in support_events[:ENTITY_ASPECT_EVENT_LIMIT]],
                    raw_refs=[event.raw_span_id for event in support_events[:ENTITY_ASPECT_EVENT_LIMIT] if event.raw_span_id],
                    child_ids=[obj.object_id],
                    entity_refs=[subject, str(obj.slot or "")] + refs,
                    canonical_entity_refs=canonicalize_entities([subject] + refs, limit=16),
                    time_range=self._time_range(support_events),
                    metadata={
                        "semantic_role": "entity_aspect_snapshot",
                        "subject": subject,
                        "slot": obj.slot,
                        "object_id": obj.object_id,
                        "memory_kind": obj.memory_kind,
                        "aspect_terms": aspect_terms,
                        "num_events": len(support_events),
                        "tags": tags,
                    },
                    embedding=self._embed_snapshot(f"{title}:{subject}:aspect-{aspect_key}", aspect_summary, tags),
                )
            )
        aspect_snapshots.sort(
            key=lambda snapshot: (
                -int((snapshot.metadata or {}).get("num_events") or 0),
                str((snapshot.metadata or {}).get("slot") or ""),
                str(snapshot.scope_id),
            )
        )
        for snapshot in aspect_snapshots:
            self.store.upsert_snapshot(snapshot)
        return aspect_snapshots

    def _build_entity_facet_snapshots(
        self,
        *,
        corpus_id: str,
        title: str,
        subject: str,
        objects: list[MemoryObjectRecord],
        parent_snapshot_id: str,
        latest_versions_by_object: dict[str, MemoryObjectVersionRecord] | None = None,
    ) -> list[MemorySnapshotRecord]:
        subject_tokens = _token_set(subject)
        facet_groups: dict[str, dict[str, Any]] = {}
        for obj in objects:
            latest_version = (latest_versions_by_object or {}).get(obj.object_id) or self.store.get_latest_version(obj.object_id)
            if latest_version is None or str(latest_version.status or "") != "active":
                continue
            support_events = self.store.get_events_for_object_ids(
                corpus_id=corpus_id,
                object_ids=[obj.object_id],
                limit=ENTITY_FACET_EVENT_LIMIT,
            )
            support_event_ids = [event.event_id for event in support_events]
            support_atoms = self.store.get_atoms_for_events(support_event_ids) if support_event_ids else []
            support_events = self._expand_support_events_from_atoms(
                corpus_id=corpus_id,
                support_events=support_events,
                support_atoms=support_atoms,
                limit=ENTITY_FACET_EVENT_LIMIT,
            )
            anchor_terms = self._derive_entity_facet_anchor_terms(
                subject=subject,
                subject_tokens=subject_tokens,
                obj=obj,
                latest_version=latest_version,
                support_events=support_events,
                support_atoms=support_atoms,
            )
            if not anchor_terms:
                continue
            anchor = anchor_terms[0]
            group = facet_groups.setdefault(
                anchor,
                {
                    "anchor": anchor,
                    "object_ids": [],
                    "event_ids": [],
                    "raw_refs": [],
                    "version_summaries": [],
                    "atom_texts": [],
                    "event_texts": [],
                    "slots": set(),
                    "memory_kinds": set(),
                    "facet_terms": set(),
                },
            )
            group["facet_terms"].update(anchor_terms)
            group["slots"].add(str(obj.slot or "").strip())
            group["memory_kinds"].add(str(obj.memory_kind or "").strip())
            if obj.object_id not in group["object_ids"]:
                group["object_ids"].append(obj.object_id)
            version_summary = str(latest_version.summary or "").strip()
            if version_summary and version_summary not in group["version_summaries"]:
                group["version_summaries"].append(version_summary)
            for atom in support_atoms:
                atom_text = str(atom.content or "").strip()
                if atom_text and atom_text not in group["atom_texts"]:
                    group["atom_texts"].append(atom_text)
            for event in support_events:
                if event.event_id not in group["event_ids"]:
                    group["event_ids"].append(event.event_id)
                if event.raw_span_id and event.raw_span_id not in group["raw_refs"]:
                    group["raw_refs"].append(event.raw_span_id)
                event_text = f"{event.speaker}: {event.text}"
                if event_text not in group["event_texts"]:
                    group["event_texts"].append(event_text)

        ranked_groups = sorted(
            facet_groups.values(),
            key=lambda item: (
                -len(item["object_ids"]),
                -len(item["event_ids"]),
                -len(item["facet_terms"]),
                str(item["anchor"]),
            ),
        )
        snapshots: list[MemorySnapshotRecord] = []
        event_map = self.store.get_events_by_ids(
            [
                event_id
                for group in ranked_groups[:ENTITY_FACET_MAX_PER_SUBJECT]
                for event_id in group["event_ids"][:ENTITY_FACET_EVENT_LIMIT]
            ]
        )
        for group in ranked_groups[:ENTITY_FACET_MAX_PER_SUBJECT]:
            anchor = str(group["anchor"]).strip()
            if not anchor:
                continue
            facet_terms = _ordered_unique_strings(list(group["facet_terms"]))[:6]
            snapshot_events = [
                event_map[event_id]
                for event_id in group["event_ids"][:ENTITY_FACET_EVENT_LIMIT]
                if event_id in event_map
            ]
            summary_inputs = [subject, anchor]
            summary_inputs.extend(group["version_summaries"][:6])
            summary_inputs.extend(group["atom_texts"][:8])
            summary_inputs.extend(group["event_texts"][:8])
            tags = build_text_tags(summary_inputs, max_tags=6)
            refs = merge_memory_refs(summary_inputs, limit=16)
            facet_summary = summarize_texts(summary_inputs, max_chars=760)
            facet_scope = _slugify_scope_term(anchor) or "facet"
            snapshots.append(
                MemorySnapshotRecord(
                    snapshot_id=stable_id("leaf_snap", corpus_id, "entity_facet", subject, facet_scope),
                    corpus_id=corpus_id,
                    parent_id=parent_snapshot_id,
                    snapshot_kind="entity_facet",
                    scope_id=f"{subject}:facet:{facet_scope}",
                    title=f"{title}:{subject}:facet-{facet_scope}",
                    synopsis=make_synopsis([anchor] + group["version_summaries"][:3] + group["event_texts"][:2], max_chars=180),
                    summary=facet_summary,
                    object_ids=group["object_ids"][:ENTITY_FACET_OBJECT_LIMIT],
                    event_ids=group["event_ids"][:ENTITY_FACET_EVENT_LIMIT],
                    raw_refs=group["raw_refs"][:ENTITY_FACET_EVENT_LIMIT],
                    child_ids=group["object_ids"][:ENTITY_FACET_OBJECT_LIMIT],
                    entity_refs=[subject, anchor] + refs,
                    canonical_entity_refs=canonicalize_entities([subject, anchor] + refs, limit=16),
                    time_range=self._time_range(snapshot_events),
                    metadata={
                        "semantic_role": "entity_facet_snapshot",
                        "subject": subject,
                        "anchor_terms": facet_terms,
                        "source_slots": sorted(item for item in group["slots"] if item),
                        "memory_kinds": sorted(item for item in group["memory_kinds"] if item),
                        "num_objects": len(group["object_ids"]),
                        "num_events": len(group["event_ids"]),
                        "tags": tags,
                    },
                    embedding=self._embed_snapshot(f"{title}:{subject}:facet-{facet_scope}", facet_summary, tags),
                )
            )
        snapshots.sort(
            key=lambda snapshot: (
                -int((snapshot.metadata or {}).get("num_objects") or 0),
                -int((snapshot.metadata or {}).get("num_events") or 0),
                str(snapshot.scope_id),
            )
        )
        for snapshot in snapshots:
            self.store.upsert_snapshot(snapshot)
        return snapshots

    def _expand_support_events_from_atoms(
        self,
        *,
        corpus_id: str,
        support_events: list[MemoryEventRecord],
        support_atoms: list[MemoryAtomRecord],
        limit: int,
    ) -> list[MemoryEventRecord]:
        ordered_raw_span_ids = _ordered_unique_strings(
            [
                span_id
                for atom in support_atoms
                for span_id in (atom.support_span_ids or [])
                if str(span_id or "").strip()
            ]
        )
        expanded_events = self.store.get_events_for_raw_span_ids(corpus_id=corpus_id, raw_span_ids=ordered_raw_span_ids)
        by_event_id: dict[str, MemoryEventRecord] = {}
        for event in list(support_events) + list(expanded_events):
            by_event_id[str(event.event_id)] = event
        return sorted(
            by_event_id.values(),
            key=lambda event: (str(event.session_id), int(event.turn_index)),
        )[:limit]

    def _derive_entity_facet_anchor_terms(
        self,
        *,
        subject: str,
        subject_tokens: set[str],
        obj: MemoryObjectRecord,
        latest_version: MemoryObjectVersionRecord,
        support_events: list[MemoryEventRecord],
        support_atoms: list[MemoryAtomRecord],
    ) -> list[str]:
        slot_tokens = _token_set(obj.slot)
        support_texts: list[str] = [str(latest_version.value or ""), str(latest_version.summary or "")]
        support_texts.extend(str(atom.content or "") for atom in support_atoms if str(atom.content or "").strip())
        support_texts.extend(str(event.text or "") for event in support_events if str(event.text or "").strip())
        if not support_texts:
            return []
        semantic_refs: list[str] = []
        for atom in support_atoms:
            semantic_refs.extend(str(item).strip() for item in (atom.metadata or {}).get("semantic_refs") or [] if str(item).strip())
        for event in support_events:
            semantic_refs.extend(str(item).strip() for item in (event.metadata or {}).get("semantic_refs") or [] if str(item).strip())
        text_tags = build_text_tags(support_texts, max_tags=8)
        normalized_text_tags = {_normalize_value(tag) for tag in text_tags}
        normalized_semantic_refs = {_normalize_value(item) for item in semantic_refs}
        ordered_candidates = _ordered_unique_strings(
            text_tags
            + semantic_refs
            + merge_memory_refs(support_texts, limit=14)
        )
        scored_candidates: list[tuple[float, str]] = []
        normalized_texts = [_normalize_value(text) for text in support_texts if str(text or "").strip()]
        for candidate in ordered_candidates:
            normalized = _normalize_value(candidate)
            if not normalized:
                continue
            tokens = [token for token in normalized.split() if len(token) >= 3]
            if not tokens:
                continue
            if set(tokens).issubset(subject_tokens.union(slot_tokens)):
                continue
            if all(token in FACET_GENERIC_TERMS for token in tokens):
                continue
            if len(tokens) == 1 and (len(tokens[0]) < 5 or tokens[0] in FACET_GENERIC_TERMS):
                continue
            mention_count = sum(1 for text in normalized_texts if normalized in text or any(token in text for token in tokens))
            score = float(mention_count)
            if " " in normalized:
                score += 0.8
            if normalized in normalized_text_tags:
                score += 0.4
            if normalized in normalized_semantic_refs:
                score += 0.3
            if any(normalized in _normalize_value(text) for text in [latest_version.value, latest_version.summary]):
                score += 0.2
            scored_candidates.append((score, normalized))
        scored_candidates.sort(key=lambda item: (-item[0], -len(item[1].split()), -len(item[1]), item[1]))
        return _ordered_unique_strings([candidate for score, candidate in scored_candidates if score >= 1.0])[:3]

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

    def backfill_entity_bridges(
        self,
        corpus_id: str,
        title: str,
        *,
        refresh: bool = True,
        mode: str = "hybrid",
    ) -> dict[str, int]:
        bridge_mode = str(mode or "hybrid").strip().lower()
        if bridge_mode not in BRIDGE_CLUSTER_MODES:
            raise ValueError(f"Unsupported bridge mode: {mode}")
        try:
            removed = 0
            if refresh:
                removed = self.store.delete_snapshots(corpus_id=corpus_id, snapshot_kind="bridge")
            events = self.store.get_events(corpus_id=corpus_id)
            known_subjects = {
                str(subject).strip().lower()
                for subject in self.store.list_subjects(corpus_id)
                if str(subject).strip()
            }
            speaker_entities = {
                canonical
                for canonical in canonicalize_entities(
                    [str(event.speaker or "").strip() for event in events if str(event.speaker or "").strip()],
                    limit=max(8, len(events) * 2),
                )
                if canonical
            }
            bridge_entities_allowed = known_subjects.union(speaker_entities)
            events_by_entity: dict[str, list[MemoryEventRecord]] = defaultdict(list)
            for event in events:
                entity_keys = _ordered_unique_strings(
                    [str(event.speaker or "").strip().lower()]
                    + [str(item).strip().lower() for item in (event.canonical_entity_refs or []) if str(item).strip()]
                )
                for entity in entity_keys:
                    if entity and (not bridge_entities_allowed or entity in bridge_entities_allowed):
                        events_by_entity[entity].append(event)

            bridge_count = 0
            entity_count = 0
            for entity, entity_events in sorted(events_by_entity.items()):
                bridge_snapshots = self._build_entity_bridge_snapshots(
                    corpus_id=corpus_id,
                    title=title,
                    entity=entity,
                    events=entity_events,
                    mode=bridge_mode,
                )
                if not bridge_snapshots:
                    continue
                entity_count += 1
                for snapshot in bridge_snapshots:
                    self.store.upsert_snapshot(snapshot)
                bridge_count += len(bridge_snapshots)
            self.store.commit()
            return {
                "bridge": bridge_count,
                "bridge_entities": entity_count,
                "bridge_removed": removed,
                "bridge_mode": bridge_mode,
            }
        except Exception:
            self.store.rollback()
            raise

    def backfill_derived_snapshots(
        self,
        corpus_id: str,
        title: str,
        *,
        refresh: bool = True,
    ) -> dict[str, int]:
        try:
            removed_counts: dict[str, int] = {}
            if refresh:
                for snapshot_kind in ("session_page", "session_block", "session", "entity_facet", "entity_aspect", "entity_slot", "entity", "root"):
                    removed_counts[snapshot_kind] = self.store.delete_snapshots(corpus_id=corpus_id, snapshot_kind=snapshot_kind)
            snapshot_counts: dict[str, int] = defaultdict(int)
            session_ids = self.store.list_session_ids(corpus_id)
            for session_id in session_ids:
                refresh_counts = self._refresh_session_snapshot(corpus_id=corpus_id, title=title, session_id=session_id)
                for kind, count in refresh_counts.items():
                    snapshot_counts[kind] += int(count)
            subjects = self.store.list_subjects(corpus_id)
            for subject in subjects:
                refresh_counts = self._refresh_entity_snapshot(corpus_id=corpus_id, title=title, subject=subject)
                for kind, count in refresh_counts.items():
                    snapshot_counts[kind] += int(count)
            root_counts = self._refresh_root_snapshot(corpus_id=corpus_id, title=title)
            for kind, count in root_counts.items():
                snapshot_counts[kind] += int(count)
            self.store.commit()
            result = dict(sorted(snapshot_counts.items()))
            for kind, count in sorted(removed_counts.items()):
                result[f"{kind}_removed"] = int(count)
            result["sessions_refreshed"] = len(session_ids)
            result["subjects_refreshed"] = len(subjects)
            return result
        except Exception:
            self.store.rollback()
            raise

    def backfill_entity_facets(
        self,
        corpus_id: str,
        title: str,
    ) -> dict[str, int]:
        try:
            snapshot_counts: dict[str, int] = defaultdict(int)
            subjects = self.store.list_subjects(corpus_id)
            for subject in subjects:
                refresh_counts = self._refresh_entity_snapshot(
                    corpus_id=corpus_id,
                    title=title,
                    subject=subject,
                    build_entity_facets=True,
                )
                for kind, count in refresh_counts.items():
                    snapshot_counts[kind] += int(count)
            self.store.commit()
            result = dict(sorted(snapshot_counts.items()))
            result["subjects_refreshed"] = len(subjects)
            return result
        except Exception:
            self.store.rollback()
            raise

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

    def _build_entity_bridge_snapshots(
        self,
        *,
        corpus_id: str,
        title: str,
        entity: str,
        events: list[MemoryEventRecord],
        mode: str = "hybrid",
    ) -> list[MemorySnapshotRecord]:
        ordered_events = [event for event in events if str(event.text or "").strip()]
        if len({str(event.session_id) for event in ordered_events}) < BRIDGE_MIN_SESSION_COUNT:
            return []
        entity_tokens = _token_set(entity)
        event_infos: list[dict[str, Any]] = []
        for event in ordered_events:
            info = self._bridge_event_info(event=event, entity=entity, entity_tokens=entity_tokens)
            if not info["terms"] and event.embedding is None:
                continue
            event_infos.append(info)
        if len(event_infos) < BRIDGE_MIN_EVENT_COUNT:
            return []
        bridge_stats = self._build_bridge_token_stats(event_infos) if mode == "graph_lexical" else {}
        if bridge_stats:
            for info in event_infos:
                self._finalize_bridge_event_info(info, bridge_stats)
        if mode == "embedding_cluster":
            clusters = self._cluster_entity_event_infos_embedding(event_infos)
        elif mode == "graph_lexical":
            clusters = self._cluster_entity_event_infos_graph_lexical(event_infos, bridge_stats=bridge_stats)
        else:
            clusters = self._cluster_entity_event_infos_hybrid(event_infos)

        bridge_snapshots: list[MemorySnapshotRecord] = []
        for cluster in clusters:
            cluster_infos = list(cluster["infos"])
            session_ids = sorted(str(item["event"].session_id) for item in cluster_infos)
            if len(set(session_ids)) < BRIDGE_MIN_SESSION_COUNT or len(cluster_infos) < BRIDGE_MIN_EVENT_COUNT:
                continue
            aspect_terms = self._bridge_aspect_terms(cluster)
            if not aspect_terms:
                continue
            representative_infos = self._select_bridge_representatives(cluster_infos)
            if len(representative_infos) < BRIDGE_MIN_EVENT_COUNT:
                continue
            aspect_key = "|".join(aspect_terms[:3])
            summary_inputs = [
                f"{info['event'].speaker}: {info['support_text']}"
                for info in representative_infos
            ]
            refs = merge_memory_refs([entity] + aspect_terms + summary_inputs, limit=16)
            tags = build_text_tags([entity] + aspect_terms + summary_inputs, max_tags=5)
            selected_events = [info["event"] for info in representative_infos]
            bridge_snapshots.append(
                MemorySnapshotRecord(
                    snapshot_id=stable_id("leaf_snap", corpus_id, "bridge", entity, aspect_key),
                    corpus_id=corpus_id,
                    parent_id=stable_id("leaf_snap", corpus_id, "root"),
                    snapshot_kind="bridge",
                    scope_id=f"{entity}:bridge:{'_'.join(aspect_terms[:3])}",
                    title=f"{title}:{entity}:bridge-{'-'.join(aspect_terms[:2])}",
                    synopsis=make_synopsis(summary_inputs[:4], max_chars=180),
                    summary=summarize_texts(summary_inputs, max_chars=760),
                    object_ids=[],
                    event_ids=[event.event_id for event in selected_events[:BRIDGE_MAX_EVENTS]],
                    raw_refs=[event.raw_span_id for event in selected_events[:BRIDGE_MAX_EVENTS] if event.raw_span_id],
                    child_ids=[event.event_id for event in selected_events[:BRIDGE_MAX_EVENTS]],
                    entity_refs=refs,
                    canonical_entity_refs=canonicalize_entities([entity] + refs, limit=16),
                    time_range=self._time_range(sorted(selected_events, key=lambda item: (str(item.session_id), int(item.turn_index)))),
                    metadata={
                        "semantic_role": "entity_bridge",
                        "entity": entity,
                        "bridge_mode": mode,
                        "aspect_terms": aspect_terms,
                        "num_events": len(cluster_infos),
                        "num_sessions": len(set(session_ids)),
                        "session_ids": _ordered_unique_strings(session_ids),
                        "tags": tags,
                    },
                    embedding=self._embed_snapshot(f"{title}:{entity}:bridge", summarize_texts(summary_inputs, max_chars=760), tags),
                )
            )
        bridge_snapshots.sort(
            key=lambda snapshot: (
                -int((snapshot.metadata or {}).get("num_sessions") or 0),
                -int((snapshot.metadata or {}).get("num_events") or 0),
                str(snapshot.scope_id),
            )
        )
        return bridge_snapshots[:BRIDGE_MAX_PER_ENTITY]

    def _cluster_entity_event_infos_hybrid(self, event_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for info in event_infos:
            best_cluster: dict[str, Any] | None = None
            best_score = 0.0
            for cluster in clusters:
                cluster_score = self._bridge_cluster_match_score(cluster=cluster, info=info)
                if cluster_score > best_score:
                    best_score = cluster_score
                    best_cluster = cluster
            if best_cluster is None or best_score < BRIDGE_HYBRID_CLUSTER_THRESHOLD:
                clusters.append(self._new_bridge_cluster(info))
                continue
            self._merge_bridge_cluster_info(best_cluster, info)
        return clusters

    def _cluster_entity_event_infos_embedding(self, event_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked_infos = sorted(
            event_infos,
            key=lambda item: (
                item["event"].embedding is None,
                -len(item.get("semantic_refs") or []),
                -len(item.get("terms") or []),
                str(item["event"].session_id),
                int(item["event"].turn_index),
            ),
        )
        clusters: list[dict[str, Any]] = []
        for info in ranked_infos:
            best_cluster: dict[str, Any] | None = None
            best_score = 0.0
            for cluster in clusters:
                cluster_score = self._bridge_cluster_embedding_match_score(cluster=cluster, info=info)
                if cluster_score > best_score:
                    best_score = cluster_score
                    best_cluster = cluster
            if best_cluster is None or best_score < BRIDGE_EMBED_CLUSTER_THRESHOLD:
                clusters.append(self._new_bridge_cluster(info))
                continue
            self._merge_bridge_cluster_info(best_cluster, info)
        return clusters

    def _cluster_entity_event_infos_graph_lexical(
        self,
        event_infos: list[dict[str, Any]],
        *,
        bridge_stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if len(event_infos) <= 1:
            return [self._new_bridge_cluster(info) for info in event_infos]
        adjacency: dict[int, set[int]] = {index: set() for index in range(len(event_infos))}
        for left_index, left_info in enumerate(event_infos):
            for right_index in range(left_index + 1, len(event_infos)):
                right_info = event_infos[right_index]
                if not self._bridge_graph_has_edge(left_info=left_info, right_info=right_info, bridge_stats=bridge_stats):
                    continue
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

        visited: set[int] = set()
        clusters: list[dict[str, Any]] = []
        for index, info in enumerate(event_infos):
            if index in visited:
                continue
            stack = [index]
            component_indices: list[int] = []
            visited.add(index)
            while stack:
                current = stack.pop()
                component_indices.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
            component_infos = [event_infos[item] for item in component_indices]
            clusters.append(self._merge_bridge_cluster_infos(component_infos))
        return clusters

    @staticmethod
    def _new_bridge_cluster(info: dict[str, Any]) -> dict[str, Any]:
        return {
            "infos": [info],
            "sessions": {str(info["event"].session_id)},
            "term_counter": Counter(info["terms"]),
            "rep_terms": list(info["terms"]),
            "semantic_refs": set(info["semantic_refs"]),
            "embeddings": [info["event"].embedding] if info["event"].embedding else [],
        }

    @staticmethod
    def _merge_bridge_cluster_info(cluster: dict[str, Any], info: dict[str, Any]) -> None:
        cluster["infos"].append(info)
        cluster["sessions"].add(str(info["event"].session_id))
        cluster["term_counter"].update(info["terms"])
        for term in info["terms"]:
            if term not in cluster["rep_terms"]:
                cluster["rep_terms"].append(term)
        cluster["semantic_refs"].update(info["semantic_refs"])
        if info["event"].embedding:
            cluster["embeddings"].append(info["event"].embedding)

    @classmethod
    def _merge_bridge_cluster_infos(cls, infos: list[dict[str, Any]]) -> dict[str, Any]:
        base = cls._new_bridge_cluster(infos[0])
        for info in infos[1:]:
            cls._merge_bridge_cluster_info(base, info)
        return base

    def _build_bridge_token_stats(self, event_infos: list[dict[str, Any]]) -> dict[str, Any]:
        event_count = len(event_infos)
        token_df: Counter[str] = Counter()
        phrase_df: Counter[str] = Counter()
        for info in event_infos:
            token_df.update(set(info.get("graph_tokens") or []))
            phrase_df.update(set(info.get("graph_phrases") or []))
        token_idf = {
            token: math.log((event_count + 1.0) / (df + 1.0)) + 1.0
            for token, df in token_df.items()
        }
        phrase_idf = {
            phrase: math.log((event_count + 1.0) / (df + 1.0)) + 1.0
            for phrase, df in phrase_df.items()
        }
        token_df_ratio = {
            token: df / max(1, event_count)
            for token, df in token_df.items()
        }
        return {
            "token_df": token_df,
            "phrase_df": phrase_df,
            "token_idf": token_idf,
            "phrase_idf": phrase_idf,
            "token_df_ratio": token_df_ratio,
            "event_count": event_count,
        }

    def _finalize_bridge_event_info(self, info: dict[str, Any], bridge_stats: dict[str, Any]) -> None:
        token_idf = dict(bridge_stats.get("token_idf") or {})
        token_df_ratio = dict(bridge_stats.get("token_df_ratio") or {})
        selected_tokens = sorted(
            (
                token
                for token in (info.get("graph_tokens") or [])
                if token not in BRIDGE_GENERIC_SUPPORT_TOKENS and token_df_ratio.get(token, 0.0) < BRIDGE_HUB_TOKEN_RATIO
            ),
            key=lambda token: (-float(token_idf.get(token, 0.0)), token),
        )[:8]
        info["terms"] = _ordered_unique_strings(list(info.get("semantic_refs") or set()) + selected_tokens)
        info["term_set"] = set(info["terms"])

    def _bridge_event_info(
        self,
        *,
        event: MemoryEventRecord,
        entity: str,
        entity_tokens: set[str],
    ) -> dict[str, Any]:
        metadata = dict(event.metadata or {})
        semantic_refs = [str(item).strip().lower() for item in (metadata.get("semantic_refs") or []) if str(item).strip()]
        support_parts = [
            str(event.text or "").strip(),
            str(metadata.get("blip_caption") or "").strip(),
            " ".join(semantic_refs),
        ]
        support_text = " ".join(part for part in support_parts if part).strip()
        graph_tokens = [
            token
            for token in _bridge_support_tokens(support_text)
            if token not in entity_tokens
        ]
        graph_phrases = _bridge_support_phrases(graph_tokens)
        terms = _ordered_unique_strings(semantic_refs + graph_tokens)[:8]
        return {
            "event": event,
            "support_text": support_text or str(event.text or "").strip(),
            "terms": terms,
            "term_set": set(terms),
            "semantic_refs": set(semantic_refs),
            "graph_tokens": graph_tokens,
            "graph_phrases": graph_phrases,
        }

    def _bridge_cluster_match_score(self, *, cluster: dict[str, Any], info: dict[str, Any]) -> float:
        cluster_terms = set(cluster.get("rep_terms") or [])
        info_terms = set(info.get("terms") or [])
        lexical_score = 0.0
        if cluster_terms and info_terms:
            lexical_score = len(cluster_terms.intersection(info_terms)) / max(1, len(cluster_terms.union(info_terms)))
        semantic_score = 0.0
        if cluster.get("semantic_refs") and info.get("semantic_refs"):
            semantic_score = 1.0 if set(cluster["semantic_refs"]).intersection(info["semantic_refs"]) else 0.0
        embedding_score = 0.0
        event_embedding = info["event"].embedding
        if event_embedding and cluster.get("embeddings"):
            embedding_score = max(
                self._cosine_similarity_safe(event_embedding, candidate)
                for candidate in cluster["embeddings"]
                if candidate
            )
        return max(
            lexical_score + semantic_score * 0.2,
            lexical_score * 0.7 + embedding_score * 0.5,
            semantic_score * 0.35 + embedding_score * 0.45,
        )

    def _bridge_cluster_embedding_match_score(self, *, cluster: dict[str, Any], info: dict[str, Any]) -> float:
        base_score = self._bridge_cluster_match_score(cluster=cluster, info=info)
        event_embedding = info["event"].embedding
        if not event_embedding or not cluster.get("embeddings"):
            return base_score * 0.9
        embedding_score = max(
            self._cosine_similarity_safe(event_embedding, candidate)
            for candidate in cluster["embeddings"]
            if candidate
        )
        semantic_overlap = 0.0
        if cluster.get("semantic_refs") and info.get("semantic_refs"):
            semantic_overlap = 1.0 if set(cluster["semantic_refs"]).intersection(info["semantic_refs"]) else 0.0
        lexical_overlap = 0.0
        cluster_terms = set(cluster.get("rep_terms") or [])
        info_terms = set(info.get("terms") or [])
        if cluster_terms and info_terms:
            lexical_overlap = len(cluster_terms.intersection(info_terms)) / max(1, len(cluster_terms.union(info_terms)))
        return max(
            embedding_score + semantic_overlap * 0.08,
            embedding_score * 0.82 + lexical_overlap * 0.18,
            base_score,
        )

    def _bridge_graph_has_edge(
        self,
        *,
        left_info: dict[str, Any],
        right_info: dict[str, Any],
        bridge_stats: dict[str, Any],
    ) -> bool:
        token_idf = dict(bridge_stats.get("token_idf") or {})
        phrase_idf = dict(bridge_stats.get("phrase_idf") or {})
        token_df_ratio = dict(bridge_stats.get("token_df_ratio") or {})

        left_tokens = set(left_info.get("graph_tokens") or [])
        right_tokens = set(right_info.get("graph_tokens") or [])
        shared_tokens = left_tokens.intersection(right_tokens)
        union_tokens = left_tokens.union(right_tokens)
        shared_non_generic = {
            token
            for token in shared_tokens
            if token not in BRIDGE_GENERIC_SUPPORT_TOKENS
        }
        union_non_generic = {
            token
            for token in union_tokens
            if token not in BRIDGE_GENERIC_SUPPORT_TOKENS
        }
        weighted_shared = sum(float(token_idf.get(token, 1.0)) for token in shared_non_generic)
        weighted_union = sum(float(token_idf.get(token, 1.0)) for token in union_non_generic)
        lexical_score = weighted_shared / max(1.0, weighted_union)

        left_phrases = set(left_info.get("graph_phrases") or [])
        right_phrases = set(right_info.get("graph_phrases") or [])
        shared_phrases = left_phrases.intersection(right_phrases)
        strong_phrase_hits = {
            phrase for phrase in shared_phrases
            if float(phrase_idf.get(phrase, 0.0)) >= BRIDGE_STRONG_ANCHOR_PHRASE_IDF
        }

        shared_semantic_refs = set(left_info.get("semantic_refs") or set()).intersection(right_info.get("semantic_refs") or set())
        strong_token_hits = {
            token
            for token in shared_non_generic
            if float(token_idf.get(token, 0.0)) >= BRIDGE_STRONG_ANCHOR_TOKEN_IDF
        }
        strong_anchor = bool(shared_semantic_refs or strong_token_hits or strong_phrase_hits)
        if not strong_anchor:
            return False

        generic_shared_count = len(
            shared_tokens.intersection(BRIDGE_GENERIC_SUPPORT_TOKENS)
        )
        hub_shared_count = len(
            [
                token
                for token in shared_tokens
                if float(token_df_ratio.get(token, 0.0)) >= BRIDGE_HUB_TOKEN_RATIO
            ]
        )
        phrase_bonus = min(
            0.28,
            len(strong_phrase_hits) * 0.14
            + sum(max(0.0, float(phrase_idf.get(phrase, 1.0)) - 2.0) for phrase in strong_phrase_hits) * 0.03,
        )
        token_bonus = min(
            0.22,
            sum(max(0.0, float(token_idf.get(token, 1.0)) - 1.6) for token in strong_token_hits) * 0.06,
        )
        semantic_bonus = 0.12 if shared_semantic_refs else 0.0
        left_embedding = left_info["event"].embedding
        right_embedding = right_info["event"].embedding
        embedding_overlap = self._cosine_similarity_safe(left_embedding, right_embedding)
        embedding_bonus = 0.06 if embedding_overlap >= BRIDGE_GRAPH_EMBED_THRESHOLD and (strong_token_hits or shared_semantic_refs) else 0.0
        edge_score = lexical_score + phrase_bonus + token_bonus + semantic_bonus + embedding_bonus
        edge_score -= generic_shared_count * 0.1
        edge_score -= hub_shared_count * 0.08
        return edge_score >= BRIDGE_GRAPH_EDGE_SCORE_THRESHOLD

    @staticmethod
    def _cosine_similarity_safe(left: list[float] | None, right: list[float] | None) -> float:
        if not left or not right:
            return 0.0
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
        return dot / (left_norm * right_norm)

    @staticmethod
    def _bridge_aspect_terms(cluster: dict[str, Any]) -> list[str]:
        ordered_terms: list[str] = []
        semantic_refs = [str(item).strip() for item in sorted(cluster.get("semantic_refs") or []) if str(item).strip()]
        for ref in semantic_refs:
            if ref not in ordered_terms:
                ordered_terms.append(ref)
        term_counter = Counter(cluster.get("term_counter") or {})
        for term, _count in term_counter.most_common(8):
            clean = str(term or "").strip()
            if clean and clean not in ordered_terms:
                ordered_terms.append(clean)
        return ordered_terms[:4]

    @staticmethod
    def _select_bridge_representatives(cluster_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked_infos = sorted(
            cluster_infos,
            key=lambda item: (
                -len(item.get("semantic_refs") or []),
                -len(item.get("terms") or []),
                str(item["event"].session_id),
                int(item["event"].turn_index),
            ),
        )
        selected: list[dict[str, Any]] = []
        covered_sessions: set[str] = set()
        for info in ranked_infos:
            session_id = str(info["event"].session_id)
            if session_id in covered_sessions:
                continue
            selected.append(info)
            covered_sessions.add(session_id)
            if len(selected) >= BRIDGE_MAX_EVENTS:
                return selected
        for info in ranked_infos:
            if info in selected:
                continue
            selected.append(info)
            if len(selected) >= BRIDGE_MAX_EVENTS:
                break
        return selected

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
        cache_key = self._reconcile_cache_key(
            candidate=candidate,
            current=current,
            current_version=current_version,
        )
        cached_action = self._load_cached_reconcile_action(cache_key=cache_key)
        if cached_action is not None:
            return cached_action
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
        prompt_tokens = estimate_message_tokens(messages)
        try:
            response = self.reconciliation_llm.chat(messages, max_tokens=120, temperature=0.0)
            provider_prompt_tokens = extract_prompt_tokens(response)
            if provider_prompt_tokens is not None:
                prompt_tokens = provider_prompt_tokens
            payload = extract_json_object(extract_chat_text(response))
        except (OpenAICompatError, ValueError):
            return None
        self._record_reconcile_prompt_tokens(
            prompt_tokens=prompt_tokens,
            from_provider_usage=provider_prompt_tokens is not None,
        )
        action = str(payload.get("action") or "").strip().upper()
        if action in {"NONE", "PATCH", "SUPERSEDE", "TENTATIVE"}:
            self._store_cached_reconcile_action(cache_key=cache_key, action=action)
            return action
        return None

    def _record_reconcile_prompt_tokens(self, *, prompt_tokens: int, from_provider_usage: bool) -> None:
        if not isinstance(getattr(self, "_ingest_runtime_metrics", None), dict):
            return
        self._ingest_runtime_metrics["reconcile_prompt_tokens_total"] = int(
            self._ingest_runtime_metrics.get("reconcile_prompt_tokens_total", 0)
        ) + int(prompt_tokens)
        key = (
            "reconcile_prompt_tokens_provider_usage_calls"
            if from_provider_usage
            else "reconcile_prompt_tokens_estimated_calls"
        )
        self._ingest_runtime_metrics[key] = int(self._ingest_runtime_metrics.get(key, 0)) + 1

    def _resolve_reconcile_cache_dir(self) -> Path | None:
        if os.environ.get("LEAF_DISABLE_RECONCILE_CACHE", "").strip().lower() in {"1", "true", "yes"}:
            return None
        raw_dir = os.environ.get("LEAF_RECONCILE_CACHE_DIR", "").strip()
        if not raw_dir:
            raw_dir = os.path.expanduser("~/.cache/leaf/reconciliation")
        try:
            path = Path(raw_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            return None

    def _reconcile_cache_key(
        self,
        *,
        candidate: StateCandidate,
        current: MemoryObjectRecord,
        current_version: MemoryObjectVersionRecord,
    ) -> str:
        model_name = str(getattr(getattr(self.reconciliation_llm, "config", None), "model_name", "") or "")
        base_url = str(getattr(getattr(self.reconciliation_llm, "config", None), "base_url", "") or "")
        prompt_revision = "reconcile_prompt_v1"
        digest = hashlib.sha1(
            "||".join(
                [
                    prompt_revision,
                    model_name,
                    base_url,
                    str(current.subject or ""),
                    str(current.slot or ""),
                    str(current.policy or ""),
                    str(current.memory_kind or ""),
                    str(current_version.value or ""),
                    str(current_version.status or ""),
                    str(candidate.value or ""),
                    str(candidate.status or ""),
                    str(candidate.memory_kind or ""),
                ]
            ).encode("utf-8")
        ).hexdigest()
        return digest

    def _reconcile_cache_file_path(self, cache_key: str) -> Path | None:
        if self._reconcile_cache_dir is None:
            return None
        return self._reconcile_cache_dir / f"{cache_key}.json"

    def _load_cached_reconcile_action(self, *, cache_key: str) -> str | None:
        cache_path = self._reconcile_cache_file_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        action = str((payload or {}).get("action") or "").strip().upper()
        if action in {"NONE", "PATCH", "SUPERSEDE", "TENTATIVE"}:
            return action
        return None

    def _store_cached_reconcile_action(self, *, cache_key: str, action: str) -> None:
        cache_path = self._reconcile_cache_file_path(cache_key)
        if cache_path is None:
            return
        payload = {"action": str(action).strip().upper()}
        temp_path = cache_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(cache_path)
        except OSError:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

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
