from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.agentic_memory import stable_hash, utc_now_iso  # noqa: E402
from leaf.clients import ChatClient, EmbeddingClient, OpenAICompatError, extract_json_object  # noqa: E402
from leaf.config import load_config  # noqa: E402
from leaf.extract import extract_entities, extract_semantic_references  # noqa: E402
from leaf.normalize import language_aware_stemmed_content_terms  # noqa: E402
from leaf.records import AdditiveMemoryRecord  # noqa: E402
from leaf.store import SQLiteMemoryStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ADD-only rich memory sidecar from existing LEAF events/atoms.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus-id", action="append", default=[])
    parser.add_argument("--all-corpora", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["event_atom", "dialog_window", "llm_facts", "both"],
        default="both",
        help="Sidecar memory family to build. llm_facts extracts compact ADD-only facts from local dialogue windows.",
    )
    parser.add_argument("--max-atoms-per-event", type=int, default=6)
    parser.add_argument("--window-radius", type=int, default=1)
    parser.add_argument("--max-window-chars", type=int, default=900)
    parser.add_argument("--llm-window-size", type=int, default=8)
    parser.add_argument("--llm-max-facts-per-window", type=int, default=6)
    parser.add_argument("--llm-limit-windows", type=int, default=0)
    parser.add_argument(
        "--llm-source",
        default="llm_fact_v1",
        help="Metadata/source label for LLM-extracted ADD-only memories.",
    )
    parser.add_argument(
        "--llm-extraction-policy",
        choices=["compact_v1", "mem0_additive_v2"],
        default="compact_v1",
        help="Prompt policy used by --mode llm_facts.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=1,
        help="Parallel LLM extraction workers. DB writes remain serialized.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=180,
        help="Per-window LLM request timeout in seconds.",
    )
    parser.add_argument(
        "--flush-every-windows",
        type=int,
        default=10,
        help="For llm_facts mode, write accumulated sidecar rows every N processed windows.",
    )
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    store = SQLiteMemoryStore(args.db)
    embedding = EmbeddingClient(config.embedding) if args.embed and config.embedding.base_url else None
    memory_llm = ChatClient(config.memory_llm) if args.mode == "llm_facts" and config.memory_llm and config.memory_llm.base_url else None
    if args.mode == "llm_facts" and memory_llm is None:
        raise SystemExit("llm_facts mode requires memory_llm/additional_llm in config.")
    try:
        corpus_ids = list(dict.fromkeys(str(item).strip() for item in args.corpus_id if str(item).strip()))
        if args.all_corpora:
            corpus_ids = store.list_corpora()
        if not corpus_ids:
            raise SystemExit("No corpus selected. Use --corpus-id or --all-corpora.")

        results: list[dict[str, Any]] = []
        for corpus_id in corpus_ids:
            result = build_for_corpus(
                store,
                corpus_id=corpus_id,
                language_mode=config.language.mode,
                mode=str(args.mode),
                max_atoms_per_event=max(1, int(args.max_atoms_per_event)),
                window_radius=max(1, int(args.window_radius)),
                max_window_chars=max(240, int(args.max_window_chars)),
                llm_window_size=max(1, int(args.llm_window_size)),
                llm_max_facts_per_window=max(1, int(args.llm_max_facts_per_window)),
                llm_limit_windows=max(0, int(args.llm_limit_windows)),
                llm_source=str(args.llm_source or "llm_fact_v1").strip() or "llm_fact_v1",
                llm_extraction_policy=str(args.llm_extraction_policy or "compact_v1"),
                llm_workers=max(1, int(args.llm_workers)),
                llm_timeout=max(10, int(args.llm_timeout)),
                flush_every_windows=max(1, int(args.flush_every_windows)),
                memory_llm=memory_llm,
                embedding=embedding,
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        store.commit()
    finally:
        store.close()

    payload = {"created_at": utc_now_iso(), "db": str(args.db), "results": results}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_for_corpus(
    store: SQLiteMemoryStore,
    *,
    corpus_id: str,
    language_mode: str,
    mode: str,
    max_atoms_per_event: int,
    window_radius: int,
    max_window_chars: int,
    llm_window_size: int,
    llm_max_facts_per_window: int,
    llm_limit_windows: int,
    llm_source: str,
    llm_extraction_policy: str,
    llm_workers: int,
    llm_timeout: int,
    flush_every_windows: int,
    memory_llm: ChatClient | None,
    embedding: EmbeddingClient | None,
) -> dict[str, Any]:
    events = store.get_events(corpus_id=corpus_id)
    atoms = store.list_atoms(corpus_id=corpus_id)
    atoms_by_event: dict[str, list[Any]] = defaultdict(list)
    for atom in atoms:
        atoms_by_event[str(atom.event_id)].append(atom)

    existing_hashes = {
        str(memory.hash)
        for memory in store.list_additive_memories(corpus_id)
        if str(memory.hash or "").strip()
    }
    records: list[AdditiveMemoryRecord] = []
    now = utc_now_iso()
    total_records_written = 0

    def flush_records(*, force: bool = False) -> int:
        nonlocal total_records_written
        if not records:
            return 0
        count = len(records)
        store.upsert_additive_memories(records)
        store.commit()
        records.clear()
        total_records_written += count
        return count

    def append_record(
        *,
        source: str,
        event: Any,
        text: str,
        linked_atoms: list[Any],
        metadata: dict[str, Any],
        canonical_entity_seed: list[str] | None = None,
    ) -> bool:
        if not text:
            return False
        memory_hash = hashlib.md5(f"{source}\n{corpus_id}\n{event.event_id}\n{text}".encode("utf-8")).hexdigest()
        if memory_hash in existing_hashes:
            return False
        existing_hashes.add(memory_hash)
        entities = extract_entities(text, mode=language_mode)
        semantic_refs = extract_semantic_references(text, mode=language_mode)
        canonical_entities = list(
            dict.fromkeys([*entities, *semantic_refs, *(canonical_entity_seed or []), *event.canonical_entity_refs])
        )
        terms = list(language_aware_stemmed_content_terms(text, mode=language_mode, include_cjk_subgrams=True))
        memory_id = f"addmem_{stable_hash(corpus_id, event.event_id, source, memory_hash, length=24)}"
        vector = embedding.embed(text) if embedding is not None else None
        records.append(
            AdditiveMemoryRecord(
                memory_id=memory_id,
                corpus_id=corpus_id,
                event_id=event.event_id,
                text=text,
                attributed_to=str(event.speaker or "memory"),
                timestamp=event.timestamp,
                linked_atom_ids=[atom.atom_id for atom in linked_atoms],
                linked_memory_ids=[],
                entities=entities,
                canonical_entities=canonical_entities,
                terms=terms,
                metadata=metadata,
                embedding=vector,
                hash=memory_hash,
                created_at=now,
            )
        )
        return True

    event_atom_records_before = len(records)
    if mode in {"event_atom", "both"}:
        for event in events:
            event_atoms = atoms_by_event.get(str(event.event_id), [])
            text = build_memory_text(event, event_atoms[:max_atoms_per_event])
            append_record(
                source="event_atom_derived_v1",
                event=event,
                text=text,
                linked_atoms=event_atoms[:max_atoms_per_event],
                canonical_entity_seed=[],
                metadata={
                    "source": "event_atom_derived_v1",
                    "session_id": event.session_id,
                    "turn_index": event.turn_index,
                    "speaker": event.speaker,
                    "atom_count": len(event_atoms),
                },
            )
    event_atom_records_written = len(records) - event_atom_records_before

    dialog_records_before = len(records)
    session_events: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        session_events[str(event.session_id)].append(event)

    if mode in {"dialog_window", "both"}:
        for session_id, session_rows in session_events.items():
            ordered_events = sorted(session_rows, key=lambda item: int(getattr(item, "turn_index", 0) or 0))
            for index, event in enumerate(ordered_events):
                left = max(0, index - window_radius)
                right = min(len(ordered_events), index + window_radius + 1)
                window_events = ordered_events[left:right]
                linked_atoms: list[Any] = []
                for window_event in window_events:
                    linked_atoms.extend(atoms_by_event.get(str(window_event.event_id), [])[:max_atoms_per_event])
                text = build_dialog_window_text(
                    anchor_event=event,
                    window_events=window_events,
                    atoms_by_event=atoms_by_event,
                    max_atoms_per_event=max_atoms_per_event,
                    max_window_chars=max_window_chars,
                )
                canonical_seed: list[str] = []
                for window_event in window_events:
                    canonical_seed.extend(list(getattr(window_event, "canonical_entity_refs", []) or []))
                    canonical_seed.extend(list(getattr(window_event, "entity_refs", []) or []))
                append_record(
                    source="dialog_window_v1",
                    event=event,
                    text=text,
                    linked_atoms=linked_atoms,
                    canonical_entity_seed=canonical_seed,
                    metadata={
                        "source": "dialog_window_v1",
                        "session_id": session_id,
                        "turn_index": event.turn_index,
                        "speaker": event.speaker,
                        "window_radius": window_radius,
                        "window_event_ids": [window_event.event_id for window_event in window_events],
                        "window_turn_indices": [
                            int(getattr(window_event, "turn_index", 0) or 0) for window_event in window_events
                        ],
                        "window_speakers": [
                            str(getattr(window_event, "speaker", "") or "") for window_event in window_events
                        ],
                        "atom_count": len(linked_atoms),
                    },
                )
    dialog_records_written = len(records) - dialog_records_before

    llm_fact_records_written = 0
    llm_fact_windows_processed = 0
    llm_fact_errors = 0
    if mode == "llm_facts":
        windows: list[tuple[int, str, list[Any]]] = []
        for session_id, session_rows in session_events.items():
            ordered_events = sorted(session_rows, key=lambda item: int(getattr(item, "turn_index", 0) or 0))
            for start in range(0, len(ordered_events), llm_window_size):
                window_events = ordered_events[start : start + llm_window_size]
                if window_events:
                    windows.append((len(windows), session_id, window_events))
                    if llm_limit_windows and len(windows) >= llm_limit_windows:
                        break
            if llm_limit_windows and len(windows) >= llm_limit_windows:
                break

        def run_window(window: tuple[int, str, list[Any]]) -> tuple[int, str, list[Any], dict[str, Any] | None, str | None]:
            window_index, session_id, window_events = window
            try:
                payload = extract_llm_facts(
                    memory_llm,
                    window_events=window_events,
                    atoms_by_event=atoms_by_event,
                    max_atoms_per_event=max_atoms_per_event,
                    max_facts=llm_max_facts_per_window,
                    extraction_policy=llm_extraction_policy,
                    timeout=llm_timeout,
                )
                return window_index, session_id, window_events, payload, None
            except (OpenAICompatError, ValueError, TypeError, json.JSONDecodeError) as exc:
                return window_index, session_id, window_events, None, str(exc)[:240]

        def consume_window_result(
            result: tuple[int, str, list[Any], dict[str, Any] | None, str | None],
        ) -> None:
            nonlocal llm_fact_errors, llm_fact_records_written, llm_fact_windows_processed
            _window_index, session_id, window_events, fact_payload, error_text = result
            if error_text is not None:
                llm_fact_errors += 1
                print(
                    json.dumps(
                        {
                            "corpus_id": corpus_id,
                            "mode": "llm_facts",
                            "progress": "window_error",
                            "windows_processed": llm_fact_windows_processed,
                            "llm_fact_errors": llm_fact_errors,
                            "llm_source": llm_source,
                            "error": error_text,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return
            llm_fact_windows_processed += 1
            facts = fact_payload.get("facts") if isinstance(fact_payload, dict) else []
            if not isinstance(facts, list):
                return
            event_by_id = {str(event.event_id): event for event in window_events}
            window_event_ids = [event.event_id for event in window_events]
            for fact_index, fact in enumerate(facts[:llm_max_facts_per_window]):
                if not isinstance(fact, dict):
                    continue
                text = " ".join(str(fact.get("text") or "").split())
                if not text:
                    continue
                linked_event_ids = [
                    str(item).strip()
                    for item in (fact.get("event_ids") or [])
                    if str(item).strip() in event_by_id
                ]
                anchor_event = event_by_id.get(linked_event_ids[0]) if linked_event_ids else window_events[0]
                linked_atoms: list[Any] = []
                canonical_seed: list[str] = []
                for linked_event_id in linked_event_ids or [anchor_event.event_id]:
                    linked_event = event_by_id.get(str(linked_event_id))
                    if linked_event is None:
                        continue
                    linked_atoms.extend(atoms_by_event.get(str(linked_event.event_id), [])[:max_atoms_per_event])
                    canonical_seed.extend(list(getattr(linked_event, "canonical_entity_refs", []) or []))
                    canonical_seed.extend(list(getattr(linked_event, "entity_refs", []) or []))
                if append_record(
                    source=llm_source,
                    event=anchor_event,
                    text=text,
                    linked_atoms=linked_atoms,
                    canonical_entity_seed=canonical_seed,
                    metadata={
                        "source": llm_source,
                        "extraction_policy": llm_extraction_policy,
                        "session_id": session_id,
                        "window_start_turn": int(getattr(window_events[0], "turn_index", 0) or 0),
                        "window_end_turn": int(getattr(window_events[-1], "turn_index", 0) or 0),
                        "window_event_ids": window_event_ids,
                        "linked_event_ids": linked_event_ids,
                        "fact_index": fact_index,
                        "kind": str(fact.get("kind") or "").strip().lower(),
                        "attributed_to": str(fact.get("attributed_to") or "").strip(),
                        "confidence": fact.get("confidence"),
                    },
                ):
                    llm_fact_records_written += 1
            if llm_fact_windows_processed % flush_every_windows == 0:
                flushed = flush_records(force=True)
                print(
                    json.dumps(
                        {
                            "corpus_id": corpus_id,
                            "mode": "llm_facts",
                            "progress": "window_flush",
                            "windows_processed": llm_fact_windows_processed,
                            "records_flushed": flushed,
                            "llm_fact_errors": llm_fact_errors,
                            "llm_source": llm_source,
                            "llm_extraction_policy": llm_extraction_policy,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if int(llm_workers) <= 1:
            for window in windows:
                consume_window_result(run_window(window))
        else:
            with ThreadPoolExecutor(max_workers=int(llm_workers)) as executor:
                future_by_index = {executor.submit(run_window, window): window[0] for window in windows}
                for future in as_completed(future_by_index):
                    consume_window_result(future.result())

    remaining_flushed = flush_records(force=True)
    if mode == "llm_facts" and remaining_flushed:
        print(
            json.dumps(
                {
                    "corpus_id": corpus_id,
                    "mode": "llm_facts",
                    "progress": "final_flush",
                    "windows_processed": llm_fact_windows_processed,
                    "records_flushed": remaining_flushed,
                    "llm_fact_errors": llm_fact_errors,
                    "llm_source": llm_source,
                    "llm_extraction_policy": llm_extraction_policy,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {
        "corpus_id": corpus_id,
        "event_count": len(events),
        "atom_count": len(atoms),
        "additive_memories_written": total_records_written,
        "event_atom_memories_written": event_atom_records_written,
        "dialog_window_memories_written": dialog_records_written,
        "llm_fact_memories_written": llm_fact_records_written,
        "llm_fact_windows_processed": llm_fact_windows_processed,
        "llm_fact_errors": llm_fact_errors,
        "llm_source": llm_source,
        "llm_extraction_policy": llm_extraction_policy,
        "llm_workers": llm_workers,
        "llm_timeout": llm_timeout,
        "existing_hash_count": len(existing_hashes),
        "mode": mode,
        "window_radius": window_radius,
        "max_window_chars": max_window_chars,
    }


def extract_llm_facts(
    memory_llm: ChatClient | None,
    *,
    window_events: list[Any],
    atoms_by_event: dict[str, list[Any]],
    max_atoms_per_event: int,
    max_facts: int,
    extraction_policy: str = "compact_v1",
    timeout: int = 180,
) -> dict[str, Any]:
    if memory_llm is None:
        return {"facts": []}
    context_lines: list[str] = []
    for event in window_events:
        event_id = str(getattr(event, "event_id", "") or "")
        speaker = " ".join(str(getattr(event, "speaker", "") or "speaker").split())
        timestamp = str(getattr(event, "timestamp", "") or "")
        text = " ".join(str(getattr(event, "text", "") or "").split())
        if not text:
            continue
        context_lines.append(f"[{event_id}] {timestamp} {speaker}: {text}")
        detail_texts: list[str] = []
        for atom in atoms_by_event.get(event_id, [])[:max_atoms_per_event]:
            content = " ".join(str(getattr(atom, "content", "") or "").split())
            if content:
                detail_texts.append(content)
        if detail_texts:
            context_lines.append(f"  existing_atom_details: {'; '.join(detail_texts[:max_atoms_per_event])}")
    if not context_lines:
        return {"facts": []}
    system, user = build_llm_fact_prompt(
        context_lines=context_lines,
        max_facts=max_facts,
        extraction_policy=extraction_policy,
    )
    response = memory_llm.text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=1200,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    payload = extract_json_object(response)
    if not isinstance(payload.get("facts"), list):
        payload["facts"] = []
    return payload


def build_llm_fact_prompt(
    *,
    context_lines: list[str],
    max_facts: int,
    extraction_policy: str,
) -> tuple[str, str]:
    schema = (
        "Schema: {\"facts\":[{\"text\":\"...\",\"event_ids\":[\"leaf_evt_...\"],"
        "\"attributed_to\":\"speaker or entity\","
        "\"kind\":\"preference|plan|relationship|biography|location|work|hobby|event|temporal|other\","
        "\"confidence\":0.0-1.0}]}"
    )
    if extraction_policy == "mem0_additive_v2":
        system = (
            "You are a precise memory extractor. Your only operation is ADD: extract self-contained, "
            "evidence-bound long-term memories from dialogue. Return JSON only. Do not answer questions. "
            "Do not invent facts."
        )
        user = "\n".join(
            [
                "Extract ADD-only memory facts from this dialogue window.",
                "Capture every memorable piece of information, including secondary details that are not the dominant topic.",
                "Extract separate memories for distinct topics, entities, events, preferences, plans, relationships, locations, work, hobbies, and dated changes.",
                "A known entity appearing again does not make the new fact redundant; new events or attributes about that entity should still be extracted.",
                "Resolve relative time references using the event timestamps shown in the window whenever possible, and preserve explicit dates.",
                "Keep each fact contextually rich but concise: include the subject, relation/action, object/location/date, and relevant change context.",
                "Skip greetings, filler, vague encouragement, and assistant meta-commentary unless the assistant speaker is a real remembered subject.",
                "Use the same language/script as the input dialogue when the dialogue is not English.",
                f"Return at most {int(max_facts)} facts.",
                schema,
                "",
                "Dialogue window:",
                "\n".join(context_lines),
            ]
        )
        return system, user

    system = (
        "You extract compact long-term memories from dialogue for retrieval. "
        "Return JSON only. Do not answer questions. Do not invent facts."
    )
    user = "\n".join(
        [
            "Extract ADD-only memory facts from this dialogue window.",
            "Keep each fact self-contained, concise, and useful for later retrieval.",
            "Focus on personal details, preferences, plans, relationships, locations, work, hobbies, activities, and dated events.",
            "Ignore greetings, generic encouragement, and assistant-only opinions unless the assistant is the remembered subject.",
            f"Return at most {int(max_facts)} facts.",
            schema,
            "",
            "Dialogue window:",
            "\n".join(context_lines),
        ]
    )
    return system, user


def build_memory_text(event: Any, atoms: list[Any]) -> str:
    event_text = " ".join(str(event.text or "").split())
    if not event_text:
        return ""
    atom_texts: list[str] = []
    seen: set[str] = set()
    for atom in atoms:
        content = " ".join(str(atom.content or "").split())
        if not content or content.lower() in seen:
            continue
        seen.add(content.lower())
        atom_texts.append(content)
    prefix = f"{event.speaker} said"
    if event.timestamp:
        prefix += f" on {event.timestamp}"
    if atom_texts:
        return f"{prefix}: {event_text} Key memory details: {'; '.join(atom_texts[:6])}."
    return f"{prefix}: {event_text}"


def build_dialog_window_text(
    *,
    anchor_event: Any,
    window_events: list[Any],
    atoms_by_event: dict[str, list[Any]],
    max_atoms_per_event: int,
    max_window_chars: int,
) -> str:
    turns: list[str] = []
    for event in window_events:
        speaker = " ".join(str(getattr(event, "speaker", "") or "speaker").split())
        text = " ".join(str(getattr(event, "text", "") or "").split())
        if not text:
            continue
        turns.append(f"{speaker}: {text}")
    if not turns:
        return ""

    anchor_turn = int(getattr(anchor_event, "turn_index", 0) or 0)
    prefix = f"Conversation window around turn {anchor_turn}"
    if getattr(anchor_event, "timestamp", None):
        prefix += f" on {getattr(anchor_event, 'timestamp')}"
    dialog = " | ".join(turns)

    detail_texts: list[str] = []
    seen: set[str] = set()
    for event in window_events:
        event_id = str(getattr(event, "event_id", ""))
        for atom in atoms_by_event.get(event_id, [])[:max_atoms_per_event]:
            content = " ".join(str(getattr(atom, "content", "") or "").split())
            key = content.lower()
            if not content or key in seen:
                continue
            seen.add(key)
            detail_texts.append(content)
    suffix = f" Key memory details: {'; '.join(detail_texts[:8])}." if detail_texts else ""
    text = f"{prefix}: {dialog}.{suffix}"
    if len(text) <= max_window_chars:
        return text

    allowed_dialog_chars = max(120, max_window_chars - len(prefix) - len(suffix) - 4)
    truncated_dialog = dialog[:allowed_dialog_chars].rsplit(" ", 1)[0].strip() or dialog[:allowed_dialog_chars].strip()
    return f"{prefix}: {truncated_dialog}.{suffix}"[:max_window_chars].rstrip()


if __name__ == "__main__":
    main()
