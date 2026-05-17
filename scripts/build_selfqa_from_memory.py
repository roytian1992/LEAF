from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf.agentic_memory import stable_hash, utc_now_iso
from leaf.clients import ChatClient, extract_json_object
from leaf.config import load_config
from leaf.store import SQLiteMemoryStore

CRITERIA_VERSION = "criteria_v1"
TEMPORAL_GRANULARITIES = {"none", "date", "month", "year", "relative", "ordering"}
CRITERIA_FAILURE_MODES = {"unknown", "miss_bridge", "wrong_time", "wrong_entity", "overbroad_topic", "wrong_topic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a frozen evidence-grounded self-QA snapshot from an existing LEAF memory DB.",
    )
    parser.add_argument("--config", required=True, help="LEAF config. Uses memory_llm when present.")
    parser.add_argument("--db", required=True, help="Path to LEAF SQLite DB.")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--summary-output", default="", help="Optional summary JSON path.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum accepted self-QA tasks.")
    parser.add_argument("--candidate-limit", type=int, default=80, help="Maximum candidate evidence groups to try.")
    parser.add_argument(
        "--per-task-type-limit",
        type=int,
        default=0,
        help="Optional accepted-task cap per task type. When set, generation continues until all caps are met or candidates run out.",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=["single_fact", "multi_hop", "temporal"],
        choices=["single_fact", "multi_hop", "temporal"],
    )
    parser.add_argument("--active-view-id", default="", help="Defaults to the active memory view for corpus.")
    parser.add_argument("--write-db", action="store_true", help="Also write tasks into leaf_selfqa_tasks.")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Append accepted tasks to the JSONL output as they are accepted instead of writing only at the end.",
    )
    parser.add_argument("--validate", action="store_true", help="Use the configured memory LLM to validate generated tasks.")
    parser.add_argument("--min-validation-score", type=float, default=0.75)
    parser.add_argument(
        "--candidate-sampling",
        choices=["round_robin_by_task_type", "stratified"],
        default="round_robin_by_task_type",
        help="round_robin_by_task_type preserves the original deterministic order; stratified spreads across topics and turns.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used by stratified candidate sampling.")
    parser.add_argument(
        "--stratified-turn-bucket-size",
        type=int,
        default=40,
        help="Turn-index bucket size used by stratified sampling.",
    )
    return parser.parse_args()


def format_evidence(evidence_items: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "\n".join(
                [
                    f"Evidence {index}:",
                    f"- event_id: {item['event_id']}",
                    f"- atom_id: {item['atom_id']}",
                    f"- timestamp: {item.get('timestamp') or ''}",
                    f"- speaker: {item.get('speaker') or ''}",
                    f"- atom: {item.get('atom_content') or ''}",
                    f"- raw text: {item.get('event_text') or ''}",
                ]
            )
            for index, item in enumerate(evidence_items, start=1)
        ]
    )


def make_messages(task_type: str, evidence_items: list[dict[str, Any]]) -> list[dict[str, str]]:
    evidence_text = format_evidence(evidence_items)
    type_instruction = {
        "single_fact": "Create a direct factual question that can be answered from the evidence.",
        "multi_hop": "Create a multi-hop question that requires combining all provided evidence items.",
        "temporal": "Create a temporal question that relies on the timestamp or ordering in the evidence.",
    }[task_type]
    return [
        {
            "role": "system",
            "content": (
                "You generate benchmark questions for a memory retrieval system. "
                "Return only one valid JSON object. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{type_instruction}\n\n"
                "Rules:\n"
                "1. Use only the evidence below.\n"
                "2. The answer must be short and directly supported.\n"
                "3. Do not reveal event_id or atom_id in the question or answer.\n"
                "4. Set gold_evidence_path to the exact event_id/atom_id pairs needed.\n"
                "5. If the task is multi_hop, every evidence item must be necessary.\n\n"
                "Return this JSON shape:\n"
                "{\n"
                '  "question": "...",\n'
                '  "answer": "...",\n'
                '  "gold_evidence_path": [\n'
                '    {"event_id": "...", "atom_id": "...", "role": "anchor"}\n'
                "  ],\n"
                '  "criteria_v1": {\n'
                '    "answer_criteria": {\n'
                '      "must_contain": ["short answer phrase or required item"],\n'
                '      "acceptable_aliases": ["optional equivalent answer"],\n'
                '      "temporal_granularity": "none|date|month|year|relative|ordering",\n'
                '      "wrong_if_contains": ["optional clearly wrong phrase"]\n'
                "    },\n"
                '    "evidence_criteria": {\n'
                '      "required_event_ids": ["..."],\n'
                '      "required_atom_ids": ["..."],\n'
                '      "all_required": true,\n'
                '      "evidence_roles": [{"event_id": "...", "atom_id": "...", "role": "anchor|bridge|temporal_anchor|support"}]\n'
                "    },\n"
                '    "retrieval_criteria": {\n'
                '      "success_at_k": 8,\n'
                '      "must_retrieve_any_event_ids": ["..."],\n'
                '      "must_retrieve_all_event_ids": ["..."],\n'
                '      "must_retrieve_any_atom_ids": ["..."],\n'
                '      "must_retrieve_all_atom_ids": ["..."]\n'
                "    },\n"
                '    "topic_criteria": {\n'
                '      "expected_topic_ids": ["optional topic_id from the evidence if present"],\n'
                '      "topic_should_help": true,\n'
                '      "failure_mode_if_missing": "miss_bridge|wrong_time|wrong_entity|overbroad_topic|wrong_topic|unknown"\n'
                "    }\n"
                "  },\n"
                '  "tags": ["single_fact|multi_hop|temporal"],\n'
                '  "rationale": "short reason why the evidence supports the answer"\n'
                "}\n\n"
                f"Evidence:\n{evidence_text}"
            ),
        },
    ]


def make_validation_messages(task: dict[str, Any], evidence_items: list[dict[str, Any]]) -> list[dict[str, str]]:
    task_type = str(task.get("metadata", {}).get("task_type") or "")
    gold_atom_ids = [str(item.get("atom_id") or "") for item in task.get("gold_evidence_path") or []]
    criteria = (task.get("metadata") or {}).get(CRITERIA_VERSION) or {}
    return [
        {
            "role": "system",
            "content": (
                "You are a strict benchmark validator for a memory retrieval task. "
                "Return only one valid JSON object. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Validate whether this generated memory QA item is usable as an evidence-grounded retrieval benchmark.\n\n"
                "Rules:\n"
                "1. The answer must be fully supported by the evidence.\n"
                "2. The question and answer must not reveal event_id or atom_id.\n"
                "3. The gold_evidence_path must point only to evidence that is actually needed.\n"
                "4. For multi_hop, at least two gold evidence atoms must be independently necessary; if one can be removed "
                "without making the answer unsupported, mark it invalid.\n"
                "5. For temporal, the timestamp or ordering must matter.\n\n"
                "6. The criteria_v1 object must be strict enough to judge retrieval success and answer correctness, "
                "but not broader than the actual gold evidence.\n\n"
                "Return this JSON shape:\n"
                "{\n"
                '  "valid": true,\n'
                '  "score": 0.0,\n'
                '  "supported": true,\n'
                '  "answerable": true,\n'
                '  "no_id_leak": true,\n'
                '  "all_gold_evidence_necessary": true,\n'
                '  "temporal_requirement_met": true,\n'
                '  "criteria_valid": true,\n'
                '  "criteria_complete": true,\n'
                '  "necessary_atom_ids": ["..."],\n'
                '  "issues": ["short issue if invalid"]\n'
                "}\n\n"
                f"Task type: {task_type}\n"
                f"Question: {task.get('question')}\n"
                f"Answer: {task.get('answer')}\n"
                f"Gold atom ids: {json.dumps(gold_atom_ids, ensure_ascii=False)}\n"
                f"Gold evidence path: {json.dumps(task.get('gold_evidence_path') or [], ensure_ascii=False)}\n\n"
                f"Criteria v1: {json.dumps(criteria, ensure_ascii=False)}\n\n"
                f"Evidence:\n{format_evidence(evidence_items)}"
            ),
        },
    ]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _string_list(values: Any, *, limit: int = 12) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []
    output: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _id_list(values: Any, allowed: set[str], *, limit: int = 32) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    raw_values = values if isinstance(values, list) else []
    for value in raw_values:
        item = str(value or "").strip()
        if not item or item not in allowed or item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def infer_temporal_granularity(task_type: str, answer: str) -> str:
    if task_type != "temporal":
        return "none"
    lowered = str(answer or "").strip().lower()
    if any(token in lowered for token in ["ago", "before", "after", "week", "weekend", "yesterday", "today"]):
        return "relative"
    if any(month in lowered for month in (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )):
        if any(char.isdigit() for char in lowered):
            return "date" if any(part.isdigit() and len(part) <= 2 for part in lowered.replace(",", " ").split()) else "month"
        return "month"
    if any(part.isdigit() and len(part) == 4 for part in lowered.replace(",", " ").split()):
        return "year"
    return "ordering"


def default_answer_must_contain(answer: str) -> list[str]:
    text = str(answer or "").strip()
    if not text:
        return []
    separators = [",", ";"]
    items = [text]
    for separator in separators:
        if separator in text:
            items = [part.strip() for part in text.split(separator) if part.strip()]
            break
    if len(items) == 1 and " and " in text.lower() and len(text.split()) <= 10:
        items = [part.strip() for part in re_split_and(text) if part.strip()]
    if len(items) <= 5 and all(len(item.split()) <= 8 for item in items):
        return items
    return [text] if len(text.split()) <= 12 else []


def re_split_and(text: str) -> list[str]:
    lowered = str(text or "")
    parts: list[str] = []
    start = 0
    for marker in [" and ", " And "]:
        if marker in lowered:
            return [part.strip() for part in lowered.split(marker)]
    parts.append(lowered[start:].strip())
    return parts


def normalize_criteria_v1(
    raw_criteria: Any,
    *,
    task_type: str,
    answer: str,
    gold_path: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    criteria = dict(raw_criteria or {}) if isinstance(raw_criteria, dict) else {}
    answer_criteria = dict(criteria.get("answer_criteria") or {})
    evidence_criteria = dict(criteria.get("evidence_criteria") or {})
    retrieval_criteria = dict(criteria.get("retrieval_criteria") or {})
    topic_criteria = dict(criteria.get("topic_criteria") or {})

    allowed_event_ids = {str(item.get("event_id") or "").strip() for item in evidence_items if str(item.get("event_id") or "").strip()}
    allowed_atom_ids = {str(item.get("atom_id") or "").strip() for item in evidence_items if str(item.get("atom_id") or "").strip()}
    gold_event_ids = [str(item.get("event_id") or "").strip() for item in gold_path if str(item.get("event_id") or "").strip()]
    gold_atom_ids = [str(item.get("atom_id") or "").strip() for item in gold_path if str(item.get("atom_id") or "").strip()]
    topic_ids = [
        str(item.get("topic_id") or "").strip()
        for item in evidence_items
        if str(item.get("topic_id") or "").strip()
    ]
    allowed_topic_ids = set(topic_ids)

    temporal_granularity = str(
        answer_criteria.get("temporal_granularity")
        or infer_temporal_granularity(task_type, answer)
    ).strip().lower()
    if temporal_granularity not in TEMPORAL_GRANULARITIES:
        temporal_granularity = infer_temporal_granularity(task_type, answer)

    evidence_roles: list[dict[str, str]] = []
    raw_roles = evidence_criteria.get("evidence_roles")
    if not isinstance(raw_roles, list):
        raw_roles = gold_path
    for item in raw_roles:
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id") or "").strip()
        atom_id = str(item.get("atom_id") or "").strip()
        if event_id not in allowed_event_ids or atom_id not in allowed_atom_ids:
            continue
        role = str(item.get("role") or "support").strip() or "support"
        if task_type == "temporal" and role == "anchor":
            role = "temporal_anchor"
        evidence_roles.append({"event_id": event_id, "atom_id": atom_id, "role": role})

    required_event_ids = _id_list(
        evidence_criteria.get("required_event_ids") or gold_event_ids,
        allowed_event_ids,
    )
    required_atom_ids = _id_list(
        evidence_criteria.get("required_atom_ids") or gold_atom_ids,
        allowed_atom_ids,
    )
    expected_topic_ids = _id_list(
        topic_criteria.get("expected_topic_ids") or topic_ids,
        allowed_topic_ids,
    )
    failure_mode = str(topic_criteria.get("failure_mode_if_missing") or "unknown").strip()
    if failure_mode not in CRITERIA_FAILURE_MODES:
        failure_mode = "unknown"
    raw_must_contain = _string_list(answer_criteria.get("must_contain"))
    answer_text = normalized_text(answer)
    must_contain = [
        phrase
        for phrase in raw_must_contain
        if normalized_text(phrase) and normalized_text(phrase) in answer_text
    ]
    if not must_contain:
        must_contain = default_answer_must_contain(answer)

    return {
        "version": CRITERIA_VERSION,
        "answer_criteria": {
            "must_contain": must_contain,
            "acceptable_aliases": _string_list(answer_criteria.get("acceptable_aliases")),
            "temporal_granularity": temporal_granularity,
            "wrong_if_contains": _string_list(answer_criteria.get("wrong_if_contains")),
        },
        "evidence_criteria": {
            "required_event_ids": required_event_ids,
            "required_atom_ids": required_atom_ids,
            "all_required": _bool(evidence_criteria.get("all_required", True)),
            "evidence_roles": evidence_roles,
        },
        "retrieval_criteria": {
            "success_at_k": max(1, int(retrieval_criteria.get("success_at_k") or 8)),
            "must_retrieve_any_event_ids": _id_list(
                retrieval_criteria.get("must_retrieve_any_event_ids") or required_event_ids[:1],
                allowed_event_ids,
            ),
            "must_retrieve_all_event_ids": _id_list(
                retrieval_criteria.get("must_retrieve_all_event_ids") or required_event_ids,
                allowed_event_ids,
            ),
            "must_retrieve_any_atom_ids": _id_list(
                retrieval_criteria.get("must_retrieve_any_atom_ids") or required_atom_ids[:1],
                allowed_atom_ids,
            ),
            "must_retrieve_all_atom_ids": _id_list(
                retrieval_criteria.get("must_retrieve_all_atom_ids") or required_atom_ids,
                allowed_atom_ids,
            ),
        },
        "topic_criteria": {
            "expected_topic_ids": expected_topic_ids,
            "topic_should_help": _bool(topic_criteria.get("topic_should_help", bool(expected_topic_ids))),
            "failure_mode_if_missing": failure_mode,
        },
    }


def normalized_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def validate_criteria_v1_locally(
    criteria: dict[str, Any],
    *,
    task_type: str,
    answer: str,
    gold_path: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[str] = []
    if criteria.get("version") != CRITERIA_VERSION:
        issues.append("missing_or_wrong_criteria_version")
    evidence_criteria = dict(criteria.get("evidence_criteria") or {})
    answer_criteria = dict(criteria.get("answer_criteria") or {})
    required_atom_ids = {str(item) for item in evidence_criteria.get("required_atom_ids") or []}
    required_event_ids = {str(item) for item in evidence_criteria.get("required_event_ids") or []}
    gold_atom_ids = {str(item.get("atom_id") or "") for item in gold_path if str(item.get("atom_id") or "")}
    gold_event_ids = {str(item.get("event_id") or "") for item in gold_path if str(item.get("event_id") or "")}
    if not required_atom_ids:
        issues.append("criteria_missing_required_atom_ids")
    if not required_event_ids:
        issues.append("criteria_missing_required_event_ids")
    if not required_atom_ids.issubset(gold_atom_ids):
        issues.append("criteria_required_atoms_not_in_gold_path")
    if not required_event_ids.issubset(gold_event_ids):
        issues.append("criteria_required_events_not_in_gold_path")
    if task_type == "multi_hop" and len(required_atom_ids) < 2:
        issues.append("multi_hop_criteria_requires_too_few_atoms")
    temporal_granularity = str(answer_criteria.get("temporal_granularity") or "none")
    if temporal_granularity not in TEMPORAL_GRANULARITIES:
        issues.append("invalid_temporal_granularity")
    if task_type == "temporal" and temporal_granularity == "none":
        issues.append("temporal_criteria_missing_granularity")
    answer_text = normalized_text(answer)
    for phrase in _string_list(answer_criteria.get("must_contain")):
        if normalized_text(phrase) and normalized_text(phrase) not in answer_text:
            issues.append("answer_must_contain_not_in_gold_answer")
            break
    return {"valid": not issues, "issues": issues}


def validate_task(
    llm: ChatClient,
    task: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    *,
    min_score: float,
) -> tuple[bool, dict[str, Any]]:
    text = llm.text(make_validation_messages(task, evidence_items), max_tokens=512, temperature=0.0)
    payload = extract_json_object(text)
    task_type = str(task.get("metadata", {}).get("task_type") or "")
    gold_atom_ids = {str(item.get("atom_id") or "").strip() for item in task.get("gold_evidence_path") or []}
    necessary_atom_ids = {
        str(atom_id).strip()
        for atom_id in (payload.get("necessary_atom_ids") or [])
        if str(atom_id).strip()
    }
    try:
        score = float(payload.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    validation = {
        "valid": _bool(payload.get("valid")),
        "score": score,
        "supported": _bool(payload.get("supported")),
        "answerable": _bool(payload.get("answerable")),
        "no_id_leak": _bool(payload.get("no_id_leak")),
        "all_gold_evidence_necessary": _bool(payload.get("all_gold_evidence_necessary")),
        "temporal_requirement_met": _bool(payload.get("temporal_requirement_met")),
        "criteria_valid": _bool(payload.get("criteria_valid", True)),
        "criteria_complete": _bool(payload.get("criteria_complete", True)),
        "necessary_atom_ids": sorted(necessary_atom_ids),
        "issues": [str(issue).strip() for issue in (payload.get("issues") or []) if str(issue).strip()],
    }
    local_criteria_validation = validate_criteria_v1_locally(
        dict((task.get("metadata") or {}).get(CRITERIA_VERSION) or {}),
        task_type=task_type,
        answer=str(task.get("answer") or ""),
        gold_path=list(task.get("gold_evidence_path") or []),
    )
    validation["criteria_local_validation"] = local_criteria_validation
    if not local_criteria_validation["valid"]:
        validation["issues"].extend(local_criteria_validation["issues"])
    passed = (
        validation["valid"]
        and validation["score"] >= min_score
        and validation["supported"]
        and validation["answerable"]
        and validation["no_id_leak"]
        and validation["criteria_valid"]
        and validation["criteria_complete"]
        and local_criteria_validation["valid"]
    )
    if task_type == "multi_hop":
        passed = (
            passed
            and validation["all_gold_evidence_necessary"]
            and len(gold_atom_ids) >= 2
            and (not necessary_atom_ids or gold_atom_ids.issubset(necessary_atom_ids))
        )
    if task_type == "temporal":
        passed = passed and validation["temporal_requirement_met"]
    return passed, validation


def normalize_task(
    payload: dict[str, Any],
    *,
    corpus_id: str,
    view_id: str | None,
    task_type: str,
    allowed_pairs: set[tuple[str, str]],
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    question = str(payload.get("question") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    if not question or not answer:
        return None
    raw_path = payload.get("gold_evidence_path") or []
    if not isinstance(raw_path, list):
        return None
    path: list[dict[str, Any]] = []
    for item in raw_path:
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id") or "").strip()
        atom_id = str(item.get("atom_id") or "").strip()
        if (event_id, atom_id) not in allowed_pairs:
            continue
        path.append(
            {
                "event_id": event_id,
                "atom_id": atom_id,
                "role": str(item.get("role") or "support").strip() or "support",
            }
        )
    if not path:
        return None
    if task_type == "multi_hop" and len({item["atom_id"] for item in path}) < 2:
        return None
    tags = [str(tag).strip() for tag in (payload.get("tags") or []) if str(tag).strip()]
    if task_type not in tags:
        tags.insert(0, task_type)
    criteria = normalize_criteria_v1(
        payload.get(CRITERIA_VERSION),
        task_type=task_type,
        answer=answer,
        gold_path=path,
        evidence_items=evidence_items,
    )
    local_criteria_validation = validate_criteria_v1_locally(
        criteria,
        task_type=task_type,
        answer=answer,
        gold_path=path,
    )
    if not local_criteria_validation["valid"]:
        return None
    created_at = utc_now_iso()
    task_id = f"selfqa_{stable_hash(corpus_id, view_id or '', task_type, question, answer, length=24)}"
    return {
        "task_id": task_id,
        "corpus_id": corpus_id,
        "view_id": view_id,
        "question": question,
        "answer": answer,
        "gold_evidence_path": path,
        "tags": list(dict.fromkeys(tags)),
        "status": "accepted",
        "metadata": {
            "generator": "build_selfqa_from_memory.py",
            "task_type": task_type,
            CRITERIA_VERSION: criteria,
            "criteria_local_validation": local_criteria_validation,
            "rationale": str(payload.get("rationale") or "").strip(),
            "source_evidence": evidence_items,
        },
        "created_at": created_at,
    }


def atom_evidence_item(atom: Any, event: Any, topic_id: str | None = None) -> dict[str, Any]:
    return {
        "event_id": str(atom.event_id),
        "atom_id": str(atom.atom_id),
        "topic_id": topic_id,
        "timestamp": event.timestamp,
        "turn_index": event.turn_index,
        "speaker": event.speaker,
        "atom_type": atom.atom_type,
        "atom_content": atom.content,
        "entities": list(atom.canonical_entities or atom.entities or []),
        "event_text": event.text,
    }


def build_candidates(store: SQLiteMemoryStore, corpus_id: str, view_id: str | None, task_types: set[str]) -> list[dict[str, Any]]:
    atoms = store.list_atoms(corpus_id)
    events = store.get_events(corpus_id)
    event_by_id = {event.event_id: event for event in events}
    atom_by_id = {atom.atom_id: atom for atom in atoms}

    topic_by_atom: dict[str, str] = {}
    if view_id:
        for assignment in store.list_topic_assignments(view_id, item_kind="atom"):
            topic_by_atom[str(assignment["item_id"])] = str(assignment["topic_id"])

    candidates: list[dict[str, Any]] = []
    usable_items: list[dict[str, Any]] = []
    for atom in atoms:
        event = event_by_id.get(atom.event_id)
        if event is None:
            continue
        item = atom_evidence_item(atom, event, topic_by_atom.get(atom.atom_id))
        usable_items.append(item)
        if "single_fact" in task_types:
            candidates.append({"task_type": "single_fact", "evidence": [item]})
        if "temporal" in task_types and item.get("timestamp"):
            candidates.append({"task_type": "temporal", "evidence": [item]})

    if "multi_hop" in task_types:
        by_entity: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        by_topic: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in usable_items:
            for entity in item.get("entities") or []:
                entity_key = str(entity).strip().lower()
                if entity_key:
                    by_entity[entity_key].append(item)
            topic_id = str(item.get("topic_id") or "").strip()
            if topic_id:
                by_topic[topic_id].append(item)
        for bucket in list(by_entity.values()) + list(by_topic.values()):
            seen_pairs: set[tuple[str, str]] = set()
            for left in bucket:
                for right in bucket:
                    if left["atom_id"] >= right["atom_id"]:
                        continue
                    if left["event_id"] == right["event_id"]:
                        continue
                    key = (left["atom_id"], right["atom_id"])
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    candidates.append({"task_type": "multi_hop", "evidence": [left, right]})
                    break
                if len(seen_pairs) >= 2:
                    break

    candidates.sort(
        key=lambda row: (
            str(row["task_type"]),
            "|".join(str(item["atom_id"]) for item in row["evidence"]),
        )
    )
    return candidates


def sample_candidates_by_task_type(
    candidates: list[dict[str, Any]],
    task_types: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {task_type: [] for task_type in task_types}
    for candidate in candidates:
        task_type = str(candidate.get("task_type") or "")
        if task_type in grouped:
            grouped[task_type].append(candidate)

    sampled: list[dict[str, Any]] = []
    offsets = {task_type: 0 for task_type in task_types}
    while len(sampled) < limit:
        progressed = False
        for task_type in task_types:
            bucket = grouped[task_type]
            offset = offsets[task_type]
            if offset >= len(bucket):
                continue
            sampled.append(bucket[offset])
            offsets[task_type] = offset + 1
            progressed = True
            if len(sampled) >= limit:
                break
        if not progressed:
            break
    return sampled


def candidate_stratum(candidate: dict[str, Any], *, turn_bucket_size: int) -> str:
    evidence_items = [item for item in (candidate.get("evidence") or []) if isinstance(item, dict)]
    topic_ids = sorted({str(item.get("topic_id") or "").strip() for item in evidence_items if str(item.get("topic_id") or "").strip()})
    topic_key = "|".join(topic_ids[:2]) if topic_ids else "no_topic"
    turn_indexes: list[int] = []
    for item in evidence_items:
        try:
            turn_indexes.append(int(item.get("turn_index")))
        except (TypeError, ValueError):
            continue
    if turn_indexes:
        bucket_size = max(1, int(turn_bucket_size))
        turn_key = f"turn_{min(turn_indexes) // bucket_size}"
    else:
        turn_key = "turn_unknown"
    return f"{topic_key}::{turn_key}"


def sample_candidates_stratified(
    candidates: list[dict[str, Any]],
    task_types: list[str],
    limit: int,
    *,
    seed: int,
    turn_bucket_size: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rng = random.Random(int(seed))
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {task_type: defaultdict(list) for task_type in task_types}
    for candidate in candidates:
        task_type = str(candidate.get("task_type") or "")
        if task_type not in grouped:
            continue
        grouped[task_type][candidate_stratum(candidate, turn_bucket_size=turn_bucket_size)].append(candidate)

    for strata in grouped.values():
        for bucket in strata.values():
            rng.shuffle(bucket)

    sampled: list[dict[str, Any]] = []
    offsets: dict[tuple[str, str], int] = {}
    while len(sampled) < limit:
        progressed = False
        for task_type in task_types:
            strata_keys = list(grouped[task_type])
            rng.shuffle(strata_keys)
            for stratum in strata_keys:
                bucket = grouped[task_type][stratum]
                key = (task_type, stratum)
                offset = offsets.get(key, 0)
                if offset >= len(bucket):
                    continue
                sampled.append(bucket[offset])
                offsets[key] = offset + 1
                progressed = True
                if len(sampled) >= limit:
                    break
            if len(sampled) >= limit:
                break
        if not progressed:
            break
    return sampled


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    memory_cfg = config.memory_llm or config.llm
    if not memory_cfg.base_url:
        raise RuntimeError("No memory_llm or llm base_url configured.")
    llm = ChatClient(memory_cfg)
    store = SQLiteMemoryStore(args.db)
    try:
        active_view = store.get_memory_view(args.active_view_id) if args.active_view_id else store.get_active_memory_view(args.corpus_id)
        view_id = str(active_view["view_id"]) if active_view else None
        candidates = build_candidates(store, args.corpus_id, view_id, set(args.task_types))
        if args.candidate_sampling == "stratified":
            candidate_sample = sample_candidates_stratified(
                candidates,
                args.task_types,
                max(0, args.candidate_limit),
                seed=int(args.seed),
                turn_bucket_size=int(args.stratified_turn_bucket_size),
            )
        else:
            candidate_sample = sample_candidates_by_task_type(candidates, args.task_types, max(0, args.candidate_limit))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.stream_output:
            output_path.write_text("", encoding="utf-8")
        accepted: list[dict[str, Any]] = []
        accepted_ids: set[str] = set()
        errors: list[dict[str, str]] = []
        per_type_limit = max(0, int(args.per_task_type_limit))
        accepted_by_type: dict[str, int] = defaultdict(int)
        target_types = list(args.task_types)
        target_total = max(0, args.limit)
        if per_type_limit > 0:
            target_total = min(target_total or per_type_limit * len(target_types), per_type_limit * len(target_types))
        for candidate in candidate_sample:
            if len(accepted) >= target_total:
                break
            candidate_task_type = str(candidate["task_type"])
            if per_type_limit > 0 and accepted_by_type[candidate_task_type] >= per_type_limit:
                continue
            evidence_items = list(candidate["evidence"])
            allowed_pairs = {(str(item["event_id"]), str(item["atom_id"])) for item in evidence_items}
            messages = make_messages(candidate_task_type, evidence_items)
            try:
                text = llm.text(messages, max_tokens=512, temperature=0.0)
                payload = extract_json_object(text)
                task = normalize_task(
                    payload,
                    corpus_id=args.corpus_id,
                    view_id=view_id,
                    task_type=candidate_task_type,
                    allowed_pairs=allowed_pairs,
                    evidence_items=evidence_items,
                )
            except Exception as exc:
                errors.append({"task_type": candidate_task_type, "error": str(exc)[:300]})
                continue
            if task is None:
                errors.append({"task_type": candidate_task_type, "error": "invalid_or_unsupported_task"})
                continue
            if args.validate:
                try:
                    validation_passed, validation = validate_task(
                        llm,
                        task,
                        evidence_items,
                        min_score=float(args.min_validation_score),
                    )
                except Exception as exc:
                    errors.append({"task_type": candidate_task_type, "error": f"validation_error:{str(exc)[:260]}"})
                    continue
                task["metadata"]["validation"] = validation
                if not validation_passed:
                    issues = ", ".join(validation.get("issues") or []) or "validation_failed"
                    errors.append({"task_type": candidate_task_type, "error": f"validation_rejected:{issues[:260]}"})
                    continue
                task["status"] = "validated"
            if task["task_id"] in accepted_ids:
                continue
            accepted.append(task)
            accepted_ids.add(str(task["task_id"]))
            accepted_by_type[candidate_task_type] += 1
            if args.stream_output:
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")

        if not args.stream_output:
            with output_path.open("w", encoding="utf-8") as handle:
                for task in accepted:
                    handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
        if args.write_db:
            for task in accepted:
                store.add_selfqa_task(
                    task_id=task["task_id"],
                    corpus_id=task["corpus_id"],
                    view_id=task["view_id"],
                    question=task["question"],
                    answer=task["answer"],
                    gold_evidence_path=task["gold_evidence_path"],
                    tags=task["tags"],
                    status=task["status"],
                    metadata=task["metadata"],
                    created_at=task["created_at"],
                )
            store.commit()

        summary = {
            "corpus_id": args.corpus_id,
            "view_id": view_id,
            "output": str(output_path),
            "candidate_count": len(candidates),
            "candidate_limit": args.candidate_limit,
            "candidate_sample_count": len(candidate_sample),
            "candidate_sampling": args.candidate_sampling,
            "per_task_type_limit": per_type_limit,
            "seed": int(args.seed),
            "stratified_turn_bucket_size": int(args.stratified_turn_bucket_size),
            "validated": bool(args.validate),
            "min_validation_score": float(args.min_validation_score),
            "accepted_count": len(accepted),
            "error_count": len(errors),
            "task_type_counts": {
                task_type: sum(1 for item in accepted if item.get("metadata", {}).get("task_type") == task_type)
                for task_type in args.task_types
            },
            "tag_counts": {
                task_type: sum(1 for item in accepted if task_type in item.get("tags", []))
                for task_type in args.task_types
            },
            "errors_sample": errors[:5],
        }
        if args.summary_output:
            summary_path = Path(args.summary_output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
