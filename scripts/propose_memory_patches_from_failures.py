from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
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


PROPOSAL_VERSION = "memory_patch_proposal_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the configured memory LLM to propose guardable memory/topic patches "
            "from a failure log. This script only writes a proposal file; it does not "
            "mutate the active memory tree."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--failure-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-id", default="")
    parser.add_argument("--max-failures", type=int, default=80)
    parser.add_argument("--max-topics", type=int, default=30)
    parser.add_argument("--max-examples-per-mode", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--record-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return round(float(statistics.mean(values)), 4) if values else 0.0


def topic_slug_from_node(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    return str(
        metadata.get("topic_slug")
        or metadata.get("evolved_slug")
        or metadata.get("seed_slug")
        or node.get("name")
        or node.get("topic_id")
        or ""
    ).strip()


def topic_role_from_node(node: dict[str, Any]) -> str:
    metadata = dict(node.get("metadata") or {})
    if metadata.get("evolved_from") or metadata.get("evolved_slug"):
        return "evolved"
    if metadata.get("seed_slug") or metadata.get("seed_role"):
        return "seed"
    return str(metadata.get("topic_role") or "unknown")


def topic_digest_for_corpus(
    store: SQLiteMemoryStore,
    corpus_id: str,
    *,
    referenced_slugs: set[str],
    max_topics: int,
) -> dict[str, Any]:
    view = store.get_active_memory_view(corpus_id)
    if view is None:
        return {"corpus_id": corpus_id, "active_view": None, "topics": []}
    nodes = store.list_topic_nodes(str(view["view_id"]))
    assignment_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()
    for assignment in store.list_topic_assignments(str(view["view_id"])):
        topic_id = str(assignment.get("topic_id") or "")
        if not topic_id:
            continue
        if str(assignment.get("item_kind") or "") == "atom_secondary":
            secondary_counts[topic_id] += 1
        else:
            assignment_counts[topic_id] += 1

    rows: list[dict[str, Any]] = []
    for node in nodes:
        metadata = dict(node.get("metadata") or {})
        slug = topic_slug_from_node(node)
        topic_id = str(node.get("topic_id") or "")
        rows.append(
            {
                "topic_id": topic_id,
                "slug": slug,
                "name": node.get("name"),
                "parent_id": node.get("parent_id"),
                "level": int(node.get("level") or 0),
                "role": topic_role_from_node(node),
                "keywords": string_list(node.get("keywords"))[:20],
                "route_keywords": string_list(metadata.get("route_keywords"))[:20],
                "profile_terms": string_list(metadata.get("profile_terms"))[:20],
                "answer_exposure": metadata.get("answer_exposure"),
                "route_exposure": metadata.get("route_exposure"),
                "assignment_count": assignment_counts[topic_id],
                "secondary_assignment_count": secondary_counts[topic_id],
                "referenced_in_failures": slug in referenced_slugs,
            }
        )
    rows.sort(
        key=lambda item: (
            not bool(item.get("referenced_in_failures")),
            int(item.get("level") or 0),
            -int(item.get("assignment_count") or 0) - int(item.get("secondary_assignment_count") or 0),
            str(item.get("slug") or ""),
        )
    )
    return {
        "corpus_id": corpus_id,
        "active_view": {
            "view_id": view.get("view_id"),
            "name": view.get("name"),
            "status": view.get("status"),
            "metadata": {
                "view_type": (view.get("metadata") or {}).get("view_type"),
                "topic_model": (view.get("metadata") or {}).get("topic_model"),
                "trigger": (view.get("metadata") or {}).get("trigger"),
            },
        },
        "topics": rows[: max(1, int(max_topics))],
    }


def slugs_from_failure(row: dict[str, Any]) -> set[str]:
    slugs: set[str] = set()
    topic_soft = row.get("topic_soft") if isinstance(row.get("topic_soft"), dict) else {}
    for key in ("active_topic_slugs", "candidate_topic_slugs", "suppressed_topic_slugs"):
        slugs.update(string_list(topic_soft.get(key)))
    topic_shadow = row.get("topic_routing_shadow") if isinstance(row.get("topic_routing_shadow"), dict) else {}
    for key in (
        "routed_topic_slugs",
        "gold_topic_slugs",
        "retrieved_topic_slugs",
        "criteria_expected_topic_slugs",
    ):
        slugs.update(string_list(topic_shadow.get(key)))
    topic_context = row.get("topic_context") if isinstance(row.get("topic_context"), dict) else {}
    for ref in topic_context.get("active_topic_refs") or []:
        if isinstance(ref, dict):
            slug = str(ref.get("slug") or "").strip()
            if slug:
                slugs.add(slug)
    return slugs


def summarize_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: Counter[str] = Counter()
    by_corpus: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    severities: list[float] = []
    for row in rows:
        by_source[str(row.get("source_type") or "")] += 1
        by_corpus[str(row.get("corpus_id") or "")] += 1
        diagnostic_type = str(row.get("diagnostic_question_type") or row.get("diagnostic_task_type") or "")
        if diagnostic_type:
            by_type[diagnostic_type] += 1
        for mode in string_list(row.get("failure_modes")):
            by_mode[mode] += 1
        for slug in slugs_from_failure(row):
            topic_counts[slug] += 1
        severities.append(as_float(row.get("severity")))
    return {
        "failure_count": len(rows),
        "by_source_type": dict(by_source.most_common()),
        "by_corpus_id": dict(by_corpus.most_common()),
        "by_failure_mode": dict(by_mode.most_common()),
        "by_diagnostic_type": dict(by_type.most_common()),
        "top_topic_slugs": dict(topic_counts.most_common(30)),
        "severity_mean": mean(severities),
        "severity_max": round(max(severities), 4) if severities else 0.0,
    }


def compact_failure(row: dict[str, Any]) -> dict[str, Any]:
    topic_soft = row.get("topic_soft") if isinstance(row.get("topic_soft"), dict) else {}
    topic_shadow = row.get("topic_routing_shadow") if isinstance(row.get("topic_routing_shadow"), dict) else {}
    baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else None
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    return {
        "failure_id": row.get("failure_id"),
        "source_type": row.get("source_type"),
        "corpus_id": row.get("corpus_id"),
        "sample_id": row.get("sample_id"),
        "task_id": row.get("task_id"),
        "question_type": row.get("diagnostic_question_type") or row.get("diagnostic_task_type"),
        "failure_modes": row.get("failure_modes"),
        "severity": row.get("severity"),
        "question": str(row.get("question") or "")[:400],
        "gold_answer": str(row.get("gold_answer") or "")[:300],
        "predicted_answer": str(row.get("predicted_answer") or "")[:300],
        "answer_f1": row.get("answer_f1"),
        "f1_delta_vs_baseline": diagnostics.get("f1_delta_vs_baseline"),
        "baseline_f1": diagnostics.get("baseline_f1") if diagnostics else (baseline or {}).get("answer_f1"),
        "topic_soft": {
            "router": topic_soft.get("router"),
            "policy_reason": topic_soft.get("policy_reason"),
            "active_topic_slugs": string_list(topic_soft.get("active_topic_slugs")),
            "candidate_topic_slugs": string_list(topic_soft.get("candidate_topic_slugs")),
            "event_count": topic_soft.get("event_count"),
            "candidate_atom_count": topic_soft.get("candidate_atom_count"),
        },
        "selfqa_routing": {
            "routed_topic_slugs": string_list(topic_shadow.get("routed_topic_slugs")),
            "gold_topic_slugs": string_list(topic_shadow.get("gold_topic_slugs")),
            "criteria_expected_topic_slugs": string_list(topic_shadow.get("criteria_expected_topic_slugs")),
            "criteria_expected_topic_route_hit": topic_shadow.get("criteria_expected_topic_route_hit"),
            "criteria_expected_topic_retrieval_hit": topic_shadow.get("criteria_expected_topic_retrieval_hit"),
            "topic_path_hit": topic_shadow.get("topic_path_hit"),
        }
        if topic_shadow
        else None,
    }


def select_failures(rows: list[dict[str, Any]], *, max_failures: int, max_examples_per_mode: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (-as_float(row.get("severity")), str(row.get("failure_id") or "")))
    selected: list[dict[str, Any]] = []
    per_mode: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        modes = string_list(row.get("failure_modes")) or ["unknown"]
        if all(per_mode[mode] >= max_examples_per_mode for mode in modes):
            continue
        failure_id = str(row.get("failure_id") or "")
        if failure_id in seen:
            continue
        selected.append(row)
        seen.add(failure_id)
        for mode in modes:
            per_mode[mode] += 1
        if len(selected) >= max_failures:
            return selected
    for row in rows:
        failure_id = str(row.get("failure_id") or "")
        if failure_id in seen:
            continue
        selected.append(row)
        seen.add(failure_id)
        if len(selected) >= max_failures:
            break
    return selected


def build_prompt(
    *,
    failure_log_path: str,
    failures: list[dict[str, Any]],
    summary: dict[str, Any],
    topic_digests: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "proposal_version": PROPOSAL_VERSION,
        "source_failure_log": failure_log_path,
        "failure_summary": summary,
        "topic_digests": topic_digests,
        "failure_examples": [compact_failure(row) for row in failures],
        "allowed_patch_types": [
            "topic_split",
            "topic_merge",
            "topic_rename",
            "topic_route_keywords",
            "topic_profile_terms",
            "topic_exposure",
            "answer_exposure",
            "retrieval_policy",
            "selfqa_criteria_policy",
        ],
        "patch_schema": {
            "patch_id": "stable short id",
            "patch_type": "one allowed patch type",
            "scope": {
                "corpus_id": "target corpus id or '*' for global policy",
                "topic_slug": "optional existing topic slug",
                "topic_id": "optional existing topic id",
            },
            "change": {
                "add_route_keywords": ["optional"],
                "remove_route_keywords": ["optional"],
                "add_profile_terms": ["optional"],
                "route_exposure": "active|shadow|disabled optional",
                "answer_exposure": "primary|supplemental|hidden optional",
                "candidate_child_topics": [
                    {
                        "name": "optional split child name",
                        "description": "short",
                        "route_keywords": ["..."],
                        "profile_terms": ["..."],
                    }
                ],
                "retrieval_policy": {
                    "topic_route_top_k": "optional int",
                    "topic_soft_event_limit": "optional int",
                    "topic_soft_per_topic_atom_limit": "optional int",
                    "topic_answer_context_policy": "optional description",
                },
                "selfqa_criteria_policy": "optional description",
            },
            "evidence_failure_ids": ["failure ids supporting the patch"],
            "expected_effect": "why this should improve memory search or QA",
            "risk": "specific regression risk",
            "guard": {
                "selfqa_required": True,
                "locomo_guard_corpora": ["corpus ids to guard"],
                "reject_if": "measurable reject rule",
            },
        },
    }
    system = (
        "You are a memory-system evolution planner for an agentic memory tree. "
        "You propose small, guardable patches from failure logs. Return only one valid JSON object. "
        "Do not include markdown. Do not use benchmark question type labels as runtime conditions. "
        "Prefer memory modeling changes: topic split/merge/profile/route/exposure and self-QA criteria policy. "
        "Avoid hard-coded English-only lexical rules. If a patch is risky, keep it shadow-only and define a guard."
    )
    user = (
        "Propose candidate memory patches from the following failure log digest. "
        "The active memory system cannot use LoCoMo question-type labels at runtime; those labels are diagnostics only. "
        "Patches must be incremental and reversible. Do not mutate any database; only propose JSON.\n\n"
        "Return this exact top-level shape:\n"
        "{\n"
        '  "proposal_version": "memory_patch_proposal_v1",\n'
        '  "rationale": "short summary",\n'
        '  "patches": [PATCH_OBJECTS],\n'
        '  "experiment_plan": [\n'
        '    {"step": "...", "command_hint": "...", "success_metric": "..."}\n'
        "  ]\n"
        "}\n\n"
        f"Failure log digest:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_proposal(payload: dict[str, Any], *, valid_failure_ids: set[str]) -> dict[str, Any]:
    proposal = dict(payload)
    proposal["proposal_version"] = str(proposal.get("proposal_version") or PROPOSAL_VERSION)
    patches = proposal.get("patches")
    if not isinstance(patches, list):
        proposal["patches"] = []
        proposal["raw_invalid_patches"] = patches
    normalized: list[dict[str, Any]] = []
    for index, patch in enumerate(proposal.get("patches") or [], start=1):
        if not isinstance(patch, dict):
            continue
        current = dict(patch)
        current.setdefault("patch_id", f"patch_{index:03d}")
        current.setdefault("patch_type", "unknown")
        current.setdefault("scope", {})
        current.setdefault("change", {})
        current.setdefault("evidence_failure_ids", [])
        current.setdefault("expected_effect", "")
        current.setdefault("risk", "")
        current.setdefault("guard", {})
        evidence_ids = [
            str(item).strip()
            for item in (current.get("evidence_failure_ids") or [])
            if str(item).strip()
        ]
        invalid_ids = [failure_id for failure_id in evidence_ids if failure_id not in valid_failure_ids]
        current["evidence_failure_ids"] = [failure_id for failure_id in evidence_ids if failure_id in valid_failure_ids]
        if invalid_ids:
            current["invalid_evidence_failure_ids"] = invalid_ids
        normalized.append(current)
    proposal["patches"] = normalized
    experiment_plan = proposal.get("experiment_plan")
    if not isinstance(experiment_plan, list):
        proposal["experiment_plan"] = []
    return proposal


def record_evolution_run(
    *,
    db_path: str,
    proposal: dict[str, Any],
    corpus_ids: list[str],
    failure_log: str,
    output: str,
) -> list[str]:
    run_ids: list[str] = []
    store = SQLiteMemoryStore(db_path)
    try:
        now = utc_now_iso()
        for corpus_id in corpus_ids:
            view = store.get_active_memory_view(corpus_id)
            run_id = f"run_{stable_hash(corpus_id, failure_log, output, now, length=20)}"
            store.add_evolution_run(
                run_id=run_id,
                corpus_id=corpus_id,
                base_view_id=str((view or {}).get("view_id") or ""),
                candidate_view_id=None,
                trigger={
                    "kind": "failure_log_memory_patch_proposal",
                    "failure_log": failure_log,
                    "proposal_output": output,
                },
                status="proposed",
                result={
                    "proposal_version": proposal.get("proposal_version"),
                    "patch_count": len(proposal.get("patches") or []),
                    "patch_ids": [patch.get("patch_id") for patch in proposal.get("patches") or []],
                },
                created_at=now,
                completed_at=utc_now_iso(),
            )
            run_ids.append(run_id)
        store.commit()
    finally:
        store.close()
    return run_ids


def main() -> None:
    args = parse_args()
    all_failures = load_jsonl(args.failure_log)
    if args.corpus_id:
        all_failures = [row for row in all_failures if str(row.get("corpus_id") or "") == args.corpus_id]
    selected = select_failures(
        all_failures,
        max_failures=max(1, int(args.max_failures)),
        max_examples_per_mode=max(1, int(args.max_examples_per_mode)),
    )
    summary = summarize_failures(all_failures)
    corpus_ids = sorted({str(row.get("corpus_id") or "") for row in all_failures if str(row.get("corpus_id") or "")})
    referenced_by_corpus: dict[str, set[str]] = defaultdict(set)
    for row in all_failures:
        corpus_id = str(row.get("corpus_id") or "")
        if corpus_id:
            referenced_by_corpus[corpus_id].update(slugs_from_failure(row))

    store = SQLiteMemoryStore(args.db)
    try:
        topic_digests = [
            topic_digest_for_corpus(
                store,
                corpus_id,
                referenced_slugs=referenced_by_corpus.get(corpus_id, set()),
                max_topics=int(args.max_topics),
            )
            for corpus_id in corpus_ids
        ]
    finally:
        store.close()

    config = load_config(args.config)
    if config.memory_llm is None:
        raise SystemExit("config.memory_llm is required for proposal generation.")
    client = ChatClient(config.memory_llm)
    messages = build_prompt(
        failure_log_path=str(args.failure_log),
        failures=selected,
        summary=summary,
        topic_digests=topic_digests,
    )
    response_text = client.text(messages, max_tokens=int(args.max_tokens), temperature=0.0)
    raw_payload = extract_json_object(response_text)
    valid_failure_ids = {str(row.get("failure_id") or "").strip() for row in all_failures if str(row.get("failure_id") or "").strip()}
    proposal = validate_proposal(raw_payload, valid_failure_ids=valid_failure_ids)
    proposal_metadata = {
        "created_at": utc_now_iso(),
        "config": str(args.config),
        "db": str(args.db),
        "failure_log": str(args.failure_log),
        "failure_count_total": len(all_failures),
        "failure_count_prompted": len(selected),
        "corpus_ids": corpus_ids,
        "memory_llm": {
            "model_name": config.memory_llm.model_name,
            "base_url": config.memory_llm.base_url,
            "temperature": config.memory_llm.temperature,
        },
        "summary": summary,
    }
    proposal["metadata"] = proposal_metadata
    if args.record_run:
        proposal["recorded_evolution_run_ids"] = record_evolution_run(
            db_path=args.db,
            proposal=proposal,
            corpus_ids=corpus_ids,
            failure_log=str(args.failure_log),
            output=str(args.output),
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposal, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "patch_count": len(proposal.get("patches") or []), "metadata": proposal_metadata}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
