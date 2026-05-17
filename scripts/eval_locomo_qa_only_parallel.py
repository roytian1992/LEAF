from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import eval_locomo as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LEAF on LoCoMo in QA-only mode with parallel workers and existing DB reuse only."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--qa-per-sample", type=int, default=0)
    parser.add_argument("--snapshot-limit", type=int, default=8)
    parser.add_argument("--raw-span-limit", type=int, default=8)
    parser.add_argument(
        "--non-temporal-raw-span-limit",
        type=int,
        default=0,
        help="Optional larger raw-span limit for non-temporal QA only. 0 keeps --raw-span-limit for every question.",
    )
    parser.add_argument(
        "--local-support-mode",
        choices=["off", "selective"],
        default="off",
        help="Experimental retrieval-time local neighbor support. off preserves the v5 baseline path.",
    )
    parser.add_argument("--answer-view-mode", choices=["heuristic", "extractive"], default="heuristic")
    parser.add_argument(
        "--answer-style",
        choices=["short", "structured_context", "structured_context_topic_labeled", "structured_context_topic_auto"],
        default="short",
    )
    parser.add_argument(
        "--answer-evidence-mode",
        choices=["auto", "baseline", "merged"],
        default="auto",
        help=(
            "Controls which evidence is used to build the answer payload. auto preserves the legacy "
            "topic-labeled behavior; merged lets accepted topic-soft evidence enter the answer context."
        ),
    )
    parser.add_argument("--ingest-mode", default="reuse_only")
    parser.add_argument("--qa-workers", type=int, default=8)
    parser.add_argument("--ingest-prepare-workers", type=int, default=0)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume from an existing output/progress file by only running missing QA rows.",
    )
    parser.add_argument("--judge-with-llm", action="store_true")
    parser.add_argument("--judge-runs", type=int, default=0)
    parser.add_argument(
        "--retrieval-mode",
        choices=["baseline", "topic_soft", "topic_soft_gated", "topic_soft_selective", "overlay_selective"],
        default="baseline",
        help="Controls retrieval-side experimental variants. baseline preserves the original path.",
    )
    parser.add_argument("--topic-view-id", default="", help="Topic view for topic_soft. Defaults to active view per corpus.")
    parser.add_argument(
        "--topic-view-map",
        default="",
        help=(
            "Optional JSON object or JSON file mapping corpus_id to topic view id. "
            "Overrides --topic-view-id for listed corpora."
        ),
    )
    parser.add_argument(
        "--topic-router",
        choices=["keyword", "profile_hybrid", "profile_quality", "evolved_profile_first", "overlay_facet_hybrid", "llm"],
        default="keyword",
    )
    parser.add_argument("--topic-route-top-k", type=int, default=3)
    parser.add_argument("--topic-soft-event-limit", type=int, default=4)
    parser.add_argument("--overlay-event-limit", type=int, default=4)
    parser.add_argument(
        "--overlay-runtime-policy",
        choices=["none", "pre_gate_v2_no_open"],
        default="none",
        help=(
            "Runtime-visible gate for overlay evidence before answer assembly. "
            "pre_gate_v2_no_open keeps overlay only for temporal/entity/local-neighbor signals "
            "and suppresses broad open-slot where/what/why/how queries."
        ),
    )
    parser.add_argument("--topic-soft-per-topic-atom-limit", type=int, default=16)
    parser.add_argument(
        "--topic-soft-deny-topic-slugs",
        default="",
        help="Comma-separated topic slugs to suppress at retrieval time. Also honors view metadata topic_soft_policy.deny_topic_slugs.",
    )
    parser.add_argument(
        "--topic-soft-allow-topic-slugs",
        default="",
        help="Comma-separated topic slugs allowed for retrieval-time topic expansion. Also honors view metadata topic_soft_policy.allow_topic_slugs.",
    )
    parser.add_argument(
        "--topic-soft-min-content-overlap",
        type=int,
        default=1,
        help="For topic_soft_gated, require each added topic atom to overlap this many non-stopword question tokens.",
    )
    parser.add_argument(
        "--topic-soft-secondary-policy",
        choices=["all", "primary_only", "strict_text_v0"],
        default="all",
        help="Controls how atom_secondary assignments from evolved topics may add retrieval evidence.",
    )
    parser.add_argument(
        "--topic-soft-secondary-min-content-overlap",
        type=int,
        default=2,
        help="For strict_text_v0, require secondary atoms to overlap this many non-stopword question tokens.",
    )
    parser.add_argument(
        "--topic-soft-secondary-min-route-keyword-overlap",
        type=int,
        default=1,
        help="For strict_text_v0, require secondary-only topic routes to overlap this many question content tokens.",
    )
    parser.add_argument(
        "--topic-soft-min-event-embedding-similarity",
        type=float,
        default=0.0,
        help=(
            "Experimental semantic gate for topic_soft evidence. When >0, candidate topic events must "
            "have event embedding cosine similarity to the question at least this value."
        ),
    )
    parser.add_argument(
        "--topic-soft-use-stemmed-content-tokens",
        action="store_true",
        help="Use English Snowball stemming in topic-soft content-token overlap and atom scoring.",
    )
    parser.add_argument(
        "--topic-soft-allow-fallback-topic",
        action="store_true",
        help="For topic_soft_gated, allow fallback/misc routes to add evidence.",
    )
    parser.add_argument(
        "--topic-soft-fallback",
        choices=["none", "baseline_on_unknown"],
        default="none",
        help="Retry answer synthesis on baseline evidence when topic_soft evidence yields an abstention.",
    )
    parser.add_argument(
        "--topic-soft-runtime-policy",
        choices=["none", "skip_temporal_query_v0"],
        default="none",
        help=(
            "Early runtime policy applied before topic routing/expansion. "
            "Uses only query text and runtime retrieval state, not benchmark labels."
        ),
    )
    parser.add_argument(
        "--topic-soft-policy-min-selected-overlap",
        type=int,
        default=2,
        help="For topic_soft_selective, require the selected topic event to overlap this many content tokens.",
    )
    parser.add_argument(
        "--topic-soft-policy-max-candidate-atoms",
        type=int,
        default=20,
        help="For topic_soft_selective, suppress topic evidence when the filtered candidate atom pool is larger.",
    )
    parser.add_argument(
        "--topic-soft-policy",
        choices=[
            "selected_overlap_and_candidate_count_v0",
            "text_temporal_suppressed_v0",
            "route_uncertainty_semantic_v0",
        ],
        default="selected_overlap_and_candidate_count_v0",
        help="For topic_soft_selective, choose the text-only policy used to decide whether topic evidence is merged.",
    )
    parser.add_argument(
        "--topic-soft-policy-min-selected-semantic-similarity",
        type=float,
        default=0.0,
        help=(
            "For route_uncertainty_semantic_v0, suppress the whole topic expansion if the selected "
            "topic event has lower question-event embedding cosine similarity."
        ),
    )
    parser.add_argument(
        "--topic-soft-policy-suppress-multi-route",
        action="store_true",
        help="For route_uncertainty_semantic_v0, suppress topic evidence when more than one route is active.",
    )
    parser.add_argument(
        "--enable-heuristic-bypass",
        action="store_true",
        help="Opt in to the old behavior where a heuristic answer can bypass the answer prompt.",
    )
    parser.add_argument(
        "--disable-heuristic-bypass",
        action="store_true",
        help="Deprecated no-op compatibility flag. Heuristic bypass is disabled by default.",
    )
    parser.add_argument(
        "--temporal-postprocess",
        choices=["off", "anchor_only", "range", "range_no_weekday", "relative_prefer"],
        default="off",
        help="Optional deterministic temporal answer rewrite using retrieved relative-time evidence.",
    )
    parser.add_argument(
        "--short-answer-postprocess",
        choices=["off", "safe", "precise"],
        default="off",
        help="Optional deterministic short-answer cleanup for safe overlong answer patterns.",
    )
    return parser.parse_args()


def load_topic_view_map(value: str) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    path = Path(text)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--topic-view-map must be a JSON object or a path to one.")
    return {str(key): str(item) for key, item in payload.items() if str(key).strip() and str(item).strip()}


def _build_ingest_row(sample: dict[str, Any], *, corpus_id: str, qas: list[dict[str, Any]]) -> dict[str, Any]:
    _, turns = base.locomo_sample_to_turns(sample)
    session_ids = base.ordered_unique([str(turn.get("session_id") or "").strip() for turn in turns])
    return {
        "sample_id": str(sample.get("sample_id") or ""),
        "corpus_id": corpus_id,
        "turn_count": len(turns),
        "session_count": len([item for item in session_ids if item]),
        "qa_count": len(qas),
        "ingested": False,
        "reused": True,
        "ingest_elapsed_ms": 0.0,
        "ingest_metrics": {},
    }


def _make_task_rows(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ingest_rows: list[dict[str, Any]] = []
    qa_tasks: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id") or "locomo-sample")
        qas = base.locomo_sample_to_qas(sample)
        corpus_id = f"locomo_{base.sanitize_sample_id(sample_id)}"
        ingest_rows.append(_build_ingest_row(sample, corpus_id=corpus_id, qas=qas))
        for qa_index, qa in enumerate(qas):
            qa_tasks.append(
                {
                    "task_index": len(qa_tasks),
                    "sample_index": sample_index,
                    "qa_index": qa_index,
                    "sample_id": sample_id,
                    "corpus_id": corpus_id,
                    "qa": qa,
                }
            )
    return ingest_rows, qa_tasks


def _task_key(task: dict[str, Any]) -> tuple[str, int, str]:
    qa = task["qa"]
    return (
        str(task["sample_id"]),
        int(qa["question_index"]),
        str(qa["qa_id"]),
    )


def _row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sample_id") or ""),
        int(row.get("question_index") or 0),
        str(row.get("qa_id") or ""),
    )


def _load_existing_rows(output_path: Path, qa_progress_path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            for row in payload.get("results") or []:
                if isinstance(row, dict):
                    rows[_row_key(row)] = row
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    if qa_progress_path.exists():
        try:
            for line in qa_progress_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows[_row_key(row)] = row
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return rows


def main() -> None:
    args = parse_args()
    topic_view_map = load_topic_view_map(args.topic_view_map)
    args.resolved_topic_view_map = topic_view_map
    output_path = Path(args.output)
    qa_progress_path = output_path.with_name(f"{output_path.stem}.qa_progress.jsonl")
    qa_progress_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume_existing:
        qa_progress_path.write_text("", encoding="utf-8")
    elif not qa_progress_path.exists():
        qa_progress_path.write_text("", encoding="utf-8")

    samples = base.load_locomo_samples(args.input)
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    for sample in samples:
        qas = base.locomo_sample_to_qas(sample)
        if args.qa_per_sample > 0:
            sample["qa"] = qas[: args.qa_per_sample]

    ingest_rows, qa_tasks = _make_task_rows(samples)

    bootstrap_service = base.LEAFService(config_path=args.config, db_path=args.db)
    try:
        existing_corpora = set(bootstrap_service.list_corpora())
    finally:
        bootstrap_service.close()

    missing_corpora = sorted({str(row["corpus_id"]) for row in ingest_rows if str(row["corpus_id"]) not in existing_corpora})
    if missing_corpora:
        raise RuntimeError(
            "QA-only parallel run requires pre-ingested corpora in the existing DB. Missing corpora: "
            + ", ".join(missing_corpora[:10])
            + (" ..." if len(missing_corpora) > 10 else "")
        )

    service_local = threading.local()
    write_lock = threading.Lock()
    ordered_results: list[dict[str, Any] | None] = [None] * len(qa_tasks)
    task_index_by_key = {_task_key(task): int(task["task_index"]) for task in qa_tasks}
    tasks_to_run = list(qa_tasks)
    if args.resume_existing:
        loaded_count = 0
        for key, row in _load_existing_rows(output_path, qa_progress_path).items():
            task_index = task_index_by_key.get(key)
            if task_index is None or ordered_results[task_index] is not None:
                continue
            ordered_results[task_index] = row
            loaded_count += 1
        tasks_to_run = [task for task in qa_tasks if ordered_results[int(task["task_index"])] is None]
        print(
            f"[locomo-qa] resume_existing loaded={loaded_count} pending={len(tasks_to_run)}/{len(qa_tasks)}",
            flush=True,
        )

    def get_worker_service() -> base.LEAFService:
        service = getattr(service_local, "service", None)
        if service is None:
            service = base.LEAFService(config_path=args.config, db_path=args.db)
            service_local.service = service
        return service

    def get_topic_context(service: base.LEAFService, corpus_id: str) -> dict[str, Any] | None:
        if args.retrieval_mode not in {"topic_soft", "topic_soft_gated", "topic_soft_selective", "overlay_selective"}:
            return None
        cache = getattr(service_local, "topic_context_cache", None)
        if cache is None:
            cache = {}
            service_local.topic_context_cache = cache
        resolved_topic_view_id = topic_view_map.get(corpus_id) or args.topic_view_id
        cache_key = f"{corpus_id}|{resolved_topic_view_id or '<active>'}"
        if cache_key not in cache:
            cache[cache_key] = base.build_topic_context(service.store, corpus_id, resolved_topic_view_id or None)
        return cache[cache_key]

    def run_single_qa(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        service = get_worker_service()
        qa = dict(task["qa"])
        sample_id = str(task["sample_id"])
        corpus_id = str(task["corpus_id"])
        question = str(qa["question"])
        gold_answer = str(qa["answer"])
        print(
            f"[locomo-qa] sample={sample_id} q={qa['question_index']} category={qa['category_name']} search_start",
            flush=True,
        )
        search_started = time.perf_counter()
        evidence = service.search(
            corpus_id=corpus_id,
            question=question,
            snapshot_limit=args.snapshot_limit,
            raw_span_limit=base.resolve_qa_raw_span_limit(args, question),
            local_support_mode=args.local_support_mode,
        )
        baseline_evidence = evidence
        topic_soft_payload = None
        overlay_payload = None
        topic_context = get_topic_context(service, corpus_id)
        if args.retrieval_mode in {"topic_soft", "topic_soft_gated", "topic_soft_selective"} and topic_context is not None:
            skip_topic_soft, skip_reason = base.should_skip_topic_soft_runtime(args, question)
            if skip_topic_soft:
                topic_soft_payload = base.empty_topic_soft_payload(
                    policy=args.topic_soft_policy,
                    reason=skip_reason,
                    runtime_policy=args.topic_soft_runtime_policy,
                )
            else:
                baseline_event_ids = {
                    str(event_id).strip()
                    for event_id in (evidence.get("selected_event_ids") or [])
                    if str(event_id).strip()
                }
                topic_query_embedding = None
                try:
                    semantic_gate_min_similarity = float(args.topic_soft_min_event_embedding_similarity or 0.0)
                    needs_query_embedding = (
                        args.topic_router in {
                            "profile_hybrid",
                            "profile_quality",
                            "evolved_profile_first",
                            "overlay_facet_hybrid",
                        }
                        or semantic_gate_min_similarity > 0.0
                    )
                    route_query_embedding = None
                    if needs_query_embedding and getattr(service, "embedding", None) is not None:
                        route_query_embedding = service.embedding.embed(question)
                        topic_query_embedding = route_query_embedding
                    routed_topics, routed_router = base.route_topics(
                        service.store,
                        topic_context,
                        question=question,
                        router=args.topic_router,
                        top_k=args.topic_route_top_k,
                        llm=service.memory_llm if args.topic_router == "llm" else None,
                        query_embedding=route_query_embedding,
                    )
                except Exception as exc:  # noqa: BLE001
                    routed_topics, routed_router = base.route_topics(
                        service.store,
                        topic_context,
                        question=question,
                        router="keyword",
                        top_k=args.topic_route_top_k,
                        llm=None,
                        query_embedding=None,
                    )
                    for route in routed_topics:
                        route["router_error"] = str(exc)[:300]
                topic_soft_payload = base.topic_soft_expand_events(
                    service.store,
                    topic_context,
                    question=question,
                    routed_topics=routed_topics,
                    exclude_event_ids=baseline_event_ids,
                    event_limit=args.topic_soft_event_limit,
                    per_topic_atom_limit=args.topic_soft_per_topic_atom_limit,
                    min_content_overlap=(
                        max(0, int(args.topic_soft_min_content_overlap))
                        if args.retrieval_mode in {"topic_soft_gated", "topic_soft_selective"}
                        else 0
                    ),
                    allow_fallback_topic=(
                        bool(args.topic_soft_allow_fallback_topic)
                        if args.retrieval_mode in {"topic_soft_gated", "topic_soft_selective"}
                        else True
                    ),
                    score_with_content_tokens=args.retrieval_mode in {"topic_soft_gated", "topic_soft_selective"},
                    deny_topic_slugs=base.parse_csv_set(args.topic_soft_deny_topic_slugs),
                    allow_topic_slugs=base.parse_csv_set(args.topic_soft_allow_topic_slugs),
                    secondary_policy=args.topic_soft_secondary_policy,
                    secondary_min_content_overlap=max(0, int(args.topic_soft_secondary_min_content_overlap)),
                    secondary_min_route_keyword_overlap=max(
                        0,
                        int(args.topic_soft_secondary_min_route_keyword_overlap),
                    ),
                    query_embedding=topic_query_embedding,
                    min_event_embedding_similarity=max(
                        0.0,
                        float(args.topic_soft_min_event_embedding_similarity or 0.0),
                    ),
                    use_stemmed_content_tokens=bool(args.topic_soft_use_stemmed_content_tokens),
                )
                topic_soft_payload["router"] = routed_router
                topic_soft_payload["runtime_policy"] = args.topic_soft_runtime_policy
                topic_soft_payload["runtime_skipped"] = False
                topic_soft_payload["runtime_skip_reason"] = ""
                if args.retrieval_mode == "topic_soft_selective":
                    topic_soft_payload = base.apply_topic_soft_policy(
                        topic_soft_payload,
                        policy=args.topic_soft_policy,
                        min_selected_overlap=max(0, int(args.topic_soft_policy_min_selected_overlap)),
                        max_candidate_atom_count=max(0, int(args.topic_soft_policy_max_candidate_atoms)),
                        suppress_for_temporal_query=base.expects_temporal_answer(question) or base.is_temporal_query(question),
                        min_selected_semantic_similarity=max(
                            0.0,
                            float(args.topic_soft_policy_min_selected_semantic_similarity or 0.0),
                        ),
                        suppress_multi_route=bool(args.topic_soft_policy_suppress_multi_route),
                    )
            evidence = base.merge_topic_soft_evidence(evidence, topic_soft_payload)
        if args.retrieval_mode == "overlay_selective" and topic_context is not None:
            baseline_event_ids = {
                str(event_id).strip()
                for event_id in (evidence.get("selected_event_ids") or [])
                if str(event_id).strip()
            }
            overlay_payload = base.overlay_expand_events(
                service.store,
                topic_context,
                question=question,
                baseline_evidence=baseline_evidence,
                exclude_event_ids=baseline_event_ids,
                event_limit=max(0, int(args.overlay_event_limit)),
                use_stemmed_content_tokens=bool(args.topic_soft_use_stemmed_content_tokens),
            )
            overlay_payload = base.apply_overlay_runtime_policy(
                overlay_payload,
                question=question,
                policy=getattr(args, "overlay_runtime_policy", "none"),
            )
            if overlay_payload.get("event_ids"):
                selected_event_ids = [
                    str(event_id)
                    for event_id in (evidence.get("selected_event_ids") or [])
                    if str(event_id).strip()
                ]
                for event_id in overlay_payload.get("event_ids") or []:
                    if event_id not in selected_event_ids:
                        selected_event_ids.append(event_id)
                seen_span_ids = {str(span.get("span_id") or "") for span in (evidence.get("raw_spans") or [])}
                raw_spans = list(evidence.get("raw_spans") or [])
                for span in overlay_payload.get("raw_spans") or []:
                    span_id = str(span.get("span_id") or "")
                    if span_id and span_id in seen_span_ids:
                        continue
                    seen_span_ids.add(span_id)
                    raw_spans.append(span)
                evidence = {**evidence, "selected_event_ids": selected_event_ids, "raw_spans": raw_spans, "memory_overlay": overlay_payload}
        search_elapsed_ms = (time.perf_counter() - search_started) * 1000.0

        answer_started = time.perf_counter()
        answer_core_evidence, answer_evidence_mode = base.select_answer_core_evidence(
            args=args,
            answer_style=args.answer_style,
            baseline_evidence=baseline_evidence,
            evidence=evidence,
        )
        effective_answer_style = base.resolve_effective_answer_style(args.answer_style, evidence)
        context_lines = base.build_answer_context_lines(answer_core_evidence)
        heuristic_answer = base.heuristic_answer_from_evidence(question=question, evidence=answer_core_evidence)
        answer_view: dict[str, Any] = {}
        answer_view_text = ""
        answer_messages: list[dict[str, str]] = []
        answer_prompt_used = False
        topic_soft_fallback_used = False
        heuristic_bypass_triggered = bool(heuristic_answer)
        heuristic_bypass_enabled = bool(args.enable_heuristic_bypass) and not bool(args.disable_heuristic_bypass)
        heuristic_bypass_used = heuristic_bypass_triggered and heuristic_bypass_enabled
        if heuristic_bypass_used:
            answer_prompt_mode = "heuristic"
        elif heuristic_bypass_triggered:
            answer_prompt_mode = "llm_bypass_disabled"
        else:
            answer_prompt_mode = "llm"
        answer_prompt_input_tokens_est = 0
        answer_max_tokens = 0
        if heuristic_bypass_used:
            predicted_answer = heuristic_answer
            answer_input_tokens_est = 0
        else:
            answer_view = base.build_compact_answer_view(
                question=question,
                evidence=answer_core_evidence,
                mode=args.answer_view_mode,
            )
            answer_view_text = base.render_answer_view_text(question=question, answer_view=answer_view)
            answer_messages = base.build_answer_messages(
                question=question,
                evidence=evidence,
                context_lines=context_lines,
                answer_view_text=answer_view_text,
                answer_view=answer_view,
                answer_style=effective_answer_style,
            )
            answer_prompt_input_tokens_est = base.estimate_message_tokens(answer_messages)
            answer_input_tokens_est = answer_prompt_input_tokens_est
            answer_prompt_used = True
            try:
                answer_max_tokens = 96 if base.is_inference_query(question) else 80
                predicted_answer = (
                    service.llm.text(answer_messages, max_tokens=answer_max_tokens, temperature=0.0).strip()
                    if service.llm
                    else ""
                )
                if (
                    args.retrieval_mode in {"topic_soft", "topic_soft_gated", "topic_soft_selective"}
                    and args.topic_soft_fallback == "baseline_on_unknown"
                    and bool((topic_soft_payload or {}).get("event_ids"))
                    and baseline_evidence is not evidence
                    and base.is_abstention_answer(predicted_answer)
                ):
                    baseline_context_lines = base.build_answer_context_lines(baseline_evidence)
                    baseline_answer_view = base.build_compact_answer_view(
                        question=question,
                        evidence=baseline_evidence,
                        mode=args.answer_view_mode,
                    )
                    baseline_answer_view_text = base.render_answer_view_text(
                        question=question,
                        answer_view=baseline_answer_view,
                    )
                    baseline_answer_messages = base.build_answer_messages(
                        question=question,
                        evidence=baseline_evidence,
                        context_lines=baseline_context_lines,
                        answer_view_text=baseline_answer_view_text,
                        answer_view=baseline_answer_view,
                        answer_style=base.resolve_effective_answer_style(args.answer_style, baseline_evidence),
                    )
                    answer_input_tokens_est += base.estimate_message_tokens(baseline_answer_messages)
                    fallback_answer = (
                        service.llm.text(
                            baseline_answer_messages,
                            max_tokens=answer_max_tokens,
                            temperature=0.0,
                        ).strip()
                        if service.llm
                        else ""
                    )
                    if fallback_answer and not base.is_abstention_answer(fallback_answer):
                        predicted_answer = fallback_answer
                        topic_soft_fallback_used = True
            except base.OpenAICompatError as exc:
                predicted_answer = f"__ERROR__: {exc}"
        answer_elapsed_ms = (time.perf_counter() - answer_started) * 1000.0
        predicted_answer = str(base.canonicalize_temporal_answer(question, predicted_answer, answer_core_evidence) or predicted_answer).strip()
        predicted_answer_before_temporal_postprocess = predicted_answer
        predicted_answer = str(
            base.temporal_anchor_postprocess(
                question,
                predicted_answer,
                answer_core_evidence,
                mode=args.temporal_postprocess,
            )
            or predicted_answer
        ).strip()
        temporal_postprocess_used = predicted_answer != predicted_answer_before_temporal_postprocess
        predicted_answer_before_short_answer_postprocess = predicted_answer
        predicted_answer = str(
            base.apply_short_answer_postprocess(
                question,
                predicted_answer,
                mode=args.short_answer_postprocess,
                evidence=answer_core_evidence,
            )
            or predicted_answer
        ).strip()
        short_answer_postprocess_used = predicted_answer != predicted_answer_before_short_answer_postprocess
        print(
            f"[locomo-qa] sample={sample_id} q={qa['question_index']} answer_done search_ms={round(search_elapsed_ms, 2)} answer_ms={round(answer_elapsed_ms, 2)}",
            flush=True,
        )

        row = {
            "sample_id": sample_id,
            "corpus_id": corpus_id,
            "question_index": int(qa["question_index"]),
            "qa_id": qa["qa_id"],
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "predicted_answer_before_temporal_postprocess": predicted_answer_before_temporal_postprocess,
            "predicted_answer_before_short_answer_postprocess": predicted_answer_before_short_answer_postprocess,
            "category": qa["category"],
            "category_name": qa["category_name"],
            "gold_evidence": list(qa["evidence"]),
            "answer_f1": round(base.answer_f1_score(gold_answer, predicted_answer), 4),
            "bleu1": round(base.bleu1_score(gold_answer, predicted_answer), 4),
            "search_elapsed_ms": round(search_elapsed_ms, 2),
            "answer_elapsed_ms": round(answer_elapsed_ms, 2),
            "elapsed_ms": round(search_elapsed_ms + answer_elapsed_ms, 2),
            "answer_input_tokens_est": answer_input_tokens_est,
            "answer_prompt_input_tokens_est": answer_prompt_input_tokens_est,
            "answer_prompt_used": answer_prompt_used,
            "answer_prompt_mode": answer_prompt_mode,
            "answer_style": args.answer_style,
            "effective_answer_style": effective_answer_style,
            "answer_evidence_mode": answer_evidence_mode,
            "heuristic_bypass_triggered": heuristic_bypass_triggered,
            "heuristic_bypass_enabled": heuristic_bypass_enabled,
            "heuristic_bypass_used": heuristic_bypass_used,
            "answer_max_tokens": answer_max_tokens,
            "heuristic_answer": heuristic_answer,
            "temporal_postprocess": args.temporal_postprocess,
            "temporal_postprocess_used": temporal_postprocess_used,
            "short_answer_postprocess": args.short_answer_postprocess,
            "short_answer_postprocess_used": short_answer_postprocess_used,
            "retrieval_mode": args.retrieval_mode,
            "raw_span_count": len(evidence.get("raw_spans") or []),
            "topic_soft_event_count": len((topic_soft_payload or {}).get("event_ids") or []),
            "topic_soft_atom_count": len((topic_soft_payload or {}).get("atom_ids") or []),
            "topic_soft_candidate_atom_count": int((topic_soft_payload or {}).get("candidate_atom_count") or 0),
            "topic_soft_raw_candidate_atom_count": int((topic_soft_payload or {}).get("raw_candidate_atom_count") or 0),
            "topic_soft_filtered_atom_count": int((topic_soft_payload or {}).get("filtered_atom_count") or 0),
            "topic_soft_skipped_low_overlap_count": int((topic_soft_payload or {}).get("skipped_low_overlap_count") or 0),
            "topic_soft_skipped_fallback_route_count": int((topic_soft_payload or {}).get("skipped_fallback_route_count") or 0),
            "topic_soft_semantic_gate_enabled": bool(
                (topic_soft_payload or {}).get("semantic_gate_enabled")
            ),
            "topic_soft_semantic_gate_min_similarity": (
                (topic_soft_payload or {}).get("semantic_gate_min_similarity")
            ),
            "topic_soft_semantic_gate_max_similarity": (
                (topic_soft_payload or {}).get("semantic_gate_max_similarity")
            ),
            "topic_soft_semantic_gate_skipped_event_count": int(
                (topic_soft_payload or {}).get("semantic_gate_skipped_event_count") or 0
            ),
            "topic_soft_semantic_gate_missing_embedding_count": int(
                (topic_soft_payload or {}).get("semantic_gate_missing_embedding_count") or 0
            ),
            "topic_soft_secondary_policy": (topic_soft_payload or {}).get("secondary_policy"),
            "topic_soft_secondary_candidate_atom_count": int(
                (topic_soft_payload or {}).get("secondary_candidate_atom_count") or 0
            ),
            "topic_soft_secondary_selected_atom_count": int(
                (topic_soft_payload or {}).get("secondary_selected_atom_count") or 0
            ),
            "topic_soft_skipped_secondary_route_count": int(
                (topic_soft_payload or {}).get("skipped_secondary_route_count") or 0
            ),
            "topic_soft_skipped_secondary_policy_count": int(
                (topic_soft_payload or {}).get("skipped_secondary_policy_count") or 0
            ),
            "topic_soft_skipped_secondary_low_overlap_count": int(
                (topic_soft_payload or {}).get("skipped_secondary_low_overlap_count") or 0
            ),
            "topic_soft_active_route_count": len((topic_soft_payload or {}).get("active_routes") or []),
            "topic_soft_router": (topic_soft_payload or {}).get("router"),
            "topic_soft_topic_slugs": [
                str(route.get("slug") or route.get("topic_id"))
                for route in ((topic_soft_payload or {}).get("routes") or [])
            ],
            "topic_soft_active_topic_slugs": [
                str(route.get("slug") or route.get("topic_id"))
                for route in ((topic_soft_payload or {}).get("active_routes") or [])
            ],
            "topic_soft_suppressed_route_count": len((topic_soft_payload or {}).get("suppressed_routes") or []),
            "topic_soft_suppressed_topic_slugs": [
                str(route.get("slug") or route.get("topic_id"))
                for route in ((topic_soft_payload or {}).get("suppressed_routes") or [])
            ],
            "topic_soft_fallback": args.topic_soft_fallback,
            "topic_soft_fallback_used": topic_soft_fallback_used,
            "topic_soft_policy": (topic_soft_payload or {}).get("policy"),
            "topic_soft_policy_applied": bool((topic_soft_payload or {}).get("policy_applied")),
            "topic_soft_policy_reason": (topic_soft_payload or {}).get("policy_reason"),
            "topic_soft_policy_max_selected_content_overlap": int(
                (topic_soft_payload or {}).get("policy_max_selected_content_overlap") or 0
            ),
            "topic_soft_policy_max_selected_semantic_similarity": (
                (topic_soft_payload or {}).get("policy_max_selected_semantic_similarity")
            ),
            "topic_soft_policy_min_selected_semantic_similarity": (
                (topic_soft_payload or {}).get("policy_min_selected_semantic_similarity")
            ),
            "topic_soft_policy_suppress_multi_route": bool(
                (topic_soft_payload or {}).get("policy_suppress_multi_route")
            ),
            "topic_soft_policy_min_selected_overlap": int(
                (topic_soft_payload or {}).get("policy_min_selected_overlap") or 0
            ),
            "topic_soft_policy_max_candidate_atom_count": int(
                (topic_soft_payload or {}).get("policy_max_candidate_atom_count") or 0
            ),
            "topic_soft_suppressed_event_count": len((topic_soft_payload or {}).get("suppressed_event_ids") or []),
            "overlay_event_count": len((overlay_payload or {}).get("event_ids") or []),
            "overlay_candidate_count": int((overlay_payload or {}).get("candidate_count") or 0),
            "overlay_suppressed": bool((overlay_payload or {}).get("suppressed")),
            "overlay_suppress_reason": (overlay_payload or {}).get("suppress_reason"),
            "overlay_primary_strength": (overlay_payload or {}).get("primary_strength"),
            "overlay_runtime_policy": (overlay_payload or {}).get("runtime_policy"),
            "overlay_runtime_policy_applied": bool((overlay_payload or {}).get("runtime_policy_applied")),
            "overlay_runtime_policy_allowed": (overlay_payload or {}).get("runtime_policy_allowed"),
            "overlay_runtime_policy_reason": (overlay_payload or {}).get("runtime_policy_reason"),
            "overlay_runtime_policy_features": (overlay_payload or {}).get("runtime_policy_features") or {},
            "overlay_sources": base.ordered_unique(
                [
                    str(source)
                    for item in ((overlay_payload or {}).get("selected") or [])
                    if isinstance(item, dict)
                    for source in (item.get("sources") or [])
                    if str(source).strip()
                ]
            ),
            "page_count": len(evidence.get("pages") or []),
            "atom_count": len(evidence.get("atoms") or []),
            "answer_context_line_count": len(context_lines),
            "answer_context_lines": list(context_lines),
            "answer_view_summary": base.summarize_answer_view(answer_view),
            "answer_view": answer_view,
            "answer_view_text": answer_view_text,
            "answer_view_text_chars": len(answer_view_text),
            "answer_prompt_messages": answer_messages,
            "retrieval": {
                "traversal_path": list(evidence.get("traversal_path") or []),
                "pages": list(evidence.get("pages") or []),
                "atoms": list(evidence.get("atoms") or []),
                "raw_spans": list(evidence.get("raw_spans") or []),
                "supporting_raw_spans": list(evidence.get("supporting_raw_spans") or []),
            },
            "retrieved_dia_ids": base.ordered_unique(
                [
                    str((span.get("metadata") or {}).get("dia_id") or "").strip()
                    for span in list(evidence.get("raw_spans") or [])
                    + list(evidence.get("supporting_raw_spans") or [])
                ]
            ),
        }
        return int(task["task_index"]), row

    worker_count = min(max(1, int(args.qa_workers)), max(1, len(tasks_to_run)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_single_qa, task) for task in tasks_to_run]
        for completed_index, future in enumerate(as_completed(futures), start=1):
            task_index, row = future.result()
            with write_lock:
                ordered_results[task_index] = row
                current_results = [item for item in ordered_results if item is not None]
                base.append_jsonl(qa_progress_path, row)
                base.write_json_atomic(
                    output_path,
                    base.build_payload(
                        args=args,
                        samples=samples,
                        ingest_rows=ingest_rows,
                        results=current_results,
                        completed=False,
                        qa_progress_path=qa_progress_path,
                    ),
                )
            if completed_index % 20 == 0 or completed_index == len(futures):
                total_completed = len([item for item in ordered_results if item is not None])
                print(
                    f"[locomo-qa] completed {completed_index}/{len(futures)} resumed_rows "
                    f"total={total_completed}/{len(qa_tasks)}",
                    flush=True,
                )
    final_results = [item for item in ordered_results if item is not None]
    base.write_json_atomic(
        output_path,
        base.build_payload(
            args=args,
            samples=samples,
            ingest_rows=ingest_rows,
            results=final_results,
            completed=True,
            qa_progress_path=qa_progress_path,
        ),
    )


if __name__ == "__main__":
    main()
