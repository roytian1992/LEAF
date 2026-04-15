from __future__ import annotations

import argparse
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
    parser.add_argument("--answer-view-mode", choices=["heuristic", "extractive"], default="heuristic")
    parser.add_argument("--qa-workers", type=int, default=8)
    parser.add_argument("--ingest-prepare-workers", type=int, default=0)
    parser.add_argument("--judge-with-llm", action="store_true")
    parser.add_argument("--judge-runs", type=int, default=0)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    qa_progress_path = output_path.with_name(f"{output_path.stem}.qa_progress.jsonl")
    qa_progress_path.parent.mkdir(parents=True, exist_ok=True)
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

    def get_worker_service() -> base.LEAFService:
        service = getattr(service_local, "service", None)
        if service is None:
            service = base.LEAFService(config_path=args.config, db_path=args.db)
            service_local.service = service
        return service

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
            raw_span_limit=args.raw_span_limit,
        )
        search_elapsed_ms = (time.perf_counter() - search_started) * 1000.0

        answer_started = time.perf_counter()
        context_lines = base.build_answer_context_lines(evidence)
        heuristic_answer = base.heuristic_answer_from_evidence(question=question, evidence=evidence)
        answer_view: dict[str, Any] = {}
        answer_view_text = ""
        answer_messages: list[dict[str, str]] = []
        answer_prompt_used = False
        answer_prompt_mode = "heuristic" if heuristic_answer else "llm"
        answer_prompt_input_tokens_est = 0
        answer_max_tokens = 0
        if heuristic_answer:
            predicted_answer = heuristic_answer
            answer_input_tokens_est = 0
        else:
            answer_view = base.build_compact_answer_view(
                question=question,
                evidence=evidence,
                mode=args.answer_view_mode,
            )
            answer_view_text = base.render_answer_view_text(question=question, answer_view=answer_view)
            answer_messages = base.build_answer_messages(
                question=question,
                evidence=evidence,
                context_lines=context_lines,
                answer_view_text=answer_view_text,
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
            except base.OpenAICompatError as exc:
                predicted_answer = f"__ERROR__: {exc}"
        answer_elapsed_ms = (time.perf_counter() - answer_started) * 1000.0
        predicted_answer = str(base.canonicalize_temporal_answer(question, predicted_answer, evidence) or predicted_answer).strip()
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
            "answer_max_tokens": answer_max_tokens,
            "heuristic_answer": heuristic_answer,
            "raw_span_count": len(evidence.get("raw_spans") or []),
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
            },
            "retrieved_dia_ids": base.ordered_unique(
                [
                    str((span.get("metadata") or {}).get("dia_id") or "").strip()
                    for span in (evidence.get("raw_spans") or [])
                ]
            ),
        }
        return int(task["task_index"]), row

    worker_count = min(max(1, int(args.qa_workers)), max(1, len(qa_tasks)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_single_qa, task) for task in qa_tasks]
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
                print(f"[locomo-qa] completed {completed_index}/{len(futures)} rows", flush=True)
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
