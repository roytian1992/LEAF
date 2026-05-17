from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import eval_locomo as locomo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute deterministic LoCoMo answer postprocessing and metrics from an existing "
            "result JSON without re-running retrieval or answer LLM calls."
        )
    )
    parser.add_argument("--input", required=True, help="Existing LoCoMo result JSON.")
    parser.add_argument("--output", required=True, help="Output JSON with recomputed answers and summary.")
    parser.add_argument(
        "--mode",
        choices=["precise"],
        default="precise",
        help="Postprocess mode to recompute. Currently only the precise path is supported.",
    )
    parser.add_argument(
        "--source-field",
        default="predicted_answer_before_short_answer_postprocess",
        help="Row field used as the pre-postprocess answer. Falls back to predicted_answer.",
    )
    parser.add_argument(
        "--note",
        default="Recomputed deterministic precise short-answer postprocess; no retrieval or LLM calls.",
    )
    return parser.parse_args()


def recompute_row(row: dict, *, source_field: str) -> dict:
    updated = copy.deepcopy(row)
    previous_answer = updated.get("predicted_answer")
    source_answer = updated.get(source_field) or previous_answer
    evidence = updated.get("retrieval") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    recomputed = locomo.precise_short_answer_postprocess(
        str(updated.get("question") or ""),
        str(source_answer or ""),
        evidence=evidence,
    )
    updated["predicted_answer_original_run"] = previous_answer
    updated["predicted_answer"] = str(recomputed or source_answer or "").strip()
    updated["answer_f1_original_run"] = updated.get("answer_f1")
    updated["bleu1_original_run"] = updated.get("bleu1")
    updated["answer_f1"] = round(
        locomo.answer_f1_score(str(updated.get("gold_answer") or ""), updated["predicted_answer"]),
        4,
    )
    updated["bleu1"] = round(
        locomo.bleu1_score(str(updated.get("gold_answer") or ""), updated["predicted_answer"]),
        4,
    )
    updated["short_answer_postprocess_recomputed"] = True
    updated["short_answer_postprocess_changed_from_original_run"] = (
        updated["predicted_answer"] != previous_answer
    )
    return updated


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    updated_payload = copy.deepcopy(payload)
    updated_results = [
        recompute_row(row, source_field=args.source_field)
        for row in payload.get("results") or []
        if isinstance(row, dict)
    ]
    changed_qa_ids = [
        str(row.get("qa_id") or "")
        for row in updated_results
        if row.get("short_answer_postprocess_changed_from_original_run")
    ]
    config = dict(payload.get("config") or {})
    updated_payload["results"] = updated_results
    updated_payload["source_result"] = str(input_path)
    updated_payload["postprocess_note"] = str(args.note)
    updated_payload["completed"] = bool(payload.get("completed", True))
    updated_payload["summary"] = locomo.build_summary(
        samples=list(payload.get("samples") or []),
        ingest_rows=list(payload.get("ingest") or []),
        results=updated_results,
        judge_with_llm=bool(config.get("judge_with_llm") or False),
        judge_runs=int(config.get("judge_runs") or 0),
    )
    updated_payload["summary"]["repostprocess_source_result"] = str(input_path)
    updated_payload["summary"]["repostprocess_changed_count"] = len(changed_qa_ids)
    updated_payload["summary"]["repostprocess_changed_qa_ids"] = changed_qa_ids
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(updated_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "summary": updated_payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
