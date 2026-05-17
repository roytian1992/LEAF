from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import eval_locomo as locomo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply deterministic postprocessing to a LoCoMo result JSON.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["off", "anchor_only", "range", "range_no_weekday"],
        default="range",
        help="Temporal postprocess policy to apply.",
    )
    parser.add_argument(
        "--short-answer-mode",
        choices=["off", "safe"],
        default="off",
        help="Optional short-answer cleanup policy to apply after temporal postprocessing.",
    )
    return parser.parse_args()


def rewrite_row(row: dict[str, Any], mode: str, short_answer_mode: str) -> dict[str, Any]:
    rewritten = dict(row)
    original_answer = str(row.get("predicted_answer") or "").strip()
    evidence = dict(row.get("retrieval") or {})
    if "raw_spans" not in evidence:
        evidence["raw_spans"] = []
    new_answer = str(
        locomo.temporal_anchor_postprocess(
            str(row.get("question") or ""),
            original_answer,
            evidence,
            mode=mode,
        )
        or original_answer
    ).strip()
    rewritten["predicted_answer_before_temporal_postprocess"] = original_answer
    rewritten["temporal_postprocess"] = mode
    rewritten["temporal_postprocess_used"] = new_answer != original_answer
    before_short = new_answer
    new_answer = str(
        locomo.apply_short_answer_postprocess(
            str(row.get("question") or ""),
            new_answer,
            mode=short_answer_mode,
            evidence=evidence,
        )
        or new_answer
    ).strip()
    rewritten["predicted_answer_before_short_answer_postprocess"] = before_short
    rewritten["short_answer_postprocess"] = short_answer_mode
    rewritten["short_answer_postprocess_used"] = new_answer != before_short
    rewritten["predicted_answer"] = new_answer
    gold_answer = str(row.get("gold_answer") or "")
    rewritten["answer_f1"] = round(locomo.answer_f1_score(gold_answer, new_answer), 4)
    rewritten["bleu1"] = round(locomo.bleu1_score(gold_answer, new_answer), 4)
    return rewritten


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = [rewrite_row(dict(row), args.mode, args.short_answer_mode) for row in payload.get("results") or []]
    payload["results"] = rows
    payload["temporal_postprocess"] = args.mode
    payload["short_answer_postprocess"] = args.short_answer_mode
    payload["temporal_postprocess_source"] = str(input_path)
    payload["summary"] = locomo.build_summary(
        samples=[],
        ingest_rows=list(payload.get("ingest") or []),
        results=rows,
        judge_with_llm=False,
        judge_runs=0,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
