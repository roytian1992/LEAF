from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans({char: " " for char in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""})


def normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower().translate(PUNCT_TABLE)
    tokens = [token for token in lowered.split() if token not in ARTICLES]
    return " ".join(tokens)


def normalized_tokens(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token}


def text_recall_score(source_text: str, candidate_text: str) -> float:
    source_tokens = normalized_tokens(source_text)
    candidate_tokens = normalized_tokens(candidate_text)
    if not source_tokens or not candidate_tokens:
        return 0.0
    return len(source_tokens.intersection(candidate_tokens)) / max(1, len(source_tokens))


def iter_answer_view_items(answer_view: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for section, values in dict(answer_view or {}).items():
        if not isinstance(values, list):
            values = [values]
        for value in values:
            if isinstance(value, dict):
                items.append(
                    {
                        "section": section,
                        "text": str(value.get("text") or "").strip(),
                        "citations": [str(item).strip() for item in value.get("citations") or [] if str(item).strip()],
                    }
                )
            else:
                items.append(
                    {
                        "section": section,
                        "text": str(value or "").strip(),
                        "citations": [],
                    }
                )
    return items


def classify_row(row: dict[str, Any], *, surface_recall_threshold: float = 0.55) -> dict[str, Any]:
    gold_dia_ids = [str(item).strip() for item in row.get("gold_evidence") or [] if str(item).strip()]
    retrieved_dia_ids = {str(item).strip() for item in row.get("retrieved_dia_ids") or [] if str(item).strip()}
    raw_spans = list(((row.get("retrieval") or {}).get("raw_spans") or []))
    answer_context_lines = [str(item or "").strip() for item in row.get("answer_context_lines") or [] if str(item or "").strip()]
    answer_view_items = iter_answer_view_items(dict(row.get("answer_view") or {}))
    answer_view_text = str(row.get("answer_view_text") or "")

    raw_span_by_dia: dict[str, list[dict[str, Any]]] = {}
    for span in raw_spans:
        dia_id = str((span.get("metadata") or {}).get("dia_id") or "").strip()
        if not dia_id:
            continue
        raw_span_by_dia.setdefault(dia_id, []).append(span)

    gold_retrieved_dia_ids = [dia_id for dia_id in gold_dia_ids if dia_id in retrieved_dia_ids]
    gold_context_dia_ids = [
        dia_id for dia_id in gold_dia_ids if any(dia_id in line for line in answer_context_lines)
    ]

    span_id_to_dia: dict[str, str] = {}
    for dia_id, spans in raw_span_by_dia.items():
        for span in spans:
            span_id = str(span.get("span_id") or "").strip()
            if span_id:
                span_id_to_dia[span_id] = dia_id

    cited_gold_dia_ids: set[str] = set()
    gold_surface_hits: dict[str, dict[str, Any]] = {}
    for dia_id in gold_dia_ids:
        gold_spans = raw_span_by_dia.get(dia_id, [])
        best_score = 0.0
        best_section = ""
        for item in answer_view_items:
            citations = set(item["citations"])
            if any(span_id_to_dia.get(citation) == dia_id for citation in citations):
                cited_gold_dia_ids.add(dia_id)
            for span in gold_spans:
                score = text_recall_score(str(span.get("text") or ""), item["text"])
                if score > best_score:
                    best_score = score
                    best_section = str(item["section"])
        if best_score <= 0.0:
            for span in gold_spans:
                best_score = max(best_score, text_recall_score(str(span.get("text") or ""), answer_view_text))
        if best_score > 0.0:
            gold_surface_hits[dia_id] = {
                "best_recall": round(best_score, 4),
                "best_section": best_section,
                "surfaced": best_score >= surface_recall_threshold or dia_id in cited_gold_dia_ids,
                "cited": dia_id in cited_gold_dia_ids,
            }

    surfaced_gold_dia_ids = [
        dia_id for dia_id in gold_dia_ids if bool(gold_surface_hits.get(dia_id, {}).get("surfaced"))
    ]

    if gold_dia_ids and not gold_retrieved_dia_ids:
        diagnosis = "missed_in_retrieval"
    elif gold_retrieved_dia_ids and not gold_context_dia_ids:
        diagnosis = "retrieved_but_not_context"
    elif gold_context_dia_ids and not surfaced_gold_dia_ids:
        diagnosis = "context_but_not_answer_view"
    elif surfaced_gold_dia_ids and float(row.get("answer_f1") or 0.0) < 0.5:
        diagnosis = "surfaced_but_answer_wrong"
    else:
        diagnosis = "ok_or_non_gold_case"

    return {
        "sample_id": row.get("sample_id"),
        "question_index": row.get("question_index"),
        "category_name": row.get("category_name"),
        "answer_f1": row.get("answer_f1"),
        "bleu1": row.get("bleu1"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "predicted_answer": row.get("predicted_answer"),
        "gold_dia_ids": gold_dia_ids,
        "gold_retrieved_dia_ids": gold_retrieved_dia_ids,
        "gold_context_dia_ids": gold_context_dia_ids,
        "gold_surfaced_dia_ids": surfaced_gold_dia_ids,
        "gold_surface_hits": gold_surface_hits,
        "diagnosis": diagnosis,
        "retrieved_dia_ids": sorted(retrieved_dia_ids),
    }


def summarize_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosis_counts = Counter(str(row["diagnosis"]) for row in rows)
    with_gold = [row for row in rows if row["gold_dia_ids"]]
    return {
        "row_count": len(rows),
        "rows_with_gold_evidence": len(with_gold),
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        "avg_answer_f1": round(sum(float(row["answer_f1"] or 0.0) for row in rows) / len(rows), 4) if rows else None,
        "avg_answer_f1_with_gold": (
            round(sum(float(row["answer_f1"] or 0.0) for row in with_gold) / len(with_gold), 4) if with_gold else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze where LoCoMo QA failures happen across retrieval/context/view layers.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--only-diagnosis", default="")
    parser.add_argument("--max-answer-f1", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    source_rows = list(report.get("results") or [])
    analyzed_rows = [classify_row(row) for row in source_rows]

    if args.sample_id:
        analyzed_rows = [row for row in analyzed_rows if str(row.get("sample_id")) == str(args.sample_id)]
    if args.question_index > 0:
        analyzed_rows = [row for row in analyzed_rows if int(row.get("question_index") or 0) == int(args.question_index)]
    if args.only_diagnosis:
        analyzed_rows = [row for row in analyzed_rows if str(row.get("diagnosis")) == str(args.only_diagnosis)]
    analyzed_rows = [row for row in analyzed_rows if float(row.get("answer_f1") or 0.0) <= float(args.max_answer_f1)]

    output_payload = {
        "report": str(Path(args.report).resolve()),
        "filters": {
            "sample_id": args.sample_id or None,
            "question_index": args.question_index or None,
            "only_diagnosis": args.only_diagnosis or None,
            "max_answer_f1": args.max_answer_f1,
            "top_n": args.top_n,
        },
        "summary": summarize_diagnostics(analyzed_rows),
        "rows": analyzed_rows[: max(0, int(args.top_n))],
    }

    rendered = json.dumps(output_payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(str(Path(args.output).resolve()))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
