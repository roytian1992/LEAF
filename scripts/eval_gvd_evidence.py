from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def build_gold_index(gold_payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = gold_payload.get("results") or []
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        index[(str(row["persona"]), int(row["question_index"]))] = row
    return index


def score_report(report_payload: dict[str, Any], gold_index: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    rows = report_payload.get("results") or []
    detailed: list[dict[str, Any]] = []
    question_hit = 0
    total_required = 0
    total_covered = 0
    elapsed_values: list[float] = []
    token_values: list[int] = []

    for row in rows:
        key = (str(row["persona"]), int(row["question_index"]))
        gold = gold_index[key]
        required = list(dict.fromkeys(str(item) for item in (gold.get("required_spans") or []) if str(item).strip()))
        retrieved = list(dict.fromkeys(str(item) for item in (row.get("retrieved_span_ids") or []) if str(item).strip()))
        required_set = set(required)
        retrieved_set = set(retrieved)
        covered = sorted(required_set & retrieved_set)
        hit = not required_set or required_set.issubset(retrieved_set)
        recall = 1.0 if not required_set else len(covered) / len(required_set)
        question_hit += int(hit)
        total_required += len(required_set)
        total_covered += len(covered)
        if row.get("elapsed_ms") is not None:
            elapsed_values.append(float(row["elapsed_ms"]))
        if row.get("answer_input_tokens_est") is not None:
            token_values.append(int(row["answer_input_tokens_est"]))
        detailed.append(
            {
                **row,
                "required_spans": required,
                "covered_required_spans": covered,
                "required_span_hit": hit,
                "required_span_recall": round(recall, 4),
            }
        )

    count = len(detailed)
    return {
        "summary": {
            "question_count": count,
            "required_span_hit_rate": round(question_hit / count, 4) if count else None,
            "required_span_recall": round(total_covered / total_required, 4) if total_required else None,
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
            "avg_answer_input_tokens_est": round(sum(token_values) / len(token_values), 2) if token_values else None,
        },
        "results": detailed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GVD evidence retrieval against required gold spans.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--leaf-old", required=True)
    parser.add_argument("--leaf-new", required=True)
    parser.add_argument("--memoryos", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_json(args.gold)
    gold_idx = build_gold_index(gold)
    methods = {
        "leaf_old": score_report(load_json(args.leaf_old), gold_idx),
        "leaf_new": score_report(load_json(args.leaf_new), gold_idx),
        "memoryos": score_report(load_json(args.memoryos), gold_idx),
    }
    payload = {
        "gold": str(args.gold),
        "methods": {name: item["summary"] for name, item in methods.items()},
        "detailed": methods,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "methods": payload["methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
