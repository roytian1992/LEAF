from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ \t\n\r]+", " ", text)
    text = text.strip(" .,!?:;\"'")
    return text


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def gold_index(gold_payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = gold_payload.get("results") or []
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["persona"]), int(row["question_index"]))
        index[key] = row
    return index


def exact_match(pred: str, target: str) -> bool:
    return normalize_text(pred) == normalize_text(target)


def accepted_match(pred: str, candidates: list[str]) -> bool:
    normalized_pred = normalize_text(pred)
    if not normalized_pred:
        return False
    normalized_candidates = [normalize_text(item) for item in candidates if normalize_text(item)]
    return normalized_pred in normalized_candidates


def score_report(report_payload: dict[str, Any], gold_map: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    rows = report_payload.get("results") or []
    scored_rows: list[dict[str, Any]] = []
    em_hits = 0
    accepted_hits = 0
    elapsed_values: list[float] = []
    token_values: list[int] = []

    for row in rows:
        key = (str(row["persona"]), int(row["question_index"]))
        gold = gold_map[key]
        pred = str(row.get("predicted_answer") or "")
        canonical = str(gold.get("canonical_answer") or "")
        acceptable = [canonical] + [str(item) for item in (gold.get("acceptable_answers") or [])]
        em = exact_match(pred, canonical)
        accepted = accepted_match(pred, acceptable)
        em_hits += int(em)
        accepted_hits += int(accepted)
        if row.get("elapsed_ms") is not None:
            elapsed_values.append(float(row["elapsed_ms"]))
        if row.get("answer_input_tokens_est") is not None:
            token_values.append(int(row["answer_input_tokens_est"]))
        scored_rows.append(
            {
                **row,
                "gold_canonical_answer": canonical,
                "gold_acceptable_answers": gold.get("acceptable_answers") or [],
                "gold_supporting_spans": gold.get("supporting_spans") or [],
                "gold_key_facts": gold.get("key_facts") or [],
                "strict_em": em,
                "accepted_match": accepted,
            }
        )

    count = len(scored_rows)
    return {
        "summary": {
            "question_count": count,
            "strict_em": round(em_hits / count, 4) if count else None,
            "accepted_match": round(accepted_hits / count, 4) if count else None,
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
            "avg_answer_input_tokens_est": round(sum(token_values) / len(token_values), 2) if token_values else None,
        },
        "results": scored_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GVD reports against a gold-reference sheet.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--leaf-old", required=True)
    parser.add_argument("--leaf-new", required=True)
    parser.add_argument("--memoryos", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_payload = load_json(args.gold)
    gold_map = gold_index(gold_payload)

    methods = {
        "leaf_old": score_report(load_json(args.leaf_old), gold_map),
        "leaf_new": score_report(load_json(args.leaf_new), gold_map),
        "memoryos": score_report(load_json(args.memoryos), gold_map),
    }

    output = {
        "gold": str(args.gold),
        "methods": {name: payload["summary"] for name, payload in methods.items()},
        "detailed": methods,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "methods": output["methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
