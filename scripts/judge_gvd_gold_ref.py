from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.clients import ChatClient, extract_json_object  # noqa: E402
from leaf.config import ModelConfig  # noqa: E402


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_judge_client(config_path: str | Path) -> ChatClient:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    llm_cfg = cfg.get("llm") or {}
    model_cfg = ModelConfig(
        provider=str(llm_cfg.get("provider") or "openai"),
        model_name=str(llm_cfg["model_name"]),
        api_key=str(llm_cfg["api_key"]),
        base_url=str(llm_cfg["base_url"]),
        timeout=int(llm_cfg.get("timeout", 180)),
        temperature=0.0,
        max_tokens=256,
    )
    return ChatClient(model_cfg)


def build_gold_index(gold_payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in gold_payload.get("results") or []:
        index[(str(row["persona"]), int(row["question_index"]))] = row
    return index


def build_judge_messages(*, gold_row: dict[str, Any], predicted_answer: str) -> list[dict[str, str]]:
    acceptable = gold_row.get("acceptable_answers") or []
    key_facts = gold_row.get("key_facts") or []
    notes = str(gold_row.get("notes") or "").strip()
    return [
        {
            "role": "system",
            "content": (
                "You are grading a model answer using a gold reference sheet, not the full history. "
                "Judge semantic correctness against the canonical answer, acceptable answers, and key facts. "
                "Return JSON only with keys: label, score, reason. "
                "label must be one of CORRECT, PARTIAL, WRONG. "
                "score must be one of 1, 0.5, 0. "
                "Use CORRECT when the answer is semantically correct even if wording differs. "
                "Use PARTIAL when the answer is incomplete but still contains a materially correct core fact. "
                "Use WRONG when the answer contradicts the reference, misses the asked fact, or says UNKNOWN despite clear support."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Persona: {gold_row['persona']}\n"
                f"Question: {gold_row['question']}\n"
                f"Canonical answer: {gold_row['canonical_answer']}\n"
                f"Acceptable answers: {json.dumps(acceptable, ensure_ascii=False)}\n"
                f"Key facts: {json.dumps(key_facts, ensure_ascii=False)}\n"
                f"Notes: {notes or '(none)'}\n"
                f"Candidate answer: {predicted_answer}\n\n"
                'Return JSON like {"label":"CORRECT","score":1,"reason":"..."}'
            ),
        },
    ]


def judge_answer(client: ChatClient, *, gold_row: dict[str, Any], predicted_answer: str, retries: int) -> dict[str, Any]:
    last_error = None
    for _ in range(max(1, retries)):
        try:
            text = client.text(
                build_judge_messages(gold_row=gold_row, predicted_answer=predicted_answer),
                max_tokens=256,
                temperature=0.0,
            ).strip()
            payload = extract_json_object(text)
            label = str(payload.get("label") or "").strip().upper()
            raw_score = payload.get("score")
            score = float(raw_score)
            if label not in {"CORRECT", "PARTIAL", "WRONG"}:
                raise ValueError(f"Unexpected label: {label}")
            if score not in {0.0, 0.5, 1.0}:
                raise ValueError(f"Unexpected score: {score}")
            return {
                "label": label,
                "score": score,
                "reason": str(payload.get("reason") or ""),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    return {"label": "ERROR", "score": None, "reason": last_error or "judge_failed"}


def aggregate_row_judgments(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    valid_scores = [float(item["score"]) for item in judgments if item.get("score") is not None]
    valid_labels = [str(item["label"]) for item in judgments if str(item.get("label") or "") != "ERROR"]
    valid_reasons = [str(item["reason"]) for item in judgments if str(item.get("reason") or "").strip()]

    if valid_labels:
        label_counts: dict[str, int] = {}
        for label in valid_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        majority_label = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    else:
        majority_label = "ERROR"

    return {
        "judge_score": round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None,
        "judge_std": round(statistics.pstdev(valid_scores), 4) if len(valid_scores) > 1 else 0.0 if valid_scores else None,
        "judge_verdict": majority_label,
        "judge_scores": [item.get("score") for item in judgments],
        "judge_verdicts": [item.get("label") for item in judgments],
        "judge_rationales": [item.get("reason") for item in judgments],
        "judge_rationale": valid_reasons[0] if valid_reasons else "",
        "judge_valid_runs": len(valid_scores),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["judge_score"]) for row in rows if row.get("judge_score") is not None]
    elapsed = [float(row["elapsed_ms"]) for row in rows if row.get("elapsed_ms") is not None]
    tokens = [int(row["answer_input_tokens_est"]) for row in rows if row.get("answer_input_tokens_est") is not None]
    max_runs = max((len(row.get("judge_scores") or []) for row in rows), default=0)
    run_means: list[float] = []
    for run_index in range(max_runs):
        run_scores = [
            float(row["judge_scores"][run_index])
            for row in rows
            if len(row.get("judge_scores") or []) > run_index and row["judge_scores"][run_index] is not None
        ]
        if run_scores:
            run_means.append(round(sum(run_scores) / len(run_scores), 4))
    return {
        "question_count": len(rows),
        "judge_count": len(scores),
        "judge_avg": round(sum(run_means) / len(run_means), 4) if run_means else round(sum(scores) / len(scores), 4) if scores else None,
        "judge_std": round(statistics.pstdev(run_means), 4) if len(run_means) > 1 else 0.0 if run_means else round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0 if scores else None,
        "judge_run_scores": run_means,
        "judge_runs": max_runs,
        "avg_elapsed_ms": round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
        "avg_answer_input_tokens_est": round(sum(tokens) / len(tokens), 2) if tokens else None,
    }


def process_report(
    report_payload: dict[str, Any],
    gold_index: dict[tuple[str, int], dict[str, Any]],
    client_factory,
    retries: int,
    runs: int,
    max_workers: int,
) -> dict[str, Any]:
    rows = report_payload.get("results") or []
    judged_rows: list[dict[str, Any] | None] = [None] * len(rows)
    client_local = threading.local()

    def get_client() -> ChatClient:
        client = getattr(client_local, "client", None)
        if client is None:
            client = client_factory()
            client_local.client = client
        return client

    def judge_single_row(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        key = (str(row["persona"]), int(row["question_index"]))
        gold_row = gold_index[key]
        judgments = [
            judge_answer(
                get_client(),
                gold_row=gold_row,
                predicted_answer=str(row.get("predicted_answer") or ""),
                retries=retries,
            )
            for _ in range(max(1, runs))
        ]
        judge = aggregate_row_judgments(judgments)
        updated = dict(row)
        updated["judge_verdict"] = judge["judge_verdict"]
        updated["judge_score"] = judge["judge_score"]
        updated["judge_std"] = judge["judge_std"]
        updated["judge_scores"] = judge["judge_scores"]
        updated["judge_verdicts"] = judge["judge_verdicts"]
        updated["judge_rationales"] = judge["judge_rationales"]
        updated["judge_rationale"] = judge["judge_rationale"]
        updated["judge_valid_runs"] = judge["judge_valid_runs"]
        return index, updated

    if max_workers <= 1 or len(rows) <= 1:
        for index, row in enumerate(rows):
            _, updated = judge_single_row(index, row)
            judged_rows[index] = updated
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(rows))) as executor:
            futures = [executor.submit(judge_single_row, index, row) for index, row in enumerate(rows)]
            for completed, future in enumerate(as_completed(futures), start=1):
                index, updated = future.result()
                judged_rows[index] = updated
                if completed % 10 == 0 or completed == len(futures):
                    print(f"judged {completed}/{len(futures)} rows", flush=True)

    finalized_rows = [row for row in judged_rows if row is not None]
    payload = dict(report_payload)
    payload["results"] = finalized_rows
    payload["summary"] = dict(payload.get("summary") or {})
    payload["summary"].update(summarize_rows(finalized_rows))
    payload["gold_ref_judge_summary"] = summarize_rows(finalized_rows)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gold-reference LLM-as-judge for a GVD report.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_json(args.gold)
    report = load_json(args.report)
    gold_index = build_gold_index(gold)
    judged = process_report(
        report,
        gold_index,
        lambda: load_judge_client(args.config),
        args.retries,
        args.runs,
        args.max_workers,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": judged["gold_ref_judge_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
