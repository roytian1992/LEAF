from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.clients import ChatClient, extract_json_object  # noqa: E402
from leaf.config import ModelConfig  # noqa: E402


def load_memory_bank(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("memory bank must be a JSON object keyed by persona")
    return payload


def load_report(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"report missing results list: {path}")
    return payload


def load_judge_client(config_path: str | Path) -> ChatClient:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
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


def render_history(persona_name: str, persona_payload: dict[str, Any]) -> str:
    history = persona_payload.get("history") or {}
    lines: list[str] = []
    for session_id in sorted(history.keys()):
        rows = history.get(session_id) or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            response = str(row.get("response") or "").strip()
            if query:
                lines.append(f"[{session_id}] {persona_name}: {query}")
            if response:
                lines.append(f"[{session_id}] AI Companion: {response}")
    return "\n".join(lines)


def build_judge_messages(*, persona_name: str, question: str, answer: str, full_history_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating a GVD memory-question answer against the full dialogue history. "
                "Use only the provided history as ground truth. "
                "Return JSON only with keys: retrieval_accuracy, response_correctness, contextual_coherence, rationale. "
                "retrieval_accuracy must be 0 or 1. "
                "response_correctness must be one of 0, 0.5, 1. "
                "contextual_coherence must be one of 0, 0.5, 1. "
                "Be strict: if the answer contradicts the history, uses the wrong fact, or says UNKNOWN when the history clearly supports an answer, score it low."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Persona: {persona_name}\n"
                f"Question: {question}\n"
                f"Candidate answer: {answer}\n\n"
                "Scoring rules:\n"
                "- retrieval_accuracy = 1 only if the answer is grounded in the correct memory from the history.\n"
                "- response_correctness = 1 for fully correct, 0.5 for partially correct / incomplete, 0 for wrong or unsupported.\n"
                "- contextual_coherence = 1 for a direct natural answer, 0.5 for understandable but awkward / evasive, 0 for incoherent or non-responsive.\n\n"
                f"Full conversation history:\n{full_history_text}\n\n"
                'Return JSON like {"retrieval_accuracy":1,"response_correctness":0.5,"contextual_coherence":1,"rationale":"..."}'
            ),
        },
    ]


def judge_row(client: ChatClient, *, persona_name: str, question: str, answer: str, full_history_text: str, retries: int) -> dict[str, Any]:
    last_error = None
    for _ in range(retries):
        try:
            text = client.text(
                build_judge_messages(
                    persona_name=persona_name,
                    question=question,
                    answer=answer,
                    full_history_text=full_history_text,
                ),
                max_tokens=256,
                temperature=0.0,
            ).strip()
            payload = extract_json_object(text)
            return {
                "retrieval_accuracy": float(payload["retrieval_accuracy"]),
                "response_correctness": float(payload["response_correctness"]),
                "contextual_coherence": float(payload["contextual_coherence"]),
                "rationale": str(payload.get("rationale") or ""),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    return {
        "retrieval_accuracy": None,
        "response_correctness": None,
        "contextual_coherence": None,
        "rationale": last_error or "judge_failed",
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [row["gvd_retrieval_accuracy"] for row in results if row.get("gvd_retrieval_accuracy") is not None]
    corr = [row["gvd_response_correctness"] for row in results if row.get("gvd_response_correctness") is not None]
    cohe = [row["gvd_contextual_coherence"] for row in results if row.get("gvd_contextual_coherence") is not None]
    elapsed = [float(row["elapsed_ms"]) for row in results if row.get("elapsed_ms") is not None]
    tokens = [int(row["answer_input_tokens_est"]) for row in results if row.get("answer_input_tokens_est") is not None]
    return {
        "question_count": len(results),
        "judge_count": len(acc),
        "avg_elapsed_ms": round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
        "avg_answer_input_tokens_est": round(sum(tokens) / len(tokens), 2) if tokens else None,
        "retrieval_accuracy_avg": round(sum(acc) / len(acc), 4) if acc else None,
        "response_correctness_avg": round(sum(corr) / len(corr), 4) if corr else None,
        "contextual_coherence_avg": round(sum(cohe) / len(cohe), 4) if cohe else None,
        "retrieval_accuracy_std": round(statistics.pstdev(acc), 4) if len(acc) > 1 else 0.0 if acc else None,
        "response_correctness_std": round(statistics.pstdev(corr), 4) if len(corr) > 1 else 0.0 if corr else None,
        "contextual_coherence_std": round(statistics.pstdev(cohe), 4) if len(cohe) > 1 else 0.0 if cohe else None,
    }


def process_report(
    *,
    report_path: Path,
    memory_bank: dict[str, Any],
    client: ChatClient,
    retries: int,
) -> dict[str, Any]:
    payload = load_report(report_path)
    results = payload["results"]
    history_cache: dict[str, str] = {}
    judged_results: list[dict[str, Any]] = []
    for row in results:
        persona_name = str(row["persona"])
        if persona_name not in history_cache:
            history_cache[persona_name] = render_history(persona_name, memory_bank[persona_name])
        scores = judge_row(
            client,
            persona_name=persona_name,
            question=str(row["question"]),
            answer=str(row["predicted_answer"]),
            full_history_text=history_cache[persona_name],
            retries=retries,
        )
        updated = dict(row)
        updated["gvd_retrieval_accuracy"] = scores["retrieval_accuracy"]
        updated["gvd_response_correctness"] = scores["response_correctness"]
        updated["gvd_contextual_coherence"] = scores["contextual_coherence"]
        updated["gvd_judge_rationale"] = scores["rationale"]
        judged_results.append(updated)
    payload["results"] = judged_results
    payload["gvd_judge_summary"] = summarize(judged_results)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-bank", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--leaf-report", required=True)
    parser.add_argument("--memoryos-report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    memory_bank = load_memory_bank(args.memory_bank)
    client = load_judge_client(args.config)

    reports = {
        "leaf": process_report(
            report_path=Path(args.leaf_report),
            memory_bank=memory_bank,
            client=client,
            retries=args.retries,
        ),
        "memoryos": process_report(
            report_path=Path(args.memoryos_report),
            memory_bank=memory_bank,
            client=client,
            retries=args.retries,
        ),
    }

    comparison: dict[str, Any] = {"methods": {}}
    for name, payload in reports.items():
        output_path = out_dir / f"{name}_gvd_judged.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        comparison["methods"][name] = payload["gvd_judge_summary"]

    comparison_path = out_dir / "gvd_compare_summary.json"
    with open(comparison_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
