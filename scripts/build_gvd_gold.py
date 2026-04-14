from __future__ import annotations

import argparse
import json
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


def load_config(path: str | Path) -> ChatClient:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    llm_cfg = cfg.get("llm") or {}
    model_cfg = ModelConfig(
        provider=str(llm_cfg.get("provider") or "openai"),
        model_name=str(llm_cfg["model_name"]),
        api_key=str(llm_cfg["api_key"]),
        base_url=str(llm_cfg["base_url"]),
        timeout=int(llm_cfg.get("timeout", 180)),
        temperature=0.0,
        max_tokens=512,
    )
    return ChatClient(model_cfg)


def load_memory_bank(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("memory bank must be an object")
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized[str(key).strip()] = value
    return normalized


def load_questions(path: str | Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for key, value in row.items():
            if isinstance(value, list):
                results[str(key)] = [str(item) for item in value]
    return results


def render_history(persona_name: str, persona_payload: dict[str, Any]) -> str:
    history = persona_payload.get("history") or {}
    lines: list[str] = []
    for session_id in sorted(history.keys()):
        rows = history.get(session_id) or []
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            response = str(row.get("response") or "").strip()
            prefix = f"[{session_id}#{idx}]"
            if query:
                lines.append(f"{prefix} USER: {query}")
            if response:
                lines.append(f"{prefix} ASSISTANT: {response}")
    return "\n".join(lines)


def build_messages(persona_name: str, question: str, history_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are building a gold-reference sheet for a memory benchmark. "
                "Use only the provided conversation history as evidence. "
                "Return JSON only. "
                "If the question is answerable from the history, provide a concise canonical answer, a few acceptable answer variants, "
                "the supporting span ids, and key facts. "
                "If the question is not answerable, set canonical_answer to UNKNOWN and explain that in notes."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Persona: {persona_name}\n"
                f"Question: {question}\n\n"
                "Return JSON with this schema:\n"
                "{\n"
                '  "canonical_answer": "string",\n'
                '  "acceptable_answers": ["string"],\n'
                '  "required_spans": ["YYYY-MM-DD#idx"],\n'
                '  "optional_spans": ["YYYY-MM-DD#idx"],\n'
                '  "key_facts": ["string"],\n'
                '  "notes": "string"\n'
                "}\n\n"
                "Requirements:\n"
                "- canonical_answer should be as short and direct as possible.\n"
                "- acceptable_answers should include short paraphrases or aliases, not long essays.\n"
                "- required_spans must be the minimal evidence spans that are necessary to answer the question correctly.\n"
                "- optional_spans can include extra supporting spans that help but are not strictly necessary.\n"
                "- All span ids must reference the ids shown in the history.\n"
                "- key_facts should list the atomic facts needed to judge correctness.\n"
                "- Do not invent facts not present in the history.\n\n"
                f"Full conversation history:\n{history_text}"
            ),
        },
    ]


def generate_gold_entry(
    client: ChatClient,
    *,
    persona_name: str,
    question: str,
    history_text: str,
    retries: int,
) -> dict[str, Any]:
    last_error = None
    for _ in range(retries):
        try:
            text = client.text(
                build_messages(persona_name, question, history_text),
                max_tokens=512,
                temperature=0.0,
            ).strip()
            payload = extract_json_object(text)
            return {
                "canonical_answer": str(payload.get("canonical_answer") or "").strip(),
                "acceptable_answers": [str(item).strip() for item in (payload.get("acceptable_answers") or []) if str(item).strip()],
                "required_spans": [str(item).strip() for item in (payload.get("required_spans") or []) if str(item).strip()],
                "optional_spans": [str(item).strip() for item in (payload.get("optional_spans") or []) if str(item).strip()],
                "key_facts": [str(item).strip() for item in (payload.get("key_facts") or []) if str(item).strip()],
                "notes": str(payload.get("notes") or "").strip(),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    return {
        "canonical_answer": "",
        "acceptable_answers": [],
        "required_spans": [],
        "optional_spans": [],
        "key_facts": [],
        "notes": f"ERROR: {last_error or 'generation_failed'}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GVD gold-reference drafts with Qwen.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--memory-bank", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--personas", nargs="+", default=[])
    parser.add_argument("--persona-limit", type=int, default=0)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = load_config(args.config)
    bank = load_memory_bank(args.memory_bank)
    questions = load_questions(args.questions)

    persona_names = list(args.personas) if args.personas else sorted(bank.keys())
    if args.persona_limit > 0:
        persona_names = persona_names[: args.persona_limit]

    results: list[dict[str, Any]] = []
    for persona_name in persona_names:
        persona_payload = bank[persona_name]
        history_text = render_history(persona_name, persona_payload)
        persona_questions = list(questions.get(persona_name, []))
        if args.question_limit > 0:
            persona_questions = persona_questions[: args.question_limit]
        for question_index, question in enumerate(persona_questions, start=1):
            gold = generate_gold_entry(
                client,
                persona_name=persona_name,
                question=question,
                history_text=history_text,
                retries=args.retries,
            )
            results.append(
                {
                    "persona": persona_name,
                    "question_index": question_index,
                    "question": question,
                    **gold,
                }
            )

    payload = {
        "memory_bank": str(args.memory_bank),
        "questions": str(args.questions),
        "personas": persona_names,
        "question_count": len(results),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "question_count": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
