from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.answer_view import build_compact_answer_view, render_answer_view_text, summarize_answer_view
from leaf.clients import ChatClient, OpenAICompatError
from leaf.grounding import is_inference_query
from leaf.service import LEAFService

LIST_QUERY_PATTERNS = (
    "what topics",
    "which topics",
    "what books",
    "which books",
    "what suggestions",
    "what tips",
    "what places",
    "what painters",
    "what bands",
    "what projects",
    "what sports",
    "what outdoor sports",
    "what problems",
    "what issues",
    "what are some",
    "what were they",
)

SINGLE_FACT_PATTERNS = (
    "what was its name",
    "what is its name",
    "which one is it",
    "what dish",
    "what movie",
    "what exhibition",
    "what attraction",
    "what city",
    "what book",
    "what was it",
    "where did i plan to go",
    "where did i go",
    "what is my favorite",
    "what's my favorite",
)

YES_NO_PREFIXES = (
    "do ",
    "does ",
    "did ",
    "is ",
    "are ",
    "was ",
    "were ",
    "can ",
    "could ",
    "would ",
    "should ",
    "have ",
    "has ",
    "had ",
)

MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

ANSWER_STOPWORDS = {
    "a",
    "an",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "would",
    "should",
    "has",
    "have",
    "had",
    "i",
    "me",
    "my",
    "you",
    "your",
    "we",
    "our",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "about",
    "with",
    "and",
    "or",
    "that",
    "this",
    "it",
    "its",
    "their",
    "there",
    "once",
    "recently",
}

TITLE_HINT_PATTERNS = (
    "name",
    "novel",
    "book",
    "movie",
    "film",
    "song",
    "concert",
    "dish",
    "exhibition",
    "attraction",
    "city",
    "place",
    "where",
    "whose",
    "who",
)


def estimate_text_tokens(text: str) -> int:
    stripped = str(text or "")
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    total = 3
    for message in messages:
        total += 4
        total += estimate_text_tokens(message.get("role", ""))
        total += estimate_text_tokens(message.get("content", ""))
    return total


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def sanitize_persona_name(name: str) -> str:
    cleaned = "_".join(str(name).strip().split())
    return cleaned or "unknown_persona"


def has_explicit_date(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    return any(month in lowered for month in MONTH_NAMES) and bool(
        re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b", lowered)
    )


def target_month_day(question: str) -> str | None:
    lowered = str(question or "").strip().lower()
    for month_index, month in enumerate(MONTH_NAMES, start=1):
        if month not in lowered:
            continue
        day_match = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{month}\b", lowered)
        if not day_match:
            day_match = re.search(rf"\b{month}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", lowered)
        if day_match:
            return f"{month_index:02d}-{int(day_match.group(1)):02d}"
    return None


def is_list_query(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    return any(pattern in lowered for pattern in LIST_QUERY_PATTERNS)


def is_yes_no_query(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    return bool(lowered) and (lowered.startswith(YES_NO_PREFIXES) or lowered.endswith("right?") or ", right?" in lowered)


def answer_query_terms(question: str) -> set[str]:
    tokens = {
        token
        for token in re.sub(r"[^a-z0-9\s]", " ", str(question or "").lower()).split()
        if len(token) > 2 and token not in ANSWER_STOPWORDS
    }
    return tokens


def score_context_line(question: str, line: str) -> float:
    lowered_question = str(question or "").lower()
    lowered_line = str(line or "").lower()
    if not lowered_line.strip():
        return 0.0
    query_terms = answer_query_terms(question)
    line_terms = answer_query_terms(line)
    score = 0.0
    if query_terms and line_terms:
        score += len(query_terms.intersection(line_terms)) / max(1, len(query_terms))
    if has_explicit_date(question):
        date = target_month_day(question)
        if date and f"-{date}" in lowered_line:
            score += 0.7
    if any(pattern in lowered_question for pattern in TITLE_HINT_PATTERNS):
        if '"' in line or "'" in line:
            score += 0.35
        if ":" in line:
            score += 0.05
    if is_yes_no_query(question):
        if "yes" in lowered_line or "no" in lowered_line or "like" in lowered_line or "don't like" in lowered_line:
            score += 0.2
    if "favorite" in lowered_question and "favorite" in lowered_line:
        score += 0.25
    if "plan" in lowered_question and "travel" in lowered_line:
        score += 0.15
    if "problem" in lowered_question and "problem" in lowered_line:
        score += 0.15
    return score


def prioritize_answer_context(question: str, context_lines: list[str]) -> tuple[list[str], float]:
    if not context_lines:
        return [], 0.0
    scored = [
        (score_context_line(question, line), index, line)
        for index, line in enumerate(context_lines)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    if is_single_fact_query(question):
        max_lines = 4 if has_explicit_date(question) else 5
    elif has_explicit_date(question):
        max_lines = 6
    else:
        max_lines = len(context_lines)
    selected = scored[:max_lines]
    selected.sort(key=lambda item: item[1])
    top_score = scored[0][0]
    return [item[2] for item in selected], top_score


def is_single_fact_query(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    if not lowered:
        return False
    if is_list_query(lowered):
        return False
    if is_yes_no_query(lowered):
        return True
    if any(pattern in lowered for pattern in SINGLE_FACT_PATTERNS):
        return True
    return bool(re.search(r"\b(what\s+(is|was)|where|which|who|when)\b", lowered))


def load_memory_bank(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("GVD memory bank must be a JSON object keyed by persona name.")
    return payload


def load_questions(path: str | Path) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if isinstance(value, list):
                    payload[str(key).strip()] = [str(item) for item in value]
    return payload


def persona_to_turns(persona_name: str, persona_payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = persona_payload.get("history") or {}
    if not isinstance(history, dict):
        raise ValueError(f"Persona {persona_name} has invalid history format.")
    turns: list[dict[str, Any]] = []
    turn_index = 0
    for session_id in sorted(history.keys()):
        session_rows = history.get(session_id) or []
        if not isinstance(session_rows, list):
            continue
        for pair_index, row in enumerate(session_rows):
            if not isinstance(row, dict):
                continue
            user_text = str(row.get("query") or "").strip()
            assistant_text = str(row.get("response") or "").strip()
            if user_text:
                turns.append(
                    {
                        "session_id": session_id,
                        "speaker": persona_name.strip() or "user",
                        "text": user_text,
                        "timestamp": session_id,
                        "turn_index": turn_index,
                        "pair_index": pair_index,
                        "role": "user",
                    }
                )
                turn_index += 1
            if assistant_text:
                turns.append(
                    {
                        "session_id": session_id,
                        "speaker": "AI Companion",
                        "text": assistant_text,
                        "timestamp": session_id,
                        "turn_index": turn_index,
                        "pair_index": pair_index,
                        "role": "assistant",
                    }
                )
                turn_index += 1
    return turns


def span_to_canonical_id(span: dict[str, Any]) -> str | None:
    session_id = str(span.get("session_id") or span.get("timestamp") or "").strip()
    metadata = dict(span.get("metadata") or {})
    pair_index = metadata.get("pair_index")
    if not session_id or pair_index is None:
        return None
    try:
        return f"{session_id}#{int(pair_index) + 1}"
    except (TypeError, ValueError):
        return None


def build_answer_context_lines(evidence: dict[str, Any]) -> list[str]:
    raw_spans = evidence.get("raw_spans") or []
    context_lines: list[str] = []
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    for span in raw_spans:
        session_id = str(span.get("session_id") or "").strip()
        speaker = str(span.get("speaker") or "unknown").strip()
        text = str(span.get("text") or "").strip()
        timestamp = str(span.get("timestamp") or "").strip()
        if not text:
            continue
        metadata = dict(span.get("metadata") or {})
        stable_span_id = (
            str(metadata.get("original_span_id") or "").strip()
            or span_to_canonical_id(span)
            or str(span.get("span_id") or "").strip()
        )
        dedupe_key = (session_id, timestamp, speaker, stable_span_id, text)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        prefix_parts = ordered_unique([session_id, timestamp])
        prefix = f"[{' | '.join(prefix_parts)}] " if prefix_parts else ""
        context_lines.append(f"{prefix}{speaker}: {text}")
    return context_lines


def extract_session_id_from_context_line(line: str) -> str | None:
    text = str(line or "").strip()
    match = re.match(r"^\[([^\]]+)\]\s*", text)
    if not match:
        return None
    prefix = match.group(1)
    parts = [part.strip() for part in prefix.split("|") if part.strip()]
    return parts[0] if parts else None


def session_sort_key(session_id: str | None) -> tuple[int, str]:
    text = str(session_id or "").strip()
    return (0, text) if re.match(r"^\d{4}-\d{2}-\d{2}$", text) else (1, text)


def filter_context_lines_for_question(question: str, context_lines: list[str]) -> list[str]:
    if not context_lines:
        return context_lines
    lowered = str(question or "").strip().lower()
    if "first conversation" in lowered:
        session_ids = [extract_session_id_from_context_line(line) for line in context_lines]
        dated_sessions = sorted({sid for sid in session_ids if sid}, key=session_sort_key)
        if dated_sessions:
            first_session = dated_sessions[0]
            first_lines = [line for line in context_lines if extract_session_id_from_context_line(line) == first_session]
            if first_lines:
                return first_lines
    if any(token in lowered for token in ["last conversation", "most recent conversation", "latest conversation"]):
        session_ids = [extract_session_id_from_context_line(line) for line in context_lines]
        dated_sessions = sorted({sid for sid in session_ids if sid}, key=session_sort_key)
        if dated_sessions:
            last_session = dated_sessions[-1]
            last_lines = [line for line in context_lines if extract_session_id_from_context_line(line) == last_session]
            if last_lines:
                return last_lines
    return context_lines


def build_answer_messages(
    question: str,
    evidence: dict[str, Any],
    *,
    answer_style: str = "short",
    context_lines: list[str] | None = None,
    answer_view_text: str | None = None,
) -> list[dict[str, str]]:
    context_lines = context_lines if context_lines is not None else build_answer_context_lines(evidence)
    single_fact_query = is_single_fact_query(question)
    explicit_date = has_explicit_date(question)
    yes_no_query = is_yes_no_query(question)
    inference_mode = is_inference_query(question)
    month_day = target_month_day(question)
    if explicit_date and month_day is not None:
        same_day_lines = [line for line in context_lines if f"-{month_day}" in line]
        if same_day_lines:
            context_lines = same_day_lines
    if answer_style == "judge_aligned":
        context_lines = filter_context_lines_for_question(question, context_lines)
        prioritized_lines, top_support_score = prioritize_answer_context(question, context_lines)
        if prioritized_lines:
            context_lines = prioritized_lines
        context = answer_view_text if answer_view_text is not None else "\n".join(context_lines).strip()
    else:
        top_support_score = 0.0
        context = "\n".join(context_lines).strip()
    context = context.strip() or "(no retrieved evidence)"
    if answer_style == "judge_aligned" and inference_mode:
        instruction_parts = [
            "Answer the question using only the provided evidence.",
            "The evidence is organized into sectioned text lists such as Direct Evidence, Facts, Timeline, Relations, Context, and Insufficient.",
            "You may make a grounded inference when the answer is implied by multiple clues across these sections even if no raw snippet states it verbatim.",
            "Prefer the best-supported concise inference over UNKNOWN.",
            "Use brief noun phrases or short clauses, not full explanations.",
            "If the evidence truly does not support a plausible answer, answer UNKNOWN.",
            "Return only the answer phrase.",
        ]
    elif answer_style == "judge_aligned":
        instruction_parts = [
            "Answer the question using only the provided evidence.",
            "The evidence is organized into sectioned text lists such as Direct Evidence, Facts, Timeline, Relations, Context, and Insufficient.",
            "Treat Direct Evidence as the strongest grounded snippets, Facts as distilled memory statements, Timeline as time anchors, Relations as graph hints, and Context as high-level support.",
            "Prefer Direct Evidence when it directly answers the question, otherwise combine Facts with Timeline and Context.",
            "If the evidence is insufficient, answer UNKNOWN.",
            "Resolve relative time references into absolute dates or times when the evidence allows.",
            "For list, set, or comparison questions, include every supported item once, comma-separated.",
            "Use the most specific short labels supported by the evidence, not a full sentence.",
            "Do not repeat the subject names from the question.",
            "For how, method, or activity questions, return only the activity phrase such as 'dancing' or 'by dancing'.",
            "Return only the shortest answer phrase, with no explanation or preamble.",
            "Do not invent or over-specify details that are not explicitly grounded by the evidence.",
        ]
    else:
        instruction_parts = [
            "Answer the question using only the provided evidence.",
            "If the evidence is insufficient, answer UNKNOWN.",
            "Return only the answer text with no explanation or preamble.",
            "Keep the answer short and directly grounded.",
        ]
        if inference_mode:
            instruction_parts.append(
                "You may make a grounded inference only when the answer is clearly implied by multiple evidence lines."
            )
    if explicit_date:
        instruction_parts.append(
            "When the question specifies a date, prioritize evidence from that exact date and ignore nearby dates unless the answer would otherwise be unsupported."
        )
    if answer_style == "judge_aligned" and single_fact_query and not inference_mode:
        instruction_parts.append(
            "This is a single-fact question. Return the shortest directly supported answer, and do not add extra items, nearby events, or background."
        )
        if top_support_score >= 0.75:
            instruction_parts.append("The retrieved evidence contains a strong direct clue. Use it and do not reply UNKNOWN.")
        else:
            instruction_parts.append(
                "If the evidence provides a concrete clue such as a named title, place, person, date, preference, or explicit statement, answer from that clue instead of replying UNKNOWN."
            )
            instruction_parts.append(
                "Only reply UNKNOWN when the evidence lacks any concrete answer and only shows weak thematic similarity."
            )
    if yes_no_query and not inference_mode:
        instruction_parts.append(
            "For confirmation questions, answer with a direct Yes or No first, then add only the shortest supporting phrase if needed."
        )
    if answer_style == "judge_aligned" and is_list_query(question):
        instruction_parts.append(
            "This is a list-style question. Include all directly supported items once, and do not collapse them into vague categories."
        )
    if answer_style == "judge_aligned" and any(token in str(question or "").lower() for token in ["recipe", "advice", "suggestions", "problems", "topics"]) and not inference_mode:
        instruction_parts.append(
            "If the question asks for steps, advice, problems, or discussed topics, include every directly supported key point and avoid underspecified summaries like 'some difficulties'."
        )
    return [
        {
            "role": "system",
            "content": " ".join(instruction_parts),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved memory evidence:\n{context}\n\n"
                "Return only the answer text."
            ),
        },
    ]

def build_revision_messages(
    question: str,
    *,
    candidate_answer: str,
    evidence_text: str,
) -> list[dict[str, str]]:
    lowered = str(question or "").strip().lower()
    instruction_parts = [
        "Revise the candidate answer using only the provided evidence.",
        "Keep every supported fact, remove unsupported details, and do not add new details.",
        "If the candidate answer is overspecified, replace it with the shortest fully supported answer.",
        "If the asked detail is not actually stated in the evidence, answer UNKNOWN.",
        "Do not explain your reasoning.",
        "Return only the revised answer text.",
    ]
    if is_yes_no_query(question):
        instruction_parts.append("For confirmation questions, answer with Yes or No first, then the shortest support phrase if needed.")
    if is_list_query(question) or any(token in lowered for token in ["recipe", "advice", "suggestions", "problems", "topics"]):
        instruction_parts.append("For list, advice, recipe, problem, or topic questions, include all directly supported items once.")
    if has_explicit_date(question):
        instruction_parts.append("Respect the exact date in the question. Do not substitute nearby dates.")
    if "first conversation" in lowered:
        instruction_parts.append("Focus on the earliest conversation only.")
    return [
        {
            "role": "system",
            "content": " ".join(instruction_parts),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Candidate answer: {candidate_answer}\n\n"
                f"Retrieved memory evidence:\n{evidence_text}\n\n"
                "Return only the revised answer text."
            ),
        },
    ]


def build_judge_messages(persona_name: str, question: str, answer: str, full_history_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating whether an answer to a memory question is supported by the conversation history. "
                "Return JSON only with keys: verdict, score, rationale. "
                "Use score 1.0 if the answer is correct or substantially correct, otherwise 0.0."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Persona: {persona_name}\n"
                f"Question: {question}\n"
                f"Candidate answer: {answer}\n\n"
                f"Full conversation history:\n{full_history_text}\n\n"
                'Return JSON like {"verdict":"correct|incorrect","score":1.0,"rationale":"..."}'
            ),
        },
    ]


def render_full_history(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in turns:
        session_id = str(turn.get("session_id") or "").strip()
        speaker = str(turn.get("speaker") or "unknown").strip()
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{session_id}] {speaker}: {text}")
    return "\n".join(lines)


def maybe_ingest_persona(
    service: LEAFService,
    corpus_id: str,
    title: str,
    turns: list[dict[str, Any]],
    *,
    refresh: bool,
) -> dict[str, Any]:
    existing = set(service.list_corpora())
    if corpus_id in existing and not refresh:
        return {"ingested": False, "reused": True, "turn_count": len(turns)}
    if corpus_id in existing and refresh:
        raise RuntimeError(
            f"Corpus {corpus_id} already exists in the SQLite store. "
            "Use a fresh DB path for refresh runs because LEAF does not yet support corpus deletion."
        )
    result = service.append_turns(corpus_id=corpus_id, title=title, turns=turns)
    return {"ingested": True, "reused": False, "turn_count": len(turns), "result": result}


def add_numeric_maps(rows: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            total[str(key)] = int(total.get(str(key), 0)) + int(value)
    return dict(sorted(total.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LEAF on MemoryBank/SiliconFriend GVD-style probing data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--memory-bank", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--personas", nargs="+", default=[])
    parser.add_argument("--persona-limit", type=int, default=0)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--snapshot-limit", type=int, default=6)
    parser.add_argument("--raw-span-limit", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--judge-with-llm", action="store_true")
    parser.add_argument(
        "--answer-style",
        choices=["short", "judge_aligned"],
        default="short",
        help="Controls only the answer synthesis layer; retrieval/memory remains unchanged.",
    )
    parser.add_argument(
        "--answer-revision",
        choices=["none", "grounded"],
        default="none",
        help="Optional second-pass answer revision over the same retrieved evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = LEAFService(config_path=args.config, db_path=args.db)
    judge_client = ChatClient(service.config.llm) if args.judge_with_llm and service.config.llm.base_url else None
    try:
        bank = load_memory_bank(args.memory_bank)
        questions = load_questions(args.questions)
        persona_names = [name for name in bank.keys() if str(name).strip() in questions]
        persona_names.sort(key=lambda item: str(item).strip().lower())
        if args.personas:
            allowed = {str(item).strip() for item in args.personas if str(item).strip()}
            persona_names = [name for name in persona_names if str(name).strip() in allowed]
        if args.persona_limit > 0:
            persona_names = persona_names[: args.persona_limit]

        results: list[dict[str, Any]] = []
        ingest_rows: list[dict[str, Any]] = []

        for persona_name in persona_names:
            normalized_name = str(persona_name).strip()
            corpus_id = f"gvd_{sanitize_persona_name(normalized_name).lower()}"
            persona_payload = bank[persona_name]
            turns = persona_to_turns(normalized_name, persona_payload)
            full_history_text = render_full_history(turns)

            ingest_started = time.perf_counter()
            ingest_result = maybe_ingest_persona(
                service,
                corpus_id=corpus_id,
                title=f"GVD Persona {normalized_name}",
                turns=turns,
                refresh=args.refresh,
            )
            ingest_elapsed_ms = (time.perf_counter() - ingest_started) * 1000.0
            ingest_rows.append(
                {
                    "persona": normalized_name,
                    "corpus_id": corpus_id,
                    "turn_count": len(turns),
                    "ingested": ingest_result["ingested"],
                    "reused": ingest_result["reused"],
                    "ingest_elapsed_ms": round(ingest_elapsed_ms, 2),
                    "ingest_metrics": ingest_result.get("result"),
                }
            )

            persona_questions = list(questions.get(normalized_name, []))
            if args.question_limit > 0:
                persona_questions = persona_questions[: args.question_limit]

            for question_index, question in enumerate(persona_questions, start=1):
                search_started = time.perf_counter()
                evidence = service.search(
                    corpus_id=corpus_id,
                    question=question,
                    snapshot_limit=args.snapshot_limit,
                    raw_span_limit=args.raw_span_limit,
                )
                search_elapsed_ms = (time.perf_counter() - search_started) * 1000.0

                context_lines = build_answer_context_lines(evidence)
                answer_view = build_compact_answer_view(question=question, evidence=evidence)
                answer_view_text = render_answer_view_text(question=question, answer_view=answer_view)
                answer_messages = build_answer_messages(
                    question=question,
                    evidence=evidence,
                    answer_style=args.answer_style,
                    context_lines=context_lines,
                    answer_view_text=answer_view_text,
                )
                answer_input_tokens_est = estimate_message_tokens(answer_messages)
                answer_started = time.perf_counter()
                try:
                    answer_max_tokens = 96 if is_inference_query(question) else 80
                    predicted_answer = service.llm.text(answer_messages, max_tokens=answer_max_tokens, temperature=0.0).strip() if service.llm else ""
                    if (
                        args.answer_revision == "grounded"
                        and service.llm
                        and predicted_answer
                        and not predicted_answer.startswith("__ERROR__:")
                    ):
                        revision_messages = build_revision_messages(
                            question=question,
                            candidate_answer=predicted_answer,
                            evidence_text=answer_view_text,
                        )
                        answer_input_tokens_est += estimate_message_tokens(revision_messages)
                        revised_answer = service.llm.text(revision_messages, max_tokens=answer_max_tokens, temperature=0.0).strip()
                        if revised_answer:
                            predicted_answer = revised_answer
                except OpenAICompatError as exc:
                    predicted_answer = f"__ERROR__: {exc}"
                answer_elapsed_ms = (time.perf_counter() - answer_started) * 1000.0

                judge_score = None
                judge_verdict = None
                judge_rationale = None
                if judge_client is not None:
                    judge_messages = build_judge_messages(
                        persona_name=normalized_name,
                        question=question,
                        answer=predicted_answer,
                        full_history_text=full_history_text,
                    )
                    try:
                        judge_text = judge_client.text(judge_messages, max_tokens=256, temperature=0.0).strip()
                        judge_payload = json.loads(judge_text)
                        judge_score = float(judge_payload.get("score", 0.0))
                        judge_verdict = str(judge_payload.get("verdict") or "")
                        judge_rationale = str(judge_payload.get("rationale") or "")
                    except Exception as exc:  # noqa: BLE001
                        judge_verdict = "error"
                        judge_rationale = str(exc)

                results.append(
                    {
                        "persona": normalized_name,
                        "corpus_id": corpus_id,
                        "question_index": question_index,
                        "question": question,
                        "predicted_answer": predicted_answer,
                        "judge_score": judge_score,
                        "judge_verdict": judge_verdict,
                        "judge_rationale": judge_rationale,
                        "search_elapsed_ms": round(search_elapsed_ms, 2),
                        "answer_elapsed_ms": round(answer_elapsed_ms, 2),
                        "elapsed_ms": round(search_elapsed_ms + answer_elapsed_ms, 2),
                        "answer_input_tokens_est": answer_input_tokens_est,
                        "answer_style": args.answer_style,
                        "answer_revision": args.answer_revision,
                        "raw_span_count": len(evidence.get("raw_spans") or []),
                        "answer_context_line_count": len(context_lines),
                        "answer_view_summary": summarize_answer_view(answer_view),
                        "answer_view_text_chars": len(answer_view_text),
                        "retrieved_span_ids": ordered_unique(
                            [
                                span_id
                                for span_id in (span_to_canonical_id(span) for span in (evidence.get("raw_spans") or []))
                                if span_id
                            ]
                        ),
                    }
                )

        token_values = [row["answer_input_tokens_est"] for row in results]
        elapsed_values = [row["elapsed_ms"] for row in results]
        judge_values = [row["judge_score"] for row in results if row["judge_score"] is not None]
        ingest_metric_rows = [
            dict(row["ingest_metrics"])
            for row in ingest_rows
            if isinstance(row.get("ingest_metrics"), dict)
        ]
        ingest_elapsed_values = [row["ingest_elapsed_ms"] for row in ingest_rows if row["ingest_elapsed_ms"] is not None]

        summary = {
            "persona_count": len(persona_names),
            "question_count": len(results),
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
            "avg_answer_input_tokens_est": round(sum(token_values) / len(token_values), 2) if token_values else None,
            "p50_answer_input_tokens_est": int(statistics.median(token_values)) if token_values else None,
            "judge_avg": round(sum(judge_values) / len(judge_values), 4) if judge_values else None,
            "judge_count": len(judge_values),
            "ingest_reused_count": sum(1 for row in ingest_rows if row["reused"]),
            "ingest_new_count": sum(1 for row in ingest_rows if row["ingested"]),
            "ingest_avg_elapsed_ms": round(sum(ingest_elapsed_values) / len(ingest_elapsed_values), 2) if ingest_elapsed_values else None,
            "ingest_elapsed_ms_total": round(sum(ingest_elapsed_values), 2) if ingest_elapsed_values else None,
            "ingest_turn_count_total": sum(int(row.get("turn_count") or 0) for row in ingest_rows),
            "ingest_events_written_total": sum(int(row.get("events_written") or 0) for row in ingest_metric_rows),
            "ingest_atoms_written_total": sum(int(row.get("atoms_written") or 0) for row in ingest_metric_rows),
            "ingest_objects_written_total": sum(int(row.get("objects_written") or 0) for row in ingest_metric_rows),
            "ingest_state_candidates_total": sum(int(row.get("state_candidates") or 0) for row in ingest_metric_rows),
            "ingest_evidence_links_written_total": sum(int(row.get("evidence_links_written") or 0) for row in ingest_metric_rows),
            "ingest_input_text_chars_total": sum(int(row.get("input_text_chars") or 0) for row in ingest_metric_rows),
            "ingest_input_text_tokens_est_total": sum(int(row.get("input_text_tokens_est") or 0) for row in ingest_metric_rows),
            "ingest_snapshot_upserts_total": sum(int(row.get("snapshot_upserts_total") or 0) for row in ingest_metric_rows),
            "ingest_memory_llm_calls_est_total": add_numeric_maps(
                [
                    dict(row.get("memory_llm_calls_est") or {})
                    for row in ingest_metric_rows
                    if isinstance(row.get("memory_llm_calls_est"), dict)
                ]
            ),
            "ingest_state_action_counts_total": add_numeric_maps(
                [
                    dict(row.get("state_action_counts") or {})
                    for row in ingest_metric_rows
                    if isinstance(row.get("state_action_counts"), dict)
                ]
            ),
            "ingest_snapshot_upserts_by_kind_total": add_numeric_maps(
                [
                    dict(row.get("snapshot_upserts_by_kind") or {})
                    for row in ingest_metric_rows
                    if isinstance(row.get("snapshot_upserts_by_kind"), dict)
                ]
            ),
        }

        payload = {
            "memory_bank": str(args.memory_bank),
            "questions": str(args.questions),
            "db": str(args.db),
            "snapshot_limit": args.snapshot_limit,
            "raw_span_limit": args.raw_span_limit,
            "personas": list(args.personas),
            "persona_limit": args.persona_limit,
            "question_limit": args.question_limit,
            "judge_with_llm": args.judge_with_llm,
            "answer_style": args.answer_style,
            "answer_revision": args.answer_revision,
            "summary": summary,
            "ingest": ingest_rows,
            "results": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
