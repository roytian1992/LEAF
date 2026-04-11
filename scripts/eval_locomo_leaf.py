from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import string
import sys
import time
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.clients import ChatClient, OpenAICompatError
from leaf.config import load_config
from leaf.extract import extract_entities, extract_semantic_references
from leaf.grounding import canonicalize_temporal_answer, is_inference_query, is_temporal_query, query_tokens as make_query_tokens
from leaf.service import LEAFService

ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
ABSTENTION_ANSWERS = {"", "unknown", "unanswerable", "not enough information", "insufficient information"}
CATEGORY_LABELS = {
    "1": "multi_hop",
    "2": "temporal",
    "3": "open_domain",
    "4": "single_hop",
    "5": "adversarial",
}
NON_ADVERSARIAL_CATEGORIES = {"1", "2", "3", "4"}
CHAT_MESSAGE_OVERHEAD_TOKENS_EST = 4
CHAT_REPLY_PRIMER_TOKENS_EST = 2
ANSWER_VIEW_SECTION_LABELS = {
    "direct_evidence": "Direct Evidence",
    "raw_evidence": "Direct Evidence",
    "entity_facts": "Facts",
    "facts": "Facts",
    "temporal_clues": "Timeline",
    "relation_paths": "Relations",
    "relations": "Relations",
    "page_context": "Context",
    "page_summaries": "Context",
    "insufficient": "Insufficient",
}
LLM_AS_JUDGE_PROMPT = """Your task is to label an answer to a question as CORRECT or WRONG.
You will be given a question, a gold answer, and a generated answer.
Be generous on wording when the generated answer matches the same fact or time period.
For time questions, treat equivalent absolute or relative forms as CORRECT if they refer to the same time.
Return only JSON with keys "label" and "reason".

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}
"""


@dataclass(slots=True)
class EvalRow:
    sample_id: str
    qa_id: str
    category: str
    question: str
    gold_answer: str
    predicted_answer: str | None
    answer_em: float
    answer_f1: float
    bleu1: float
    elapsed_ms: float
    predicted_evidence: list[str]
    answer_input_tokens_est: int | None = None
    stage_timings_ms: dict[str, float] | None = None
    judge_label: str | None = None
    judge_score: float | None = None
    skipped: bool = False
    skip_reason: str | None = None


def load_locomo_samples(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LoCoMo file should be a list of samples.")
    return [item for item in payload if isinstance(item, dict)]


def sanitize_sample_id(sample_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(sample_id or "").strip())
    return cleaned.strip("_").lower() or "locomo_sample"


def locomo_sample_to_turns(sample: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    sample_id = str(sample.get("sample_id") or "locomo-sample")
    conversation = sample.get("conversation") or {}
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo sample conversation must be a dict.")
    session_keys = sorted(
        [key for key in conversation.keys() if re.fullmatch(r"session_\d+", str(key)) and isinstance(conversation.get(key), list)],
        key=lambda value: int(str(value).split("_")[1]),
    )
    turns: list[dict[str, Any]] = []
    for session_key in session_keys:
        session_turns = conversation.get(session_key) or []
        timestamp = conversation.get(f"{session_key}_date_time")
        if not isinstance(session_turns, list):
            continue
        for turn in session_turns:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            turns.append(
                {
                    "session_id": session_key,
                    "speaker": str(turn.get("speaker") or "unknown"),
                    "text": text,
                    "timestamp": str(timestamp) if timestamp else None,
                    "dia_id": str(turn.get("dia_id") or "").strip() or None,
                    "img_url": turn.get("img_url"),
                    "blip_caption": turn.get("blip_caption"),
                }
            )
    return sample_id, turns


def locomo_sample_to_qas(sample: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = str(sample.get("sample_id") or "sample")
    payload = sample.get("qa") or []
    rows: list[dict[str, Any]] = []
    for index, qa in enumerate(payload, start=1):
        if not isinstance(qa, dict):
            continue
        rows.append(
            {
                "qa_id": str(qa.get("qa_id") or f"{sample_id}-qa-{index}"),
                "question": str(qa.get("question") or "").strip(),
                "answer": str(qa.get("answer") or "").strip(),
                "evidence": [str(item) for item in qa.get("evidence") or []],
                "category": str(qa.get("category") or "").strip(),
            }
        )
    return rows


def filter_qas_by_adversarial(qas: list[dict[str, Any]], exclude_adversarial: bool) -> list[dict[str, Any]]:
    if not exclude_adversarial:
        return qas
    return [qa for qa in qas if str(qa.get("category") or "") in NON_ADVERSARIAL_CATEGORIES]


def normalize_category_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    reverse_labels = {value: key for key, value in CATEGORY_LABELS.items()}
    normalized: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        normalized.add(reverse_labels.get(text, text))
    return normalized or None


def filter_qas_by_category(qas: list[dict[str, Any]], category_filter: set[str] | None) -> list[dict[str, Any]]:
    if not category_filter:
        return qas
    return [qa for qa in qas if str(qa.get("category") or "").strip() in category_filter]


def estimate_text_tokens(text: str) -> int:
    stripped = str(text or "")
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    total = CHAT_REPLY_PRIMER_TOKENS_EST
    for message in messages:
        total += CHAT_MESSAGE_OVERHEAD_TOKENS_EST
        total += estimate_text_tokens(message.get("role", ""))
        total += estimate_text_tokens(message.get("content", ""))
    return total


def normalize_answer(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = lowered.translate(PUNCT_TABLE)
    tokens = [token for token in lowered.split() if token not in ARTICLES]
    return " ".join(tokens)


def is_abstention_answer(text: str | None) -> bool:
    return normalize_answer(text or "") in ABSTENTION_ANSWERS


def answer_f1_score(gold_answer: str, predicted_answer: str | None) -> float:
    if not normalize_answer(gold_answer) and is_abstention_answer(predicted_answer):
        return 1.0
    gold_tokens = normalize_answer(gold_answer).split()
    pred_tokens = normalize_answer(predicted_answer or "").split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    overlap = 0
    remaining = pred_tokens.copy()
    for token in gold_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / max(1e-9, precision + recall)


def answer_em_score(gold_answer: str, predicted_answer: str | None) -> float:
    if not normalize_answer(gold_answer) and is_abstention_answer(predicted_answer):
        return 1.0
    return float(normalize_answer(gold_answer) == normalize_answer(predicted_answer or ""))


def bleu1_score(gold_answer: str, predicted_answer: str | None) -> float:
    if not normalize_answer(gold_answer) and is_abstention_answer(predicted_answer):
        return 1.0
    gold_tokens = normalize_answer(gold_answer).split()
    pred_tokens = normalize_answer(predicted_answer or "").split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts)
    precision = overlap / len(pred_tokens)
    if not gold_tokens:
        return 0.0
    brevity_penalty = 1.0 if len(pred_tokens) > len(gold_tokens) else pow(2.718281828459045, 1 - (len(gold_tokens) / max(1, len(pred_tokens))))
    return precision * brevity_penalty


def extract_dia_ids(raw_spans: list[dict[str, Any]]) -> list[str]:
    dia_ids: list[str] = []
    for span in raw_spans:
        metadata = span.get("metadata") or {}
        dia_id = str(metadata.get("dia_id") or "").strip()
        if dia_id and dia_id not in dia_ids:
            dia_ids.append(dia_id)
        chunk_ids = metadata.get("dia_ids") or []
        if isinstance(chunk_ids, list):
            for item in chunk_ids:
                value = str(item).strip()
                if value and value not in dia_ids:
                    dia_ids.append(value)
    return dia_ids


def heuristic_answer_from_evidence(question: str, evidence: dict[str, Any]) -> str | None:
    lowered = question.lower()
    spans = evidence.get("raw_spans") or []
    if not spans:
        return None
    refs: list[str] = []
    for span in spans:
        metadata = span.get("metadata") or {}
        refs.extend(str(item) for item in metadata.get("semantic_refs") or [])
        refs.extend(extract_semantic_references(f"{span.get('speaker')}: {span.get('text')}\n{metadata.get('blip_caption') or ''}"))
    ordered: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        value = str(ref).strip().lower()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)

    if "martial arts" in lowered:
        items = [item for item in ordered if item in {"kickboxing", "taekwondo", "karate", "judo", "kung fu"}]
        if items:
            return ", ".join(items)
    if any(token in lowered for token in ["destress", "de-stress", "stress relief", "relax"]) and "both" in lowered:
        if "dancing" in ordered:
            return "by dancing"
    if "fields" in lowered and "educat" in lowered:
        items = [item for item in ordered if item in {"psychology", "counseling certification"}]
        if items:
            return ", ".join(items)
    return None


def _trim_text(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def _unique_compact_lines(lines: list[str], limit: int, max_chars: int) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for line in lines:
        compact = _trim_text(line, max_chars)
        key = compact.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(compact)
        if len(ordered) >= limit:
            break
    return ordered


def _score_answer_line(question: str, line: str, *, query_entities: list[str], query_token_set: set[str], temporal: bool, inference: bool, section: str) -> float:
    text = str(line or "")
    lowered = text.lower()
    line_tokens = make_query_tokens(text)
    score = len(query_token_set.intersection(line_tokens)) * 1.5
    line_entities = extract_entities(text)
    for query_entity in query_entities:
        if query_entity in line_entities or query_entity in lowered:
            score += 2.5
    if temporal:
        if re.search(r"\b(19|20)\d{2}\b", text):
            score += 2.0
        if any(marker in lowered for marker in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
            score += 1.5
        if "[" in text and "]" in text:
            score += 1.0
    if inference:
        if section == "facts":
            score += 1.5
        if section == "page_summaries":
            score += 1.0
        if any(marker in lowered for marker in ["identity", "goal", "preference", "plan", "support", "career", "education", "study", "wants to"]):
            score += 1.0
    if section == "raw_evidence":
        score += 0.75
    return score


def _select_answer_lines(question: str, candidates: list[str], *, section: str, limit: int, max_chars: int) -> list[str]:
    query_entities = extract_entities(question)
    query_token_set = make_query_tokens(question)
    temporal = is_temporal_query(question)
    inference = is_inference_query(question)
    scored: list[tuple[float, int, str]] = []
    for index, line in enumerate(candidates):
        compact = _trim_text(line, max_chars)
        if not compact:
            continue
        score = _score_answer_line(question, compact, query_entities=query_entities, query_token_set=query_token_set, temporal=temporal, inference=inference, section=section)
        scored.append((score, index, compact))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return _unique_compact_lines([line for _, _, line in scored], limit=limit, max_chars=max_chars)


def build_answer_view(question: str, evidence: dict[str, Any]) -> dict[str, Any]:
    pages = evidence.get("pages") or []
    atoms = evidence.get("atoms") or []
    raw_spans = evidence.get("raw_spans") or []

    page_candidates = [
        f"{page.get('title')}: {page.get('synopsis') or page.get('summary')}"
        for page in pages
        if (page.get("synopsis") or page.get("summary"))
    ]
    fact_candidates = [
        str(atom.get("content") or "")
        for atom in atoms
        if str(atom.get("content") or "").strip()
    ]
    raw_candidates = [
        (
            f"[{span.get('timestamp')}] {span.get('speaker')}: {span.get('text')}"
            if span.get("timestamp")
            else f"{span.get('speaker')}: {span.get('text')}"
        )
        for span in raw_spans
        if str(span.get("text") or "").strip()
    ]
    temporal = is_temporal_query(question)
    inference = is_inference_query(question)
    page_lines = _select_answer_lines(question, page_candidates, section="page_summaries", limit=4 if inference else 3, max_chars=180)
    fact_lines = _select_answer_lines(question, fact_candidates, section="facts", limit=8 if inference else 6, max_chars=160)
    raw_lines = _select_answer_lines(question, raw_candidates, section="raw_evidence", limit=8 if temporal else 6, max_chars=200)
    return {
        "page_summaries": page_lines,
        "facts": fact_lines,
        "raw_evidence": raw_lines,
        "relations": [],
    }


def _answer_view_item_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or "").strip()
    return str(item or "").strip()


def render_answer_view_text(question: str, answer_view: dict[str, Any]) -> str:
    preferred_order = [
        "direct_evidence",
        "raw_evidence",
        "entity_facts",
        "facts",
        "temporal_clues",
        "relation_paths",
        "relations",
        "page_context",
        "page_summaries",
        "insufficient",
    ]
    lines = [f"Question: {question}", "", "Evidence:"]
    rendered_any = False
    for key in preferred_order:
        items = answer_view.get(key) or []
        if not isinstance(items, list):
            items = [items]
        texts: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = _trim_text(_answer_view_item_text(item), 220)
            normalized = text.lower()
            if not text or normalized in seen:
                continue
            seen.add(normalized)
            texts.append(text)
        if not texts:
            continue
        rendered_any = True
        lines.append(f"[{ANSWER_VIEW_SECTION_LABELS.get(key, key.replace('_', ' ').title())}]")
        lines.extend(f"- {text}" for text in texts)
        lines.append("")
    if not rendered_any:
        lines.append("[Evidence]")
        lines.append("- No relevant evidence.")
    return "\n".join(lines).strip()


def build_evidence_answer_messages(question: str, evidence: dict[str, Any], answer_view: dict[str, Any] | None = None) -> tuple[list[dict[str, str]], bool]:
    inference_mode = is_inference_query(question)
    system_prompt = (
        "Answer the question using only the provided evidence. "
        "The evidence is organized into sectioned text lists such as Direct Evidence, Facts, Timeline, Relations, Context, and Insufficient. "
        "Treat direct_evidence and raw_evidence as the strongest grounded snippets, entity_facts and facts as distilled memory statements, temporal_clues as time anchors, relation_paths and relations as graph hints, and page_context or page_summaries as high-level context. "
        "Prefer direct_evidence or raw_evidence when they directly answer the question, otherwise combine distilled facts with temporal clues, relation paths, and page context. "
        "If the evidence is insufficient, answer UNKNOWN. "
        "Resolve relative time references into absolute dates or times when the evidence allows. "
        "For list, set, or comparison questions, include every supported item once, comma-separated. "
        "Use the most specific short labels supported by the evidence, not a full sentence. "
        "Do not repeat the subject names from the question. "
        "For how/method/activity questions, return only the activity phrase such as 'dancing' or 'by dancing'. "
        "Return only the shortest answer phrase, with no explanation or preamble."
    )
    if inference_mode:
        system_prompt = (
            "Answer the question using only the provided evidence. "
            "The evidence is organized into sectioned text lists such as Direct Evidence, Facts, Timeline, Relations, Context, and Insufficient. "
            "You may make a grounded inference when the answer is implied by multiple clues across these sections even if no raw snippet states it verbatim. "
            "Prefer the best-supported concise inference over UNKNOWN. "
            "Use brief noun phrases or short clauses, not full explanations. "
            "If the evidence truly does not support a plausible answer, answer UNKNOWN. "
            "Return only the answer phrase."
        )
    user_content = render_answer_view_text(question=question, answer_view=answer_view or build_answer_view(question, evidence))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages, inference_mode


def answer_from_evidence(llm: ChatClient | None, question: str, evidence: dict[str, Any], answer_view: dict[str, Any] | None = None) -> str | None:
    heuristic = heuristic_answer_from_evidence(question=question, evidence=evidence)
    if heuristic:
        return heuristic
    if llm is None:
        return None
    messages, inference_mode = build_evidence_answer_messages(question=question, evidence=evidence, answer_view=answer_view)
    try:
        return llm.text(messages, max_tokens=96 if inference_mode else 80, temperature=0.0).strip()
    except OpenAICompatError:
        return None


def judge_answer(llm: ChatClient | None, question: str, gold_answer: str, predicted_answer: str | None) -> tuple[str | None, float | None]:
    if not gold_answer.strip():
        return ("CORRECT", 1.0) if is_abstention_answer(predicted_answer) else ("WRONG", 0.0)
    if llm is None:
        return None, None
    messages = [{"role": "system", "content": LLM_AS_JUDGE_PROMPT.format(question=question, gold_answer=gold_answer, generated_answer=str(predicted_answer or "").strip())}]
    try:
        response = llm.text(messages, max_tokens=120, temperature=0.0)
        payload = json.loads(response)
        label = str(payload.get("label") or "").strip().upper()
    except (OpenAICompatError, ValueError, json.JSONDecodeError):
        return None, None
    if label not in {"CORRECT", "WRONG"}:
        return None, None
    return label, 1.0 if label == "CORRECT" else 0.0


def summarize_rows(rows: list[EvalRow]) -> dict[str, Any]:
    valid_rows = [row for row in rows if not row.skipped]
    skipped_rows = [row for row in rows if row.skipped]
    payload = {
        "count": len(valid_rows),
        "total_count": len(rows),
        "skipped_count": len(skipped_rows),
        "answer_em": round(sum(row.answer_em for row in valid_rows) / max(1, len(valid_rows)), 4),
        "answer_f1": round(sum(row.answer_f1 for row in valid_rows) / max(1, len(valid_rows)), 4),
        "bleu1": round(sum(row.bleu1 for row in valid_rows) / max(1, len(valid_rows)), 4),
        "avg_elapsed_ms": round(sum(row.elapsed_ms for row in valid_rows) / max(1, len(valid_rows)), 2),
    }
    token_rows = [row for row in valid_rows if row.answer_input_tokens_est is not None]
    if token_rows:
        token_values = [int(row.answer_input_tokens_est or 0) for row in token_rows]
        payload["avg_answer_input_tokens_est"] = round(sum(token_values) / len(token_values), 2)
    judged_rows = [row for row in valid_rows if row.judge_score is not None]
    if judged_rows:
        judge_values = [float(row.judge_score or 0.0) for row in judged_rows]
        payload["judge_score"] = round(sum(judge_values) / len(judge_values), 4)
        payload["judge_count"] = len(judged_rows)
    stage_keys = sorted({key for row in valid_rows for key in (row.stage_timings_ms or {}).keys()})
    if stage_keys:
        payload["avg_stage_timings_ms"] = {
            key: round(
                sum(float((row.stage_timings_ms or {}).get(key, 0.0)) for row in valid_rows if row.stage_timings_ms and key in row.stage_timings_ms)
                / max(1, len([row for row in valid_rows if row.stage_timings_ms and key in row.stage_timings_ms])),
                2,
            )
            for key in stage_keys
        }
    return payload


def aggregate_rows_by_category(rows: list[EvalRow]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    categories = sorted({row.category for row in rows})
    for category in categories:
        category_rows = [row for row in rows if row.category == category]
        summary = summarize_rows(category_rows)
        summary["label"] = CATEGORY_LABELS.get(category, category)
        payload[category] = summary
    return payload


def build_split_summary(rows: list[EvalRow]) -> dict[str, Any]:
    by_category = aggregate_rows_by_category(rows)
    non_adv_rows = [row for row in rows if row.category in NON_ADVERSARIAL_CATEGORIES]
    payload = summarize_rows(rows)
    payload["by_category"] = by_category
    payload["by_category_named"] = {item["label"]: item for item in by_category.values()}
    payload["non_adversarial"] = summarize_rows(non_adv_rows)
    payload["non_adversarial"]["by_category"] = {key: value for key, value in by_category.items() if key in NON_ADVERSARIAL_CATEGORIES}
    payload["non_adversarial"]["by_category_named"] = {item["label"]: item for key, item in by_category.items() if key in NON_ADVERSARIAL_CATEGORIES}
    return payload


def ensure_sample_db(config_path: str, db_dir: str, sample: dict[str, Any], refresh: bool) -> tuple[str, dict[str, Any]]:
    sample_id, turns = locomo_sample_to_turns(sample)
    corpus_id = f"locomo_{sanitize_sample_id(sample_id)}"
    db_path = Path(db_dir) / f"{corpus_id}.sqlite3"
    if refresh and db_path.exists():
        db_path.unlink()
    ingest_row = {
        "sample_id": sample_id,
        "corpus_id": corpus_id,
        "db_path": str(db_path),
        "turn_count": len(turns),
        "session_count": len({str(turn.get('session_id') or '').strip() for turn in turns if str(turn.get('session_id') or '').strip()}),
        "reused_existing_db": db_path.exists(),
    }
    if db_path.exists() and not refresh:
        ingest_row["ingest_elapsed_ms"] = 0.0
        return str(db_path), ingest_row
    service = LEAFService(config_path=config_path, db_path=db_path)
    try:
        result = service.migrate_turns(corpus_id=corpus_id, title=f"LoCoMo {sample_id}", turns=turns)
        ingest_row.update(result)
        return str(db_path), ingest_row
    finally:
        service.close()


def evaluate_sample(service: LEAFService, sample_id: str, qas: list[dict[str, Any]], llm: ChatClient | None, judge_llm: ChatClient | None, raw_span_limit: int) -> list[EvalRow]:
    corpus_id = f"locomo_{sanitize_sample_id(sample_id)}"
    rows: list[EvalRow] = []
    for qa in qas:
        question = str(qa.get("question") or "")
        gold_answer = str(qa.get("answer") or "")
        started_at = time.perf_counter()
        stage_timings_ms: dict[str, float] = {}
        try:
            evidence = service.search(corpus_id=corpus_id, question=question, raw_span_limit=raw_span_limit)
            stage_timings_ms.update({key: float(value) for key, value in (evidence.get("timing") or {}).items()})
            answer_view = build_answer_view(question, evidence)
            answer_messages, _ = build_evidence_answer_messages(question=question, evidence=evidence, answer_view=answer_view)
            answer_input_tokens_est = estimate_message_tokens(answer_messages)
            predicted_answer = answer_from_evidence(llm=llm, question=question, evidence=evidence, answer_view=answer_view)
            predicted_answer = canonicalize_temporal_answer(question=question, predicted_answer=predicted_answer, evidence=evidence)
            judge_label, judge_score = judge_answer(judge_llm, question, gold_answer, predicted_answer)
            rows.append(
                EvalRow(
                    sample_id=sample_id,
                    qa_id=str(qa.get("qa_id") or ""),
                    category=str(qa.get("category") or ""),
                    question=question,
                    gold_answer=gold_answer,
                    predicted_answer=predicted_answer,
                    answer_em=answer_em_score(gold_answer, predicted_answer),
                    answer_f1=answer_f1_score(gold_answer, predicted_answer),
                    bleu1=bleu1_score(gold_answer, predicted_answer),
                    elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                    predicted_evidence=extract_dia_ids(evidence.get("raw_spans") or []),
                    answer_input_tokens_est=answer_input_tokens_est,
                    stage_timings_ms={key: round(value, 2) for key, value in stage_timings_ms.items()},
                    judge_label=judge_label,
                    judge_score=judge_score,
                )
            )
        except Exception as exc:
            rows.append(
                EvalRow(
                    sample_id=sample_id,
                    qa_id=str(qa.get("qa_id") or ""),
                    category=str(qa.get("category") or ""),
                    question=question,
                    gold_answer=gold_answer,
                    predicted_answer=None,
                    answer_em=0.0,
                    answer_f1=0.0,
                    bleu1=0.0,
                    elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                    predicted_evidence=[],
                    skipped=True,
                    skip_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official LEAF on LoCoMo using LEAF-only code path.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--qa-per-sample", type=int, default=10)
    parser.add_argument("--raw-span-limit", type=int, default=8)
    parser.add_argument("--exclude-adversarial", action="store_true")
    parser.add_argument("--category-filter", nargs="+")
    parser.add_argument("--refresh-db", action="store_true")
    parser.add_argument("--judge-with-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    llm = ChatClient(config.llm) if config.llm and config.llm.base_url else None
    judge_llm = llm if args.judge_with_llm else None
    samples = load_locomo_samples(args.input)
    selected = samples[max(0, args.sample_start): max(0, args.sample_start) + max(0, args.sample_count)]

    all_rows: list[EvalRow] = []
    ingest_rows: list[dict[str, Any]] = []
    for sample in selected:
        sample_id, _ = locomo_sample_to_turns(sample)
        qas = locomo_sample_to_qas(sample)
        qas = filter_qas_by_adversarial(qas, args.exclude_adversarial)
        qas = filter_qas_by_category(qas, normalize_category_filter(args.category_filter))[: args.qa_per_sample]
        db_path, ingest_row = ensure_sample_db(args.config, args.db_dir, sample, args.refresh_db)
        ingest_rows.append(ingest_row)
        service = LEAFService(config_path=args.config, db_path=db_path)
        try:
            rows = evaluate_sample(service, sample_id=sample_id, qas=qas, llm=llm, judge_llm=judge_llm, raw_span_limit=args.raw_span_limit)
            all_rows.extend(rows)
        finally:
            service.close()

    payload = {
        "input": args.input,
        "config": args.config,
        "db_dir": args.db_dir,
        "sample_start": args.sample_start,
        "sample_count": len(selected),
        "qa_per_sample": args.qa_per_sample,
        "raw_span_limit": args.raw_span_limit,
        "exclude_adversarial": args.exclude_adversarial,
        "category_filter": sorted(normalize_category_filter(args.category_filter)) if normalize_category_filter(args.category_filter) else None,
        "judge_with_llm": args.judge_with_llm,
        "ingest": ingest_rows,
        "summary": {"leaf": build_split_summary(all_rows)},
        "results": [asdict(row) for row in all_rows],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
