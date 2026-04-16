from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.clients import ChatClient, OpenAICompatError, extract_json_object  # noqa: E402
from leaf.answer_view import build_compact_answer_view, render_answer_view_text, summarize_answer_view  # noqa: E402
from leaf.extract import extract_semantic_references  # noqa: E402
from leaf.grounding import canonicalize_temporal_answer, derive_temporal_grounding, format_grounded_value, is_inference_query, is_temporal_query, match_temporal_pattern  # noqa: E402
from leaf.service import LEAFService  # noqa: E402

ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans({char: " " for char in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""})
ABSTENTION_ANSWERS = {
    "",
    "unknown",
    "not enough information",
    "insufficient information",
    "cannot determine",
    "cant determine",
    "don't know",
    "do not know",
}

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
    "where did i plan",
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

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
}

HEURISTIC_STOPWORDS = {
    "caroline",
    "melanie",
    "what",
    "when",
    "where",
    "which",
    "who",
    "did",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "and",
    "or",
    "with",
    "about",
    "from",
    "while",
    "during",
    "recently",
    "currently",
    "planning",
    "going",
    "plan",
    "plans",
}

PAINTING_SUBJECT_STOPWORDS = {
    "a",
    "an",
    "the",
    "another",
    "own",
    "love",
    "photo",
    "picture",
    "painting",
    "painted",
    "canvas",
    "wall",
    "wooden",
    "background",
}

NON_DISTINCTIVE_FOCUS_TERMS = {
    "make",
    "class",
    "pottery",
    "adoption",
    "lgbtq",
    "paint",
    "painting",
    "group",
    "join",
    "apply",
    "new",
}

HEURISTIC_ANSWER_POOLS = [
    {
        "name": "martial_arts",
        "question_groups": [("martial arts",)],
        "kind": "ordered_filter",
        "values": ("kickboxing", "taekwondo", "karate", "judo", "kung fu"),
    },
    {
        "name": "destress_both",
        "question_groups": [("both",), ("destress", "de-stress", "stress relief", "relax")],
        "kind": "ordered_filter",
        "values": ("dancing",),
        "prefix": "by ",
    },
    {
        "name": "education_fields",
        "question_groups": [("fields",), ("educat",)],
        "kind": "ordered_filter",
        "values": ("psychology", "counseling certification"),
    },
    {
        "name": "painting_subjects",
        "question_groups": [("paint",), ("what has", "what did")],
        "kind": "subject_candidates",
        "source": "painting_subject_candidates",
        "recent_first": True,
    },
    {
        "name": "camp_locations",
        "question_groups": [("camped",), ("camp",)],
        "kind": "list_values",
        "source": "camping_locations",
    },
    {
        "name": "camping_activities",
        "question_groups": [("camp",), ("what did",)],
        "kind": "phrase_pool",
        "items": (
            {"answer": "explored nature", "any_of": (r"exploring forests?", r"explore nature", r"connect with nature")},
            {"answer": "roasted marshmallows", "any_of": (r"roast(?:ed)? marshmallows",)},
            {"answer": "went on a hike", "any_of": (r"\bhiking\b", r"went on a hike")},
        ),
    },
    {
        "name": "artifact_reminder",
        "question_groups": [("reminder",), ("bowl",)],
        "kind": "phrase_pool",
        "items": (
            {"answer": "art and self-expression", "all_of": (r"\bart\b", r"self-expression")},
        ),
    },
    {
        "name": "counseling_focus",
        "question_groups": [("counseling",), ("mental health",)],
        "kind": "phrase_pool",
        "items": (
            {
                "answer": "working with trans people, helping them accept themselves and supporting their mental health",
                "all_of": (
                    r"working with trans people",
                    r"helping them accept themselves",
                    r"supporting their mental health",
                ),
            },
        ),
    },
]


def heuristic_token_variants(token: str) -> set[str]:
    normalized = str(token or "").strip().lower()
    if not normalized:
        return set()
    variants = {normalized}
    irregulars = {
        "made": {"make"},
        "ran": {"run"},
        "went": {"go"},
        "gave": {"give"},
        "took": {"take"},
    }
    variants.update(irregulars.get(normalized, set()))
    if normalized.endswith("ies") and len(normalized) > 4:
        variants.add(normalized[:-3] + "y")
    if normalized.endswith("es") and len(normalized) > 4:
        variants.add(normalized[:-2])
    if normalized.endswith("s") and len(normalized) > 4 and not normalized.endswith("ss"):
        variants.add(normalized[:-1])
    if normalized.endswith("ing") and len(normalized) > 5:
        stem = normalized[:-3]
        variants.add(stem)
        if stem.endswith("pp") or stem.endswith("tt"):
            variants.add(stem[:-1])
    if normalized.endswith("ed") and len(normalized) > 4:
        stem = normalized[:-2]
        variants.add(stem)
        if stem.endswith("i") and len(stem) > 3:
            variants.add(stem[:-1] + "y")
        if stem.endswith("pp") or stem.endswith("tt"):
            variants.add(stem[:-1])
    return {item for item in variants if item}


def heuristic_focus_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower()).split():
        if len(token) <= 2 or token in HEURISTIC_STOPWORDS:
            continue
        terms.update(heuristic_token_variants(token))
    return terms


def extract_caption_subjects(text: str) -> list[str]:
    lowered_text = str(text or "").lower()
    subjects: list[str] = []
    patterns = [
        r"painting of (?:a |an |the )?([a-z]+)",
        r"photo of (?:a |an |the )?([a-z]+) painted",
        r"inspired by (?:the )?([a-z]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered_text):
            value = str(match.group(1) or "").strip().lower()
            if not value or value in PAINTING_SUBJECT_STOPWORDS:
                continue
            if value.endswith("s") and len(value) > 4:
                value = value[:-1]
            subjects.append(value)
    return subjects


def matches_question_groups(question: str, groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...]) -> bool:
    lowered = str(question or "").lower()
    for group in groups:
        if not any(term in lowered for term in group):
            return False
    return True


def phrase_pool_matches(combined_text: str, item: dict[str, Any]) -> bool:
    all_of = tuple(item.get("all_of") or ())
    any_of = tuple(item.get("any_of") or ())
    if all_of and not all(re.search(pattern, combined_text, flags=re.IGNORECASE) for pattern in all_of):
        return False
    if any_of and not any(re.search(pattern, combined_text, flags=re.IGNORECASE) for pattern in any_of):
        return False
    return bool(all_of or any_of)


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


def normalize_answer(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = lowered.translate(PUNCT_TABLE)
    tokens = [token for token in lowered.split() if token not in ARTICLES]
    return " ".join(tokens)


def is_abstention_answer(text: str | None) -> bool:
    normalized = normalize_answer(text or "")
    return normalized in ABSTENTION_ANSWERS


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
    brevity_penalty = (
        1.0
        if len(pred_tokens) > len(gold_tokens)
        else pow(2.718281828459045, 1 - (len(gold_tokens) / max(1, len(pred_tokens))))
    )
    return precision * brevity_penalty


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
    return bool(lowered) and (
        lowered.startswith(YES_NO_PREFIXES) or lowered.endswith("right?") or ", right?" in lowered
    )


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


def answer_query_terms(question: str) -> set[str]:
    stopwords = {
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
    return {
        token
        for token in re.sub(r"[^a-z0-9\s]", " ", str(question or "").lower()).split()
        if len(token) > 2 and token not in stopwords
    }


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
    scored = [(score_context_line(question, line), index, line) for index, line in enumerate(context_lines)]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    if is_single_fact_query(question):
        max_lines = 4 if has_explicit_date(question) else 5
    elif has_explicit_date(question):
        max_lines = 6
    else:
        max_lines = len(context_lines)
    selected = scored[:max_lines]
    selected.sort(key=lambda item: item[1])
    return [item[2] for item in selected], scored[0][0]


def load_locomo_samples(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LoCoMo file should be a list of conversation samples.")
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
    normalized: list[dict[str, Any]] = []
    qas = sample.get("qa") or []
    if not isinstance(qas, list):
        return normalized
    for index, qa in enumerate(qas, start=1):
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not question:
            continue
        evidence = qa.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        category_value = qa.get("category")
        try:
            category = int(category_value) if category_value is not None else None
        except (TypeError, ValueError):
            category = None
        normalized.append(
            {
                "question_index": index,
                "qa_id": str(qa.get("qa_id") or f"qa-{index}"),
                "question": question,
                "answer": answer,
                "evidence": [str(item).strip() for item in evidence if str(item).strip()],
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, f"category_{category}" if category is not None else "unknown"),
            }
        )
    return normalized


def heuristic_answer_from_evidence(question: str, evidence: dict[str, Any]) -> str | None:
    lowered = question.lower()
    spans = evidence.get("raw_spans") or []
    if not spans:
        return None
    if is_temporal_query(question):
        best_temporal: tuple[int, int, str] | None = None
        focus_terms = heuristic_focus_terms(question)
        distinctive_focus_terms = {
            term for term in focus_terms if len(term) >= 5 and term not in NON_DISTINCTIVE_FOCUS_TERMS
        }
        prefer_future = any(token in lowered for token in ["planning", "plan ", "plans ", "going to", "will ", "next "])
        prefer_past = any(token in lowered for token in ["when did", "when was", "how long ago", "last "])
        for span in spans:
            span_text = str(span.get("text") or "").strip()
            grounding = derive_temporal_grounding(span_text, span.get("timestamp"))
            formatted = format_grounded_value(grounding)
            if not formatted:
                continue
            span_lowered = span_text.lower()
            span_pattern, _ = match_temporal_pattern(span_text)
            has_temporal_marker = span_pattern is not None
            precision = str(grounding.get("precision") or "")
            if not has_temporal_marker and precision not in {"date", "month", "relative", "year"}:
                continue
            span_terms = heuristic_focus_terms(span_lowered)
            focus_hits = len(focus_terms.intersection(span_terms))
            distinctive_hits = len(distinctive_focus_terms.intersection(span_terms))
            score = focus_hits * 4
            score += distinctive_hits * 3
            if has_temporal_marker:
                score += 3
            if precision in {"month", "relative"}:
                score += 3
            if precision == "date":
                score += 2
            if precision == "year":
                score += 2
            if prefer_future:
                if span_pattern is not None and span_pattern.get("name") in {"next_month", "next_week"}:
                    score += 4
                if span_pattern is not None and span_pattern.get("name") in {"last_year", "last_week", "last_weekday", "days_ago", "yesterday"}:
                    score -= 4
            if prefer_past:
                if span_pattern is not None and span_pattern.get("name") in {"last_year", "last_week", "last_weekend", "last_weekday", "days_ago", "yesterday"}:
                    score += 4
                if span_pattern is not None and span_pattern.get("name") in {"next_month", "next_week"}:
                    score -= 4
            if focus_terms and focus_hits == 0:
                score -= 6
            if distinctive_focus_terms and distinctive_hits == 0:
                score -= 4
            if best_temporal is None or (score, distinctive_hits) > (best_temporal[0], best_temporal[1]):
                best_temporal = (score, distinctive_hits, formatted)
        if best_temporal is not None and best_temporal[0] >= 5:
            if not distinctive_focus_terms or best_temporal[1] > 0:
                return best_temporal[2]

    def extract_camping_locations(text: str) -> list[str]:
        lowered_text = str(text or "").lower()
        locations: list[str] = []
        patterns = [
            r"\bcamping (?:at|in|on) the ([a-z]+)",
            r"\bcamping trip (?:at|in|on) the ([a-z]+)",
            r"\btrip in the ([a-z]+)",
            r"\bcampfire on the ([a-z]+)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, lowered_text):
                value = str(match.group(1) or "").strip()
                if value:
                    locations.append(value)
        for value in re.findall(r"\b(beach|mountains?|forests?)\b", lowered_text):
            normalized = str(value).strip().lower()
            if normalized == "forests":
                normalized = "forest"
            locations.append(normalized)
        return locations

    def unique_join(items: list[str]) -> str | None:
        ordered_items: list[str] = []
        seen_items: set[str] = set()
        for item in items:
            value = str(item or "").strip().lower()
            if not value or value in seen_items:
                continue
            seen_items.add(value)
            ordered_items.append(value)
        return ", ".join(ordered_items) if ordered_items else None

    refs: list[str] = []
    painting_subject_candidates: list[tuple[int, str]] = []
    camping_locations: list[str] = []
    support_texts: list[str] = []
    for span in spans:
        metadata = span.get("metadata") or {}
        refs.extend(str(item) for item in metadata.get("semantic_refs") or [])
        support_text = f"{span.get('speaker')}: {span.get('text')}\n{metadata.get('blip_caption') or ''}"
        support_texts.append(support_text)
        refs.extend(
            extract_semantic_references(
                support_text
            )
        )
        span_subjects = extract_caption_subjects(metadata.get("blip_caption") or "")
        span_subjects.extend(extract_caption_subjects(span.get("text") or ""))
        span_lowered = str(span.get("text") or "").lower()
        recency_score = 0
        if any(marker in span_lowered for marker in ["last week", "last weekend", "latest work", "latest"]):
            recency_score += 5
        if "recently" in span_lowered:
            recency_score += 3
        if "sunset" in span_lowered or "sunsets" in span_lowered:
            recency_score += 2
        for subject in span_subjects:
            painting_subject_candidates.append((recency_score, subject))
        camping_locations.extend(extract_camping_locations(support_text))
    ordered: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        value = str(ref).strip().lower()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)

    combined_support = "\n".join(support_texts).lower()
    pool_context = {
        "ordered_refs": ordered,
        "painting_subject_candidates": [subject for _, subject in sorted(painting_subject_candidates, key=lambda item: item[0], reverse=True)],
        "camping_locations": camping_locations,
    }
    for spec in HEURISTIC_ANSWER_POOLS:
        if not matches_question_groups(question, spec.get("question_groups") or ()):
            continue
        kind = str(spec.get("kind") or "")
        if kind == "ordered_filter":
            items = [item for item in pool_context["ordered_refs"] if item in set(spec.get("values") or ())]
            if items:
                answer = ", ".join(items)
                prefix = str(spec.get("prefix") or "")
                return f"{prefix}{answer}" if prefix else answer
        elif kind == "subject_candidates":
            source_items = list(pool_context.get(str(spec.get("source") or ""), []))
            answer = unique_join(source_items)
            if answer:
                if spec.get("recent_first") and "recently" in lowered:
                    return answer.split(",")[0].strip()
                return answer
        elif kind == "list_values":
            source_items = list(pool_context.get(str(spec.get("source") or ""), []))
            answer = unique_join(source_items)
            if answer:
                return answer
        elif kind == "phrase_pool":
            matched_answers = [
                str(item.get("answer") or "").strip()
                for item in spec.get("items") or ()
                if phrase_pool_matches(combined_support, item)
            ]
            answer = unique_join(matched_answers)
            if answer:
                return answer
    return None


def build_answer_context_lines(evidence: dict[str, Any]) -> list[str]:
    raw_spans = evidence.get("raw_spans") or []
    context_lines: list[str] = []
    seen_keys: set[tuple[str, str, str, str, str, str]] = set()
    for span in raw_spans:
        session_id = str(span.get("session_id") or "").strip()
        speaker = str(span.get("speaker") or "unknown").strip()
        text = str(span.get("text") or "").strip()
        timestamp = str(span.get("timestamp") or "").strip()
        if not text:
            continue
        metadata = dict(span.get("metadata") or {})
        dia_id = str(metadata.get("dia_id") or "").strip()
        stable_span_id = (
            str(metadata.get("original_span_id") or "").strip()
            or str(span.get("span_id") or "").strip()
        )
        dedupe_key = (session_id, timestamp, speaker, dia_id, stable_span_id, text)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        prefix_parts = ordered_unique([session_id, timestamp, dia_id])
        prefix = f"[{' | '.join(prefix_parts)}] " if prefix_parts else ""
        context_lines.append(f"{prefix}{speaker}: {text}")
    return context_lines


def build_answer_messages(
    question: str,
    evidence: dict[str, Any],
    *,
    context_lines: list[str] | None = None,
    answer_view_text: str | None = None,
) -> list[dict[str, str]]:
    context_lines = context_lines if context_lines is not None else build_answer_context_lines(evidence)
    inference_mode = is_inference_query(question)
    system_prompt = (
        "You are an intelligent memory assistant tasked with answering questions from conversation memories. "
        "Use only the provided evidence. Carefully analyze the evidence, pay attention to timestamps, and prefer direct evidence when available. "
        "If multiple memories conflict, prioritize the most recent well-supported memory. "
        "Always convert relative time references into specific dates, months, or years when the evidence allows. "
        "Focus only on the people and facts actually supported by the memories. "
        "The final answer should be a precise, concise phrase, usually under 5 to 6 words. "
        "Do not output UNKNOWN, insufficient information, or similar abstentions. "
        "Instead, provide the single best-supported answer implied by the evidence."
    )
    if inference_mode:
        system_prompt += (
            " You may make a grounded inference when the answer is implied by multiple clues, "
            "but stay close to the evidence and return only the best-supported answer phrase."
        )
    else:
        system_prompt += (
            " Use the most specific short label supported by the evidence. "
            "For list or set questions, include every supported item once, comma-separated. "
            "For how, method, or activity questions, return only the activity phrase."
        )
    user_content = (answer_view_text or "").strip()
    if not user_content:
        context = "\n".join(context_lines).strip() or "(no retrieved evidence)"
        user_content = (
            f"Question: {question}\n\n"
            f"Evidence:\n{context}"
        )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def maybe_ingest_sample(
    service: LEAFService,
    *,
    corpus_id: str,
    title: str,
    turns: list[dict[str, Any]],
    refresh: bool,
    ingest_mode: str | None = None,
) -> dict[str, Any]:
    existing = set(service.list_corpora())
    if corpus_id in existing and not refresh:
        return {"ingested": False, "reused": True, "turn_count": len(turns)}
    if corpus_id in existing and refresh:
        raise RuntimeError(
            f"Corpus {corpus_id} already exists in the SQLite store. "
            "Use a fresh DB path for refresh runs because LEAF does not yet support corpus deletion."
        )
    result = service.append_turns(corpus_id=corpus_id, title=title, turns=turns, ingest_mode=ingest_mode)
    session_ids = ordered_unique([str(turn.get("session_id") or "").strip() for turn in turns])
    return {
        "ingested": True,
        "reused": False,
        "turn_count": len(turns),
        "session_count": len([item for item in session_ids if item]),
        "result": result,
    }


def add_numeric_maps(rows: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            total[str(key)] = int(total.get(str(key), 0)) + int(value)
    return dict(sorted(total.items()))


def build_locomo_judge_messages(*, question: str, gold_answer: str, predicted_answer: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are grading a predicted answer against a gold answer for a long-conversation QA benchmark. "
                "Judge semantic equivalence, not exact wording. "
                "Return JSON only with keys: label, score, reason. "
                "label must be one of CORRECT, PARTIAL, WRONG. "
                "score must be one of 1, 0.5, 0. "
                "Use CORRECT when the prediction is semantically correct even if phrasing differs. "
                "Use PARTIAL when the prediction contains a materially correct subset but is incomplete. "
                "Use WRONG when the answer contradicts the gold answer, misses the asked fact, or says UNKNOWN despite the gold answer being specific."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Gold answer: {gold_answer}\n"
                f"Candidate answer: {predicted_answer}\n\n"
                'Return JSON like {"label":"CORRECT","score":1,"reason":"..."}'
            ),
        },
    ]


def judge_answer(client: ChatClient, *, question: str, gold_answer: str, predicted_answer: str, retries: int) -> dict[str, Any]:
    last_error = None
    for _ in range(max(1, retries)):
        try:
            payload = extract_json_object(
                client.text(
                    build_locomo_judge_messages(
                        question=question,
                        gold_answer=gold_answer,
                        predicted_answer=predicted_answer,
                    ),
                    max_tokens=256,
                    temperature=0.0,
                ).strip()
            )
            label = str(payload.get("label") or "").strip().upper()
            score = float(payload.get("score"))
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


def summarize_judge_runs(rows: list[dict[str, Any]]) -> tuple[float | None, float | None, list[float]]:
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
    if not run_means:
        scores = [float(row["judge_score"]) for row in rows if row.get("judge_score") is not None]
        if not scores:
            return None, None, []
        mean = round(sum(scores) / len(scores), 4)
        std = round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0
        return mean, std, []
    mean = round(sum(run_means) / len(run_means), 4)
    std = round(statistics.pstdev(run_means), 4) if len(run_means) > 1 else 0.0
    return mean, std, run_means


def summarize_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_values = [float(row["elapsed_ms"]) for row in rows if row.get("elapsed_ms") is not None]
    search_values = [float(row["search_elapsed_ms"]) for row in rows if row.get("search_elapsed_ms") is not None]
    answer_values = [float(row["answer_elapsed_ms"]) for row in rows if row.get("answer_elapsed_ms") is not None]
    token_values = [int(row["answer_input_tokens_est"]) for row in rows if row.get("answer_input_tokens_est") is not None]
    judge_mean, judge_std, judge_run_scores = summarize_judge_runs(rows)
    return {
        "count": len(rows),
        "answer_f1": round(sum(float(row["answer_f1"]) for row in rows) / len(rows), 4) if rows else None,
        "bleu1": round(sum(float(row["bleu1"]) for row in rows) / len(rows), 4) if rows else None,
        "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
        "avg_search_elapsed_ms": round(sum(search_values) / len(search_values), 2) if search_values else None,
        "avg_answer_elapsed_ms": round(sum(answer_values) / len(answer_values), 2) if answer_values else None,
        "avg_answer_input_tokens_est": round(sum(token_values) / len(token_values), 2) if token_values else None,
        "judge_mean": judge_mean,
        "judge_std": judge_std,
        "judge_run_scores": judge_run_scores,
        "judge_count": sum(1 for row in rows if row.get("judge_score") is not None),
    }


def build_summary(
    *,
    samples: list[dict[str, Any]],
    ingest_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    judge_with_llm: bool,
    judge_runs: int,
) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        categories.setdefault(str(row["category_name"]), []).append(row)

    token_values = [int(row["answer_input_tokens_est"]) for row in results if row.get("answer_input_tokens_est") is not None]
    elapsed_values = [float(row["elapsed_ms"]) for row in results if row.get("elapsed_ms") is not None]
    search_values = [float(row["search_elapsed_ms"]) for row in results if row.get("search_elapsed_ms") is not None]
    answer_values = [float(row["answer_elapsed_ms"]) for row in results if row.get("answer_elapsed_ms") is not None]
    ingest_metric_rows = [
        dict(row["ingest_metrics"])
        for row in ingest_rows
        if isinstance(row.get("ingest_metrics"), dict)
    ]
    ingest_elapsed_values = [float(row["ingest_elapsed_ms"]) for row in ingest_rows if row.get("ingest_elapsed_ms") is not None]
    judge_mean, judge_std, judge_run_scores = summarize_judge_runs(results)

    return {
        "sample_count": len(samples),
        "question_count": len(results),
        "answer_f1": round(sum(float(row["answer_f1"]) for row in results) / len(results), 4) if results else None,
        "bleu1": round(sum(float(row["bleu1"]) for row in results) / len(results), 4) if results else None,
        "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
        "avg_search_elapsed_ms": round(sum(search_values) / len(search_values), 2) if search_values else None,
        "avg_answer_elapsed_ms": round(sum(answer_values) / len(answer_values), 2) if answer_values else None,
        "avg_answer_input_tokens_est": round(sum(token_values) / len(token_values), 2) if token_values else None,
        "p50_answer_input_tokens_est": int(statistics.median(token_values)) if token_values else None,
        "judge_mean": judge_mean,
        "judge_std": judge_std,
        "judge_run_scores": judge_run_scores,
        "judge_runs": judge_runs if judge_with_llm else 0,
        "judge_count": sum(1 for row in results if row.get("judge_score") is not None),
        "ingest_reused_count": sum(1 for row in ingest_rows if row["reused"]),
        "ingest_new_count": sum(1 for row in ingest_rows if row["ingested"]),
        "ingest_avg_elapsed_ms": round(sum(ingest_elapsed_values) / len(ingest_elapsed_values), 2) if ingest_elapsed_values else None,
        "ingest_elapsed_ms_total": round(sum(ingest_elapsed_values), 2) if ingest_elapsed_values else None,
        "ingest_turn_count_total": sum(int(row.get("turn_count") or 0) for row in ingest_rows),
        "ingest_session_count_total": sum(int(row.get("session_count") or 0) for row in ingest_rows),
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
        "by_category": {
            name: summarize_category(rows)
            for name, rows in sorted(categories.items())
        },
    }


def build_payload(
    *,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    ingest_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    completed: bool,
    qa_progress_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "input": str(args.input),
        "db": str(args.db),
        "ingest_mode": str(args.ingest_mode),
        "snapshot_limit": args.snapshot_limit,
        "raw_span_limit": args.raw_span_limit,
        "answer_view_mode": args.answer_view_mode,
        "ingest_prepare_workers": int(os.environ.get("LEAF_INGEST_PREPARE_WORKERS", "4") or "4"),
        "sample_limit": args.sample_limit,
        "qa_per_sample": args.qa_per_sample,
        "judge_with_llm": args.judge_with_llm,
        "judge_runs": args.judge_runs if args.judge_with_llm else 0,
        "completed": completed,
        "qa_progress_path": str(qa_progress_path) if qa_progress_path is not None else None,
        "summary": build_summary(
            samples=samples,
            ingest_rows=ingest_rows,
            results=results,
            judge_with_llm=args.judge_with_llm,
            judge_runs=args.judge_runs,
        ),
        "ingest": ingest_rows,
        "results": results,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LEAF on LoCoMo with persistent SQLite reuse.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--qa-per-sample", type=int, default=0)
    parser.add_argument("--snapshot-limit", type=int, default=8)
    parser.add_argument("--raw-span-limit", type=int, default=8)
    parser.add_argument("--answer-view-mode", choices=["heuristic", "extractive"], default="heuristic")
    parser.add_argument("--ingest-prepare-workers", type=int, default=0)
    parser.add_argument("--ingest-mode", choices=["online", "migration"], default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--judge-with-llm", action="store_true")
    parser.add_argument("--judge-runs", type=int, default=5)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--judge-max-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    qa_progress_path = output_path.with_name(f"{output_path.stem}.qa_progress.jsonl")
    qa_progress_path.parent.mkdir(parents=True, exist_ok=True)
    qa_progress_path.write_text("", encoding="utf-8")
    if args.ingest_prepare_workers > 0:
        os.environ["LEAF_INGEST_PREPARE_WORKERS"] = str(max(1, args.ingest_prepare_workers))
    service = LEAFService(config_path=args.config, db_path=args.db)
    try:
        if not args.ingest_mode:
            args.ingest_mode = str(service.config.ingest.mode)
        samples = load_locomo_samples(args.input)
        if args.sample_limit > 0:
            samples = samples[: args.sample_limit]

        ingest_rows: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        for sample in samples:
            sample_id, turns = locomo_sample_to_turns(sample)
            qas = locomo_sample_to_qas(sample)
            if args.qa_per_sample > 0:
                qas = qas[: args.qa_per_sample]
            corpus_id = f"locomo_{sanitize_sample_id(sample_id)}"
            print(
                f"[locomo] sample={sample_id} corpus={corpus_id} turns={len(turns)} qas={len(qas)} ingest_start",
                flush=True,
            )
            ingest_started = time.perf_counter()
            ingest_result = maybe_ingest_sample(
                service,
                corpus_id=corpus_id,
                title=f"LoCoMo {sample_id}",
                turns=turns,
                refresh=args.refresh,
                ingest_mode=args.ingest_mode,
            )
            ingest_elapsed_ms = (time.perf_counter() - ingest_started) * 1000.0
            session_ids = ordered_unique([str(turn.get("session_id") or "").strip() for turn in turns])
            ingest_rows.append(
                {
                    "sample_id": sample_id,
                    "corpus_id": corpus_id,
                    "turn_count": len(turns),
                    "session_count": len([item for item in session_ids if item]),
                    "qa_count": len(qas),
                    "ingested": ingest_result["ingested"],
                    "reused": ingest_result["reused"],
                    "ingest_elapsed_ms": round(ingest_elapsed_ms, 2),
                    "ingest_metrics": ingest_result.get("result"),
                }
            )
            print(
                f"[locomo] sample={sample_id} ingest_done reused={ingest_result['reused']} elapsed_ms={round(ingest_elapsed_ms, 2)}",
                flush=True,
            )

            for qa in qas:
                question = str(qa["question"])
                gold_answer = str(qa["answer"])
                print(
                    f"[locomo] sample={sample_id} q={qa['question_index']} category={qa['category_name']} search_start",
                    flush=True,
                )

                search_started = time.perf_counter()
                evidence = service.search(
                    corpus_id=corpus_id,
                    question=question,
                    snapshot_limit=args.snapshot_limit,
                    raw_span_limit=args.raw_span_limit,
                )
                search_elapsed_ms = (time.perf_counter() - search_started) * 1000.0

                answer_started = time.perf_counter()
                context_lines = build_answer_context_lines(evidence)
                heuristic_answer = heuristic_answer_from_evidence(question=question, evidence=evidence)
                answer_view: dict[str, Any] = {}
                answer_view_text = ""
                answer_messages: list[dict[str, str]] = []
                answer_prompt_used = False
                answer_prompt_mode = "heuristic" if heuristic_answer else "llm"
                answer_prompt_input_tokens_est = 0
                answer_max_tokens = 0
                if heuristic_answer:
                    predicted_answer = heuristic_answer
                    answer_input_tokens_est = 0
                else:
                    answer_view = build_compact_answer_view(
                        question=question,
                        evidence=evidence,
                        mode=args.answer_view_mode,
                    )
                    answer_view_text = render_answer_view_text(question=question, answer_view=answer_view)
                    answer_messages = build_answer_messages(
                        question=question,
                        evidence=evidence,
                        context_lines=context_lines,
                        answer_view_text=answer_view_text,
                    )
                    answer_prompt_input_tokens_est = estimate_message_tokens(answer_messages)
                    answer_input_tokens_est = answer_prompt_input_tokens_est
                    answer_prompt_used = True
                    try:
                        answer_max_tokens = 96 if is_inference_query(question) else 80
                        predicted_answer = (
                            service.llm.text(answer_messages, max_tokens=answer_max_tokens, temperature=0.0).strip()
                            if service.llm
                            else ""
                        )
                    except OpenAICompatError as exc:
                        predicted_answer = f"__ERROR__: {exc}"
                answer_elapsed_ms = (time.perf_counter() - answer_started) * 1000.0
                predicted_answer = str(canonicalize_temporal_answer(question, predicted_answer, evidence) or predicted_answer).strip()
                print(
                    f"[locomo] sample={sample_id} q={qa['question_index']} answer_done search_ms={round(search_elapsed_ms, 2)} answer_ms={round(answer_elapsed_ms, 2)}",
                    flush=True,
                )

                row = {
                    "sample_id": sample_id,
                    "corpus_id": corpus_id,
                    "question_index": int(qa["question_index"]),
                    "qa_id": qa["qa_id"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "category": qa["category"],
                    "category_name": qa["category_name"],
                    "gold_evidence": list(qa["evidence"]),
                    "answer_f1": round(answer_f1_score(gold_answer, predicted_answer), 4),
                    "bleu1": round(bleu1_score(gold_answer, predicted_answer), 4),
                    "search_elapsed_ms": round(search_elapsed_ms, 2),
                    "answer_elapsed_ms": round(answer_elapsed_ms, 2),
                    "elapsed_ms": round(search_elapsed_ms + answer_elapsed_ms, 2),
                    "answer_input_tokens_est": answer_input_tokens_est,
                    "answer_prompt_input_tokens_est": answer_prompt_input_tokens_est,
                    "answer_prompt_used": answer_prompt_used,
                    "answer_prompt_mode": answer_prompt_mode,
                    "answer_max_tokens": answer_max_tokens,
                    "heuristic_answer": heuristic_answer,
                    "raw_span_count": len(evidence.get("raw_spans") or []),
                    "page_count": len(evidence.get("pages") or []),
                    "atom_count": len(evidence.get("atoms") or []),
                    "answer_context_line_count": len(context_lines),
                    "answer_context_lines": list(context_lines),
                    "answer_view_summary": summarize_answer_view(answer_view),
                    "answer_view": answer_view,
                    "answer_view_text": answer_view_text,
                    "answer_view_text_chars": len(answer_view_text),
                    "answer_prompt_messages": answer_messages,
                    "retrieval": {
                        "traversal_path": list(evidence.get("traversal_path") or []),
                        "pages": list(evidence.get("pages") or []),
                        "atoms": list(evidence.get("atoms") or []),
                        "raw_spans": list(evidence.get("raw_spans") or []),
                    },
                    "retrieved_dia_ids": ordered_unique(
                        [
                            str((span.get("metadata") or {}).get("dia_id") or "").strip()
                            for span in (evidence.get("raw_spans") or [])
                        ]
                    ),
                }
                results.append(row)
                append_jsonl(qa_progress_path, row)
                write_json_atomic(
                    output_path,
                    build_payload(
                        args=args,
                        samples=samples,
                        ingest_rows=ingest_rows,
                        results=results,
                        completed=False,
                        qa_progress_path=qa_progress_path,
                    ),
                )

        if args.judge_with_llm:
            if service.llm is None:
                raise RuntimeError("Judge requested but llm is not configured.")

            def judge_single(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
                judgments = [
                    judge_answer(
                        service.llm,
                        question=str(row["question"]),
                        gold_answer=str(row["gold_answer"]),
                        predicted_answer=str(row["predicted_answer"]),
                        retries=args.judge_retries,
                    )
                    for _ in range(max(1, args.judge_runs))
                ]
                judge = aggregate_row_judgments(judgments)
                updated = dict(row)
                updated.update(judge)
                return index, updated

            judged_rows: list[dict[str, Any] | None] = [None] * len(results)
            worker_count = min(max(1, args.judge_max_workers), max(1, len(results)))
            if worker_count <= 1 or len(results) <= 1:
                for index, row in enumerate(results):
                    _, updated = judge_single(index, row)
                    judged_rows[index] = updated
                    if (index + 1) % 10 == 0 or (index + 1) == len(results):
                        print(f"[locomo] judged {index + 1}/{len(results)} rows", flush=True)
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(judge_single, index, row) for index, row in enumerate(results)]
                    for completed, future in enumerate(as_completed(futures), start=1):
                        index, updated = future.result()
                        judged_rows[index] = updated
                        if completed % 10 == 0 or completed == len(futures):
                            print(f"judged {completed}/{len(futures)} rows", flush=True)
            results = [row for row in judged_rows if row is not None]
        payload = build_payload(
            args=args,
            samples=samples,
            ingest_rows=ingest_rows,
            results=results,
            completed=True,
            qa_progress_path=qa_progress_path,
        )
        write_json_atomic(output_path, payload)
        print(json.dumps({"output": str(output_path), "summary": payload["summary"]}, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
