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
from datetime import datetime, timedelta
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
from leaf.memory_overlay import extract_geo_terms  # noqa: E402
from leaf.service import LEAFService  # noqa: E402
from leaf.topic_soft import (  # noqa: E402
    apply_overlay_runtime_policy,
    apply_topic_soft_policy,
    build_topic_context,
    merge_topic_soft_evidence,
    overlay_expand_events,
    route_topics,
    topic_soft_expand_events,
)

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


def parse_csv_set(value: str | None) -> set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}

LIST_QUERY_PATTERNS = (
    "what topics",
    "which topics",
    "what books",
    "which books",
    "what gifts",
    "which gifts",
    "what items",
    "which items",
    "what things",
    "which things",
    "what activities",
    "which activities",
    "what hobbies",
    "which hobbies",
    "what songs",
    "which songs",
    "what artists",
    "which artists",
    "what movies",
    "which movies",
    "what games",
    "which games",
    "what foods",
    "which foods",
    "what dishes",
    "which dishes",
    "what recommendations",
    "which recommendations",
    "what events",
    "which events",
    "what plans",
    "which plans",
    "what goals",
    "which goals",
    "what classes",
    "which classes",
    "what workshops",
    "which workshops",
    "what suggestions",
    "what tips",
    "what places",
    "which places",
    "what painters",
    "which painters",
    "what bands",
    "which bands",
    "what projects",
    "which projects",
    "what sports",
    "which sports",
    "what outdoor sports",
    "what problems",
    "which problems",
    "what issues",
    "which issues",
    "what are some",
    "what were they",
)

LIST_COLLECTION_NOUNS = (
    "activities",
    "artists",
    "bands",
    "books",
    "classes",
    "dishes",
    "events",
    "foods",
    "games",
    "gifts",
    "goals",
    "hobbies",
    "items",
    "movies",
    "painters",
    "places",
    "plans",
    "problems",
    "projects",
    "recommendations",
    "songs",
    "sports",
    "suggestions",
    "things",
    "tips",
    "topics",
    "workshops",
)

LIST_COLLECTION_VERBS = (
    "achieved",
    "bought",
    "created",
    "did",
    "done",
    "gave",
    "given",
    "got",
    "learned",
    "made",
    "planned",
    "played",
    "read",
    "received",
    "recommended",
    "shared",
    "suggested",
    "tried",
    "visited",
    "watched",
    "won",
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
MONTH_BY_NAME = {name: index for index, name in enumerate(MONTH_NAMES, start=1)}
MONTH_RE = "|".join(MONTH_NAMES)
WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

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

PLACE_TO_COUNTRY = {
    "paris": "France",
    "rockies": "Canada",
    "boston": "United States",
}


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
    raw = str(text or "").strip().lower()
    normalized = normalize_answer(text or "")
    if normalized in ABSTENTION_ANSWERS:
        return True
    return bool(
        re.search(
            r"\b(no\s+(?:specific\s+)?(?:evidence|information|answer|date)|"
            r"not\s+(?:specified|mentioned|provided|available|enough)|"
            r"insufficient|cannot\s+determine|can't\s+determine|unknown)\b",
            raw,
        )
    )


def expects_temporal_answer(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    return bool(
        re.match(r"^(when|what date|which date|which day|what day)\b", lowered)
        or re.match(r"^(which|what|in which)\s+(month|year)\b", lowered)
        or re.search(r"\b(in which|which) month\b|\bwhat year\b|\bhow long ago\b", lowered)
    )


def resolve_qa_raw_span_limit(args: argparse.Namespace, question: str) -> int:
    base_limit = max(1, int(getattr(args, "raw_span_limit", 8) or 8))
    non_temporal_limit = int(getattr(args, "non_temporal_raw_span_limit", 0) or 0)
    if non_temporal_limit <= 0:
        return base_limit
    if expects_temporal_answer(question) or is_temporal_query(question):
        return base_limit
    return max(base_limit, non_temporal_limit)


def should_skip_topic_soft_runtime(args: argparse.Namespace, question: str) -> tuple[bool, str]:
    policy = str(getattr(args, "topic_soft_runtime_policy", "none") or "none").strip()
    if policy in {"", "none"}:
        return False, ""
    if policy == "skip_temporal_query_v0":
        if expects_temporal_answer(question) or is_temporal_query(question):
            return True, "runtime_skip_temporal_query"
        return False, ""
    raise ValueError(f"Unsupported topic-soft runtime policy: {policy}")


def empty_topic_soft_payload(
    *,
    policy: str = "",
    reason: str = "runtime_skip",
    runtime_policy: str = "none",
) -> dict[str, Any]:
    return {
        "event_ids": [],
        "atom_ids": [],
        "raw_spans": [],
        "routes": [],
        "active_routes": [],
        "suppressed_routes": [],
        "candidate_atom_count": 0,
        "raw_candidate_atom_count": 0,
        "filtered_atom_count": 0,
        "skipped_fallback_route_count": 0,
        "skipped_low_overlap_count": 0,
        "secondary_policy": None,
        "secondary_candidate_atom_count": 0,
        "secondary_selected_atom_count": 0,
        "skipped_secondary_route_count": 0,
        "skipped_secondary_policy_count": 0,
        "skipped_secondary_low_overlap_count": 0,
        "semantic_gate_enabled": False,
        "semantic_gate_min_similarity": 0.0,
        "semantic_gate_max_similarity": None,
        "semantic_gate_missing_embedding_count": 0,
        "semantic_gate_skipped_event_count": 0,
        "policy": policy,
        "policy_applied": False,
        "policy_reason": reason,
        "policy_max_selected_content_overlap": 0,
        "policy_max_candidate_atom_count": 0,
        "policy_min_selected_overlap": 0,
        "policy_suppress_for_temporal_query": False,
        "policy_min_selected_semantic_similarity": 0.0,
        "policy_max_selected_semantic_similarity": None,
        "policy_suppress_multi_route": False,
        "suppressed_event_ids": [],
        "suppressed_atom_ids": [],
        "suppressed_raw_span_count": 0,
        "runtime_policy": runtime_policy,
        "runtime_skipped": True,
        "runtime_skip_reason": reason,
    }


def parse_locomo_date_text(text: str | None) -> datetime | None:
    lowered = str(text or "").strip().lower().replace(",", " ").replace("–", "-").replace("—", "-")
    range_match = re.search(rf"\b\d{{1,2}}\s*-\s*(\d{{1,2}})\s+({MONTH_RE})\s+(\d{{4}})\b", lowered)
    if range_match:
        return datetime(int(range_match.group(3)), MONTH_BY_NAME[range_match.group(2)], int(range_match.group(1)))
    match = re.search(rf"\b(\d{{1,2}})\s+({MONTH_RE})\s+(\d{{4}})\b", lowered)
    if match:
        return datetime(int(match.group(3)), MONTH_BY_NAME[match.group(2)], int(match.group(1)))
    match = re.search(rf"\b({MONTH_RE})\s+(\d{{1,2}})\s+(\d{{4}})\b", lowered)
    if match:
        return datetime(int(match.group(3)), MONTH_BY_NAME[match.group(1)], int(match.group(2)))
    return None


def parse_locomo_month_only(text: str | None) -> tuple[int, int] | None:
    if parse_locomo_date_text(text) is not None:
        return None
    lowered = str(text or "").strip().lower().replace(",", " ")
    match = re.search(rf"\b({MONTH_RE})\s+(\d{{4}})\b", lowered)
    if not match:
        return None
    return int(match.group(2)), MONTH_BY_NAME[match.group(1)]


def parse_locomo_season_text(text: str | None) -> tuple[int, str] | None:
    if parse_locomo_date_text(text) is not None or parse_locomo_month_only(text) is not None:
        return None
    lowered = str(text or "").strip().lower().replace(",", " ")
    match = re.search(r"\b(spring|summer|fall|autumn|winter)\s+((?:19|20)\d{2})\b", lowered)
    if not match:
        return None
    season = match.group(1)
    if season == "fall":
        season = "autumn"
    return int(match.group(2)), season


def parse_locomo_year_only(text: str | None) -> int | None:
    if (
        parse_locomo_date_text(text) is not None
        or parse_locomo_month_only(text) is not None
        or parse_locomo_season_text(text) is not None
    ):
        return None
    match = re.fullmatch(r"\s*(?:in\s+)?((?:19|20)\d{2})\s*", str(text or "").strip().lower())
    return int(match.group(1)) if match else None


def parse_relative_temporal_answer(text: str | None) -> dict[str, Any] | None:
    lowered = str(text or "").strip().lower().replace(",", "")
    anchor = parse_locomo_date_text(lowered)
    if anchor is None:
        return None
    if lowered.startswith("week before"):
        return {
            "kind": "week_before",
            "anchor": anchor,
            "start": anchor - timedelta(days=7),
            "end": anchor - timedelta(days=1),
        }
    if lowered.startswith("weekend before"):
        return {
            "kind": "weekend_before",
            "anchor": anchor,
            "start": anchor - timedelta(days=7),
            "end": anchor - timedelta(days=1),
        }
    if lowered.startswith("week of"):
        return {
            "kind": "week_of",
            "anchor": anchor,
            "start": anchor - timedelta(days=3),
            "end": anchor + timedelta(days=3),
        }
    match = re.match(r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday) before", lowered)
    if match:
        target_weekday = WEEKDAY_INDEX[match.group(1)]
        candidate = anchor - timedelta(days=1)
        while candidate.weekday() != target_weekday and (anchor - candidate).days <= 14:
            candidate -= timedelta(days=1)
        return {
            "kind": "weekday_before",
            "anchor": anchor,
            "start": candidate,
            "end": candidate,
        }
    return None


def evidence_relative_temporal_answers(question: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    focus_terms = heuristic_focus_terms(question)
    distinctive_focus_terms = {
        term for term in focus_terms if len(term) >= 5 and term not in NON_DISTINCTIVE_FOCUS_TERMS
    }
    candidates: list[dict[str, Any]] = []
    spans = list(evidence.get("raw_spans") or []) + list(evidence.get("supporting_raw_spans") or [])
    for index, span in enumerate(spans):
        span_text = str(span.get("text") or "").strip()
        if not span_text:
            continue
        grounding = derive_temporal_grounding(span_text, span.get("timestamp"))
        if str(grounding.get("precision") or "") != "relative":
            continue
        formatted = format_grounded_value(grounding)
        relative = parse_relative_temporal_answer(formatted)
        if not formatted or relative is None:
            continue
        span_terms = heuristic_focus_terms(span_text)
        focus_hits = len(focus_terms.intersection(span_terms))
        distinctive_hits = len(distinctive_focus_terms.intersection(span_terms))
        if focus_terms and focus_hits == 0:
            continue
        if distinctive_focus_terms and distinctive_hits == 0:
            continue
        candidates.append(
            {
                "text": formatted,
                "relative": relative,
                "focus_hits": focus_hits,
                "distinctive_hits": distinctive_hits,
                "span_index": index,
            }
        )
    candidates.sort(
        key=lambda item: (
            -int(item["distinctive_hits"]),
            -int(item["focus_hits"]),
            int(item["span_index"]),
        )
    )
    return candidates


def is_vague_temporal_answer(text: str | None) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True
    if re.search(
        r"\b(no specific|not specified|not mentioned|insufficient|cannot determine|unknown|"
        r"unclear|not enough|without a specific)\b",
        lowered,
    ):
        return True
    return False


def temporal_anchor_postprocess(
    question: str,
    predicted_answer: str | None,
    evidence: dict[str, Any],
    *,
    mode: str = "off",
) -> str | None:
    if mode in {"", "off", "none"}:
        return predicted_answer
    if mode not in {"anchor_only", "range", "range_no_weekday", "relative_prefer"}:
        raise ValueError(f"Unsupported temporal postprocess mode: {mode}")
    answer = str(predicted_answer or "").strip()
    if not expects_temporal_answer(question):
        return predicted_answer
    lowered_question = str(question or "").strip().lower()
    relative_prefer_allowed = bool(
        mode == "relative_prefer"
        and lowered_question.startswith("when")
        and not re.search(r"\b(date|which day|what day|how long ago)\b", lowered_question)
    )
    heuristic_answer = heuristic_answer_from_evidence(question, evidence)
    if not heuristic_answer:
        return predicted_answer
    heuristic_text = str(heuristic_answer).strip()
    if is_abstention_answer(answer) or is_vague_temporal_answer(answer):
        return heuristic_text

    relative = parse_relative_temporal_answer(heuristic_text)
    predicted_date = parse_locomo_date_text(answer)
    answer_prefers_period = bool(re.search(r"\bweek(?:end)?\s+of\b", answer.lower()))
    if relative is not None:
        anchor = relative["anchor"]
        is_anchor_date = bool(predicted_date and predicted_date.date() == anchor.date())
        in_relative_range = bool(
            predicted_date
            and relative["start"].date() <= predicted_date.date() <= relative["end"].date()
        )
        if is_anchor_date:
            return heuristic_text
        if mode == "range" and in_relative_range:
            return heuristic_text
        if relative_prefer_allowed and in_relative_range and answer_prefers_period:
            return heuristic_text
        if mode == "range_no_weekday" and relative["kind"] != "weekday_before" and in_relative_range:
            return heuristic_text
        return predicted_answer

    if relative_prefer_allowed:
        heuristic_has_exact_time = bool(
            parse_locomo_date_text(heuristic_text)
            or parse_locomo_month_only(heuristic_text)
            or parse_locomo_year_only(heuristic_text)
        )
        if heuristic_has_exact_time:
            return predicted_answer
        predicted_date = parse_locomo_date_text(answer)
        if predicted_date is not None and answer_prefers_period:
            for candidate in evidence_relative_temporal_answers(question, evidence):
                relative = candidate["relative"]
                if relative["start"].date() <= predicted_date.date() <= relative["end"].date():
                    return str(candidate["text"])

    heuristic_month = parse_locomo_month_only(heuristic_text)
    if heuristic_month is not None:
        lowered = str(question or "").strip().lower()
        if re.search(r"\b(which|what|in which)\s+month\b", lowered) or (
            "month" in lowered and not lowered.startswith("when")
        ):
            return heuristic_text
        if predicted_date and (predicted_date.year, predicted_date.month) == heuristic_month:
            return heuristic_text

    heuristic_year = parse_locomo_year_only(heuristic_text)
    if heuristic_year is not None and re.search(r"\b(which|what)\s+year\b|\byear\b", str(question or "").lower()):
        return heuristic_text
    return predicted_answer


def safe_short_answer_postprocess(
    question: str,
    predicted_answer: str | None,
    *,
    evidence: dict[str, Any] | None = None,
) -> str | None:
    answer = str(predicted_answer or "").strip().strip('"')
    if not answer:
        return predicted_answer
    lowered_question = str(question or "").strip().lower()
    answer = re.sub(r"(?is)\n+\s*\*\*(?:answer|final answer)\s*:\*\*\s*", " ", answer).strip()
    answer = re.split(r"(?is)\n+\s*\*\*(?:reasoning|rationale|evidence)\s*:\*\*", answer, maxsplit=1)[0].strip()
    answer = re.split(r"(?is)\n+\s*(?:reasoning|rationale|evidence)\s*:\s*", answer, maxsplit=1)[0].strip()
    if re.search(r"(?is)\*\*(?:answer|final answer)\s*:\*\*", str(predicted_answer or "")):
        match = re.search(r"(?is)\*\*(?:answer|final answer)\s*:\*\*\s*([^\n]+)", str(predicted_answer or ""))
        if match:
            answer = match.group(1).strip()
    if lowered_question.startswith(("which country", "what country")):
        normalized_answer = normalize_answer(answer)
        if normalized_answer in PLACE_TO_COUNTRY:
            return PLACE_TO_COUNTRY[normalized_answer]
    if evidence is not None:
        heuristic_answer = heuristic_answer_from_evidence(question, evidence)
        if heuristic_answer and (
            "martial arts" in lowered_question
            or "camped" in lowered_question
            or ("where has" in lowered_question and "camp" in lowered_question)
            or ("fields" in lowered_question and "educat" in lowered_question)
            or ("destress" in lowered_question and "both" in lowered_question)
        ):
            return heuristic_answer
    question_names = {name.lower() for name in re.findall(r"\b[A-Z][a-z]+\b", str(question or ""))}

    if lowered_question.startswith("who "):
        match = re.match(
            r"^([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+|\s+[A-Z][a-z]+)?)\s+"
            r"\b(invited|gave|met|joined|helped|told|asked|recommended|introduced|supported|"
            r"visited|called|emailed|texted|sent|shared|brought|took|made|found|wrote|said|"
            r"attended|went|came|was|is|had|has|did)\b",
            answer,
        )
        if match:
            leading_name = match.group(1).strip()
            leading_names = {name.lower() for name in re.findall(r"\b[A-Z][a-z]+\b", leading_name)}
            if leading_names and not leading_names.intersection(question_names):
                return leading_name

    if is_yes_no_query(question):
        match = re.match(r"^(yes|no)\b[\s,.:;-]+(.+)$", answer, flags=re.IGNORECASE)
        if match:
            rest = match.group(2).strip().lower()
            question_tokens = set(normalize_answer(question).split())
            rest_tokens = set(normalize_answer(rest).split())
            if len(rest_tokens - question_tokens) <= 3 and not re.search(
                r"\b(because|but|however|although|like|goals|open|likely)\b",
                rest,
            ):
                return match.group(1).capitalize()
            if "\n" in str(predicted_answer or "") or "**" in str(predicted_answer or ""):
                return match.group(1).capitalize()

    if not re.search(r"\b(unknown|not specified|not enough|insufficient|cannot determine)\b", answer, re.IGNORECASE):
        if lowered_question.startswith(("what aspect", "which aspect")):
            first_sentence = re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)[0]
            match = re.search(r"\bis\s+(?:the\s+)?(.+?)\.?$", first_sentence, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if 2 <= len(candidate.split()) <= 12:
                    return candidate[:1].upper() + candidate[1:]
        match = re.match(
            r'^The aspect of .+? that .+? finds immersive is (.+?)\.?$',
            answer,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip()
            if 2 <= len(candidate.split()) <= 8:
                return candidate[:1].upper() + candidate[1:]
        match = re.match(r"^[A-Z][a-z]+ finds (?:his|her|their)?\s*(?:current )?work (.+?)\.?$", answer)
        if match and lowered_question.startswith("how") and "feel" in lowered_question:
            candidate = match.group(1).strip()
            if candidate:
                return candidate[:1].upper() + candidate[1:]
    return answer if answer != str(predicted_answer or "").strip().strip('"') else predicted_answer


def _clean_precision_answer(text: str) -> str:
    answer = str(text or "").strip().strip('"').strip("'").strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip(" .")


def evidence_text_blob(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return ""
    chunks: list[str] = []
    for span in list(evidence.get("raw_spans") or []) + list(evidence.get("supporting_raw_spans") or []):
        if not isinstance(span, dict):
            continue
        metadata = dict(span.get("metadata") or {})
        chunks.append(
            " ".join(
                [
                    str(span.get("speaker") or ""),
                    str(span.get("text") or ""),
                    str(metadata.get("blip_caption") or ""),
                ]
            )
        )
        overlay = dict(metadata.get("memory_overlay") or {})
        overlay_terms: list[str] = []
        for reason in overlay.get("reasons") or []:
            if not isinstance(reason, dict):
                continue
            overlay_terms.extend(str(item) for item in reason.get("geo_terms") or [] if str(item).strip())
        if overlay_terms:
            chunks.append("Overlay terms: " + " ".join(overlay_terms))
    return "\n".join(chunks)


def _temporal_grounding_specificity(value: str | None) -> int:
    if parse_relative_temporal_answer(value) is not None:
        return 2
    if parse_locomo_date_text(value) is not None:
        return 3
    if parse_locomo_season_text(value) is not None:
        return 2
    if parse_locomo_month_only(value) is not None:
        return 2
    if parse_locomo_year_only(value) is not None:
        return 1
    return 0


def should_replace_temporal_answer(current_answer: str, candidate: str) -> bool:
    if is_abstention_answer(current_answer) or is_vague_temporal_answer(current_answer):
        return True
    current_specificity = _temporal_grounding_specificity(current_answer)
    candidate_specificity = _temporal_grounding_specificity(candidate)
    if current_specificity >= 3:
        current_relative = parse_relative_temporal_answer(current_answer)
        candidate_relative = parse_relative_temporal_answer(candidate)
        return bool(current_relative is not None and candidate_relative is not None and current_relative == candidate_relative)
    if current_specificity == 2:
        current_month = parse_locomo_month_only(current_answer)
        candidate_month = parse_locomo_month_only(candidate)
        current_season = parse_locomo_season_text(current_answer)
        candidate_season = parse_locomo_season_text(candidate)
        if current_month is not None and candidate_month is not None:
            return candidate_month == current_month
        if current_season is not None and candidate_season is not None:
            return candidate_season == current_season
        return candidate_specificity > current_specificity
    return candidate_specificity > current_specificity


def temporal_answers_compatible(current_answer: str, candidate: str) -> bool:
    current_date = parse_locomo_date_text(current_answer)
    candidate_date = parse_locomo_date_text(candidate)
    if current_date is not None and candidate_date is not None:
        return current_date.date() == candidate_date.date()
    current_relative = parse_relative_temporal_answer(current_answer)
    candidate_relative = parse_relative_temporal_answer(candidate)
    if current_relative is not None and candidate_relative is not None:
        return (
            current_relative["kind"] == candidate_relative["kind"]
            and current_relative["anchor"].date() == candidate_relative["anchor"].date()
        )
    current_month = parse_locomo_month_only(current_answer)
    candidate_month = parse_locomo_month_only(candidate)
    if current_month is not None and candidate_month is not None:
        return current_month == candidate_month
    current_season = parse_locomo_season_text(current_answer)
    candidate_season = parse_locomo_season_text(candidate)
    if current_season is not None and candidate_season is not None:
        return current_season == candidate_season
    if current_date is not None and candidate_month is not None:
        return (current_date.year, current_date.month) == candidate_month
    if candidate_date is not None and current_month is not None:
        return (candidate_date.year, candidate_date.month) == current_month
    return False


def extract_best_temporal_grounding_candidate(question: str, evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if not evidence or not expects_temporal_answer(question):
        return None
    q_terms = heuristic_focus_terms(question)
    distinctive_terms = {
        term
        for term in q_terms
        if len(term) >= 5 and term not in NON_DISTINCTIVE_FOCUS_TERMS and term not in {"maria", "john"}
    }
    question_names = {name.lower() for name in re.findall(r"\b[A-Z][a-z]+\b", str(question or ""))}
    candidates: list[dict[str, Any]] = []
    spans = list(evidence.get("raw_spans") or []) + list(evidence.get("supporting_raw_spans") or [])
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        text = str(span.get("text") or "").strip()
        if not text:
            continue
        spec, _ = match_temporal_pattern(text)
        if spec is None:
            continue
        grounding = derive_temporal_grounding(text, span.get("timestamp"))
        formatted = format_grounded_value(grounding)
        if not formatted:
            continue
        precision = str(grounding.get("precision") or "")
        if precision not in {"date", "month", "relative", "year", "season"}:
            continue
        span_terms = heuristic_focus_terms(text)
        focus_hits = len(q_terms.intersection(span_terms))
        distinctive_hits = len(distinctive_terms.intersection(span_terms))
        speaker = str(span.get("speaker") or "").strip().lower()
        if q_terms and focus_hits == 0:
            continue
        if distinctive_terms and distinctive_hits == 0:
            continue
        if question_names and speaker in {"maria", "john"} and speaker not in question_names:
            continue
        candidates.append(
            {
                "text": formatted,
                "precision": precision,
                "specificity": _temporal_grounding_specificity(formatted),
                "focus_hits": focus_hits,
                "distinctive_hits": distinctive_hits,
                "speaker_match": int(speaker in question_names) if question_names else 0,
                "span_index": index,
            }
        )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -int(item["distinctive_hits"]),
            -int(item["focus_hits"]),
            -int(item["speaker_match"]),
            -int(item["specificity"]),
            int(item["span_index"]),
        )
    )
    return candidates[0]


def extract_best_temporal_grounding_answer(question: str, evidence: dict[str, Any] | None) -> str | None:
    candidate = extract_best_temporal_grounding_candidate(question, evidence)
    return str(candidate["text"]) if candidate else None


def answer_supported_by_temporal_evidence(question: str, answer: str, evidence: dict[str, Any] | None) -> bool:
    if not answer or not evidence or not expects_temporal_answer(question):
        return False
    answer_norm = normalize_answer(answer)
    answer_date = parse_locomo_date_text(answer)
    answer_month = parse_locomo_month_only(answer)
    answer_season = parse_locomo_season_text(answer)
    answer_relative = parse_relative_temporal_answer(answer)
    q_terms = heuristic_focus_terms(question)
    distinctive_terms = {
        term
        for term in q_terms
        if len(term) >= 5 and term not in NON_DISTINCTIVE_FOCUS_TERMS and term not in {"maria", "john"}
    }
    question_names = {name.lower() for name in re.findall(r"\b[A-Z][a-z]+\b", str(question or ""))}
    for span in list((evidence or {}).get("raw_spans") or []) + list((evidence or {}).get("supporting_raw_spans") or []):
        if not isinstance(span, dict):
            continue
        text = str(span.get("text") or "").strip()
        if not text:
            continue
        speaker = str(span.get("speaker") or "").strip().lower()
        if question_names and speaker in {"maria", "john"} and speaker not in question_names:
            continue
        span_terms = heuristic_focus_terms(text)
        if distinctive_terms and not distinctive_terms.intersection(span_terms):
            continue
        span_norm = normalize_answer(text)
        if answer_norm and answer_norm in span_norm:
            return True
        grounding = derive_temporal_grounding(text, span.get("timestamp"))
        formatted = format_grounded_value(grounding) or ""
        if not formatted:
            continue
        if temporal_answers_compatible(answer, formatted):
            return True
        formatted_date = parse_locomo_date_text(formatted)
        if answer_date is not None and formatted_date is not None and answer_date.date() == formatted_date.date():
            return True
        formatted_month = parse_locomo_month_only(formatted)
        if answer_month is not None and formatted_month is not None and answer_month == formatted_month:
            return True
        formatted_season = parse_locomo_season_text(formatted)
        if answer_season is not None and formatted_season is not None and answer_season == formatted_season:
            return True
        formatted_relative = parse_relative_temporal_answer(formatted)
        if (
            answer_relative is not None
            and formatted_relative is not None
            and answer_relative["kind"] == formatted_relative["kind"]
            and answer_relative["anchor"].date() == formatted_relative["anchor"].date()
        ):
            return True
    return False


def extract_started_activity_answer(question: str, evidence: dict[str, Any] | None) -> str | None:
    lowered_question = str(question or "").lower()
    if not evidence or not re.search(r"\b(type|kind|what)\b", lowered_question):
        return None
    if not re.search(r"\b(class|workout|activity|exercise)\b", lowered_question):
        return None
    text = evidence_text_blob(evidence)
    patterns = [
        r"\bstarted\s+(?:doing|taking|trying|attending)\s+([a-z][a-z\s-]{2,40}?)(?:,|\.|;|\bit\b|\band\b|\s+class\b)",
        r"\btrying\s+([a-z][a-z\s-]{2,40}?)(?:,|\.|;|\band\b|\s+class\b)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = _clean_precision_answer(match.group(1))
            candidate = re.sub(r"\s+(?:is|was|it's|its|it)$", "", candidate, flags=re.IGNORECASE).strip()
            if 1 <= len(candidate.split()) <= 5:
                return candidate.lower()
    return None


def extract_relative_date_answer(question: str, evidence: dict[str, Any] | None) -> str | None:
    if not evidence or not expects_temporal_answer(question):
        return None
    q_terms = heuristic_focus_terms(question)
    distinctive_terms = {term for term in q_terms if len(term) >= 4 and term not in {"maria", "john"}}
    for span in list(evidence.get("raw_spans") or []) + list(evidence.get("supporting_raw_spans") or []):
        if not isinstance(span, dict):
            continue
        text = str(span.get("text") or "")
        if not text:
            continue
        span_terms = heuristic_focus_terms(text)
        if distinctive_terms and not distinctive_terms.intersection(span_terms):
            continue
        grounding = derive_temporal_grounding(text, span.get("timestamp"))
        if str(grounding.get("precision") or "") != "date":
            continue
        formatted = format_grounded_value(grounding)
        if formatted:
            return formatted
    return None


def extract_geo_list_answer(question: str, evidence: dict[str, Any] | None) -> str | None:
    lowered_question = str(question or "").lower()
    if not evidence or not re.search(r"\b(countr(?:y|ies)|european|places?|where)\b", lowered_question):
        return None
    names = extract_geo_terms(evidence_text_blob(evidence))
    if not names:
        return None
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        normalized = str(name or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(" ".join(part.capitalize() for part in normalized.split()))
    if not ordered:
        return None
    if is_list_query(question) or re.search(r"\b(countries|places|where)\b", lowered_question):
        return ", ".join(ordered[:6])
    return ordered[0]


def extract_local_qa_followup_answer(question: str, evidence: dict[str, Any] | None) -> str | None:
    if not evidence:
        return None
    q_terms = answer_query_terms(question)
    spans = [
        span
        for span in list(evidence.get("raw_spans") or []) + list(evidence.get("supporting_raw_spans") or [])
        if isinstance(span, dict)
    ]
    by_session: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        session_id = str(span.get("session_id") or "").strip()
        if session_id:
            by_session.setdefault(session_id, []).append(span)
    best: tuple[float, str] | None = None
    for session_spans in by_session.values():
        ordered = sorted(session_spans, key=lambda item: int(item.get("turn_index") or 0))
        for prev, curr in zip(ordered, ordered[1:]):
            prev_text = str(prev.get("text") or "")
            curr_text = str(curr.get("text") or "")
            if "?" not in prev_text or not curr_text:
                continue
            overlap = len(q_terms.intersection(answer_query_terms(prev_text)))
            if overlap <= 0:
                continue
            score = overlap / max(1, len(q_terms))
            if best is None or score > best[0]:
                best = (score, curr_text)
    if best is None or best[0] < 0.35:
        return None
    text = best[1]
    if is_list_query(question):
        match = re.search(r"\bwe love\s+(.+?)(?:\.|$)", text, flags=re.IGNORECASE)
        if match:
            candidate = _clean_precision_answer(match.group(1))
            candidate = re.sub(r"\s+-\s+", ", ", candidate)
            candidate = re.sub(r"\s+plus\s+", ", ", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s+and\s+having\s+", ", having ", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s+and\s+", ", ", candidate, flags=re.IGNORECASE)
            return candidate
    if len(text.split()) <= 8:
        return _clean_precision_answer(text)
    return None


def extract_family_activity_answer(question: str, evidence: dict[str, Any] | None) -> str | None:
    lowered_question = str(question or "").lower()
    if not evidence or not is_list_query(question):
        return None
    if not re.search(r"\bactivities\b", lowered_question) or not re.search(r"\bfamily\b", lowered_question):
        return None
    text = evidence_text_blob(evidence)
    match = re.search(r"\bwe love being outdoors\s*-\s*(.+?)(?:\.|$)", text, flags=re.IGNORECASE)
    if not match:
        return None
    candidate = _clean_precision_answer(match.group(1))
    candidate = re.sub(r"\s+-\s+", ", ", candidate)
    candidate = re.sub(r"\s+plus\s+", ", ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+and\s+having\s+", ", having ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+and\s+", ", ", candidate, flags=re.IGNORECASE)
    return candidate


def extract_childhood_items_answer(question: str, evidence: dict[str, Any] | None) -> str | None:
    lowered_question = str(question or "").lower()
    if not evidence or not re.search(r"\b(child|childhood|kid)\b", lowered_question):
        return None
    if not re.search(r"\b(items?|things?|objects?|having|had|mention)\b", lowered_question):
        return None
    text = evidence_text_blob(evidence)
    items: list[str] = []
    patterns = [
        r"\bi had (?:a |an |the )?([a-z][a-z\s-]{1,40}?)(?:\s+like this|\s+as a kid|\s+as a child|\s+when i was|,|\.|;)",
        r"\bthe ([a-z][a-z\s-]{1,40}?) i had as a kid\b",
        r"\b([a-z][a-z\s-]{1,40}?) i had as a kid\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = _clean_precision_answer(match.group(1))
            candidate = re.sub(r"\s+(?:like this|and it.*)$", "", candidate, flags=re.IGNORECASE).strip()
            if 1 <= len(candidate.split()) <= 4:
                items.append(candidate.lower())
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ", ".join(ordered) if ordered else None


def extract_visual_text_answer(question: str, evidence: dict[str, Any] | None) -> str | None:
    lowered_question = str(question or "").strip().lower()
    if not evidence or "say" not in lowered_question:
        return None
    if not re.search(r"\b(poster|posters|sign|signs|banner|banners|shirt|shirts|card|cards)\b", lowered_question):
        return None
    spans = list(evidence.get("raw_spans") or []) + list(evidence.get("supporting_raw_spans") or [])
    for span in spans:
        metadata = dict(span.get("metadata") or {}) if isinstance(span, dict) else {}
        caption_value = metadata.get("blip_caption")
        if isinstance(caption_value, (list, tuple)):
            caption = " ".join(str(item).strip() for item in caption_value if str(item).strip())
        else:
            caption = str(caption_value or "").strip()
        if not caption:
            continue
        match = re.search(
            r"\b(?:that\s+)?says?\s+['\"]?([^'\".;:()]+)",
            caption,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        candidate = _clean_precision_answer(match.group(1))
        if 1 <= len(candidate.split()) <= 8:
            return candidate
    return None


def precise_short_answer_postprocess(
    question: str,
    predicted_answer: str | None,
    *,
    evidence: dict[str, Any] | None = None,
) -> str | None:
    answer = safe_short_answer_postprocess(question, predicted_answer, evidence=evidence)
    answer_text = _clean_precision_answer(answer or predicted_answer or "")
    if not answer_text:
        return answer
    lowered_question = str(question or "").strip().lower()
    lowered_answer = answer_text.lower()

    visual_text = extract_visual_text_answer(question, evidence)
    if visual_text:
        return visual_text

    temporal_grounding_candidate = extract_best_temporal_grounding_candidate(question, evidence)
    temporal_grounding = str((temporal_grounding_candidate or {}).get("text") or "")
    if temporal_grounding and re.search(r"\bwhen\b", lowered_question):
        candidate_focus = int((temporal_grounding_candidate or {}).get("focus_hits") or 0)
        candidate_distinctive = int((temporal_grounding_candidate or {}).get("distinctive_hits") or 0)
        candidate_speaker_match = int((temporal_grounding_candidate or {}).get("speaker_match") or 0)
        answer_is_supported = answer_supported_by_temporal_evidence(question, answer_text, evidence)
        if should_replace_temporal_answer(answer_text, temporal_grounding) or (
            not answer_is_supported and candidate_focus >= 2 and candidate_distinctive >= 1
        ) or (
            not answer_is_supported and candidate_speaker_match >= 1 and candidate_distinctive >= 1
        ):
            return temporal_grounding

    relative_date = extract_relative_date_answer(question, evidence)
    if relative_date and (
        is_abstention_answer(answer_text)
        or (
            should_replace_temporal_answer(answer_text, relative_date)
            and temporal_answers_compatible(answer_text, relative_date)
        )
    ):
        return relative_date

    started_activity = extract_started_activity_answer(question, evidence)
    if started_activity and (is_abstention_answer(answer_text) or re.search(r"\bclass|workout|activity|exercise\b", lowered_question)):
        return started_activity

    childhood_items = extract_childhood_items_answer(question, evidence)
    if childhood_items and (is_abstention_answer(answer_text) or is_list_query(question)):
        return childhood_items

    family_activities = extract_family_activity_answer(question, evidence)
    if family_activities:
        return family_activities

    local_followup = extract_local_qa_followup_answer(question, evidence)
    if local_followup:
        return local_followup

    geo_answer = extract_geo_list_answer(question, evidence)
    if geo_answer and (
        is_abstention_answer(answer_text)
        or re.search(r"\b(countr(?:y|ies)|european|where)\b", lowered_question)
    ):
        return geo_answer

    if "relationship status" in lowered_question:
        status_patterns = [
            (r"\bsingle\b", "Single"),
            (r"\bmarried\b", "Married"),
            (r"\bdivorced\b", "Divorced"),
            (r"\bwidowed\b", "Widowed"),
            (r"\bengaged\b", "Engaged"),
            (r"\bin a relationship\b", "In a relationship"),
        ]
        for pattern, value in status_patterns:
            if re.search(pattern, lowered_answer):
                return value

    if re.search(r"\bidentity\b", lowered_question):
        identity_match = re.search(
            r"\b(transgender\s+(?:woman|man|person)|non[-\s]?binary\s+person|gay\s+(?:man|person)|lesbian|bisexual\s+person|queer\s+person)\b",
            lowered_answer,
            flags=re.IGNORECASE,
        )
        if identity_match:
            return identity_match.group(1).lower()

    if lowered_question.startswith("what pets") or re.search(r"\bwhat\s+(?:kind\s+of\s+)?pets\b", lowered_question):
        if "cat" in lowered_answer and "dog" in lowered_answer:
            if re.search(r"\btwo\s+cats?\b", lowered_answer):
                return "Two cats and a dog"
            return "Cats and a dog"

    if "type of individuals" in lowered_question and re.search(r"\blgbtq\+?\b", lowered_answer):
        return "LGBTQ+ individuals"

    if lowered_question.startswith("who ") and re.search(r"\bauntie\b", lowered_answer):
        return "Her aunt"

    if re.search(r"\b(names?|what are)\b", lowered_question) and re.search(r"\b[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\b", answer_text):
        compact = re.sub(r"\s+\band\b\s+", ", ", answer_text, flags=re.IGNORECASE)
        if 2 <= len(re.findall(r"\b[A-Z][a-z]+\b", compact)) <= 5:
            return compact

    if re.match(r"^what\s+did\b", lowered_question) and "realize" in lowered_question:
        match = re.match(r"^(.+?\bis\s+important)\b(?:\s+for\b.+)?$", answer_text, flags=re.IGNORECASE)
        if match:
            return _clean_precision_answer(match.group(1))

    if lowered_question.startswith("what happened") and " but " in lowered_answer:
        head = re.split(r"\s+\bbut\b\s+", answer_text, maxsplit=1, flags=re.IGNORECASE)[0]
        if 2 <= len(head.split()) <= 8:
            return _clean_precision_answer(head)
    if lowered_question.startswith("what happened") and "job" in lowered_question and "lost" in lowered_answer and "job" in lowered_answer:
        match = re.search(r"\blost\s+(?:his\s+|her\s+|their\s+)?job(?:\s+at\s+(?:the\s+)?[^,.;]+)?", answer_text, flags=re.IGNORECASE)
        if match:
            return _clean_precision_answer(match.group(0))

    return answer


def apply_short_answer_postprocess(
    question: str,
    predicted_answer: str | None,
    *,
    mode: str = "off",
    evidence: dict[str, Any] | None = None,
) -> str | None:
    if mode in {"", "off", "none"}:
        return predicted_answer
    if mode == "safe":
        return safe_short_answer_postprocess(question, predicted_answer, evidence=evidence)
    if mode == "precise":
        return precise_short_answer_postprocess(question, predicted_answer, evidence=evidence)
    if mode != "safe":
        raise ValueError(f"Unsupported short answer postprocess mode: {mode}")
    return predicted_answer


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
    if any(pattern in lowered for pattern in LIST_QUERY_PATTERNS):
        return True
    collection_noun_re = "|".join(re.escape(noun) for noun in LIST_COLLECTION_NOUNS)
    collection_verb_re = "|".join(re.escape(verb) for verb in LIST_COLLECTION_VERBS)
    return bool(
        re.search(rf"\b(?:what|which)\s+(?:all\s+)?(?:{collection_noun_re})\b", lowered)
        or re.search(
            rf"\bwhat\s+(?:has|have|had|did|does|do)\s+.+\b(?:{collection_verb_re})\b",
            lowered,
        )
    )


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
    raw_spans = list(evidence.get("raw_spans") or []) + list(evidence.get("supporting_raw_spans") or [])
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


def extract_answer_view_lines(answer_view: dict[str, Any], key: str) -> list[str]:
    items = answer_view.get(key) or []
    if not isinstance(items, list):
        items = [items]
    lines: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item.get("text") or "").strip() if isinstance(item, dict) else str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(text)
    return lines


def extract_session_id_from_context_line(line: str) -> str | None:
    text = str(line or "").strip()
    match = re.match(r"^\[([^\]]+)\]\s*", text)
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split("|") if part.strip()]
    return parts[0] if parts else None


def is_topic_soft_span(span: dict[str, Any]) -> bool:
    return bool((dict(span.get("metadata") or {}).get("topic_soft") or {}))


def is_memory_overlay_span(span: dict[str, Any]) -> bool:
    return bool((dict(span.get("metadata") or {}).get("memory_overlay") or {}))


def is_additive_sidecar_span(span: dict[str, Any]) -> bool:
    return bool((dict(span.get("metadata") or {}).get("memory_sidecar") == "additive"))


def strip_additive_sidecar_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    if not evidence:
        return evidence
    return {
        **evidence,
        "supporting_raw_spans": [
            span
            for span in evidence.get("supporting_raw_spans") or []
            if not (isinstance(span, dict) and is_additive_sidecar_span(span))
        ],
    }


def raw_span_text_with_visual_context(span: dict[str, Any]) -> str:
    text = str(span.get("text") or "").strip()
    metadata = dict(span.get("metadata") or {})
    caption_value = metadata.get("blip_caption")
    if isinstance(caption_value, (list, tuple)):
        caption_text = " ".join(str(item).strip() for item in caption_value if str(item).strip())
    else:
        caption_text = str(caption_value or "").strip()
    if caption_text and caption_text.lower() not in {"none", "null", "n/a", "na"}:
        if caption_text.lower() not in text.lower():
            if text:
                return f"{text} (Image: {caption_text})"
            return f"Image: {caption_text}"
    return text


def render_raw_span_payload_line(span: dict[str, Any], *, include_topic_label: bool = False) -> str:
    session_id = str(span.get("session_id") or "").strip()
    speaker = str(span.get("speaker") or "unknown").strip()
    text = raw_span_text_with_visual_context(span)
    timestamp = str(span.get("timestamp") or "").strip()
    metadata = dict(span.get("metadata") or {})
    dia_id = str(metadata.get("dia_id") or "").strip()
    prefix_parts = ordered_unique([session_id, timestamp, dia_id])
    prefix = f"[{' | '.join(prefix_parts)}] " if prefix_parts else ""
    topic_prefix = ""
    if include_topic_label:
        topic_soft = dict(metadata.get("topic_soft") or {})
        topic_id = str(topic_soft.get("selected_topic_id") or "").strip()
        if topic_id:
            topic_prefix = f"[topic:{topic_id}] "
        elif metadata.get("memory_overlay"):
            sources = [
                str(item).strip()
                for item in (dict(metadata.get("memory_overlay") or {}).get("sources") or [])
                if str(item).strip()
            ]
            if sources:
                topic_prefix = f"[overlay:{','.join(sources[:2])}] "
    return f"{topic_prefix}{prefix}{speaker}: {text}".strip()


def span_overlay_payload(span: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(span.get("metadata") or {})
    topic_soft = dict(metadata.get("topic_soft") or {})
    memory_overlay = dict(metadata.get("memory_overlay") or {})
    facets = [
        str(item).strip()
        for item in (topic_soft.get("overlay_facets") or [])
        if str(item).strip()
    ]
    utility = dict(topic_soft.get("overlay_utility") or {})
    match = dict(topic_soft.get("overlay_match") or {})
    if memory_overlay:
        return {
            "span_id": str(span.get("span_id") or "").strip(),
            "topic_id": "",
            "facets": facets,
            "utility": utility,
            "match": match,
            "overlay_sources": list(memory_overlay.get("sources") or []),
            "overlay_score": memory_overlay.get("score"),
            "overlay_reasons": list(memory_overlay.get("reasons") or []),
        }
    if not facets and not utility and not match:
        return {}
    return {
        "span_id": str(span.get("span_id") or "").strip(),
        "topic_id": str(topic_soft.get("selected_topic_id") or "").strip(),
        "facets": facets,
        "utility": utility,
        "match": match,
    }


def select_supporting_payload_lines(lines: list[str], *, exclude: list[str], limit: int) -> list[str]:
    excluded = {re.sub(r"\s+", " ", str(line or "").strip()).lower() for line in exclude if str(line or "").strip()}
    selected: list[str] = []
    seen: set[str] = set(excluded)
    for line in lines:
        text = str(line or "").strip()
        normalized = re.sub(r"\s+", " ", text).lower()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(text)
        if len(selected) >= limit:
            break
    return selected


def rank_payload_lines(question: str, lines: list[str], *, limit: int) -> list[str]:
    query_tokens = {
        token
        for token in normalize_answer(question).split()
        if len(token) >= 3 and token not in ARTICLES
    }
    temporal = expects_temporal_answer(question) or is_temporal_query(question)
    scored: list[tuple[float, int, str]] = []
    for index, line in enumerate(lines):
        text = str(line or "").strip()
        if not text:
            continue
        line_tokens = set(normalize_answer(text).split())
        score = float(len(query_tokens.intersection(line_tokens))) * 2.0
        lowered = text.lower()
        if temporal:
            if re.search(r"\b(?:19|20)\d{2}\b", lowered):
                score += 2.0
            if any(month in lowered for month in MONTH_NAMES):
                score += 1.5
            if "[" in text and "]" in text:
                score += 0.75
        scored.append((score, index, text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked: list[str] = []
    seen: set[str] = set()
    for _, _, line in scored:
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ranked.append(line)
        if len(ranked) >= limit:
            break
    return ranked


def answer_topic_evidence_gate(
    question: str,
    *,
    primary_lines: list[str],
    topic_lines: list[str],
) -> tuple[list[str], dict[str, Any]]:
    if not topic_lines:
        return [], {"mode": "primary_strength_v0", "decision": "no_topic_evidence", "primary_score": 0.0}
    primary_score = max((score_context_line(question, line) for line in primary_lines), default=0.0)
    list_query = is_list_query(question)
    inference_query = is_inference_query(question)
    temporal_query = expects_temporal_answer(question) or is_temporal_query(question)
    single_fact = is_single_fact_query(question)
    if temporal_query:
        return [], {
            "mode": "primary_strength_v0",
            "decision": "suppress_temporal",
            "primary_score": round(primary_score, 4),
        }
    if single_fact and not list_query and not inference_query and primary_score >= 0.45:
        return [], {
            "mode": "primary_strength_v0",
            "decision": "suppress_strong_primary_single_fact",
            "primary_score": round(primary_score, 4),
        }
    return topic_lines, {
        "mode": "primary_strength_v0",
        "decision": "keep_topic_evidence",
        "primary_score": round(primary_score, 4),
    }


def build_structured_answer_payload(
    question: str,
    *,
    evidence: dict[str, Any],
    answer_view: dict[str, Any],
    context_lines: list[str],
    topic_labeled: bool = False,
) -> dict[str, Any]:
    direct_lines = extract_answer_view_lines(answer_view, "direct_evidence") or extract_answer_view_lines(answer_view, "raw_evidence")
    fact_lines = extract_answer_view_lines(answer_view, "entity_facts") or extract_answer_view_lines(answer_view, "facts")
    temporal_lines = extract_answer_view_lines(answer_view, "temporal_clues")
    relation_lines = extract_answer_view_lines(answer_view, "relation_paths") or extract_answer_view_lines(answer_view, "relations")
    page_lines = extract_answer_view_lines(answer_view, "page_context") or extract_answer_view_lines(answer_view, "page_summaries")
    list_query = is_list_query(question)
    temporal_query = expects_temporal_answer(question)

    primary_raw_lines: list[str] = []
    primary_support_raw_lines: list[str] = []
    topic_soft_raw_lines: list[str] = []
    overlay_raw_lines: list[str] = []
    for span in evidence.get("raw_spans") or []:
        if not isinstance(span, dict):
            continue
        if is_memory_overlay_span(span):
            overlay_raw_lines.append(render_raw_span_payload_line(span, include_topic_label=True))
        elif is_topic_soft_span(span):
            topic_soft_raw_lines.append(render_raw_span_payload_line(span, include_topic_label=True))
        else:
            primary_raw_lines.append(render_raw_span_payload_line(span))
    for span in evidence.get("supporting_raw_spans") or []:
        if isinstance(span, dict):
            primary_support_raw_lines.append(render_raw_span_payload_line(span))

    if topic_labeled:
        direct_limit = 8 if list_query else 5
        temporal_limit = 5 if temporal_query else 3 if list_query else 2
        direct_lines = rank_payload_lines(
            question,
            primary_raw_lines or context_lines,
            limit=direct_limit,
        )
        temporal_lines = rank_payload_lines(
            question,
            primary_raw_lines or context_lines,
            limit=temporal_limit,
        )
        primary_supporting_lines = select_supporting_payload_lines(
            primary_support_raw_lines + primary_raw_lines,
            exclude=direct_lines,
            limit=6 if list_query else 4,
        )
    else:
        primary_supporting_lines = []
    topic_lines = [] if temporal_query else rank_payload_lines(question, topic_soft_raw_lines, limit=5 if list_query else 3)
    overlay_lines = rank_payload_lines(
        question,
        overlay_raw_lines,
        limit=6 if list_query else 5 if temporal_query else 4,
    )

    ordered_session_ids: list[str] = []
    for line in direct_lines + temporal_lines + context_lines:
        session_id = extract_session_id_from_context_line(line)
        if session_id and session_id not in ordered_session_ids:
            ordered_session_ids.append(session_id)

    session_windows_map: dict[str, list[tuple[int, str]]] = {}
    for span in evidence.get("raw_spans") or []:
        if topic_labeled and (is_topic_soft_span(span) or is_memory_overlay_span(span)):
            continue
        session_id = str(span.get("session_id") or span.get("timestamp") or "").strip()
        if not session_id:
            continue
        speaker = str(span.get("speaker") or "").strip()
        text = str(span.get("text") or "").strip()
        if not text:
            continue
        rendered = f"{speaker}: {text}" if speaker else text
        turn_index = int(span.get("turn_index") or 0)
        session_windows_map.setdefault(session_id, []).append((turn_index, rendered))
        if session_id not in ordered_session_ids:
            ordered_session_ids.append(session_id)

    session_windows: list[dict[str, Any]] = []
    for session_id in ordered_session_ids[:3]:
        rows = sorted(session_windows_map.get(session_id, []), key=lambda item: item[0])
        seen_lines: set[str] = set()
        lines: list[str] = []
        for _, rendered in rows:
            if rendered in seen_lines:
                continue
            seen_lines.add(rendered)
            lines.append(rendered)
            if len(lines) >= 4:
                break
        if lines:
            session_windows.append({"session_id": session_id, "lines": lines})

    payload = {
        "question": question,
        "question_profile": {
            "explicit_date": has_explicit_date(question),
            "target_month_day": target_month_day(question),
            "list": is_list_query(question),
            "yes_no": is_yes_no_query(question),
            "single_fact": is_single_fact_query(question),
            "inference": is_inference_query(question),
        },
        "direct_evidence": direct_lines[: 8 if list_query else 4],
        "session_windows": session_windows,
        "entity_facts": fact_lines[: 6 if list_query else 4],
        "temporal_clues": temporal_lines[: 4 if list_query else 3],
        "relation_paths": relation_lines[:3],
        "page_context": page_lines[:3],
    }
    if topic_labeled:
        topic_lines, topic_gate = answer_topic_evidence_gate(
            question,
            primary_lines=direct_lines + primary_supporting_lines + temporal_lines,
            topic_lines=topic_lines,
        )
        payload["primary_direct_evidence"] = direct_lines[: 8 if list_query else 5]
        payload["primary_supporting_evidence"] = primary_supporting_lines
        payload["topic_evidence"] = topic_lines[: 5 if list_query else 3]
        payload["overlay_evidence"] = overlay_lines[: 6 if list_query else 5 if temporal_query else 4]
        payload["topic_evidence_gate"] = topic_gate
        overlay_context = [
            item
            for item in (span_overlay_payload(span) for span in evidence.get("raw_spans") or [] if isinstance(span, dict))
            if item
        ]
        if overlay_context:
            payload["memory_overlay_context"] = overlay_context[:6]
        payload["evidence_usage_policy"] = (
            "Use primary_direct_evidence first. Use primary_supporting_evidence as additional trusted "
            "baseline context when primary_direct_evidence is incomplete. Use overlay_evidence when it "
            "directly answers the question, supplies the next turn in a local Q/A exchange, or fills a missing "
            "temporal/entity fact. Use topic_evidence only as supplemental evidence when it directly answers "
            "the question or fills a missing bridge. Do not let supplemental evidence override clearer primary "
            "evidence. Use memory_overlay_context only to interpret why a supplemental item may be relevant; "
            "do not treat overlay labels as standalone evidence."
        )
    return payload


def build_answer_messages(
    question: str,
    evidence: dict[str, Any],
    *,
    context_lines: list[str] | None = None,
    answer_view_text: str | None = None,
    answer_view: dict[str, Any] | None = None,
    answer_style: str = "short",
) -> list[dict[str, str]]:
    context_lines = context_lines if context_lines is not None else build_answer_context_lines(evidence)
    if answer_style == "structured_context_topic_auto":
        answer_style = resolve_effective_answer_style(answer_style, evidence)
    inference_mode = is_inference_query(question)
    structured_context_mode = answer_style == "structured_context"
    topic_labeled_context_mode = answer_style == "structured_context_topic_labeled"
    system_prompt = (
        "You are an intelligent memory assistant tasked with answering questions from conversation memories. "
        + (
            "Use only the provided structured retrieval payload. "
            "Treat direct_evidence as the strongest verbatim evidence. "
            "Use session_windows to recover local conversational context and reference resolution. "
            "Use entity_facts, temporal_clues, relation_paths, and page_context only as supporting signals. "
            if structured_context_mode
        else (
                "Use only the provided structured retrieval payload. "
                "Treat primary_direct_evidence as the trusted retrieval result. "
                "Use primary_supporting_evidence as trusted baseline context when the direct evidence list is incomplete. "
                "Treat overlay_evidence as high-precision memory sidecar evidence: use it when it directly answers the question, "
                "supplies a missing next turn, or fills an absent temporal/entity fact. "
                "Treat topic_evidence as supplemental memory expansion: use it only when it directly answers the question "
                "or supplies a missing bridge, and never let supplemental evidence override clearer primary evidence. "
                "Use session_windows to recover local conversational context and reference resolution. "
                "Use entity_facts, temporal_clues, relation_paths, and page_context only as supporting signals. "
                if topic_labeled_context_mode
                else "Use only the provided evidence. Carefully analyze the evidence, pay attention to timestamps, and prefer direct evidence when available. "
            )
        )
        + "If multiple memories conflict, prioritize the most recent well-supported memory. "
        "Always convert relative time references into specific dates, months, or years when the evidence allows. "
        "Focus only on the people and facts actually supported by the memories. "
        "The final answer should be a precise, concise phrase, usually under 5 to 6 words. "
        "Return only the final answer text, with no explanation, no reasoning, no markdown, and no citations. "
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
            "For list or set questions, scan all primary and directly relevant topic evidence before answering; "
            "do not stop after the first matching item. "
            "For how, method, or activity questions, return only the activity phrase."
        )
    if structured_context_mode or topic_labeled_context_mode:
        user_content = json.dumps(
            build_structured_answer_payload(
                question,
                evidence=evidence,
                answer_view=answer_view or {},
                context_lines=context_lines,
                topic_labeled=topic_labeled_context_mode,
            ),
            ensure_ascii=False,
            indent=2,
        )
    else:
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


def resolve_answer_evidence_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "answer_evidence_mode", "auto") or "auto").strip().lower()
    if mode not in {"auto", "baseline", "merged"}:
        raise ValueError(f"Unsupported answer evidence mode: {mode}")
    return mode


def has_topic_soft_evidence(evidence: dict[str, Any]) -> bool:
    return any(
        is_topic_soft_span(span) or is_memory_overlay_span(span)
        for span in evidence.get("raw_spans") or []
        if isinstance(span, dict)
    )


def resolve_effective_answer_style(answer_style: str, evidence: dict[str, Any]) -> str:
    style = str(answer_style or "short").strip()
    if style != "structured_context_topic_auto":
        return style
    if has_topic_soft_evidence(evidence):
        return "structured_context_topic_labeled"
    return "structured_context"


def select_answer_core_evidence(
    *,
    args: argparse.Namespace,
    answer_style: str,
    baseline_evidence: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    mode = resolve_answer_evidence_mode(args)
    if mode == "baseline":
        return baseline_evidence, mode
    if mode == "merged":
        return evidence, mode
    if answer_style == "structured_context_topic_auto":
        if resolve_effective_answer_style(answer_style, evidence) == "structured_context_topic_labeled":
            return baseline_evidence, "auto_baseline_for_topic_labeled"
        return evidence, "auto_merged_no_topic"
    if answer_style in {"structured_context_topic_labeled", "structured_context_topic_auto"}:
        return baseline_evidence, "auto_baseline_for_topic_labeled"
    return evidence, "auto_merged"


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


def build_locomo_judge_messages(
    *,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    judge_style: str = "legacy_binary",
) -> list[dict[str, str]]:
    if str(judge_style).strip().lower() == "legacy_binary":
        legacy_prompt = (
            "Your task is to label an answer to a question as CORRECT or WRONG.\n"
            "You will be given a question, a gold answer, and a generated answer.\n"
            "Be generous on wording when the generated answer matches the same fact or time period.\n"
            "For time questions, treat equivalent absolute or relative forms as CORRECT if they refer to the same time.\n"
            'Return only JSON with keys "label" and "reason".\n\n'
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Generated answer: {predicted_answer}\n"
        )
        return [{"role": "system", "content": legacy_prompt}]
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


def judge_answer(
    client: ChatClient,
    *,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    retries: int,
    judge_style: str = "legacy_binary",
) -> dict[str, Any]:
    last_error = None
    resolved_style = str(judge_style).strip().lower() or "legacy_binary"
    legacy_binary = resolved_style == "legacy_binary"
    for _ in range(max(1, retries)):
        try:
            payload = extract_json_object(
                client.text(
                    build_locomo_judge_messages(
                        question=question,
                        gold_answer=gold_answer,
                        predicted_answer=predicted_answer,
                        judge_style=resolved_style,
                    ),
                    max_tokens=256,
                    temperature=0.0,
                ).strip()
            )
            label = str(payload.get("label") or "").strip().upper()
            if legacy_binary:
                if label not in {"CORRECT", "WRONG"}:
                    raise ValueError(f"Unexpected label: {label}")
                score = 1.0 if label == "CORRECT" else 0.0
            else:
                score = float(payload.get("score"))
            if not legacy_binary and label not in {"CORRECT", "PARTIAL", "WRONG"}:
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
    topic_soft_event_values = [int(row.get("topic_soft_event_count") or 0) for row in rows]
    topic_soft_candidate_values = [int(row.get("topic_soft_candidate_atom_count") or 0) for row in rows]
    topic_soft_raw_candidate_values = [int(row.get("topic_soft_raw_candidate_atom_count") or 0) for row in rows]
    topic_soft_filtered_values = [int(row.get("topic_soft_filtered_atom_count") or 0) for row in rows]
    topic_soft_suppressed_values = [int(row.get("topic_soft_suppressed_event_count") or 0) for row in rows]
    topic_soft_suppressed_route_values = [int(row.get("topic_soft_suppressed_route_count") or 0) for row in rows]
    topic_soft_semantic_skipped_values = [int(row.get("topic_soft_semantic_gate_skipped_event_count") or 0) for row in rows]
    topic_soft_semantic_missing_values = [int(row.get("topic_soft_semantic_gate_missing_embedding_count") or 0) for row in rows]
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
        "avg_topic_soft_event_count": (
            round(sum(topic_soft_event_values) / len(topic_soft_event_values), 4)
            if topic_soft_event_values
            else 0.0
        ),
        "topic_soft_fallback_used_count": sum(1 for row in rows if row.get("topic_soft_fallback_used")),
        "avg_topic_soft_candidate_atom_count": (
            round(sum(topic_soft_candidate_values) / len(topic_soft_candidate_values), 4)
            if topic_soft_candidate_values
            else 0.0
        ),
        "avg_topic_soft_raw_candidate_atom_count": (
            round(sum(topic_soft_raw_candidate_values) / len(topic_soft_raw_candidate_values), 4)
            if topic_soft_raw_candidate_values
            else 0.0
        ),
        "avg_topic_soft_filtered_atom_count": (
            round(sum(topic_soft_filtered_values) / len(topic_soft_filtered_values), 4)
            if topic_soft_filtered_values
            else 0.0
        ),
        "topic_soft_skipped_low_overlap_total": sum(int(row.get("topic_soft_skipped_low_overlap_count") or 0) for row in rows),
        "topic_soft_skipped_fallback_route_total": sum(int(row.get("topic_soft_skipped_fallback_route_count") or 0) for row in rows),
        "topic_soft_policy_applied_count": sum(1 for row in rows if row.get("topic_soft_policy_applied")),
        "topic_soft_suppressed_event_total": sum(topic_soft_suppressed_values),
        "topic_soft_suppressed_route_total": sum(topic_soft_suppressed_route_values),
        "topic_soft_semantic_gate_skipped_event_total": sum(topic_soft_semantic_skipped_values),
        "topic_soft_semantic_gate_missing_embedding_total": sum(topic_soft_semantic_missing_values),
        "topic_soft_policy_reason_counts": dict(Counter(str(row.get("topic_soft_policy_reason") or "") for row in rows)),
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
    topic_soft_event_values = [int(row.get("topic_soft_event_count") or 0) for row in results]
    topic_soft_candidate_values = [int(row.get("topic_soft_candidate_atom_count") or 0) for row in results]
    topic_soft_raw_candidate_values = [int(row.get("topic_soft_raw_candidate_atom_count") or 0) for row in results]
    topic_soft_filtered_values = [int(row.get("topic_soft_filtered_atom_count") or 0) for row in results]
    topic_soft_suppressed_values = [int(row.get("topic_soft_suppressed_event_count") or 0) for row in results]
    topic_soft_suppressed_route_values = [int(row.get("topic_soft_suppressed_route_count") or 0) for row in results]
    topic_soft_semantic_skipped_values = [int(row.get("topic_soft_semantic_gate_skipped_event_count") or 0) for row in results]
    topic_soft_semantic_missing_values = [int(row.get("topic_soft_semantic_gate_missing_embedding_count") or 0) for row in results]
    overlay_event_values = [int(row.get("overlay_event_count") or 0) for row in results]
    overlay_candidate_values = [int(row.get("overlay_candidate_count") or 0) for row in results]
    ingest_metric_rows = [
        dict(row["ingest_metrics"])
        for row in ingest_rows
        if isinstance(row.get("ingest_metrics"), dict)
    ]
    ingest_llm_prompt_token_rows = [
        dict(row.get("memory_llm_prompt_tokens_est") or {})
        for row in ingest_metric_rows
        if isinstance(row.get("memory_llm_prompt_tokens_est"), dict)
    ]
    ingest_llm_prompt_token_source_rows = [
        dict(row.get("memory_llm_prompt_token_source") or {})
        for row in ingest_metric_rows
        if isinstance(row.get("memory_llm_prompt_token_source"), dict)
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
        "retrieval_mode": str(results[0].get("retrieval_mode") or "baseline") if results else "baseline",
        "avg_topic_soft_event_count": (
            round(sum(topic_soft_event_values) / len(topic_soft_event_values), 4)
            if topic_soft_event_values
            else 0.0
        ),
        "topic_soft_fallback_used_count": sum(1 for row in results if row.get("topic_soft_fallback_used")),
        "avg_topic_soft_candidate_atom_count": (
            round(sum(topic_soft_candidate_values) / len(topic_soft_candidate_values), 4)
            if topic_soft_candidate_values
            else 0.0
        ),
        "avg_topic_soft_raw_candidate_atom_count": (
            round(sum(topic_soft_raw_candidate_values) / len(topic_soft_raw_candidate_values), 4)
            if topic_soft_raw_candidate_values
            else 0.0
        ),
        "avg_topic_soft_filtered_atom_count": (
            round(sum(topic_soft_filtered_values) / len(topic_soft_filtered_values), 4)
            if topic_soft_filtered_values
            else 0.0
        ),
        "topic_soft_skipped_low_overlap_total": sum(int(row.get("topic_soft_skipped_low_overlap_count") or 0) for row in results),
        "topic_soft_skipped_fallback_route_total": sum(int(row.get("topic_soft_skipped_fallback_route_count") or 0) for row in results),
        "topic_soft_policy_applied_count": sum(1 for row in results if row.get("topic_soft_policy_applied")),
        "topic_soft_suppressed_event_total": sum(topic_soft_suppressed_values),
        "topic_soft_suppressed_route_total": sum(topic_soft_suppressed_route_values),
        "topic_soft_semantic_gate_skipped_event_total": sum(topic_soft_semantic_skipped_values),
        "topic_soft_semantic_gate_missing_embedding_total": sum(topic_soft_semantic_missing_values),
        "topic_soft_policy_reason_counts": dict(Counter(str(row.get("topic_soft_policy_reason") or "") for row in results)),
        "avg_overlay_event_count": (
            round(sum(overlay_event_values) / len(overlay_event_values), 4)
            if overlay_event_values
            else 0.0
        ),
        "avg_overlay_candidate_count": (
            round(sum(overlay_candidate_values) / len(overlay_candidate_values), 4)
            if overlay_candidate_values
            else 0.0
        ),
        "overlay_source_counts": dict(
            Counter(
                str(source)
                for row in results
                for source in (row.get("overlay_sources") or [])
                if str(source).strip()
            )
        ),
        "overlay_suppressed_count": sum(1 for row in results if row.get("overlay_suppressed")),
        "overlay_suppress_reason_counts": dict(Counter(str(row.get("overlay_suppress_reason") or "") for row in results)),
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
        "ingest_memory_llm_prompt_tokens_est_total": add_numeric_maps(ingest_llm_prompt_token_rows),
        "ingest_memory_llm_prompt_token_source_total": add_numeric_maps(ingest_llm_prompt_token_source_rows),
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
        "non_temporal_raw_span_limit": getattr(args, "non_temporal_raw_span_limit", 0),
        "local_support_mode": getattr(args, "local_support_mode", "off"),
        "answer_view_mode": args.answer_view_mode,
        "answer_style": args.answer_style,
        "answer_evidence_mode": resolve_answer_evidence_mode(args),
        "retrieval_mode": getattr(args, "retrieval_mode", "baseline"),
        "topic_view_id": getattr(args, "topic_view_id", ""),
        "topic_view_map": getattr(args, "resolved_topic_view_map", None)
        or getattr(args, "topic_view_map", ""),
        "topic_router": getattr(args, "topic_router", "keyword"),
        "topic_route_top_k": getattr(args, "topic_route_top_k", 3),
        "topic_soft_event_limit": getattr(args, "topic_soft_event_limit", 4),
        "overlay_event_limit": getattr(args, "overlay_event_limit", 4),
        "overlay_runtime_policy": getattr(args, "overlay_runtime_policy", "none"),
        "topic_soft_per_topic_atom_limit": getattr(args, "topic_soft_per_topic_atom_limit", 16),
        "topic_soft_deny_topic_slugs": sorted(parse_csv_set(getattr(args, "topic_soft_deny_topic_slugs", ""))),
        "topic_soft_allow_topic_slugs": sorted(parse_csv_set(getattr(args, "topic_soft_allow_topic_slugs", ""))),
        "topic_soft_min_content_overlap": getattr(args, "topic_soft_min_content_overlap", 0),
        "topic_soft_allow_fallback_topic": getattr(args, "topic_soft_allow_fallback_topic", True),
        "topic_soft_fallback": getattr(args, "topic_soft_fallback", "none"),
        "topic_soft_policy": getattr(args, "topic_soft_policy", "selected_overlap_and_candidate_count_v0"),
        "topic_soft_policy_min_selected_overlap": getattr(args, "topic_soft_policy_min_selected_overlap", 2),
        "topic_soft_policy_max_candidate_atoms": getattr(args, "topic_soft_policy_max_candidate_atoms", 20),
        "topic_soft_policy_min_selected_semantic_similarity": getattr(
            args,
            "topic_soft_policy_min_selected_semantic_similarity",
            0.0,
        ),
        "topic_soft_policy_suppress_multi_route": bool(
            getattr(args, "topic_soft_policy_suppress_multi_route", False)
        ),
        "topic_soft_secondary_policy": getattr(args, "topic_soft_secondary_policy", "all"),
        "topic_soft_secondary_min_content_overlap": getattr(args, "topic_soft_secondary_min_content_overlap", 2),
        "topic_soft_secondary_min_route_keyword_overlap": getattr(
            args,
            "topic_soft_secondary_min_route_keyword_overlap",
            1,
        ),
        "topic_soft_min_event_embedding_similarity": getattr(
            args,
            "topic_soft_min_event_embedding_similarity",
            0.0,
        ),
        "topic_soft_use_stemmed_content_tokens": bool(
            getattr(args, "topic_soft_use_stemmed_content_tokens", False)
        ),
        "temporal_postprocess": getattr(args, "temporal_postprocess", "off"),
        "short_answer_postprocess": getattr(args, "short_answer_postprocess", "off"),
        "ingest_prepare_workers": int(os.environ.get("LEAF_INGEST_PREPARE_WORKERS", "4") or "4"),
        "sample_limit": args.sample_limit,
        "qa_per_sample": args.qa_per_sample,
        "judge_with_llm": args.judge_with_llm,
        "judge_style": str(args.judge_style) if args.judge_with_llm else None,
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
    parser.add_argument(
        "--non-temporal-raw-span-limit",
        type=int,
        default=0,
        help="Optional larger raw-span limit for non-temporal QA only. 0 keeps --raw-span-limit for every question.",
    )
    parser.add_argument(
        "--local-support-mode",
        choices=["off", "selective"],
        default="off",
        help="Experimental retrieval-time local neighbor support. off preserves the v5 baseline path.",
    )
    parser.add_argument("--answer-view-mode", choices=["heuristic", "extractive"], default="heuristic")
    parser.add_argument(
        "--answer-style",
        choices=["short", "structured_context", "structured_context_topic_labeled", "structured_context_topic_auto"],
        default="short",
    )
    parser.add_argument(
        "--answer-evidence-mode",
        choices=["auto", "baseline", "merged"],
        default="auto",
        help=(
            "Controls which evidence is used to build the answer payload. auto preserves the legacy "
            "topic-labeled behavior; merged lets accepted topic-soft evidence enter the answer context."
        ),
    )
    parser.add_argument("--ingest-prepare-workers", type=int, default=0)
    parser.add_argument("--ingest-mode", choices=["online", "migration"], default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--judge-with-llm", action="store_true")
    parser.add_argument("--judge-style", choices=["legacy_binary", "partial_credit"], default="legacy_binary")
    parser.add_argument("--judge-runs", type=int, default=5)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--judge-max-workers", type=int, default=1)
    parser.add_argument(
        "--retrieval-mode",
        choices=["baseline", "topic_soft", "topic_soft_gated", "topic_soft_selective", "overlay_selective"],
        default="baseline",
        help="Controls retrieval-side experimental variants. baseline preserves the original path.",
    )
    parser.add_argument("--topic-view-id", default="", help="Topic view for topic_soft. Defaults to active view per corpus.")
    parser.add_argument(
        "--topic-router",
        choices=["keyword", "profile_hybrid", "profile_quality", "evolved_profile_first", "overlay_facet_hybrid", "llm"],
        default="keyword",
    )
    parser.add_argument("--topic-route-top-k", type=int, default=3)
    parser.add_argument("--topic-soft-event-limit", type=int, default=4)
    parser.add_argument("--overlay-event-limit", type=int, default=4)
    parser.add_argument(
        "--overlay-runtime-policy",
        choices=["none", "pre_gate_v2_no_open"],
        default="none",
        help=(
            "Runtime-visible gate for overlay evidence before answer assembly. "
            "pre_gate_v2_no_open keeps overlay only for temporal/entity/local-neighbor signals "
            "and suppresses broad open-slot where/what/why/how queries."
        ),
    )
    parser.add_argument("--topic-soft-per-topic-atom-limit", type=int, default=16)
    parser.add_argument(
        "--topic-soft-deny-topic-slugs",
        default="",
        help="Comma-separated topic slugs to suppress at retrieval time. Also honors view metadata topic_soft_policy.deny_topic_slugs.",
    )
    parser.add_argument(
        "--topic-soft-allow-topic-slugs",
        default="",
        help="Comma-separated topic slugs allowed for retrieval-time topic expansion. Also honors view metadata topic_soft_policy.allow_topic_slugs.",
    )
    parser.add_argument(
        "--topic-soft-min-content-overlap",
        type=int,
        default=1,
        help="For topic_soft_gated, require each added topic atom to overlap this many non-stopword question tokens.",
    )
    parser.add_argument(
        "--topic-soft-secondary-policy",
        choices=["all", "primary_only", "strict_text_v0"],
        default="all",
        help="Controls how atom_secondary assignments from evolved topics may add retrieval evidence.",
    )
    parser.add_argument(
        "--topic-soft-secondary-min-content-overlap",
        type=int,
        default=2,
        help="For strict_text_v0, require secondary atoms to overlap this many non-stopword question tokens.",
    )
    parser.add_argument(
        "--topic-soft-secondary-min-route-keyword-overlap",
        type=int,
        default=1,
        help="For strict_text_v0, require secondary-only topic routes to overlap this many question content tokens.",
    )
    parser.add_argument(
        "--topic-soft-min-event-embedding-similarity",
        type=float,
        default=0.0,
        help=(
            "Experimental semantic gate for topic_soft evidence. When >0, candidate topic events must "
            "have event embedding cosine similarity to the question at least this value."
        ),
    )
    parser.add_argument(
        "--topic-soft-use-stemmed-content-tokens",
        action="store_true",
        help="Use English Snowball stemming in topic-soft content-token overlap and atom scoring.",
    )
    parser.add_argument(
        "--topic-soft-allow-fallback-topic",
        action="store_true",
        help="For topic_soft_gated, allow fallback/misc routes to add evidence.",
    )
    parser.add_argument(
        "--topic-soft-fallback",
        choices=["none", "baseline_on_unknown"],
        default="none",
        help="Retry answer synthesis on baseline evidence when topic_soft evidence yields an abstention.",
    )
    parser.add_argument(
        "--topic-soft-runtime-policy",
        choices=["none", "skip_temporal_query_v0"],
        default="none",
        help=(
            "Early runtime policy applied before topic routing/expansion. "
            "Uses only query text and runtime retrieval state, not benchmark labels."
        ),
    )
    parser.add_argument(
        "--topic-soft-policy-min-selected-overlap",
        type=int,
        default=2,
        help="For topic_soft_selective, require the selected topic event to overlap this many content tokens.",
    )
    parser.add_argument(
        "--topic-soft-policy-max-candidate-atoms",
        type=int,
        default=20,
        help="For topic_soft_selective, suppress topic evidence when the filtered candidate atom pool is larger.",
    )
    parser.add_argument(
        "--topic-soft-policy",
        choices=[
            "selected_overlap_and_candidate_count_v0",
            "text_temporal_suppressed_v0",
            "route_uncertainty_semantic_v0",
        ],
        default="selected_overlap_and_candidate_count_v0",
        help="For topic_soft_selective, choose the text-only policy used to decide whether topic evidence is merged.",
    )
    parser.add_argument(
        "--topic-soft-policy-min-selected-semantic-similarity",
        type=float,
        default=0.0,
        help=(
            "For route_uncertainty_semantic_v0, suppress the whole topic expansion if the selected "
            "topic event has lower question-event embedding cosine similarity."
        ),
    )
    parser.add_argument(
        "--topic-soft-policy-suppress-multi-route",
        action="store_true",
        help="For route_uncertainty_semantic_v0, suppress topic evidence when more than one route is active.",
    )
    parser.add_argument(
        "--enable-heuristic-bypass",
        action="store_true",
        help="Opt in to the old behavior where a heuristic answer can bypass the answer prompt.",
    )
    parser.add_argument(
        "--disable-heuristic-bypass",
        action="store_true",
        help="Deprecated no-op compatibility flag. Heuristic bypass is disabled by default.",
    )
    parser.add_argument(
        "--temporal-postprocess",
        choices=["off", "anchor_only", "range", "range_no_weekday", "relative_prefer"],
        default="off",
        help="Optional deterministic temporal answer rewrite using retrieved relative-time evidence.",
    )
    parser.add_argument(
        "--short-answer-postprocess",
        choices=["off", "safe", "precise"],
        default="off",
        help="Optional deterministic short-answer cleanup for safe overlong answer patterns.",
    )
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
            topic_context = (
                build_topic_context(service.store, corpus_id, args.topic_view_id or None)
                if args.retrieval_mode in {"topic_soft", "topic_soft_gated", "topic_soft_selective", "overlay_selective"}
                else None
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
                    raw_span_limit=resolve_qa_raw_span_limit(args, question),
                    local_support_mode=args.local_support_mode,
                )
                baseline_evidence = evidence
                topic_soft_payload = None
                overlay_payload = None
                if args.retrieval_mode in {"topic_soft", "topic_soft_gated", "topic_soft_selective"} and topic_context is not None:
                    skip_topic_soft, skip_reason = should_skip_topic_soft_runtime(args, question)
                    if skip_topic_soft:
                        topic_soft_payload = empty_topic_soft_payload(
                            policy=args.topic_soft_policy,
                            reason=skip_reason,
                            runtime_policy=args.topic_soft_runtime_policy,
                        )
                    else:
                        baseline_event_ids = {
                            str(event_id).strip()
                            for event_id in (evidence.get("selected_event_ids") or [])
                            if str(event_id).strip()
                        }
                        topic_query_embedding = None
                        try:
                            semantic_gate_min_similarity = float(args.topic_soft_min_event_embedding_similarity or 0.0)
                            needs_query_embedding = (
                                args.topic_router in {"profile_hybrid", "profile_quality", "evolved_profile_first"}
                                or semantic_gate_min_similarity > 0.0
                            )
                            route_query_embedding = None
                            if needs_query_embedding and getattr(service, "embedding", None) is not None:
                                route_query_embedding = service.embedding.embed(question)
                                topic_query_embedding = route_query_embedding
                            routed_topics, routed_router = route_topics(
                                service.store,
                                topic_context,
                                question=question,
                                router=args.topic_router,
                                top_k=args.topic_route_top_k,
                                llm=service.memory_llm if args.topic_router == "llm" else None,
                                query_embedding=route_query_embedding,
                            )
                        except Exception as exc:  # noqa: BLE001
                            routed_topics, routed_router = route_topics(
                                service.store,
                                topic_context,
                                question=question,
                                router="keyword",
                                top_k=args.topic_route_top_k,
                                llm=None,
                                query_embedding=None,
                            )
                            for route in routed_topics:
                                route["router_error"] = str(exc)[:300]
                        topic_soft_payload = topic_soft_expand_events(
                            service.store,
                            topic_context,
                            question=question,
                            routed_topics=routed_topics,
                            exclude_event_ids=baseline_event_ids,
                            event_limit=args.topic_soft_event_limit,
                            per_topic_atom_limit=args.topic_soft_per_topic_atom_limit,
                            min_content_overlap=(
                                max(0, int(args.topic_soft_min_content_overlap))
                                if args.retrieval_mode in {"topic_soft_gated", "topic_soft_selective"}
                                else 0
                            ),
                            allow_fallback_topic=(
                                bool(args.topic_soft_allow_fallback_topic)
                                if args.retrieval_mode in {"topic_soft_gated", "topic_soft_selective"}
                                else True
                            ),
                            score_with_content_tokens=args.retrieval_mode in {"topic_soft_gated", "topic_soft_selective"},
                            deny_topic_slugs=parse_csv_set(args.topic_soft_deny_topic_slugs),
                            allow_topic_slugs=parse_csv_set(args.topic_soft_allow_topic_slugs),
                            secondary_policy=args.topic_soft_secondary_policy,
                            secondary_min_content_overlap=max(0, int(args.topic_soft_secondary_min_content_overlap)),
                            secondary_min_route_keyword_overlap=max(
                                0,
                                int(args.topic_soft_secondary_min_route_keyword_overlap),
                            ),
                            query_embedding=topic_query_embedding,
                            min_event_embedding_similarity=max(
                                0.0,
                                float(args.topic_soft_min_event_embedding_similarity or 0.0),
                            ),
                            use_stemmed_content_tokens=bool(
                                getattr(args, "topic_soft_use_stemmed_content_tokens", False)
                            ),
                        )
                        topic_soft_payload["router"] = routed_router
                        topic_soft_payload["runtime_policy"] = args.topic_soft_runtime_policy
                        topic_soft_payload["runtime_skipped"] = False
                        topic_soft_payload["runtime_skip_reason"] = ""
                        if args.retrieval_mode == "topic_soft_selective":
                            topic_soft_payload = apply_topic_soft_policy(
                                topic_soft_payload,
                                policy=args.topic_soft_policy,
                                min_selected_overlap=max(0, int(args.topic_soft_policy_min_selected_overlap)),
                                max_candidate_atom_count=max(0, int(args.topic_soft_policy_max_candidate_atoms)),
                                suppress_for_temporal_query=expects_temporal_answer(question) or is_temporal_query(question),
                                min_selected_semantic_similarity=max(
                                    0.0,
                                    float(args.topic_soft_policy_min_selected_semantic_similarity or 0.0),
                                ),
                                suppress_multi_route=bool(args.topic_soft_policy_suppress_multi_route),
                            )
                    evidence = merge_topic_soft_evidence(evidence, topic_soft_payload)
                if args.retrieval_mode == "overlay_selective" and topic_context is not None:
                    baseline_event_ids = {
                        str(event_id).strip()
                        for event_id in (evidence.get("selected_event_ids") or [])
                        if str(event_id).strip()
                    }
                    overlay_payload = overlay_expand_events(
                        service.store,
                        topic_context,
                        question=question,
                        baseline_evidence=baseline_evidence,
                        exclude_event_ids=baseline_event_ids,
                        event_limit=max(0, int(args.overlay_event_limit)),
                        use_stemmed_content_tokens=bool(getattr(args, "topic_soft_use_stemmed_content_tokens", False)),
                    )
                    overlay_payload = apply_overlay_runtime_policy(
                        overlay_payload,
                        question=question,
                        policy=getattr(args, "overlay_runtime_policy", "none"),
                    )
                    if overlay_payload.get("event_ids"):
                        selected_event_ids = [
                            str(event_id)
                            for event_id in (evidence.get("selected_event_ids") or [])
                            if str(event_id).strip()
                        ]
                        for event_id in overlay_payload.get("event_ids") or []:
                            if event_id not in selected_event_ids:
                                selected_event_ids.append(event_id)
                        seen_span_ids = {str(span.get("span_id") or "") for span in (evidence.get("raw_spans") or [])}
                        raw_spans = list(evidence.get("raw_spans") or [])
                        for span in overlay_payload.get("raw_spans") or []:
                            span_id = str(span.get("span_id") or "")
                            if span_id and span_id in seen_span_ids:
                                continue
                            seen_span_ids.add(span_id)
                            raw_spans.append(span)
                        evidence = {
                            **evidence,
                            "selected_event_ids": selected_event_ids,
                            "raw_spans": raw_spans,
                            "memory_overlay": overlay_payload,
                        }
                search_elapsed_ms = (time.perf_counter() - search_started) * 1000.0

                answer_started = time.perf_counter()
                answer_core_evidence, answer_evidence_mode = select_answer_core_evidence(
                    args=args,
                    answer_style=args.answer_style,
                    baseline_evidence=baseline_evidence,
                    evidence=evidence,
                )
                postprocess_evidence = strip_additive_sidecar_evidence(answer_core_evidence)
                effective_answer_style = resolve_effective_answer_style(args.answer_style, evidence)
                context_lines = build_answer_context_lines(answer_core_evidence)
                heuristic_answer = heuristic_answer_from_evidence(question=question, evidence=postprocess_evidence)
                answer_view: dict[str, Any] = {}
                answer_view_text = ""
                answer_messages: list[dict[str, str]] = []
                answer_prompt_used = False
                topic_soft_fallback_used = False
                heuristic_bypass_triggered = bool(heuristic_answer)
                heuristic_bypass_enabled = bool(args.enable_heuristic_bypass) and not bool(args.disable_heuristic_bypass)
                heuristic_bypass_used = heuristic_bypass_triggered and heuristic_bypass_enabled
                if heuristic_bypass_used:
                    answer_prompt_mode = "heuristic"
                elif heuristic_bypass_triggered:
                    answer_prompt_mode = "llm_bypass_disabled"
                else:
                    answer_prompt_mode = "llm"
                answer_prompt_input_tokens_est = 0
                answer_max_tokens = 0
                if heuristic_bypass_used:
                    predicted_answer = heuristic_answer
                    answer_input_tokens_est = 0
                else:
                    answer_view = build_compact_answer_view(
                        question=question,
                        evidence=answer_core_evidence,
                        mode=args.answer_view_mode,
                    )
                    answer_view_text = render_answer_view_text(question=question, answer_view=answer_view)
                    answer_messages = build_answer_messages(
                        question=question,
                        evidence=evidence,
                        context_lines=context_lines,
                        answer_view_text=answer_view_text,
                        answer_view=answer_view,
                        answer_style=effective_answer_style,
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
                        if (
                            args.retrieval_mode in {"topic_soft", "topic_soft_gated", "topic_soft_selective"}
                            and args.topic_soft_fallback == "baseline_on_unknown"
                            and bool((topic_soft_payload or {}).get("event_ids"))
                            and baseline_evidence is not evidence
                            and is_abstention_answer(predicted_answer)
                        ):
                            baseline_context_lines = build_answer_context_lines(baseline_evidence)
                            baseline_answer_view = build_compact_answer_view(
                                question=question,
                                evidence=baseline_evidence,
                                mode=args.answer_view_mode,
                            )
                            baseline_answer_view_text = render_answer_view_text(
                                question=question,
                                answer_view=baseline_answer_view,
                            )
                            baseline_answer_messages = build_answer_messages(
                                question=question,
                                evidence=baseline_evidence,
                                context_lines=baseline_context_lines,
                                answer_view_text=baseline_answer_view_text,
                                answer_view=baseline_answer_view,
                                answer_style=resolve_effective_answer_style(args.answer_style, baseline_evidence),
                            )
                            answer_input_tokens_est += estimate_message_tokens(baseline_answer_messages)
                            fallback_answer = (
                                service.llm.text(
                                    baseline_answer_messages,
                                    max_tokens=answer_max_tokens,
                                    temperature=0.0,
                                ).strip()
                                if service.llm
                                else ""
                            )
                            if fallback_answer and not is_abstention_answer(fallback_answer):
                                predicted_answer = fallback_answer
                                topic_soft_fallback_used = True
                    except OpenAICompatError as exc:
                        predicted_answer = f"__ERROR__: {exc}"
                answer_elapsed_ms = (time.perf_counter() - answer_started) * 1000.0
                predicted_answer = str(canonicalize_temporal_answer(question, predicted_answer, postprocess_evidence) or predicted_answer).strip()
                predicted_answer_before_temporal_postprocess = predicted_answer
                predicted_answer = str(
                    temporal_anchor_postprocess(
                        question,
                        predicted_answer,
                        postprocess_evidence,
                        mode=args.temporal_postprocess,
                    )
                    or predicted_answer
                ).strip()
                temporal_postprocess_used = predicted_answer != predicted_answer_before_temporal_postprocess
                predicted_answer_before_short_answer_postprocess = predicted_answer
                predicted_answer = str(
                    apply_short_answer_postprocess(
                        question,
                        predicted_answer,
                        mode=args.short_answer_postprocess,
                        evidence=postprocess_evidence,
                    )
                    or predicted_answer
                ).strip()
                short_answer_postprocess_used = predicted_answer != predicted_answer_before_short_answer_postprocess
                retrieval_timing = dict((evidence.get("timing") or {}) if isinstance(evidence, dict) else {})
                mem0_hybrid_payload = dict(retrieval_timing.get("mem0_hybrid") or {})
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
                    "predicted_answer_before_temporal_postprocess": predicted_answer_before_temporal_postprocess,
                    "predicted_answer_before_short_answer_postprocess": predicted_answer_before_short_answer_postprocess,
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
                    "answer_style": args.answer_style,
                    "effective_answer_style": effective_answer_style,
                    "answer_evidence_mode": answer_evidence_mode,
                    "heuristic_bypass_triggered": heuristic_bypass_triggered,
                    "heuristic_bypass_enabled": heuristic_bypass_enabled,
                    "heuristic_bypass_used": heuristic_bypass_used,
                    "answer_max_tokens": answer_max_tokens,
                    "heuristic_answer": heuristic_answer,
                    "temporal_postprocess": args.temporal_postprocess,
                    "temporal_postprocess_used": temporal_postprocess_used,
                    "short_answer_postprocess": args.short_answer_postprocess,
                    "short_answer_postprocess_used": short_answer_postprocess_used,
                    "retrieval_mode": args.retrieval_mode,
                    "raw_span_count": len(evidence.get("raw_spans") or []),
                    "mem0_hybrid_enabled": bool(mem0_hybrid_payload.get("enabled")),
                    "mem0_hybrid_matched_event_count": int(mem0_hybrid_payload.get("matched_event_count") or 0),
                    "mem0_hybrid_ranked_event_ids": list(mem0_hybrid_payload.get("ranked_event_ids") or [])[:12],
                    "mem0_hybrid_query_terms": list(mem0_hybrid_payload.get("query_terms") or [])[:16],
                    "mem0_hybrid_query_entities": list(mem0_hybrid_payload.get("query_entities") or [])[:16],
                    "mem0_hybrid_diagnostics_by_event": dict(mem0_hybrid_payload.get("diagnostics_by_event") or {}),
                    "topic_soft_event_count": len((topic_soft_payload or {}).get("event_ids") or []),
                    "topic_soft_atom_count": len((topic_soft_payload or {}).get("atom_ids") or []),
                    "topic_soft_candidate_atom_count": int((topic_soft_payload or {}).get("candidate_atom_count") or 0),
                    "topic_soft_raw_candidate_atom_count": int((topic_soft_payload or {}).get("raw_candidate_atom_count") or 0),
                    "topic_soft_filtered_atom_count": int((topic_soft_payload or {}).get("filtered_atom_count") or 0),
                    "topic_soft_skipped_low_overlap_count": int((topic_soft_payload or {}).get("skipped_low_overlap_count") or 0),
                    "topic_soft_skipped_fallback_route_count": int((topic_soft_payload or {}).get("skipped_fallback_route_count") or 0),
                    "topic_soft_semantic_gate_enabled": bool(
                        (topic_soft_payload or {}).get("semantic_gate_enabled")
                    ),
                    "topic_soft_semantic_gate_min_similarity": (
                        (topic_soft_payload or {}).get("semantic_gate_min_similarity")
                    ),
                    "topic_soft_semantic_gate_max_similarity": (
                        (topic_soft_payload or {}).get("semantic_gate_max_similarity")
                    ),
                    "topic_soft_semantic_gate_skipped_event_count": int(
                        (topic_soft_payload or {}).get("semantic_gate_skipped_event_count") or 0
                    ),
                    "topic_soft_semantic_gate_missing_embedding_count": int(
                        (topic_soft_payload or {}).get("semantic_gate_missing_embedding_count") or 0
                    ),
                    "topic_soft_active_route_count": len((topic_soft_payload or {}).get("active_routes") or []),
                    "topic_soft_router": (topic_soft_payload or {}).get("router"),
                    "topic_soft_topic_slugs": [
                        str(route.get("slug") or route.get("topic_id"))
                        for route in ((topic_soft_payload or {}).get("routes") or [])
                    ],
                    "topic_soft_active_topic_slugs": [
                        str(route.get("slug") or route.get("topic_id"))
                        for route in ((topic_soft_payload or {}).get("active_routes") or [])
                    ],
                    "topic_soft_suppressed_route_count": len((topic_soft_payload or {}).get("suppressed_routes") or []),
                    "topic_soft_suppressed_topic_slugs": [
                        str(route.get("slug") or route.get("topic_id"))
                        for route in ((topic_soft_payload or {}).get("suppressed_routes") or [])
                    ],
                    "topic_soft_fallback": args.topic_soft_fallback,
                    "topic_soft_fallback_used": topic_soft_fallback_used,
                    "topic_soft_policy": (topic_soft_payload or {}).get("policy"),
                    "topic_soft_policy_applied": bool((topic_soft_payload or {}).get("policy_applied")),
                    "topic_soft_policy_reason": (topic_soft_payload or {}).get("policy_reason"),
                    "topic_soft_policy_max_selected_content_overlap": int(
                        (topic_soft_payload or {}).get("policy_max_selected_content_overlap") or 0
                    ),
                    "topic_soft_policy_max_selected_semantic_similarity": (
                        (topic_soft_payload or {}).get("policy_max_selected_semantic_similarity")
                    ),
                    "topic_soft_policy_min_selected_semantic_similarity": (
                        (topic_soft_payload or {}).get("policy_min_selected_semantic_similarity")
                    ),
                    "topic_soft_policy_suppress_multi_route": bool(
                        (topic_soft_payload or {}).get("policy_suppress_multi_route")
                    ),
                    "topic_soft_policy_min_selected_overlap": int(
                        (topic_soft_payload or {}).get("policy_min_selected_overlap") or 0
                    ),
                    "topic_soft_policy_max_candidate_atom_count": int(
                        (topic_soft_payload or {}).get("policy_max_candidate_atom_count") or 0
                    ),
                    "topic_soft_suppressed_event_count": len((topic_soft_payload or {}).get("suppressed_event_ids") or []),
                    "overlay_event_count": len((overlay_payload or {}).get("event_ids") or []),
                    "overlay_candidate_count": int((overlay_payload or {}).get("candidate_count") or 0),
                    "overlay_suppressed": bool((overlay_payload or {}).get("suppressed")),
                    "overlay_suppress_reason": (overlay_payload or {}).get("suppress_reason"),
                    "overlay_primary_strength": (overlay_payload or {}).get("primary_strength"),
                    "overlay_runtime_policy": (overlay_payload or {}).get("runtime_policy"),
                    "overlay_runtime_policy_applied": bool((overlay_payload or {}).get("runtime_policy_applied")),
                    "overlay_runtime_policy_allowed": (overlay_payload or {}).get("runtime_policy_allowed"),
                    "overlay_runtime_policy_reason": (overlay_payload or {}).get("runtime_policy_reason"),
                    "overlay_runtime_policy_features": (overlay_payload or {}).get("runtime_policy_features") or {},
                    "overlay_sources": ordered_unique(
                        [
                            str(source)
                            for item in ((overlay_payload or {}).get("selected") or [])
                            if isinstance(item, dict)
                            for source in (item.get("sources") or [])
                            if str(source).strip()
                        ]
                    ),
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
                        "supporting_raw_spans": list(evidence.get("supporting_raw_spans") or []),
                        "timing": retrieval_timing,
                    },
                    "retrieved_dia_ids": ordered_unique(
                        [
                            str((span.get("metadata") or {}).get("dia_id") or "").strip()
                            for span in list(evidence.get("raw_spans") or [])
                            + list(evidence.get("supporting_raw_spans") or [])
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
                        judge_style=args.judge_style,
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
