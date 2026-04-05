from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Any

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

TEMPORAL_QUERY_HINTS = {"when", "date", "day", "month", "year", "time"}
INFERENCE_QUERY_PATTERNS = (
    "would ",
    "likely",
    "might ",
    "personality trait",
    "personality traits",
    "be considered",
    "financial status",
    "underlying condition",
    "what job",
    "what career",
    "what fields",
    "what state did",
    "what console",
    "what pets",
    "how many",
    "is it likely",
)


def span_surface_text(speaker: str, text: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    parts = [f"{speaker}: {text.strip()}"]
    caption = str(metadata.get("blip_caption") or "").strip()
    if caption:
        parts.append(f"Image: {caption}")
    return "\n".join(part for part in parts if part.strip())


def parse_anchor_datetime(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    cleaned = str(timestamp).strip()
    patterns = [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%d %B, %Y",
        "%d %B %Y",
        "%B %d, %Y",
    ]
    for pattern in patterns:
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
    return None


def derive_temporal_grounding(text: str, timestamp: str | None) -> dict[str, Any]:
    lowered = str(text or "").strip().lower()
    anchor = parse_anchor_datetime(timestamp)
    payload: dict[str, Any] = {}
    if anchor is not None:
        payload["anchor_date"] = anchor.strftime("%Y-%m-%d")
        payload["anchor_month"] = anchor.strftime("%Y-%m")
        payload["anchor_year"] = anchor.year

    if anchor is None:
        return payload

    if "yesterday" in lowered:
        grounded = anchor - timedelta(days=1)
        payload["grounded_date"] = grounded.strftime("%Y-%m-%d")
        payload["precision"] = "date"
    elif "today" in lowered:
        payload["grounded_date"] = anchor.strftime("%Y-%m-%d")
        payload["precision"] = "date"
    elif "last year" in lowered:
        payload["grounded_year"] = anchor.year - 1
        payload["precision"] = "year"
    elif "this year" in lowered:
        payload["grounded_year"] = anchor.year
        payload["precision"] = "year"
    elif "last month" in lowered:
        year = anchor.year
        month = anchor.month - 1
        if month == 0:
            month = 12
            year -= 1
        payload["grounded_month"] = f"{year:04d}-{month:02d}"
        payload["precision"] = "month"
    elif "this month" in lowered:
        payload["grounded_month"] = anchor.strftime("%Y-%m")
        payload["precision"] = "month"
    else:
        payload["grounded_date"] = anchor.strftime("%Y-%m-%d")
        payload["precision"] = "anchor"
    return payload


def format_grounded_value(grounding: dict[str, Any]) -> str | None:
    if grounding.get("precision") == "year" and grounding.get("grounded_year"):
        return str(grounding["grounded_year"])
    if grounding.get("precision") == "month" and grounding.get("grounded_month"):
        year, month = str(grounding["grounded_month"]).split("-", 1)
        return f"{MONTH_NAMES.get(int(month), month)}, {year}"
    grounded_date = grounding.get("grounded_date")
    if grounded_date:
        year, month, day = grounded_date.split("-", 2)
        return f"{int(day)} {MONTH_NAMES.get(int(month), month)} {year}"
    return None


def is_temporal_query(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(token in lowered.split() or token in lowered for token in TEMPORAL_QUERY_HINTS)


def is_inference_query(query: str) -> bool:
    lowered = f" {str(query or '').strip().lower()} "
    return any(pattern in lowered for pattern in INFERENCE_QUERY_PATTERNS)


def canonicalize_temporal_answer(question: str, predicted_answer: str | None, evidence: dict[str, Any]) -> str | None:
    answer = str(predicted_answer or "").strip()
    if not answer or answer.upper() == "UNKNOWN" or not is_temporal_query(question):
        return predicted_answer
    lowered = answer.lower()
    if not any(marker in lowered for marker in ["yesterday", "today", "last year", "this year", "last month", "this month"]):
        return predicted_answer

    raw_spans = evidence.get("raw_spans") or []
    candidates: list[tuple[int, str]] = []
    for span in raw_spans:
        metadata = dict(span.get("metadata") or {})
        grounding = metadata.get("temporal_grounding")
        if not isinstance(grounding, dict):
            grounding = derive_temporal_grounding(
                text=str(span.get("text") or ""),
                timestamp=span.get("timestamp"),
            )
        formatted = format_grounded_value(grounding)
        if not formatted:
            continue
        score = 0
        span_text = str(span.get("text") or "").lower()
        if "yesterday" in lowered and "yesterday" in span_text:
            score += 4
        if "last year" in lowered and "last year" in span_text:
            score += 4
        if "this month" in lowered and "this month" in span_text:
            score += 4
        if "today" in lowered and "today" in span_text:
            score += 4
        if score == 0 and grounding.get("precision") in {"date", "month", "year"}:
            score += 1
        candidates.append((score, formatted))

    if not candidates:
        return predicted_answer
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def query_tokens(query: str) -> set[str]:
    return {
        token
        for token in re.sub(r"[^a-z0-9\s]", " ", query.lower()).split()
        if len(token) > 2
    }
