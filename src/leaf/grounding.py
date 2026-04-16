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

WEEKDAY_NAMES = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "weds": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

WEEKDAY_PATTERN = r"(mon|monday|tue|tues|tuesday|wed|weds|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)"

TEMPORAL_PATTERN_SPECS = (
    {
        "name": "days_ago",
        "pattern": re.compile(r"\b(?P<value>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago\b"),
        "kind": "days_ago",
        "precision": "date",
    },
    {
        "name": "yesterday",
        "pattern": re.compile(r"\byesterday\b"),
        "kind": "day_offset",
        "precision": "date",
        "offset_days": -1,
    },
    {
        "name": "today",
        "pattern": re.compile(r"\btoday\b"),
        "kind": "day_offset",
        "precision": "date",
        "offset_days": 0,
    },
    {
        "name": "last_year",
        "pattern": re.compile(r"\blast year\b"),
        "kind": "year_offset",
        "precision": "year",
        "offset_years": -1,
    },
    {
        "name": "this_year",
        "pattern": re.compile(r"\bthis year\b"),
        "kind": "year_offset",
        "precision": "year",
        "offset_years": 0,
    },
    {
        "name": "last_month",
        "pattern": re.compile(r"\blast month\b"),
        "kind": "month_offset",
        "precision": "month",
        "offset_months": -1,
    },
    {
        "name": "next_month",
        "pattern": re.compile(r"\bnext month\b"),
        "kind": "month_offset",
        "precision": "month",
        "offset_months": 1,
    },
    {
        "name": "this_month",
        "pattern": re.compile(r"\bthis month\b"),
        "kind": "current_month",
        "precision": "month",
    },
    {
        "name": "this_week",
        "pattern": re.compile(r"\bthis week\b"),
        "kind": "relative_label",
        "precision": "relative",
        "label": "week of",
    },
    {
        "name": "last_week",
        "pattern": re.compile(r"\blast week\b"),
        "kind": "relative_label",
        "precision": "relative",
        "label": "week before",
    },
    {
        "name": "next_week",
        "pattern": re.compile(r"\bnext week\b"),
        "kind": "relative_label",
        "precision": "relative",
        "label": "week after",
    },
    {
        "name": "last_weekend",
        "pattern": re.compile(r"\blast weekend\b"),
        "kind": "relative_label",
        "precision": "relative",
        "label": "weekend before",
    },
    {
        "name": "last_weekday",
        "pattern": re.compile(rf"\blast\s+(?P<weekday>{WEEKDAY_PATTERN})\b"),
        "kind": "last_weekday",
        "precision": "relative",
    },
)

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
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    for pattern in patterns:
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
    return None


def match_temporal_pattern(text: str) -> tuple[dict[str, Any] | None, re.Match[str] | None]:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None, None
    for spec in TEMPORAL_PATTERN_SPECS:
        match = spec["pattern"].search(lowered)
        if match:
            return spec, match
    return None, None


def has_temporal_pattern(text: str) -> bool:
    spec, _ = match_temporal_pattern(text)
    return spec is not None


def _relative_anchor_text(anchor: datetime) -> str:
    return f"{anchor.day} {MONTH_NAMES.get(anchor.month, anchor.month)} {anchor.year}"


def _month_offset(anchor: datetime, offset_months: int) -> tuple[int, int]:
    total = (anchor.year * 12 + (anchor.month - 1)) + offset_months
    year = total // 12
    month = (total % 12) + 1
    return year, month


def _apply_temporal_pattern(
    spec: dict[str, Any],
    match: re.Match[str] | None,
    anchor: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "anchor_month": anchor.strftime("%Y-%m"),
        "anchor_year": anchor.year,
    }
    kind = str(spec.get("kind") or "")
    precision = str(spec.get("precision") or "")
    if kind == "days_ago":
        raw_value = str((match.group("value") if match is not None else "") or "").strip().lower()
        delta_days = int(raw_value) if raw_value.isdigit() else NUMBER_WORDS.get(raw_value)
        if delta_days is None:
            return payload
        grounded = anchor - timedelta(days=delta_days)
        payload["grounded_date"] = grounded.strftime("%Y-%m-%d")
        payload["precision"] = precision
        return payload
    if kind == "day_offset":
        grounded = anchor + timedelta(days=int(spec.get("offset_days") or 0))
        payload["grounded_date"] = grounded.strftime("%Y-%m-%d")
        payload["precision"] = precision
        return payload
    if kind == "year_offset":
        payload["grounded_year"] = anchor.year + int(spec.get("offset_years") or 0)
        payload["precision"] = precision
        return payload
    if kind == "month_offset":
        year, month = _month_offset(anchor, int(spec.get("offset_months") or 0))
        payload["grounded_month"] = f"{year:04d}-{month:02d}"
        payload["precision"] = precision
        return payload
    if kind == "current_month":
        payload["grounded_month"] = anchor.strftime("%Y-%m")
        payload["precision"] = precision
        return payload
    if kind == "relative_label":
        payload["grounded_relative"] = f"{spec.get('label')} {_relative_anchor_text(anchor)}"
        payload["precision"] = precision
        return payload
    if kind == "last_weekday":
        weekday_key = str((match.group("weekday") if match is not None else "") or "").strip().lower()
        weekday_name = WEEKDAY_NAMES.get(weekday_key)
        if not weekday_name:
            return payload
        payload["grounded_relative"] = f"{weekday_name} before {_relative_anchor_text(anchor)}"
        payload["precision"] = precision
        return payload
    payload["grounded_date"] = anchor.strftime("%Y-%m-%d")
    payload["precision"] = "anchor"
    return payload


def derive_temporal_grounding(text: str, timestamp: str | None) -> dict[str, Any]:
    anchor = parse_anchor_datetime(timestamp)
    payload: dict[str, Any] = {}
    if anchor is not None:
        payload["anchor_date"] = anchor.strftime("%Y-%m-%d")
        payload["anchor_month"] = anchor.strftime("%Y-%m")
        payload["anchor_year"] = anchor.year

    if anchor is None:
        return payload
    spec, match = match_temporal_pattern(text)
    if spec is None:
        payload["grounded_date"] = anchor.strftime("%Y-%m-%d")
        payload["precision"] = "anchor"
        return payload
    return _apply_temporal_pattern(spec, match, anchor)


def format_grounded_value(grounding: dict[str, Any]) -> str | None:
    if grounding.get("precision") == "relative" and grounding.get("grounded_relative"):
        value = str(grounding["grounded_relative"]).strip()
        return value[:1].upper() + value[1:] if value else None
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
    answer_spec, _ = match_temporal_pattern(answer)
    if answer_spec is None:
        return predicted_answer

    raw_spans = evidence.get("raw_spans") or []
    candidates: list[tuple[int, str]] = []
    for span in raw_spans:
        grounding = derive_temporal_grounding(
            text=str(span.get("text") or ""),
            timestamp=span.get("timestamp"),
        )
        formatted = format_grounded_value(grounding)
        if not formatted:
            continue
        score = 0
        span_spec, _ = match_temporal_pattern(str(span.get("text") or ""))
        if span_spec is not None and span_spec.get("name") == answer_spec.get("name"):
            score += 4
        if span_spec is not None and span_spec.get("precision") == answer_spec.get("precision"):
            score += 1
        if score == 0 and grounding.get("precision") in {"date", "month", "year", "relative"}:
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


TEXT_FRAGMENT_MAX_UNITS = 400
TEXT_FRAGMENT_OVERLAP_UNITS = 40
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?。！？])\s+|\n+")
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[,;:，；：、])\s*")


def text_unit_count(text: str) -> int:
    stripped = str(text or "").strip()
    if not stripped:
        return 0
    cjk_count = len(_CJK_CHAR_RE.findall(stripped))
    latin_count = len(_LATIN_WORD_RE.findall(stripped))
    return cjk_count + latin_count


def _split_with_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    pieces = [part.strip() for part in pattern.split(str(text or "").strip()) if part and part.strip()]
    return pieces


def _split_oversized_piece(text: str, max_units: int) -> list[str]:
    clauses = _split_with_pattern(text, _CLAUSE_BOUNDARY_RE)
    if len(clauses) > 1:
        return _pack_units(clauses, max_units=max_units)
    words = str(text or "").split()
    if len(words) > 1:
        chunks: list[str] = []
        current: list[str] = []
        current_units = 0
        for word in words:
            units = max(1, text_unit_count(word))
            if current and current_units + units > max_units:
                chunks.append(" ".join(current).strip())
                current = [word]
                current_units = units
            else:
                current.append(word)
                current_units += units
        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]
    raw = str(text or "").strip()
    if text_unit_count(raw) <= max_units:
        return [raw]
    chars = list(raw)
    chunks = []
    start = 0
    while start < len(chars):
        end = min(len(chars), start + max_units)
        chunks.append("".join(chars[start:end]).strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def _pack_units(parts: list[str], max_units: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_units = 0
    for part in parts:
        units = text_unit_count(part)
        if units > max_units:
            oversized = _split_oversized_piece(part, max_units=max_units)
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_units = 0
            chunks.extend(oversized)
            continue
        if current and current_units + units > max_units:
            chunks.append(" ".join(current).strip())
            current = [part]
            current_units = units
        else:
            current.append(part)
            current_units += units
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _tail_overlap_segments(text: str, overlap_units: int) -> list[str]:
    segments = _split_with_pattern(text, _SENTENCE_BOUNDARY_RE)
    if not segments:
        return []
    selected: list[str] = []
    total_units = 0
    for segment in reversed(segments):
        seg_units = text_unit_count(segment)
        if selected and total_units + seg_units > overlap_units:
            break
        selected.insert(0, segment)
        total_units += seg_units
        if total_units >= overlap_units:
            break
    return selected


def _rebalance_chunks(chunks: list[str], max_units: int) -> list[str]:
    if len(chunks) < 2:
        return chunks
    balanced = list(chunks)
    while len(balanced) >= 2:
        tail_units = text_unit_count(balanced[-1])
        if tail_units >= int(max_units * 0.35):
            break
        prev_segments = _split_with_pattern(balanced[-2], _SENTENCE_BOUNDARY_RE)
        if len(prev_segments) <= 1:
            break
        moved = prev_segments.pop()
        candidate_tail = f"{moved} {balanced[-1]}".strip()
        if text_unit_count(candidate_tail) > max_units:
            break
        balanced[-2] = " ".join(prev_segments).strip()
        balanced[-1] = candidate_tail
        if not balanced[-2]:
            balanced.pop(-2)
            break
    return [chunk for chunk in balanced if chunk]


def split_text_fragments(
    text: str,
    *,
    max_units: int = TEXT_FRAGMENT_MAX_UNITS,
    overlap_units: int = TEXT_FRAGMENT_OVERLAP_UNITS,
) -> list[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []
    if text_unit_count(stripped) <= max_units:
        return [stripped]

    sentence_parts = _split_with_pattern(stripped, _SENTENCE_BOUNDARY_RE)
    if not sentence_parts:
        sentence_parts = [stripped]
    chunks = _pack_units(sentence_parts, max_units=max_units)
    chunks = _rebalance_chunks(chunks, max_units=max_units)
    if len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        overlap_segments = _tail_overlap_segments(overlapped[-1], overlap_units=overlap_units)
        prefix = " ".join(overlap_segments).strip()
        candidate = f"{prefix} {chunk}".strip() if prefix else chunk
        if prefix and text_unit_count(candidate) > max_units:
            candidate = chunk
        overlapped.append(candidate)
    return overlapped
