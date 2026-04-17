from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Any

from .normalize import language_aware_terms, normalize_surface_text

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

ZH_NUMBER_WORDS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

ZH_WEEKDAY_NAMES = {
    "一": "Monday",
    "二": "Tuesday",
    "三": "Wednesday",
    "四": "Thursday",
    "五": "Friday",
    "六": "Saturday",
    "日": "Sunday",
    "天": "Sunday",
}

_LANGUAGE_MODE = "en"


def set_language_mode(mode: str) -> None:
    global _LANGUAGE_MODE
    normalized = str(mode or "en").strip().lower()
    if normalized not in {"en", "zh"}:
        raise ValueError(f"Unsupported language mode: {mode}")
    _LANGUAGE_MODE = normalized


def get_language_mode() -> str:
    return _LANGUAGE_MODE

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
    {
        "name": "zh_days_ago",
        "pattern": re.compile(r"(?P<value>\d+|[一二两三四五六七八九十]+)天前"),
        "kind": "days_ago",
        "precision": "date",
    },
    {
        "name": "zh_day_before_yesterday",
        "pattern": re.compile(r"前天"),
        "kind": "day_offset",
        "precision": "date",
        "offset_days": -2,
    },
    {
        "name": "zh_yesterday",
        "pattern": re.compile(r"昨天"),
        "kind": "day_offset",
        "precision": "date",
        "offset_days": -1,
    },
    {
        "name": "zh_today",
        "pattern": re.compile(r"今天"),
        "kind": "day_offset",
        "precision": "date",
        "offset_days": 0,
    },
    {
        "name": "zh_last_year",
        "pattern": re.compile(r"去年"),
        "kind": "year_offset",
        "precision": "year",
        "offset_years": -1,
    },
    {
        "name": "zh_this_year",
        "pattern": re.compile(r"今年"),
        "kind": "year_offset",
        "precision": "year",
        "offset_years": 0,
    },
    {
        "name": "zh_last_month",
        "pattern": re.compile(r"上个月"),
        "kind": "month_offset",
        "precision": "month",
        "offset_months": -1,
    },
    {
        "name": "zh_next_month",
        "pattern": re.compile(r"下个月"),
        "kind": "month_offset",
        "precision": "month",
        "offset_months": 1,
    },
    {
        "name": "zh_this_month",
        "pattern": re.compile(r"(这个月|本月)"),
        "kind": "current_month",
        "precision": "month",
    },
    {
        "name": "zh_this_week",
        "pattern": re.compile(r"(这周|本周)"),
        "kind": "relative_label",
        "precision": "relative",
        "label": "week of",
    },
    {
        "name": "zh_last_week",
        "pattern": re.compile(r"上周"),
        "kind": "relative_label",
        "precision": "relative",
        "label": "week before",
    },
    {
        "name": "zh_next_week",
        "pattern": re.compile(r"下周"),
        "kind": "relative_label",
        "precision": "relative",
        "label": "week after",
    },
    {
        "name": "zh_last_weekend",
        "pattern": re.compile(r"上周末"),
        "kind": "relative_label",
        "precision": "relative",
        "label": "weekend before",
    },
    {
        "name": "zh_last_weekday",
        "pattern": re.compile(r"上周(?P<weekday>[一二三四五六日天])"),
        "kind": "last_weekday",
        "precision": "relative",
    },
)

TEMPORAL_QUERY_PATTERNS = {
    "when", "date", "day", "month", "year", "time",
    "什么时候", "哪天", "几月", "几号", "日期", "时间", "当天", "那天", "当时",
}
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
    "性格",
    "特质",
    "可能",
    "会不会",
    "倾向",
    "职业",
    "工作",
    "领域",
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
        delta_days = _parse_relative_number(raw_value)
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
        weekday_name = WEEKDAY_NAMES.get(weekday_key) or ZH_WEEKDAY_NAMES.get(weekday_key)
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
    if any(token in lowered for token in TEMPORAL_QUERY_PATTERNS):
        return True
    return bool(
        re.search(r"\b(19|20)\d{2}\b", lowered)
        or re.search(r"(19|20)\d{2}年", str(query or ""))
        or re.search(r"\d{1,2}月\d{1,2}[日号]?", str(query or ""))
    )


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
    normalized = normalize_surface_text(query).lower()
    if not normalized:
        return set()
    if get_language_mode() == "zh":
        return _query_tokens_zh(normalized)
    return {
        token
        for token in re.sub(r"[^a-z0-9\s]", " ", normalized).split()
        if len(token) > 2
    }


TEXT_FRAGMENT_MAX_UNITS = 400
TEXT_FRAGMENT_OVERLAP_UNITS = 40
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?。！？])\s+|\n+")
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[,;:，；：、])\s*")


def _parse_relative_number(raw_value: str) -> int | None:
    text = str(raw_value or "").strip().lower()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in NUMBER_WORDS:
        return NUMBER_WORDS[text]
    if text in ZH_NUMBER_WORDS:
        return ZH_NUMBER_WORDS[text]
    if text == "十一":
        return 11
    if text == "十二":
        return 12
    if len(text) == 2 and text.startswith("十") and text[1] in ZH_NUMBER_WORDS:
        return 10 + ZH_NUMBER_WORDS[text[1]]
    if len(text) == 2 and text.endswith("十") and text[0] in ZH_NUMBER_WORDS:
        return ZH_NUMBER_WORDS[text[0]] * 10
    return None


def _query_tokens_zh(query: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", query):
        if len(token) > 1:
            tokens.add(token)
    for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", query):
        clean = str(run).strip()
        if len(clean) >= 2:
            if len(clean) <= 8:
                tokens.add(clean)
            upper = min(4, len(clean))
            for size in range(2, upper + 1):
                for index in range(0, len(clean) - size + 1):
                    tokens.add(clean[index : index + size])
    return {token for token in tokens if token}


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
