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
from leaf.grounding import is_inference_query, is_temporal_query, query_tokens as make_query_tokens
from leaf.normalize import contains_cjk, language_aware_terms, normalize_surface_text, strip_edge_punctuation
from leaf.service import LEAFService

try:
    import regex as unicode_re
except ImportError:  # pragma: no cover - optional dependency fallback
    unicode_re = re

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

TITLE_QUERY_PATTERNS = (
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

EXACT_SHORT_QUERY_PATTERNS = (
    "what was its name",
    "what is its name",
    "what movie",
    "what novel",
    "what book",
    "what film",
    "what song",
    "what exhibition",
    "what attraction",
    "what city",
    "what is my favorite",
    "what's my favorite",
)

ZH_LIST_QUERY_PATTERNS = (
    "哪些",
    "哪几",
    "哪两",
    "几个",
    "几部",
    "几本",
    "几种",
)

ZH_LIST_TAIL_PATTERNS = (
    "建议",
    "方法",
    "技巧",
    "话题",
    "书",
    "书目",
    "电影",
    "比赛",
    "菜系",
    "地方",
    "城市",
    "项目",
    "运动",
    "问题",
    "礼物",
)

ZH_TITLE_QUERY_PATTERNS = (
    "名字",
    "书",
    "电影",
    "歌曲",
    "电视剧",
    "展览",
    "景点",
    "地方",
    "城市",
    "国家",
)

ZH_EXACT_SHORT_QUERY_PATTERNS = (
    "名字是",
    "叫什么",
    "它的名字",
    "它是",
    "哪个国家",
    "哪个地方",
    "哪个城市",
    "什么颜色",
    "它的颜色",
    "哪一部电影",
    "哪部电影",
    "哪本书",
    "哪本小说",
    "哪一天",
    "哪天",
    "多少人",
)

ZH_SUPPORT_QUERY_PATTERNS = (
    "为什么",
    "问题",
    "困难",
    "建议",
    "方法",
    "技巧",
    "话题",
    "灵感",
    "怎么",
    "如何",
)

TITLE_SPAN_RE = unicode_re.compile(r"[《“\"「](.{1,60}?)[》”\"」]")
NUMBER_SPAN_RE = unicode_re.compile(r"(?:大约|约|大概)?\s*([一二两三四五六七八九十百千万0-9]+)\s*(?:个人左右|人左右|个人|人)")


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
    text = normalize_surface_text(question)
    lowered = text.lower()
    if any(month in lowered for month in MONTH_NAMES) and bool(re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b", lowered)):
        return True
    return bool(re.search(r"(19|20)\d{2}年\d{1,2}月\d{1,2}[日号]?|\d{1,2}月\d{1,2}[日号]?", text))


def target_month_day(question: str) -> str | None:
    text = normalize_surface_text(question)
    lowered = text.lower()
    zh_match = re.search(r"(?:(19|20)\d{2}年)?(\d{1,2})月(\d{1,2})[日号]?", text)
    if zh_match:
        return f"{int(zh_match.group(2)):02d}-{int(zh_match.group(3)):02d}"
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
    text = normalize_surface_text(question)
    lowered = text.lower()
    if any(pattern in lowered for pattern in LIST_QUERY_PATTERNS):
        return True
    if any(pattern in text for pattern in ZH_LIST_QUERY_PATTERNS):
        return True
    return bool(re.search(r"什么.*(" + "|".join(re.escape(token) for token in ZH_LIST_TAIL_PATTERNS) + r")", text))


def is_yes_no_query(question: str) -> bool:
    text = normalize_surface_text(question)
    lowered = text.lower()
    if bool(lowered) and (lowered.startswith(YES_NO_PREFIXES) or lowered.endswith("right?") or ", right?" in lowered):
        return True
    return bool(re.search(r"(对吗|是吗|是不是|有没有|是否|会不会|还记得.*吗|记得.*吗)[\?？]?$", text))


def use_exact_short_mode(question: str) -> bool:
    text = normalize_surface_text(question)
    lowered = text.lower()
    if not lowered or is_inference_query(question):
        return False
    if contains_cjk(text):
        if any(token in text for token in ["建议", "方法", "技巧", "怎么", "如何", "为什么", "话题", "具体的做法", "主要内容"]):
            return False
        return is_yes_no_query(text) or is_single_fact_query(text) or any(pattern in text for pattern in ZH_EXACT_SHORT_QUERY_PATTERNS)
    if is_list_query(question):
        return False
    if any(
        token in lowered
        for token in [
            "recipe",
            "advice",
            "suggestions",
            "methods",
            "what were",
            "what problems",
            "what issues",
            "what topics",
            "content",
            "what did i do",
            "why was",
            "what inspired",
            "which books",
            "what books",
            "how am i feeling",
            "living a healthy life",
        ]
    ):
        return False
    if is_yes_no_query(question):
        return True
    return any(pattern in lowered for pattern in EXACT_SHORT_QUERY_PATTERNS)


def answer_query_terms(question: str) -> set[str]:
    text = normalize_surface_text(question)
    mode = "zh" if contains_cjk(text) else "en"
    tokens = set(language_aware_terms(text, mode=mode, include_cjk_subgrams=False))
    tokens.update(token for token in make_query_tokens(text) if token not in ANSWER_STOPWORDS)
    tokens = {token for token in tokens if token not in ANSWER_STOPWORDS}
    if contains_cjk(text):
        return tokens
    return {token for token in tokens if len(token) > 2}


def score_context_line(question: str, line: str) -> float:
    question_text = str(question or "")
    lowered_question = question_text.lower()
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
    if any(pattern in lowered_question for pattern in TITLE_QUERY_PATTERNS):
        if '"' in line or "'" in line:
            score += 0.35
        if ":" in line:
            score += 0.05
    if contains_cjk(question_text) and any(pattern in question_text for pattern in ZH_TITLE_QUERY_PATTERNS):
        if any(marker in str(line or "") for marker in ["《", "》", "“", "”", "「", "」"]):
            score += 0.35
        if "：" in str(line or "") or ":" in str(line or ""):
            score += 0.05
    if is_yes_no_query(question):
        if (
            "yes" in lowered_line
            or "no" in lowered_line
            or "like" in lowered_line
            or "don't like" in lowered_line
            or any(token in str(line or "") for token in ["对", "不对", "不是", "喜欢", "不喜欢", "讨厌"])
        ):
            score += 0.2
    if ("favorite" in lowered_question and "favorite" in lowered_line) or ("最喜欢" in question_text and "最喜欢" in str(line or "")):
        score += 0.25
    if ("plan" in lowered_question and "travel" in lowered_line) or ("计划" in question_text and any(token in str(line or "") for token in ["计划", "打算", "想去"])):
        score += 0.15
    if ("problem" in lowered_question and "problem" in lowered_line) or ("问题" in question_text and "问题" in str(line or "")):
        score += 0.15
    return score


def normalize_context_text(text: str) -> str:
    return normalize_surface_text(text)


def strip_context_prefix(line: str) -> str:
    text = normalize_context_text(line)
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = re.sub(r"^[^:：]{1,40}[:：]\s*", "", text)
    return text.strip()


def clean_answer_phrase(text: str) -> str:
    value = normalize_context_text(text)
    value = strip_edge_punctuation(value)
    value = re.sub(r"^(是|就是|应该是)\s*", "", value)
    return value.strip()


def dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in lines:
        compact = normalize_context_text(line)
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(compact)
    return ordered


def extract_answer_view_lines(answer_view: dict[str, Any], key: str) -> list[str]:
    items = answer_view.get(key) or []
    if not isinstance(items, list):
        items = [items]
    lines: list[str] = []
    for item in items:
        text = str(item.get("text") or "").strip() if isinstance(item, dict) else str(item or "").strip()
        compact = normalize_context_text(text)
        if compact:
            lines.append(compact)
    return dedupe_preserve_order(lines)


def trim_display_line(text: str, max_chars: int = 220) -> str:
    compact = normalize_context_text(text)
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def line_has_temporal_signal(line: str) -> bool:
    text = str(line or "").strip()
    lowered = text.lower()
    if not lowered:
        return False
    if re.search(r"\b(19|20)\d{2}\b", lowered):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\b", lowered):
        return True
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", lowered):
        return True
    if re.search(r"(19|20)\d{2}年|\d{1,2}月\d{1,2}[日号]?", text):
        return True
    if any(month in lowered for month in MONTH_NAMES):
        return True
    return any(
        token in lowered or token in text
        for token in [
            "today",
            "yesterday",
            "tomorrow",
            "last week",
            "last weekend",
            "this week",
            "next week",
            "last month",
            "this month",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "今天",
            "昨天",
            "前天",
            "那天",
            "当时",
            "上周",
            "本周",
            "这周",
            "下周",
            "上周末",
        ]
    )


def score_heuristic_bucket_line(question: str, line: str, *, bucket: str) -> float:
    score = score_context_line(question, line)
    question_text = str(question or "").strip()
    line_text = str(line or "").strip()
    lowered = str(line or "").strip().lower()
    if not lowered:
        return score
    if bucket == "direct":
        if line_supports_exact_answer(question, line):
            score += 0.45
        if "?" in line_text or "？" in line_text:
            score -= 0.3
        if re.match(r"^\[[^\]]+\]\s+", str(line or "").strip()):
            score += 0.08
        if ":" in line:
            score += 0.04
        if contains_cjk(question_text):
            if "多少" in question_text and re.search(r"[一二两三四五六七八九十百千万0-9]+(个|人|位|名)", line_text):
                score += 0.5
            if any(token in question_text for token in ["哪个国家", "什么国家", "哪个地方", "什么地方", "哪里", "哪儿"]):
                if any(token in line_text for token in ["去", "在", "附近", "海滩", "山", "湖", "公园", "国家", "城市"]):
                    score += 0.25
            if "话题" in question_text or "聊了什么" in question_text:
                if any(token in line_text for token in ["你好", "很高兴", "朋友", "谢谢", "好朋友", "聆听"]):
                    score -= 0.25
                elif len(line_text) >= 24 and "？" not in line_text and "?" not in line_text:
                    score += 0.12
    elif bucket == "fact":
        if any(token in lowered for token in ["favorite", "plan", "goal", "problem", "issue", "career", "study", "want", "wants to"]):
            score += 0.18
        if is_inference_query(question):
            score += 0.12
    elif bucket == "page":
        if any(token in lowered for token in ["summary:", "context", "overlap"]):
            score -= 0.04
    elif bucket == "temporal":
        if line_has_temporal_signal(line):
            score += 0.3
        if has_explicit_date(question):
            date = target_month_day(question)
            if date and line_date_token(line) == date:
                score += 0.5
    elif bucket == "relation":
        if " " in lowered:
            score += 0.05
    return score


def select_diverse_scored_lines(
    question: str,
    candidates: list[str],
    *,
    bucket: str,
    limit: int,
    anchor_session: str | None = None,
) -> list[str]:
    scored: list[tuple[float, int, str, str | None]] = []
    for index, line in enumerate(dedupe_preserve_order(candidates)):
        compact = trim_display_line(line)
        if not compact:
            continue
        score = score_heuristic_bucket_line(question, compact, bucket=bucket)
        session_id = extract_session_id_from_context_line(compact)
        if anchor_session and session_id == anchor_session and bucket in {"direct", "temporal"}:
            score += 0.08
        scored.append((score, index, compact, session_id))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected: list[str] = []
    selected_sessions: set[str] = set()
    selected_norms: list[set[str]] = []
    for score, _, compact, session_id in scored:
        norm_terms = answer_query_terms(compact)
        redundancy = 0.0
        for existing_terms in selected_norms:
            overlap = len(norm_terms.intersection(existing_terms))
            redundancy = max(redundancy, overlap * 0.03)
        adjusted = score - redundancy
        if adjusted <= 0 and selected:
            continue
        if session_id and session_id in selected_sessions and bucket in {"direct", "temporal"} and len(selected) >= max(2, limit - 1):
            continue
        selected.append(compact)
        if session_id:
            selected_sessions.add(session_id)
        selected_norms.append(norm_terms)
        if len(selected) >= limit:
            break
    return selected


def reshape_heuristic_answer_view_for_gvd(
    question: str,
    *,
    answer_view: dict[str, Any],
    context_lines: list[str],
) -> dict[str, Any]:
    if any(key in answer_view for key in ("direct_evidence", "entity_facts", "temporal_clues", "relation_paths", "page_context")):
        return answer_view

    raw_lines = dedupe_preserve_order(context_lines + extract_answer_view_lines(answer_view, "raw_evidence"))
    fact_lines = extract_answer_view_lines(answer_view, "facts")
    page_lines = extract_answer_view_lines(answer_view, "page_summaries")
    relation_lines = extract_answer_view_lines(answer_view, "relations")

    explicit_date = has_explicit_date(question)
    inference_mode = is_inference_query(question)
    list_query = is_list_query(question)
    yes_no_query = is_yes_no_query(question)
    single_fact_query = is_single_fact_query(question)

    direct_seed_limit = 4 if (single_fact_query or yes_no_query) else 5 if list_query else 4
    direct_evidence = select_diverse_scored_lines(
        question,
        raw_lines,
        bucket="direct",
        limit=direct_seed_limit,
    )

    anchor_session = extract_session_id_from_context_line(direct_evidence[0]) if direct_evidence else None
    if anchor_session and (single_fact_query or yes_no_query):
        same_session_lines = [
            line
            for line in raw_lines
            if extract_session_id_from_context_line(line) == anchor_session
        ]
        anchored_direct = select_diverse_scored_lines(
            question,
            same_session_lines,
            bucket="direct",
            limit=direct_seed_limit,
            anchor_session=anchor_session,
        )
        if len(anchored_direct) >= max(2, min(3, direct_seed_limit)):
            direct_evidence = anchored_direct

    temporal_sources = raw_lines + fact_lines + page_lines
    temporal_clues = select_diverse_scored_lines(
        question,
        [line for line in temporal_sources if line_has_temporal_signal(line)] or temporal_sources,
        bucket="temporal",
        limit=4 if explicit_date else 2,
        anchor_session=anchor_session,
    )

    fact_sources = fact_lines + [line for line in page_lines if line not in fact_lines]
    entity_facts = select_diverse_scored_lines(
        question,
        fact_sources,
        bucket="fact",
        limit=5 if inference_mode else 4 if list_query else 3,
        anchor_session=anchor_session,
    )

    page_context = select_diverse_scored_lines(
        question,
        [line for line in page_lines if line not in entity_facts],
        bucket="page",
        limit=3 if inference_mode else 2,
        anchor_session=anchor_session,
    )

    relation_paths = select_diverse_scored_lines(
        question,
        relation_lines,
        bucket="relation",
        limit=3 if list_query else 2,
    )

    reshaped = {
        "direct_evidence": direct_evidence,
        "entity_facts": entity_facts,
        "temporal_clues": temporal_clues,
        "relation_paths": relation_paths,
        "page_context": page_context,
        "insufficient": [],
    }
    if not any(reshaped[key] for key in ("direct_evidence", "entity_facts", "temporal_clues", "relation_paths", "page_context")):
        reshaped["insufficient"] = ["No strong grounded evidence selected."]
    return reshaped


def line_date_token(line: str) -> str | None:
    text = str(line or "")
    match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if not match:
        return None
    return f"{match.group(2)}-{match.group(3)}"


def line_supports_exact_answer(question: str, line: str) -> bool:
    question_text = str(question or "").strip()
    lowered_question = question_text.lower()
    line_text = str(line or "")
    lowered_line = line_text.strip().lower()
    if not lowered_line:
        return False
    if is_yes_no_query(question):
        polarity_markers = [
            " like ",
            " likes ",
            " don't like",
            " doesn't like",
            "love",
            "loved",
            "favorite",
            "prefer",
            "preferred",
            "enjoy",
            "enjoyed",
            "dislike",
            "hate",
            "喜欢",
            "不喜欢",
            "不对",
            "不是",
            "讨厌",
        ]
        return any(marker in f" {lowered_line} " for marker in polarity_markers)
    if any(pattern in lowered_question for pattern in TITLE_QUERY_PATTERNS) or any(
        pattern in question_text for pattern in ZH_TITLE_QUERY_PATTERNS
    ):
        if any(marker in line_text for marker in ['"', "'", "《", "》", "“", "”", "「", "」"]):
            return True
        if re.search(r"\b[A-Z][A-Za-z0-9'&-]+(?:\s+[A-Z][A-Za-z0-9'&-]+){0,6}\b", line_text):
            return True
    if any(token in lowered_question for token in ["favorite", "problem", "issue", "inspired", "why", "what did i do"]):
        salient_markers = [
            "favorite",
            "problem",
            "issue",
            "difficulty",
            "struggling",
            "inspired",
            "because",
            "due to",
            "asked",
            "went",
            "read",
            "watched",
            "liked",
            "loved",
        ]
        return any(marker in lowered_line for marker in salient_markers)
    if any(token in question_text for token in ["最喜欢", "问题", "困难", "灵感", "为什么", "做法"]):
        return any(
            marker in line_text
            for marker in ["最喜欢", "问题", "困难", "因为", "由于", "灵感", "做法", "步骤", "推荐", "建议", "去了", "看了", "读了"]
        )
    return score_context_line(question, line) >= 0.6


def select_short_answer_context(
    question: str,
    *,
    evidence: dict[str, Any],
    answer_view: dict[str, Any],
    context_lines: list[str],
) -> tuple[str, dict[str, Any]]:
    single_fact_query = is_single_fact_query(question)
    yes_no_query = is_yes_no_query(question)
    list_query = is_list_query(question)
    explicit_date = has_explicit_date(question)
    inference_mode = is_inference_query(question)
    month_day = target_month_day(question)

    direct_lines = extract_answer_view_lines(answer_view, "direct_evidence")
    if not direct_lines:
        direct_lines = context_lines
    direct_lines = filter_context_lines_for_question(question, direct_lines)
    if explicit_date and month_day is not None:
        dated = [line for line in direct_lines if line_date_token(line) == month_day]
        if dated:
            direct_lines = dated
    direct_scored = [
        (score_heuristic_bucket_line(question, line, bucket="direct"), index, normalize_context_text(line))
        for index, line in enumerate(direct_lines)
        if normalize_context_text(line)
    ]
    direct_scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    top_direct_score = direct_scored[0][0] if direct_scored else 0.0
    dominant_session = extract_session_id_from_context_line(direct_scored[0][2]) if direct_scored else None
    if dominant_session and not inference_mode and (single_fact_query or yes_no_query or list_query):
        same_session = [item for item in direct_scored if extract_session_id_from_context_line(item[2]) == dominant_session]
        if len(same_session) >= 2 or (same_session and top_direct_score >= 0.45):
            direct_scored = same_session
    max_direct = 3 if (single_fact_query or yes_no_query) else 4 if list_query else 4
    selected_direct = [item[2] for item in direct_scored[:max_direct]]

    support_lines: list[str] = []
    if list_query or inference_mode or any(
        token in str(question or "").lower() for token in ["why", "problem", "issue", "inspired", "what did i do"]
    ) or any(token in str(question or "") for token in ZH_SUPPORT_QUERY_PATTERNS):
        support_lines.extend(extract_answer_view_lines(answer_view, "entity_facts")[:2])
    if explicit_date:
        support_lines.extend(extract_answer_view_lines(answer_view, "temporal_clues")[:1])
    support_lines = [line for line in dedupe_preserve_order(support_lines) if line not in selected_direct]

    if not selected_direct:
        fallback_lines, fallback_score = prioritize_answer_context(question, context_lines)
        selected_direct = fallback_lines[:max_direct]
        top_direct_score = max(top_direct_score, fallback_score)

    selected_direct = dedupe_preserve_order(selected_direct)
    has_concrete_direct_clue = any(line_supports_exact_answer(question, line) for line in selected_direct)

    rendered: list[str] = ["[Direct Evidence]"]
    rendered.extend(f"- {line}" for line in selected_direct)
    if support_lines:
        rendered.extend(["", "[Supporting Facts]"])
        rendered.extend(f"- {line}" for line in support_lines)
    return "\n".join(rendered).strip(), {
        "top_direct_score": round(top_direct_score, 4),
        "direct_count": len(selected_direct),
        "support_count": len(support_lines),
        "has_concrete_direct_clue": has_concrete_direct_clue,
        "dominant_session": dominant_session,
    }


def extract_structured_answer_from_context(
    question: str,
    *,
    answer_view: dict[str, Any],
    context_lines: list[str],
    language_mode: str,
) -> str | None:
    if language_mode != "zh":
        return None

    def normalize_place_answer(text: str) -> str:
        candidate = clean_answer_phrase(text)
        if not candidate:
            return ""
        candidate = re.split(r"[，,。；;！？!?]", candidate, maxsplit=1)[0].strip()
        candidate = re.sub(r"(旅行|旅游|游玩|度假|玩儿|玩)$", "", candidate).strip()
        candidate = re.sub(r"(的)?(地方|城市|国家)$", "", candidate).strip()
        if not candidate:
            return ""
        if re.search(r"(哪里|哪儿|哪个|什么地方|什么城市|什么国家|何处)", candidate):
            return ""
        return candidate

    direct_lines = extract_answer_view_lines(answer_view, "direct_evidence")
    base_lines = filter_context_lines_for_question(question, list(context_lines))
    if direct_lines:
        merged_lines = dedupe_preserve_order(direct_lines + base_lines)
    else:
        merged_lines = base_lines
    if not merged_lines:
        return None

    explicit_date = has_explicit_date(question)
    month_day = target_month_day(question)
    if explicit_date and month_day is not None:
        dated = [line for line in merged_lines if line_date_token(line) == month_day]
        if dated:
            merged_lines = dated

    line_rows: list[dict[str, Any]] = []
    for index, line in enumerate(merged_lines):
        payload = strip_context_prefix(line)
        if not payload:
            continue
        line_rows.append(
            {
                "raw": line,
                "payload": payload,
                "index": index,
                "is_assistant": "AI Companion:" in line,
            }
        )
    if not line_rows:
        return None

    question_text = str(question or "").strip()
    list_query = is_list_query(question_text)
    prefers_user_statement = bool(
        contains_cjk(question_text)
        and (
            question_text.startswith("我")
            or "我" in question_text
            or "我的" in question_text
            or "我们" in question_text
        )
    )

    if is_yes_no_query(question_text):
        negative_match = re.search(r"我不喜欢(.+?)(?:，|,|。|\?|？|对吗|是吗|是不是|有没有|是否|会不会)", question_text)
        if negative_match:
            target = clean_answer_phrase(negative_match.group(1))
            for row in line_rows:
                payload = str(row["payload"])
                if target and target in payload and "喜欢" in payload and "不喜欢" not in payload:
                    return f"不对，你喜欢{target}"
        generic_negative_match = re.search(r"不是(.+?)(?:，|,|。|\?|？|对吗|是吗|是不是|有没有|是否|会不会)", question_text)
        if generic_negative_match:
            target = clean_answer_phrase(generic_negative_match.group(1))
            for row in line_rows:
                payload = str(row["payload"])
                if target and target in payload:
                    return f"不是，{target}"

    if "多少人" in question_text or "几个人" in question_text or "人数" in question_text:
        for row in line_rows:
            payload = str(row["payload"])
            match = NUMBER_SPAN_RE.search(payload)
            if match:
                phrase = clean_answer_phrase(match.group(0))
                if phrase:
                    return phrase

    title_like_query = any(
        token in question_text
        for token in [
            "哪本书",
            "哪本小说",
            "哪部电影",
            "哪一部电影",
            "什么歌",
            "最喜欢他的什么歌",
            "它的名字",
            "名字是",
            "叫什么",
        ]
    ) or bool(re.search(r"(哪|什么).*(书|小说|电影|歌曲|歌|名字)|最喜欢.*歌", question_text))
    if title_like_query and not list_query:
        title_candidates: list[tuple[float, str]] = []
        for row in line_rows:
            payload = str(row["payload"])
            base_score = score_context_line(question_text, payload) + max(0.0, 0.2 - int(row["index"]) * 0.02)
            if prefers_user_statement and not bool(row["is_assistant"]):
                base_score += 0.18
            if "书" in question_text and not any(token in payload for token in ["书", "小说", "读", "买"]):
                base_score -= 0.18
            if "电影" in question_text and "电影" not in payload and "片" not in payload:
                base_score -= 0.18
            if "歌" in question_text and not any(token in payload for token in ["歌", "歌曲", "演唱会", "最喜欢"]):
                base_score -= 0.18
            if "名字" in question_text and not any(token in payload for token in ["叫做", "名叫", "名字", "是《"]):
                base_score -= 0.12
            for match in TITLE_SPAN_RE.finditer(payload):
                title = clean_answer_phrase(match.group(1))
                if not title:
                    continue
                score = base_score
                left_context = payload[: match.start()]
                if "最喜欢" in left_context:
                    score += 0.25
                if any(token in left_context for token in ["推荐", "看了", "读了", "分享", "提到"]):
                    score += 0.12
                if any(token in payload for token in ["我看的是", "我买了一本", "我去看了", "我读了", "我分享过"]):
                    score += 0.22
                title_candidates.append((score, f"《{title}》"))
        if title_candidates:
            title_candidates.sort(key=lambda item: item[0], reverse=True)
            seen: set[str] = set()
            ordered_titles: list[str] = []
            for _, title in title_candidates:
                if title in seen:
                    continue
                seen.add(title)
                ordered_titles.append(title)
            return ordered_titles[0]

    if any(token in question_text for token in ["哪个国家", "什么国家", "哪个城市", "什么城市", "哪个地方", "什么地方", "哪里", "哪儿"]):
        fact_payloads = [
            strip_context_prefix(line)
            for line in extract_answer_view_lines(answer_view, "entity_facts")
        ]
        search_payloads = [str(row["payload"]) for row in line_rows] + [payload for payload in fact_payloads if payload]
        for payload in search_payloads:
            patterns = [
                r"(?:计划去|打算去|想去|准备去)([^，。；！？\s]{1,20})",
                r"(?:想去|计划去|打算去|准备去)[^，。；！？]{0,20}?([一-龥]{2,12})(?:旅行|旅游|游玩|度假)",
                r"(?:去了|去过)([^，。；！？\s]{1,20})",
                r"在(.{1,20}?)拍摄",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, payload):
                    candidate = normalize_place_answer(match.group(1))
                    if candidate:
                        return candidate

    return None


def build_structured_answer_payload(
    question: str,
    *,
    evidence: dict[str, Any],
    answer_view: dict[str, Any],
    context_lines: list[str],
    language_mode: str,
) -> dict[str, Any]:
    direct_lines = extract_answer_view_lines(answer_view, "direct_evidence")
    fact_lines = extract_answer_view_lines(answer_view, "entity_facts")
    temporal_lines = extract_answer_view_lines(answer_view, "temporal_clues")
    relation_lines = extract_answer_view_lines(answer_view, "relation_paths")
    page_lines = extract_answer_view_lines(answer_view, "page_context")

    ordered_session_ids: list[str] = []
    for line in direct_lines + temporal_lines + context_lines:
        session_id = extract_session_id_from_context_line(line)
        if session_id and session_id not in ordered_session_ids:
            ordered_session_ids.append(session_id)

    session_windows_map: dict[str, list[tuple[int, str]]] = {}
    raw_spans = evidence.get("raw_spans") or []
    for span in raw_spans:
        session_id = str(span.get("session_id") or span.get("timestamp") or "").strip()
        if not session_id:
            continue
        speaker = str(span.get("speaker") or "").strip()
        text = normalize_context_text(span.get("text") or "")
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
            normalized = rendered.lower()
            if normalized in seen_lines:
                continue
            seen_lines.add(normalized)
            lines.append(rendered)
            if len(lines) >= 4:
                break
        if lines:
            session_windows.append({"session_id": session_id, "lines": lines})

    return {
        "question": question,
        "language": language_mode,
        "question_profile": {
            "explicit_date": has_explicit_date(question),
            "target_month_day": target_month_day(question),
            "list": is_list_query(question),
            "yes_no": is_yes_no_query(question),
            "single_fact": is_single_fact_query(question),
            "inference": is_inference_query(question),
        },
        "direct_evidence": direct_lines[:4],
        "session_windows": session_windows,
        "entity_facts": fact_lines[:4],
        "temporal_clues": temporal_lines[:3],
        "relation_paths": relation_lines[:3],
        "page_context": page_lines[:3],
    }


def render_structured_answer_text(payload: dict[str, Any], *, language_mode: str) -> str:
    is_zh = language_mode == "zh"
    profile = dict(payload.get("question_profile") or {})
    tags: list[str] = []
    if profile.get("single_fact"):
        tags.append("single_fact")
    if profile.get("list"):
        tags.append("list")
    if profile.get("yes_no"):
        tags.append("yes_no")
    if profile.get("inference"):
        tags.append("inference")
    if profile.get("explicit_date"):
        month_day = str(profile.get("target_month_day") or "").strip()
        tags.append(f"date={month_day}" if month_day else "explicit_date")

    section_specs: list[tuple[str, list[str]]] = []
    if tags:
        section_specs.append(("问题类型" if is_zh else "Question Profile", tags))
    direct_lines = [str(item).strip() for item in (payload.get("direct_evidence") or []) if str(item).strip()]
    if direct_lines:
        section_specs.append(("直接证据" if is_zh else "Direct Evidence", direct_lines))

    session_windows = payload.get("session_windows") or []
    for row in session_windows:
        if not isinstance(row, dict):
            continue
        session_id = str(row.get("session_id") or "").strip()
        lines = [str(item).strip() for item in (row.get("lines") or []) if str(item).strip()]
        if not lines:
            continue
        if is_zh:
            label = f"会话窗口: {session_id}" if session_id else "会话窗口"
        else:
            label = f"Session Window: {session_id}" if session_id else "Session Window"
        section_specs.append((label, lines))

    fact_lines = [str(item).strip() for item in (payload.get("entity_facts") or []) if str(item).strip()]
    if fact_lines:
        section_specs.append(("补充事实" if is_zh else "Supporting Facts", fact_lines))
    temporal_lines = [str(item).strip() for item in (payload.get("temporal_clues") or []) if str(item).strip()]
    if temporal_lines:
        section_specs.append(("时间线索" if is_zh else "Temporal Clues", temporal_lines))
    relation_lines = [str(item).strip() for item in (payload.get("relation_paths") or []) if str(item).strip()]
    if relation_lines:
        section_specs.append(("关系线索" if is_zh else "Relation Paths", relation_lines))
    page_lines = [str(item).strip() for item in (payload.get("page_context") or []) if str(item).strip()]
    if page_lines:
        section_specs.append(("页面上下文" if is_zh else "Page Context", page_lines))

    lines: list[str] = []
    for index, (label, items) in enumerate(section_specs):
        if index:
            lines.append("")
        lines.append(f"[{label}]")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines).strip()


def prioritize_answer_context(question: str, context_lines: list[str]) -> tuple[list[str], float]:
    if not context_lines:
        return [], 0.0
    scored = [
        (score_heuristic_bucket_line(question, line, bucket="direct"), index, line)
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
    text = str(question or "").strip()
    lowered = text.lower()
    if not lowered:
        return False
    if is_list_query(lowered):
        return False
    if is_yes_no_query(lowered):
        return True
    if any(pattern in lowered for pattern in SINGLE_FACT_PATTERNS):
        return True
    if contains_cjk(text):
        if any(pattern in text for pattern in ZH_EXACT_SHORT_QUERY_PATTERNS):
            return True
        if re.search(r"(哪个|哪座|哪部|哪本|哪里|哪天|谁|多少人|什么颜色|什么地方|什么国家|什么城市|名字是|叫什么)", text):
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
    text = str(question or "").strip()
    lowered = text.lower()
    if "first conversation" in lowered or any(token in text for token in ["第一次对话", "第一次聊天", "最开始的对话"]):
        session_ids = [extract_session_id_from_context_line(line) for line in context_lines]
        dated_sessions = sorted({sid for sid in session_ids if sid}, key=session_sort_key)
        if dated_sessions:
            first_session = dated_sessions[0]
            first_lines = [line for line in context_lines if extract_session_id_from_context_line(line) == first_session]
            if first_lines:
                return first_lines
    if any(token in lowered for token in ["last conversation", "most recent conversation", "latest conversation"]) or any(
        token in text for token in ["最后一次对话", "最近一次对话", "最新一次对话"]
    ):
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
    answer_view: dict[str, Any] | None = None,
    language_mode: str = "en",
) -> list[dict[str, str]]:
    context_lines = context_lines if context_lines is not None else build_answer_context_lines(evidence)
    exact_short_mode = answer_style == "short" and (
        use_exact_short_mode(question) or (language_mode == "zh" and is_list_query(question))
    )
    memoryos_aligned_mode = answer_style == "memoryos_aligned"
    structured_context_mode = answer_style == "structured_context"
    structured_compact_mode = answer_style == "structured_compact"
    single_fact_query = is_single_fact_query(question)
    explicit_date = has_explicit_date(question)
    yes_no_query = is_yes_no_query(question)
    inference_mode = is_inference_query(question)
    month_day = target_month_day(question)
    support_profile = {
        "top_direct_score": 0.0,
        "direct_count": 0,
        "support_count": 0,
        "has_concrete_direct_clue": False,
        "dominant_session": None,
    }
    if explicit_date and month_day is not None:
        same_day_lines = [line for line in context_lines if f"-{month_day}" in line]
        if same_day_lines:
            context_lines = same_day_lines
    if (
        language_mode == "zh"
        and explicit_date
        and month_day is not None
        and any(token in str(question or "") for token in ["聊了什么话题", "什么话题"])
    ):
        same_day_spans = [
            span
            for span in (evidence.get("raw_spans") or [])
            if line_date_token(str(span.get("timestamp") or "")) == month_day
        ]
        same_day_spans.sort(key=lambda span: int(span.get("turn_index") or 0))
        topical_lines: list[str] = []
        for span in same_day_spans:
            text = normalize_context_text(span.get("text") or "")
            if not text:
                continue
            if any(token in text for token in ["你好", "很高兴", "好朋友", "下次再聊", "谢谢你的分享"]):
                continue
            speaker = str(span.get("speaker") or "").strip()
            topical_lines.append(f"[{span.get('timestamp')}] {speaker}: {text}")
        if topical_lines:
            context_lines = topical_lines[:6]
    if answer_style == "judge_aligned":
        context_lines = filter_context_lines_for_question(question, context_lines)
        prioritized_lines, top_support_score = prioritize_answer_context(question, context_lines)
        if prioritized_lines:
            context_lines = prioritized_lines
        context = answer_view_text if answer_view_text is not None else "\n".join(context_lines).strip()
    elif memoryos_aligned_mode:
        top_support_score = 0.0
        context = answer_view_text if answer_view_text is not None else "\n".join(context_lines).strip()
    elif structured_context_mode:
        top_support_score = 0.0
        context = json.dumps(
            build_structured_answer_payload(
                question,
                evidence=evidence,
                answer_view=answer_view or {},
                context_lines=context_lines,
                language_mode=language_mode,
            ),
            ensure_ascii=False,
            indent=2,
        )
    elif structured_compact_mode:
        top_support_score = 0.0
        context = render_structured_answer_text(
            build_structured_answer_payload(
                question,
                evidence=evidence,
                answer_view=answer_view or {},
                context_lines=context_lines,
                language_mode=language_mode,
            ),
            language_mode=language_mode,
        )
    elif exact_short_mode:
        context, support_profile = select_short_answer_context(
            question,
            evidence=evidence,
            answer_view=answer_view or {},
            context_lines=context_lines,
        )
        top_support_score = float(support_profile.get("top_direct_score") or 0.0)
    else:
        top_support_score = 0.0
        context = "\n".join(context_lines).strip()
    context = context.strip() or "(no retrieved evidence)"
    if language_mode == "zh":
        context = context.replace("[Direct Evidence]", "[直接证据]").replace("[Supporting Facts]", "[补充事实]")
    if memoryos_aligned_mode:
        instruction_parts = [
            "Answer the question using only the provided MemoryOS retrieval context.",
            "Prefer directly retrieved conversation pages.",
            "Use user profile and long-term knowledge only as supporting context.",
            "Return a short answer phrase, not a full explanation.",
            "If the context is insufficient, return UNKNOWN.",
        ]
    elif structured_context_mode:
        instruction_parts = [
            "Answer the question using only the provided structured retrieval payload.",
            "Treat direct_evidence as the strongest verbatim evidence.",
            "Use session_windows to recover local conversation flow and resolve nearby references.",
            "Use entity_facts, temporal_clues, relation_paths, and page_context only as supporting context.",
            "Prefer direct grounded answers over vague summaries.",
            "For list or support questions, combine all non-conflicting supported items from the payload.",
            "For title, slot, or exact-value questions, return the shortest exact value supported by the payload.",
            "If the payload is insufficient, return UNKNOWN.",
            "Return only the answer text with no explanation or preamble.",
        ]
    elif structured_compact_mode:
        instruction_parts = [
            "Answer the question using only the provided structured retrieval context.",
            "Direct Evidence contains the strongest verbatim evidence.",
            "Session Window contains a short local conversation window from the same retrieved sessions.",
            "Supporting Facts, Temporal Clues, Relation Paths, and Page Context are supporting signals only.",
            "Prefer direct grounded answers over vague summaries.",
            "For list or support questions, combine all non-conflicting supported items from the context.",
            "For title, slot, or exact-value questions, return the shortest exact value supported by the context.",
            "If the context is insufficient, return UNKNOWN.",
            "Return only the answer text with no explanation or preamble.",
        ]
    elif answer_style == "judge_aligned" and inference_mode:
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
            "Treat Direct Evidence as the strongest grounded snippets, Facts as distilled memory statements, Timeline as time anchors, Relations as graph relations, and Context as high-level support.",
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
    elif exact_short_mode:
        instruction_parts = [
            "Answer the question using only the provided evidence.",
            "Use Direct Evidence as the primary source.",
            "Use Supporting Facts only to complete a missing detail, not to broaden the answer.",
            "Respect the asked scope exactly and do not add nearby but different facts, plans, or examples.",
            "If the evidence is insufficient, answer UNKNOWN.",
            "Return only the shortest exact answer text with no explanation or preamble.",
            "Keep the answer directly grounded.",
        ]
        if inference_mode:
            instruction_parts.append(
                "You may make a grounded inference only when the answer is clearly implied by multiple evidence lines."
            )
        if list_query := is_list_query(question):
            instruction_parts.append(
                "For list questions, include only items explicitly supported by Direct Evidence and do not add related extras."
            )
        if yes_no_query:
            instruction_parts.append(
                "For confirmation questions, answer with a direct Yes or No first, then only the shortest supporting phrase if needed."
            )
            instruction_parts.append(
                "Evaluate whether the proposition in the question is true or false based on the evidence, and do not merely repeat the question's wording."
            )
            instruction_parts.append(
                "If the correct answer is No, state the corrected fact briefly instead of repeating the false negative wording from the question."
            )
        if support_profile.get("has_concrete_direct_clue"):
            instruction_parts.append(
                "The evidence already contains a concrete direct clue. Extract it and do not answer UNKNOWN."
            )
        if single_fact_query and top_support_score >= 0.55:
            instruction_parts.append(
                "This is an exact single-fact question. Prefer the most specific directly stated value over a vague summary."
            )
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
    if language_mode == "zh":
        instruction_parts.append("Answer in concise Chinese and prefer directly copied short phrases from the evidence.")
        instruction_parts.append("Do not restate the full question or the speaker unless needed for disambiguation.")
        if yes_no_query:
            instruction_parts.append("For Chinese confirmation questions, prefer compact forms like '不对，户外运动' or '不是'.")
    return [
        {
            "role": "system",
            "content": " ".join(instruction_parts),
        },
        {
            "role": "user",
            "content": (
                context
                if structured_context_mode
                else (
                    f"Question: {question}\n\n"
                    f"Retrieved memory evidence:\n{context}\n\n"
                    "Return only the answer text."
                )
            ),
        },
    ]


def build_unknown_recovery_messages(
    question: str,
    *,
    focused_context: str,
    yes_no_query: bool,
    list_query: bool,
    language_mode: str = "en",
) -> list[dict[str, str]]:
    instruction_parts = [
        "Extract the answer from the provided direct evidence only.",
        "A concrete clue is already present in the evidence, so do not answer UNKNOWN.",
        "Return only the shortest exact answer text.",
        "Do not explain.",
    ]
    if yes_no_query:
        instruction_parts.append("For confirmation questions, answer with Yes or No first, then only the shortest support phrase if needed.")
        instruction_parts.append("Judge whether the question's proposition is true or false from the evidence, and do not mirror the question's negation.")
        instruction_parts.append("If the answer is No, briefly state the corrected fact rather than repeating the false wording from the question.")
    if list_query:
        instruction_parts.append("For list questions, include only the items directly stated in the evidence and nothing extra.")
    if language_mode == "zh":
        instruction_parts.append("Answer in concise Chinese and copy the shortest supported phrase directly from the evidence.")
    return [
        {"role": "system", "content": " ".join(instruction_parts)},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Focused evidence:\n{focused_context}\n\n"
                "Return only the answer text."
            ),
        },
    ]

def build_revision_messages(
    question: str,
    *,
    candidate_answer: str,
    evidence_text: str,
    language_mode: str = "en",
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
    if language_mode == "zh":
        instruction_parts.append("Keep the final answer as a concise Chinese phrase and avoid restating the full question.")
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
    parser.add_argument("--answer-view-mode", choices=["heuristic", "extractive"], default="heuristic")
    parser.add_argument("--ingest-mode", choices=["online", "migration"], default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--judge-with-llm", action="store_true")
    parser.add_argument(
        "--answer-style",
        choices=["short", "judge_aligned", "memoryos_aligned", "structured_context", "structured_compact"],
        default="short",
        help="Controls only the answer synthesis layer; retrieval/memory remains unchanged.",
    )
    parser.add_argument(
        "--answer-revision",
        choices=["none", "grounded"],
        default="none",
        help="Optional second-pass answer revision over the same retrieved evidence.",
    )
    parser.add_argument(
        "--unknown-recovery",
        choices=["none", "direct_clue"],
        default="none",
        help="Optional second-pass recovery when the first answer is UNKNOWN despite a strong direct clue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = LEAFService(config_path=args.config, db_path=args.db)
    language_mode = str((service.config.language.mode if service.config.language is not None else "en") or "en").strip().lower()
    judge_client = ChatClient(service.config.llm) if args.judge_with_llm and service.config.llm.base_url else None
    try:
        if not args.ingest_mode:
            args.ingest_mode = str(service.config.ingest.mode)
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
                ingest_mode=args.ingest_mode,
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
                answer_view = build_compact_answer_view(
                    question=question,
                    evidence=evidence,
                    mode=args.answer_view_mode,
                    language_mode=language_mode,
                )
                if args.answer_view_mode == "heuristic":
                    answer_view = reshape_heuristic_answer_view_for_gvd(
                        question,
                        answer_view=answer_view,
                        context_lines=context_lines,
                    )
                answer_view_text = render_answer_view_text(
                    question=question,
                    answer_view=answer_view,
                    language_mode=language_mode,
                )
                answer_messages = build_answer_messages(
                    question=question,
                    evidence=evidence,
                    answer_style=args.answer_style,
                    context_lines=context_lines,
                    answer_view_text=answer_view_text,
                    answer_view=answer_view,
                    language_mode=language_mode,
                )
                answer_input_tokens_est = estimate_message_tokens(answer_messages)
                answer_started = time.perf_counter()
                unknown_recovery_triggered = False
                unknown_recovery_used = False
                deterministic_answer_used = False
                answer_llm_call_count = 0
                try:
                    answer_max_tokens = 96 if is_inference_query(question) else 80
                    predicted_answer = extract_structured_answer_from_context(
                        question,
                        answer_view=answer_view,
                        context_lines=context_lines,
                        language_mode=language_mode,
                    )
                    if predicted_answer:
                        deterministic_answer_used = True
                    else:
                        predicted_answer = service.llm.text(answer_messages, max_tokens=answer_max_tokens, temperature=0.0).strip() if service.llm else ""
                        if service.llm:
                            answer_llm_call_count += 1
                    if (
                        args.unknown_recovery == "direct_clue"
                        and
                        args.answer_style == "short"
                        and use_exact_short_mode(question)
                        and service.llm
                        and str(predicted_answer or "").strip().upper() == "UNKNOWN"
                    ):
                        focused_context, support_profile = select_short_answer_context(
                            question,
                            evidence=evidence,
                            answer_view=answer_view,
                            context_lines=context_lines,
                        )
                        if support_profile.get("has_concrete_direct_clue"):
                            unknown_recovery_triggered = True
                            recovery_messages = build_unknown_recovery_messages(
                                question,
                                focused_context=focused_context,
                                yes_no_query=is_yes_no_query(question),
                                list_query=is_list_query(question),
                                language_mode=language_mode,
                            )
                            answer_input_tokens_est += estimate_message_tokens(recovery_messages)
                            recovered_answer = service.llm.text(
                                recovery_messages,
                                max_tokens=answer_max_tokens,
                                temperature=0.0,
                            ).strip()
                            answer_llm_call_count += 1
                            if recovered_answer and not recovered_answer.startswith("__ERROR__:"):
                                predicted_answer = recovered_answer
                                unknown_recovery_used = True
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
                            language_mode=language_mode,
                        )
                        answer_input_tokens_est += estimate_message_tokens(revision_messages)
                        revised_answer = service.llm.text(revision_messages, max_tokens=answer_max_tokens, temperature=0.0).strip()
                        answer_llm_call_count += 1
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
                        "unknown_recovery": args.unknown_recovery,
                        "unknown_recovery_triggered": unknown_recovery_triggered,
                        "unknown_recovery_used": unknown_recovery_used,
                        "deterministic_answer_used": deterministic_answer_used,
                        "answer_llm_call_count": answer_llm_call_count,
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
        ingest_elapsed_values = [row["ingest_elapsed_ms"] for row in ingest_rows if row["ingest_elapsed_ms"] is not None]

        summary = {
            "persona_count": len(persona_names),
            "question_count": len(results),
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
            "avg_answer_input_tokens_est": round(sum(token_values) / len(token_values), 2) if token_values else None,
            "p50_answer_input_tokens_est": int(statistics.median(token_values)) if token_values else None,
            "judge_avg": round(sum(judge_values) / len(judge_values), 4) if judge_values else None,
            "judge_count": len(judge_values),
            "unknown_recovery_mode": args.unknown_recovery,
            "unknown_recovery_triggered_count": sum(1 for row in results if row.get("unknown_recovery_triggered")),
            "unknown_recovery_used_count": sum(1 for row in results if row.get("unknown_recovery_used")),
            "answer_llm_call_count_total": sum(int(row.get("answer_llm_call_count") or 0) for row in results),
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
        }

        payload = {
            "memory_bank": str(args.memory_bank),
            "questions": str(args.questions),
            "db": str(args.db),
            "ingest_mode": str(args.ingest_mode),
            "snapshot_limit": args.snapshot_limit,
            "raw_span_limit": args.raw_span_limit,
            "personas": list(args.personas),
            "persona_limit": args.persona_limit,
            "question_limit": args.question_limit,
            "judge_with_llm": args.judge_with_llm,
            "answer_style": args.answer_style,
            "answer_view_mode": args.answer_view_mode,
            "answer_revision": args.answer_revision,
            "unknown_recovery": args.unknown_recovery,
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
