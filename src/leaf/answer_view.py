from __future__ import annotations

import re
from typing import Any

from .extract import extract_entities
from .grounding import get_language_mode, is_inference_query, is_temporal_query, query_tokens as make_query_tokens

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

ANSWER_VIEW_SECTION_LABELS_ZH = {
    "direct_evidence": "直接证据",
    "raw_evidence": "直接证据",
    "entity_facts": "事实",
    "facts": "事实",
    "temporal_clues": "时间线索",
    "relation_paths": "关系线索",
    "relations": "关系线索",
    "page_context": "上下文",
    "page_summaries": "上下文",
    "insufficient": "证据不足",
}

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def _resolve_language_mode(question: str, language_mode: str | None = None) -> str:
    if language_mode is not None:
        normalized = str(language_mode or "en").strip().lower()
        if normalized in {"en", "zh"}:
            return normalized
    if _contains_cjk(question):
        return "zh"
    return str(get_language_mode() or "en").strip().lower()


def _trim_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _unique_compact_lines(lines: list[str], limit: int, max_chars: int) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for line in lines:
        compact = _trim_text(line, max_chars)
        normalized = compact.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(compact)
        if len(ordered) >= limit:
            break
    return ordered


def _score_answer_line(
    question: str,
    line: str,
    *,
    query_entities: list[str],
    query_token_set: set[str],
    temporal: bool,
    inference: bool,
    section: str,
    language_mode: str | None = None,
) -> float:
    lowered = str(line or "").lower()
    line_tokens = make_query_tokens(str(line or ""))
    score = len(query_token_set.intersection(line_tokens)) * 1.5
    resolved_language = _resolve_language_mode(question, language_mode)
    for query_entity in query_entities:
        if query_entity and query_entity in lowered:
            score += 2.5
    if temporal:
        if re.search(r"\b(19|20)\d{2}\b", lowered):
            score += 2.0
        if re.search(r"(19|20)\d{2}年|\d{1,2}月\d{1,2}[日号]?", str(line or "")):
            score += 2.0
        if any(
            month in lowered
            for month in [
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
            ]
        ):
            score += 1.5
        if "[" in lowered and "]" in lowered:
            score += 1.0
        if resolved_language == "zh" and any(
            token in str(line or "")
            for token in ["今天", "昨天", "前天", "那天", "当时", "上周", "本周", "这周", "下周", "上周末"]
        ):
            score += 1.0
    if inference:
        if section == "facts":
            score += 1.5
        if section == "page_summaries":
            score += 1.0
        if any(
            marker in lowered
            for marker in [
                "identity",
                "goal",
                "preference",
                "plan",
                "support",
                "career",
                "education",
                "study",
                "wants to",
                "性格",
                "目标",
                "计划",
                "支持",
                "职业",
                "学习",
                "喜欢",
                "想要",
            ]
        ):
            score += 1.0
    if section == "raw_evidence":
        score += 0.75
    return score


def _select_answer_lines(
    question: str,
    candidates: list[str],
    *,
    section: str,
    limit: int,
    max_chars: int,
    language_mode: str | None = None,
) -> list[str]:
    query_entities = extract_entities(question)
    query_token_set = make_query_tokens(question)
    temporal = is_temporal_query(question)
    inference = is_inference_query(question)
    scored: list[tuple[float, int, str]] = []
    for index, line in enumerate(candidates):
        compact = _trim_text(line, max_chars)
        if not compact:
            continue
        score = _score_answer_line(
            question,
            compact,
            query_entities=query_entities,
            query_token_set=query_token_set,
            temporal=temporal,
            inference=inference,
            section=section,
            language_mode=language_mode,
        )
        scored.append((score, index, compact))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return _unique_compact_lines([line for _, _, line in scored], limit=limit, max_chars=max_chars)


def _dedupe_view_items(items: list[dict[str, Any]], limit: int, max_chars: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in items:
        text = _trim_text(str(item.get("text") or ""), max_chars)
        citations = [str(value).strip() for value in item.get("citations") or [] if str(value).strip()]
        key = (text.lower(), tuple(citations))
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append({"text": text, "citations": citations})
        if len(deduped) >= limit:
            break
    return deduped


def _split_text_sentences(text: str) -> list[str]:
    compact = " ".join(str(text or "").split())
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?;。！？；])\s*|\s+\|\s+", compact)
    sentences = [part.strip() for part in parts if part and part.strip()]
    return sentences or [compact]


def _extract_relevant_text(
    question: str,
    text: str,
    *,
    section: str,
    max_chars: int,
    preserve_prefix: str = "",
    language_mode: str | None = None,
) -> str:
    compact = " ".join(str(text or "").split())
    if not compact:
        return ""
    body = compact
    if preserve_prefix and compact.startswith(preserve_prefix):
        body = compact[len(preserve_prefix) :].strip()
    body = body or compact
    if len(compact) <= max_chars:
        return compact
    query_entities = extract_entities(question)
    query_token_set = make_query_tokens(question)
    temporal = is_temporal_query(question)
    inference = is_inference_query(question)
    candidates = _split_text_sentences(body)
    scored: list[tuple[float, int, str]] = []
    for index, candidate in enumerate(candidates):
        score = _score_answer_line(
            question,
            candidate,
            query_entities=query_entities,
            query_token_set=query_token_set,
            temporal=temporal,
            inference=inference,
            section=section,
            language_mode=language_mode,
        )
        scored.append((score, index, candidate))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    chosen: list[str] = []
    total_len = len(preserve_prefix)
    for _, _, candidate in scored:
        candidate = candidate.strip()
        if not candidate:
            continue
        extra_len = len(candidate) + (1 if chosen else 0)
        if chosen and total_len + extra_len > max_chars:
            continue
        chosen.append(candidate)
        total_len += extra_len
        if len(chosen) >= (2 if inference or temporal else 1):
            break
    result = " ".join(chosen) if chosen else body
    if preserve_prefix:
        result = f"{preserve_prefix}{result}".strip()
    return _trim_text(result, max_chars)


def _score_view_items(
    question: str,
    items: list[dict[str, Any]],
    *,
    section: str,
    limit: int,
    max_chars: int,
    language_mode: str | None = None,
) -> list[dict[str, Any]]:
    query_entities = extract_entities(question)
    query_token_set = make_query_tokens(question)
    temporal = is_temporal_query(question)
    inference = is_inference_query(question)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, item in enumerate(items):
        text = _trim_text(str(item.get("text") or ""), max_chars)
        if not text:
            continue
        score = _score_answer_line(
            question,
            text,
            query_entities=query_entities,
            query_token_set=query_token_set,
            temporal=temporal,
            inference=inference,
            section=section,
            language_mode=language_mode,
        )
        citations = [str(value).strip() for value in item.get("citations") or [] if str(value).strip()]
        if citations:
            score += 0.2
        scored.append((score, index, {"text": text, "citations": citations}))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return _dedupe_view_items([item for _, _, item in scored], limit=limit, max_chars=max_chars)


def build_extractive_answer_view(question: str, evidence: dict[str, Any], *, language_mode: str | None = None) -> dict[str, Any]:
    pages = evidence.get("pages") or []
    atoms = evidence.get("atoms") or []
    raw_spans = evidence.get("raw_spans") or []
    edges = list(evidence.get("edges") or [])

    direct_candidates: list[dict[str, Any]] = []
    temporal_candidates: list[dict[str, Any]] = []
    fact_candidates: list[dict[str, Any]] = []
    relation_candidates: list[dict[str, Any]] = []
    page_candidates: list[dict[str, Any]] = []

    for span in raw_spans:
        span_id = str(span.get("span_id") or "").strip()
        prefix = (
            f"[{span.get('timestamp')}] {span.get('speaker')}: "
            if span.get("timestamp")
            else f"{span.get('speaker')}: "
        )
        text = _extract_relevant_text(
            question,
            prefix + str(span.get("text") or ""),
            section="raw_evidence",
            max_chars=170,
            preserve_prefix=prefix,
            language_mode=language_mode,
        )
        item = {"text": text, "citations": [span_id] if span_id else []}
        direct_candidates.append(item)
        if span.get("timestamp"):
            temporal_candidates.append(item)

    for atom in atoms:
        atom_id = str(atom.get("atom_id") or atom.get("span_id") or "").strip()
        support_ids = [str(raw_id).strip() for raw_id in atom.get("support_span_ids") or [] if str(raw_id).strip()]
        item = {
            "text": _extract_relevant_text(
                question,
                str(atom.get("content") or ""),
                section="facts",
                max_chars=150,
                language_mode=language_mode,
            ),
            "citations": ([atom_id] if atom_id else []) + support_ids[:2],
        }
        fact_candidates.append(item)
        if atom.get("time_range"):
            temporal_candidates.append(
                {
                    "text": _extract_relevant_text(
                        question,
                        f"{atom.get('content')} | time={atom.get('time_range')}",
                        section="raw_evidence",
                        max_chars=150,
                        language_mode=language_mode,
                    ),
                    "citations": ([atom_id] if atom_id else []) + support_ids[:2],
                }
            )

    for edge in edges:
        relation_candidates.append(
            {
                "text": f"{edge.get('src')} {edge.get('relation')} {edge.get('dst')}",
                "citations": [str(edge.get("edge_id") or "").strip()] if str(edge.get("edge_id") or "").strip() else [],
            }
        )

    for page in pages:
        page_id = str(page.get("page_id") or "").strip()
        summary = str(page.get("synopsis") or page.get("summary") or "").strip()
        if not summary:
            continue
        anchor_ids = [str(raw_id).strip() for raw_id in page.get("anchor_span_ids") or [] if str(raw_id).strip()]
        citations = ([page_id] if page_id else []) + anchor_ids[:2]
        page_candidates.append(
            {
                "text": _extract_relevant_text(
                    question,
                    f"{page.get('title')}: {summary}",
                    section="page_summaries",
                    max_chars=180,
                    language_mode=language_mode,
                ),
                "citations": citations,
            }
        )
        if page.get("time_range"):
            temporal_candidates.append(
                {
                    "text": _extract_relevant_text(
                        question,
                        f"{page.get('title')}: {summary} | time={page.get('time_range')}",
                        section="raw_evidence",
                        max_chars=150,
                        language_mode=language_mode,
                    ),
                    "citations": citations,
                }
            )

    temporal = is_temporal_query(question)
    inference = is_inference_query(question)
    view = {
        "direct_evidence": _score_view_items(
            question,
            direct_candidates,
            section="raw_evidence",
            limit=5 if temporal else 4,
            max_chars=200,
            language_mode=language_mode,
        ),
        "entity_facts": _score_view_items(
            question,
            fact_candidates,
            section="facts",
            limit=6 if inference else 5,
            max_chars=170,
            language_mode=language_mode,
        ),
        "temporal_clues": _score_view_items(
            question,
            temporal_candidates,
            section="raw_evidence",
            limit=5 if temporal else 2,
            max_chars=170,
            language_mode=language_mode,
        ),
        "relation_paths": _score_view_items(
            question,
            relation_candidates,
            section="relations",
            limit=4,
            max_chars=140,
            language_mode=language_mode,
        ),
        "page_context": _score_view_items(
            question,
            page_candidates,
            section="page_summaries",
            limit=3 if inference else 2,
            max_chars=180,
            language_mode=language_mode,
        ),
        "insufficient": [],
    }
    if not any(
        view[key]
        for key in ("direct_evidence", "entity_facts", "temporal_clues", "relation_paths", "page_context")
    ):
        view["insufficient"] = [{"text": "No strong grounded evidence selected.", "citations": []}]
    return view


def build_answer_view(question: str, evidence: dict[str, Any], *, language_mode: str | None = None) -> dict[str, Any]:
    pages = evidence.get("pages") or []
    atoms = evidence.get("atoms") or []
    raw_spans = evidence.get("raw_spans") or []
    edges = list(evidence.get("edges") or [])

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
    relation_candidates = [
        f"{edge.get('src')} {edge.get('relation')} {edge.get('dst')}"
        for edge in edges
        if str(edge.get("src") or "").strip()
        and str(edge.get("relation") or "").strip()
        and str(edge.get("dst") or "").strip()
    ]

    temporal = is_temporal_query(question)
    inference = is_inference_query(question)
    page_lines = _select_answer_lines(
        question,
        page_candidates,
        section="page_summaries",
        limit=4 if inference else 3,
        max_chars=180,
        language_mode=language_mode,
    )
    fact_lines = _select_answer_lines(
        question,
        fact_candidates,
        section="facts",
        limit=8 if inference else 6,
        max_chars=160,
        language_mode=language_mode,
    )
    raw_lines = _select_answer_lines(
        question,
        raw_candidates,
        section="raw_evidence",
        limit=8 if temporal else 6,
        max_chars=200,
        language_mode=language_mode,
    )
    relation_lines = _select_answer_lines(
        question,
        relation_candidates,
        section="relations",
        limit=4,
        max_chars=130,
        language_mode=language_mode,
    )
    return {
        "page_summaries": page_lines,
        "facts": fact_lines,
        "raw_evidence": raw_lines,
        "relations": relation_lines,
    }


def build_compact_answer_view(
    question: str,
    evidence: dict[str, Any],
    *,
    mode: str = "extractive",
    language_mode: str | None = None,
) -> dict[str, Any]:
    if mode == "heuristic":
        return build_answer_view(question, evidence, language_mode=language_mode)
    if mode == "extractive":
        return build_extractive_answer_view(question, evidence, language_mode=language_mode)
    raise ValueError(f"Unsupported answer view mode: {mode}")


def render_answer_view_text(question: str, answer_view: dict[str, Any], *, language_mode: str | None = None) -> str:
    resolved_language = _resolve_language_mode(question, language_mode)
    section_labels = ANSWER_VIEW_SECTION_LABELS_ZH if resolved_language == "zh" else ANSWER_VIEW_SECTION_LABELS
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
    lines = [f"问题：{question}", "", "证据："] if resolved_language == "zh" else [f"Question: {question}", "", "Evidence:"]
    rendered_any = False
    for key in preferred_order:
        items = answer_view.get(key) or []
        if not isinstance(items, list):
            items = [items]
        texts: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item.get("text") or "").strip() if isinstance(item, dict) else str(item or "").strip()
            compact = _trim_text(text, 220)
            normalized = compact.lower()
            if not compact or normalized in seen:
                continue
            seen.add(normalized)
            texts.append(compact)
        if not texts:
            continue
        rendered_any = True
        lines.append(f"[{section_labels.get(key, key.replace('_', ' ').title())}]")
        lines.extend(f"- {text}" for text in texts)
        lines.append("")
    if not rendered_any:
        if resolved_language == "zh":
            lines.append("[证据]")
            lines.append("- 没有检索到相关证据。")
        else:
            lines.append("[Evidence]")
            lines.append("- No relevant evidence.")
    return "\n".join(lines).strip()


def summarize_answer_view(answer_view: dict[str, Any]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for key, items in answer_view.items():
        if isinstance(items, list):
            summary[key] = len(items)
    return summary
