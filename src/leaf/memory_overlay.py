from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Any

from .normalize import language_aware_content_terms, language_aware_stemmed_content_terms

try:  # Optional gazetteer; experiments can install it in tracenav_nlp, but code has a fallback.
    import pycountry  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pycountry = None  # type: ignore

_TEMPORAL_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b|\b(?:today|yesterday|tomorrow|week|"
    r"month|year|before|after|recent|latest|last|next|when)\b",
    re.IGNORECASE,
)
_PROFILE_RE = re.compile(
    r"\b(?:like|likes|liked|love|loves|favorite|prefer|prefers|want|wants|plan|"
    r"plans|goal|goals|work|job|career|study|family|friend|partner|pet|habit|"
    r"hobby|hobbies|interested|enjoy|enjoys)\b",
    re.IGNORECASE,
)
_RELATION_RE = re.compile(
    r"\b(?:friend|friends|family|cousin|partner|mother|father|sister|brother|"
    r"colleague|coworker|roommate|neighbor|mentor|teacher|manager|boss)\b",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(r"\b(?:plan|plans|planned|will|going to|hope|hopes|want|wants|intend|intends)\b", re.IGNORECASE)
_MEDIA_RE = re.compile(r"\b(?:book|movie|film|song|music|game|screenplay|show|series|artist|band)\b", re.IGNORECASE)
_PLACE_RE = re.compile(r"\b(?:place|city|town|restaurant|studio|gym|park|museum|venue|trip|travel|visit|visited|go to)\b", re.IGNORECASE)
_ACTIVITY_RE = re.compile(r"\b(?:class|workout|volunteer|volunteering|cook|bake|run|walk|hike|march|workshop|meeting)\b", re.IGNORECASE)
_COUNTRY_QUERY_RE = re.compile(
    r"\b(?:countr(?:y|ies)|nation|nations|european|asian|african|american|continent)\b",
    re.IGNORECASE,
)
_CHILDHOOD_RE = re.compile(r"\b(?:child|children|childhood|kid|kids|younger|growing up)\b", re.IGNORECASE)
_REACTION_RE = re.compile(r"\b(?:react|reaction|respond|response|describe|described|feel|felt)\b", re.IGNORECASE)
_GEO_FALLBACK_NAMES = {
    "argentina",
    "australia",
    "brazil",
    "canada",
    "china",
    "england",
    "france",
    "germany",
    "india",
    "ireland",
    "italy",
    "japan",
    "mexico",
    "netherlands",
    "scotland",
    "spain",
    "united kingdom",
    "united states",
    "wales",
}


def content_terms(text: str, *, stemmed: bool = True) -> set[str]:
    extractor = language_aware_stemmed_content_terms if stemmed else language_aware_content_terms
    return set(extractor(text, mode="auto", include_cjk_subgrams=True))


@lru_cache(maxsize=1)
def geo_name_lexicon() -> tuple[str, ...]:
    names: set[str] = set(_GEO_FALLBACK_NAMES)
    if pycountry is not None:
        for country in getattr(pycountry, "countries", []):
            for attr in ("name", "official_name", "common_name"):
                value = str(getattr(country, attr, "") or "").strip().lower()
                if len(value) > 3:
                    names.add(value)
        for subdivision in getattr(pycountry, "subdivisions", []):
            value = str(getattr(subdivision, "name", "") or "").strip().lower()
            if len(value) > 3:
                names.add(value)
    return tuple(sorted(names, key=lambda item: (-len(item), item)))


def extract_geo_terms(text: str) -> list[str]:
    lowered = f" {str(text or '').lower()} "
    hits: list[str] = []
    seen: set[str] = set()
    for name in geo_name_lexicon():
        pattern = r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])"
        if not re.search(pattern, lowered):
            continue
        if name in seen:
            continue
        seen.add(name)
        hits.append(name)
    return hits[:24]


def infer_text_facets(text: str, *, memory_kind: str = "") -> list[str]:
    raw = str(text or "")
    facets: list[str] = []
    if _TEMPORAL_RE.search(raw):
        facets.append("temporal")
    if _PROFILE_RE.search(raw) or memory_kind in {"state", "preference"}:
        facets.append("profile")
    if _RELATION_RE.search(raw) or memory_kind == "relation":
        facets.append("relationship")
    if _PLAN_RE.search(raw) or memory_kind == "plan":
        facets.append("plan")
    if _MEDIA_RE.search(raw):
        facets.append("media_hobby")
    if _PLACE_RE.search(raw) or extract_geo_terms(raw):
        facets.append("place_travel")
    if _ACTIVITY_RE.search(raw):
        facets.append("activity")
    return facets


def score_text_utility(text: str, *, has_timestamp: bool = False) -> dict[str, Any]:
    raw = str(text or "")
    terms = content_terms(raw, stemmed=True)
    facets = infer_text_facets(raw)
    specificity = min(1.0, len(terms) / 24.0)
    answerability = specificity
    if has_timestamp or "temporal" in facets:
        answerability += 0.08
    if "profile" in facets:
        answerability += 0.08
    if "relationship" in facets:
        answerability += 0.06
    if "place_travel" in facets or "activity" in facets:
        answerability += 0.05
    return {
        "specificity": round(min(1.0, specificity), 4),
        "temporal_salience": 1.0 if has_timestamp or "temporal" in facets else 0.0,
        "profile_salience": 1.0 if "profile" in facets else 0.0,
        "relation_salience": 1.0 if "relationship" in facets else 0.0,
        "answerability": round(min(1.0, answerability), 4),
        "term_count": len(terms),
        "entity_count": 0,
    }


def score_atom_utility(atom: Any) -> dict[str, Any]:
    text = str(getattr(atom, "content", "") or "")
    entities = [str(item).strip() for item in (getattr(atom, "canonical_entities", []) or getattr(atom, "entities", []) or []) if str(item).strip()]
    terms = content_terms(" ".join([text, " ".join(entities)]), stemmed=True)
    specificity = min(1.0, (len(terms) / 18.0) + (len(entities) * 0.08))
    temporal = 1.0 if (getattr(atom, "time_range", None) or _TEMPORAL_RE.search(text)) else 0.0
    profile = 1.0 if _PROFILE_RE.search(text) else 0.0
    relation = 1.0 if _RELATION_RE.search(text) else 0.0
    memory_kind = str(getattr(atom, "memory_kind", "") or "")
    salience = specificity
    if memory_kind in {"state", "preference", "plan", "relation"}:
        salience += 0.12
    salience += 0.08 * temporal + 0.08 * profile + 0.06 * relation
    return {
        "specificity": round(min(1.0, specificity), 4),
        "temporal_salience": round(temporal, 4),
        "profile_salience": round(profile, 4),
        "relation_salience": round(relation, 4),
        "answerability": round(min(1.0, salience), 4),
        "term_count": len(terms),
        "entity_count": len(entities),
    }


def infer_atom_facets(atom: Any) -> list[str]:
    text = " ".join(
        [
            str(getattr(atom, "content", "") or ""),
            " ".join(str(item) for item in (getattr(atom, "canonical_entities", []) or getattr(atom, "entities", []) or [])),
            str(getattr(atom, "memory_kind", "") or ""),
        ]
    )
    facets = infer_text_facets(text, memory_kind=str(getattr(atom, "memory_kind", "") or ""))
    if getattr(atom, "time_range", None):
        facets.append("temporal")
    facets = list(dict.fromkeys(facets))
    if not facets:
        facets.append("general_fact")
    return facets


def build_event_overlay(
    *,
    event: Any,
    atoms: list[Any],
    assignment_slugs: list[str] | None = None,
) -> dict[str, Any]:
    event_text = str(getattr(event, "text", "") or "")
    atom_text = " ".join(str(getattr(atom, "content", "") or "") for atom in atoms)
    speaker = str(getattr(event, "speaker", "") or "").strip()
    entities = [
        str(item).strip()
        for item in (getattr(event, "canonical_entity_refs", []) or getattr(event, "entity_refs", []) or [])
        if str(item).strip()
    ]
    event_terms = content_terms(" ".join([speaker, event_text, " ".join(entities)]), stemmed=True)
    atom_terms = content_terms(atom_text, stemmed=True)
    terms = set(event_terms) | set(atom_terms)
    geo_terms = extract_geo_terms(" ".join([speaker, event_text, atom_text, " ".join(entities)]))
    facet_counts: Counter[str] = Counter()
    facet_counts.update(infer_text_facets(event_text))
    utility_rows: list[dict[str, Any]] = []
    for atom in atoms:
        facet_counts.update(infer_atom_facets(atom))
        utility_rows.append(score_atom_utility(atom))
    timestamp = str(getattr(event, "timestamp", "") or "")
    event_utility = score_text_utility(event_text, has_timestamp=bool(timestamp))
    utility_rows.append(event_utility)
    answerability = max((float(row.get("answerability") or 0.0) for row in utility_rows), default=0.0)
    temporal_key = ""
    if timestamp:
        parsed = _parse_date(timestamp)
        temporal_key = parsed.strftime("%Y-%m-%d") if parsed else timestamp[:10]
    return {
        "event_id": str(getattr(event, "event_id", "") or ""),
        "session_id": str(getattr(event, "session_id", "") or ""),
        "turn_index": int(getattr(event, "turn_index", 0) or 0),
        "speaker": speaker,
        "timestamp": timestamp,
        "temporal_key": temporal_key,
        "entities": entities[:12],
        "terms": sorted(terms)[:128],
        "event_terms": sorted(event_terms)[:64],
        "atom_terms": sorted(atom_terms)[:96],
        "geo_terms": geo_terms,
        "facets": sorted(facet_counts),
        "facet_counts": dict(sorted(facet_counts.items())),
        "topic_slugs": sorted({str(item).strip() for item in (assignment_slugs or []) if str(item).strip()}),
        "utility": {
            "answerability": round(answerability, 4),
            "atom_count": len(atoms),
            "term_count": len(terms),
            "has_timestamp": bool(timestamp),
            "has_profile_signal": bool(facet_counts.get("profile")),
            "has_temporal_signal": bool(facet_counts.get("temporal") or timestamp),
        },
    }


def build_entity_profile_overlay(events: list[Any], atoms: list[Any]) -> dict[str, Any]:
    by_entity: dict[str, dict[str, Any]] = {}
    atoms_by_event = group_atoms_by_event(atoms)
    for event in events:
        entities = [
            str(item).strip().lower()
            for item in (getattr(event, "canonical_entity_refs", []) or getattr(event, "entity_refs", []) or [])
            if str(item).strip()
        ]
        if not entities:
            speaker = str(getattr(event, "speaker", "") or "").strip().lower()
            if speaker:
                entities = [speaker]
        event_atoms = atoms_by_event.get(str(getattr(event, "event_id", "") or ""), [])
        text = " ".join([str(getattr(event, "text", "") or "")] + [str(getattr(atom, "content", "") or "") for atom in event_atoms])
        terms = content_terms(text, stemmed=True)
        facets = Counter(facet for atom in event_atoms for facet in infer_atom_facets(atom))
        for entity in entities:
            profile = by_entity.setdefault(
                entity,
                {
                    "entity": entity,
                    "aliases": [entity],
                    "term_counts": Counter(),
                    "facet_counts": Counter(),
                    "event_ids": [],
                    "recent_event_ids": [],
                },
            )
            profile["term_counts"].update(terms)
            profile["facet_counts"].update(facets)
            event_id = str(getattr(event, "event_id", "") or "")
            if event_id and event_id not in profile["event_ids"]:
                profile["event_ids"].append(event_id)
                profile["recent_event_ids"].append(event_id)
                profile["recent_event_ids"] = profile["recent_event_ids"][-12:]
    output: dict[str, Any] = {}
    for entity, profile in by_entity.items():
        term_counts: Counter[str] = profile.pop("term_counts")
        facet_counts: Counter[str] = profile.pop("facet_counts")
        profile["terms"] = [term for term, _ in term_counts.most_common(48)]
        profile["facets"] = dict(facet_counts.most_common(12))
        profile["event_count"] = len(profile.get("event_ids") or [])
        output[entity] = profile
    return output


def build_temporal_overlay(events: list[Any], atoms: list[Any]) -> dict[str, Any]:
    atoms_by_event = group_atoms_by_event(atoms)
    rows: list[dict[str, Any]] = []
    by_date: dict[str, list[str]] = defaultdict(list)
    by_month: dict[str, list[str]] = defaultdict(list)
    rows_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        event_id = str(getattr(event, "event_id", "") or "")
        timestamp = str(getattr(event, "timestamp", "") or "")
        if not event_id or not timestamp:
            continue
        parsed = _parse_date(timestamp)
        date_key = parsed.strftime("%Y-%m-%d") if parsed else timestamp[:10]
        month_key = date_key[:7] if len(date_key) >= 7 else ""
        facets = Counter(facet for atom in atoms_by_event.get(event_id, []) for facet in infer_atom_facets(atom))
        row = {
            "event_id": event_id,
            "timestamp": timestamp,
            "date_key": date_key,
            "month_key": month_key,
            "session_id": str(getattr(event, "session_id", "") or ""),
            "turn_index": int(getattr(event, "turn_index", 0) or 0),
            "facets": dict(facets.most_common(8)),
        }
        rows.append(row)
        rows_by_session[str(getattr(event, "session_id", "") or "")].append(row)
        by_date[date_key].append(event_id)
        if month_key:
            by_month[month_key].append(event_id)
    rows.sort(key=lambda item: (str(item.get("timestamp") or ""), str(item.get("session_id") or ""), int(item.get("turn_index") or 0)))
    previous_by_event: dict[str, str] = {}
    next_by_event: dict[str, str] = {}
    for prev, curr in zip(rows, rows[1:]):
        previous_by_event[str(curr["event_id"])] = str(prev["event_id"])
        next_by_event[str(prev["event_id"])] = str(curr["event_id"])
    previous_in_session_by_event: dict[str, str] = {}
    next_in_session_by_event: dict[str, str] = {}
    session_event_ids: dict[str, list[str]] = {}
    for session_id, session_rows in rows_by_session.items():
        ordered_rows = sorted(session_rows, key=lambda item: int(item.get("turn_index") or 0))
        session_event_ids[session_id] = [str(item["event_id"]) for item in ordered_rows]
        for prev, curr in zip(ordered_rows, ordered_rows[1:]):
            previous_in_session_by_event[str(curr["event_id"])] = str(prev["event_id"])
            next_in_session_by_event[str(prev["event_id"])] = str(curr["event_id"])
    return {
        "event_count": len(rows),
        "timeline": rows,
        "by_date": {key: values[:24] for key, values in sorted(by_date.items())},
        "by_month": {key: values[:48] for key, values in sorted(by_month.items())},
        "previous_by_event": previous_by_event,
        "next_by_event": next_by_event,
        "previous_in_session_by_event": previous_in_session_by_event,
        "next_in_session_by_event": next_in_session_by_event,
        "session_event_ids": session_event_ids,
    }


def default_retrieval_policy_overlay() -> dict[str, Any]:
    return {
        "version": "retrieval_policy_overlay_v2",
        "use_runtime_features_only": True,
        "topic_soft": {
            "enabled": True,
            "skip_temporal": True,
            "max_candidate_atoms": 20,
            "min_selected_overlap": 2,
        },
        "entity_profile": {
            "enabled": True,
            "max_events": 3,
            "min_query_term_overlap": 1,
            "require_query_entity_match": True,
            "min_event_score": 0.42,
            "run_when_primary_strong": False,
        },
        "event_lexical": {
            "enabled": True,
            "max_events": 4,
            "min_query_term_overlap": 2,
            "min_distinctive_overlap": 1,
            "min_score": 0.36,
            "geo_request_min_score": 0.24,
            "run_when_primary_strong": True,
            "suppress_profile_attribute_queries": True,
            "include_atom_context": True,
            "max_atom_contexts": 2,
        },
        "local_neighbors": {
            "enabled": True,
            "max_events": 4,
            "forward_window": 2,
            "backward_window": 0,
            "run_when_primary_strong": True,
            "require_source_question": True,
            "allow_bridge_question": True,
            "min_source_overlap": 0.45,
            "min_source_distinctive_overlap": 1,
            "score": 0.64,
        },
        "temporal": {
            "enabled": True,
            "max_events": 4,
            "date_match_bonus": 1.0,
            "month_match_bonus": 0.5,
            "run_when_primary_strong": False,
        },
        "fallback": {
            "baseline_on_unknown": False,
            "suppress_when_primary_strong": True,
            "suppress_expansion_when_primary_has_relative_temporal": True,
            "suppress_inference_queries": False,
            "suppress_profile_attribute_queries": False,
            "primary_strong_min_raw_spans": 4,
            "primary_strong_min_content_overlap": 2,
            "primary_strong_min_top3_overlap_sum": 4,
            "primary_strong_min_distinctive_hits": 2,
        },
    }


def overlay_query_features(question: str) -> dict[str, Any]:
    lowered = str(question or "").lower()
    terms = set(content_terms(question, stemmed=True))
    if _CHILDHOOD_RE.search(lowered):
        if re.search(r"\b(items?|things?|object|objects|had|having|mention)\b", lowered):
            terms.update(content_terms("child childhood kid", stemmed=True))
        else:
            terms.update(content_terms("child childhood kid kids children growing up", stemmed=True))
    if _COUNTRY_QUERY_RE.search(lowered):
        terms.update(content_terms("country countries nation travel trip visit visited place", stemmed=True))
    if re.search(r"\b(?:activity|activities|enjoy|together|family)\b", lowered):
        terms.update(content_terms("activity activities enjoy love together family", stemmed=True))
    if _REACTION_RE.search(lowered):
        terms.update(content_terms("react reaction response describe described felt", stemmed=True))
    return {
        "terms": sorted(terms),
        "temporal": bool(_TEMPORAL_RE.search(lowered)),
        "profile": bool(_PROFILE_RE.search(lowered)),
        "relationship": bool(_RELATION_RE.search(lowered)),
        "plan": bool(_PLAN_RE.search(lowered)),
        "media_hobby": bool(_MEDIA_RE.search(lowered)),
        "place_travel": bool(_PLACE_RE.search(lowered)),
        "activity": bool(_ACTIVITY_RE.search(lowered)),
        "geo_request": bool(_COUNTRY_QUERY_RE.search(lowered)),
        "childhood": bool(_CHILDHOOD_RE.search(lowered)),
        "reaction": bool(_REACTION_RE.search(lowered)),
        "inference": bool(
            re.search(
                r"\b(might|would|could|likely|considered|financial status|what job|what career|degree be|open to)\b",
                lowered,
            )
        ),
    }


def overlay_event_bonus(
    *,
    question: str,
    overlay: dict[str, Any],
    max_bonus: float = 0.18,
) -> tuple[float, dict[str, Any]]:
    features = overlay_query_features(question)
    query_terms = set(features.get("terms") or [])
    event_terms = set(overlay.get("terms") or [])
    facets = set(overlay.get("facets") or [])
    utility = dict(overlay.get("utility") or {})
    term_hits = sorted(query_terms.intersection(event_terms))
    facet_hits: list[str] = []
    for facet in ["temporal", "profile", "relationship", "plan", "media_hobby", "place_travel", "activity"]:
        if features.get(facet) and facet in facets:
            facet_hits.append(facet)
    bonus = 0.0
    bonus += min(0.08, 0.015 * len(term_hits))
    bonus += min(0.08, 0.035 * len(facet_hits))
    bonus += min(0.04, float(utility.get("answerability") or 0.0) * 0.035)
    if features.get("temporal") and utility.get("has_timestamp"):
        bonus += 0.025
    capped = min(float(max_bonus), bonus)
    return capped, {
        "term_hits": term_hits[:8],
        "facet_hits": facet_hits,
        "answerability": utility.get("answerability"),
        "bonus": round(capped, 4),
    }


def _parse_date(value: str) -> datetime | None:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt)
        except ValueError:
            continue
    return None


def merge_overlay_terms(existing: list[str], additions: list[str], *, limit: int = 64) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in list(existing or []) + list(additions or []):
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def top_profile_terms_for_topic(atom_rows: list[Any], *, limit: int = 32) -> list[str]:
    counts: Counter[str] = Counter()
    for atom in atom_rows:
        text = " ".join(
            [
                str(getattr(atom, "content", "") or ""),
                " ".join(str(item) for item in (getattr(atom, "canonical_entities", []) or getattr(atom, "entities", []) or [])),
            ]
        )
        counts.update(content_terms(text, stemmed=True))
    return [term for term, _ in counts.most_common(limit)]


def group_atoms_by_event(atoms: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for atom in atoms:
        grouped[str(getattr(atom, "event_id", "") or "")].append(atom)
    return grouped
