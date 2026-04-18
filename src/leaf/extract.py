from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .normalize import EntityResolver, canonicalize_entity
from .clients import (
    ChatClient,
    OpenAICompatError,
    estimate_message_tokens,
    extract_chat_text,
    extract_json_object,
    extract_prompt_tokens,
)
from .grounding import derive_temporal_grounding, span_surface_text
from .schemas import GraphEdge, MemoryAtom, RawSpan

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in",
    "on", "for", "with", "this", "that", "it", "we", "i", "you", "they", "he", "she",
    "as", "at", "by", "from", "our", "your", "their", "his", "her",
}
ENTITY_NOISE_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "did", "does", "do", "is", "are", "was", "were", "can", "could", "would", "should",
    "may", "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december",
    "hey", "hi", "hello", "thanks", "thank", "wow", "cool", "great", "awesome", "okay",
    "ok", "yeah", "yep", "yup", "aww", "woah", "whoa",
}
ZH_ENTITY_NOISE_WORDS = {
    "什么", "哪些", "哪个", "谁", "哪里", "哪天", "什么时候", "为何", "为什么", "如何",
    "我们", "你们", "他们", "她们", "这个", "那个", "这些", "那些",
    "事情", "经历", "话题", "问题", "方法", "建议", "技巧", "内容", "名字", "时间",
    "一下", "一下子", "真的", "就是", "还是", "已经", "当时", "最近", "之前", "之后",
    "感觉", "有点", "非常", "特别", "超级", "可以", "一下", "然后", "因为",
}

_LANGUAGE_MODE = "en"

ENTITY_RESOLVER = EntityResolver()
SEMANTIC_LEXICON_BY_MODE = {
    "en": [
    {
        "concept": "martial arts",
        "aliases": ["martial arts", "combat sport", "fighting style"],
        "items": {
            "kickboxing": "kickboxing",
            "taekwondo": "taekwondo",
            "karate": "karate",
            "judo": "judo",
            "kung fu": "kung fu",
        },
    },
    {
        "concept": "stress relief",
        "aliases": ["destress", "de-stress", "stress relief", "relax", "unwind", "escape"],
        "items": {
            "dance": "dancing",
            "dancing": "dancing",
            "run": "running",
            "running": "running",
            "read": "reading",
            "reading": "reading",
            "paint": "painting",
            "painting": "painting",
            "violin": "playing violin",
        },
    },
    {
        "concept": "mental health career",
        "aliases": ["mental health", "counseling", "counselling", "counselor", "counsellor", "psychology"],
        "items": {
            "mental health": "psychology",
            "psychology": "psychology",
            "counseling": "counseling certification",
            "counselling": "counseling certification",
            "counselor": "counseling certification",
            "counsellor": "counseling certification",
        },
    },
    ],
    "zh": [
        {
            "concept": "减压放松",
            "aliases": ["减压", "放松", "缓解压力", "舒缓压力", "放松心情", "释放压力", "舒压"],
            "items": {
                "冥想": "冥想",
                "深呼吸": "深呼吸",
                "瑜伽": "瑜伽",
                "散步": "散步",
                "听音乐": "听音乐",
                "看电影": "看电影",
                "读书": "读书",
                "聊天": "聊天",
            },
        },
        {
            "concept": "艺术创作",
            "aliases": ["画画", "绘画", "摄影", "拍照", "艺术", "展览", "博物馆", "展品"],
            "items": {
                "画画": "绘画",
                "绘画": "绘画",
                "摄影": "摄影",
                "拍照": "摄影",
                "博物馆": "博物馆",
                "展览": "展览",
                "展品": "展品",
            },
        },
        {
            "concept": "烹饪美食",
            "aliases": ["做饭", "做菜", "菜谱", "做法", "食材", "调料", "烹饪", "美食"],
            "items": {
                "菜谱": "菜谱",
                "做法": "做法",
                "食材": "食材",
                "调料": "调料",
                "烹饪": "烹饪",
                "做菜": "做菜",
            },
        },
        {
            "concept": "职业学习",
            "aliases": ["工作", "职业", "事业", "学习", "课程", "训练", "英语", "编程"],
            "items": {
                "工作": "工作",
                "职业": "职业",
                "学习": "学习",
                "课程": "课程",
                "训练": "训练",
                "英语": "英语",
                "编程": "编程",
            },
        },
        {
            "concept": "音乐影视",
            "aliases": ["音乐", "歌曲", "演唱会", "电影", "电视剧", "演员", "场景", "剧情"],
            "items": {
                "音乐": "音乐",
                "歌曲": "歌曲",
                "演唱会": "演唱会",
                "电影": "电影",
                "电视剧": "电视剧",
                "剧情": "剧情",
                "场景": "场景",
            },
        },
    ],
}
SPACY_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT"}
_SPACY_NLP_BY_MODE: dict[str, Any] = {}
_SPACY_INIT_ATTEMPTED: set[str] = set()
_NLTK_READY = None
_YAKE_KEYWORD_EXTRACTOR_BY_MODE: dict[str, Any] = {}
_YAKE_INIT_ATTEMPTED: set[str] = set()
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?。！？])\s+|\n+")


def set_language_mode(mode: str) -> None:
    global _LANGUAGE_MODE
    normalized = str(mode or "en").strip().lower()
    if normalized not in {"en", "zh"}:
        raise ValueError(f"Unsupported language mode: {mode}")
    _LANGUAGE_MODE = normalized


def get_language_mode() -> str:
    return _LANGUAGE_MODE


def _env_enabled(*names: str) -> bool:
    for name in names:
        if os.environ.get(name, "").lower() in {"1", "true", "yes"}:
            return True
    return False


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _get_spacy_nlp(mode: str | None = None):
    language_mode = str(mode or get_language_mode() or "en").strip().lower()
    if language_mode in _SPACY_INIT_ATTEMPTED:
        return _SPACY_NLP_BY_MODE.get(language_mode)
    _SPACY_INIT_ATTEMPTED.add(language_mode)
    if _env_enabled("LEAF_DISABLE_SPACY"):
        return None
    try:
        import spacy
    except ImportError:
        return None
    model_names = (
        [_env_first("LEAF_SPACY_MODEL_ZH"), _env_first("LEAF_SPACY_MODEL"), "zh_core_web_sm"]
        if language_mode == "zh"
        else [_env_first("LEAF_SPACY_MODEL_EN"), _env_first("LEAF_SPACY_MODEL"), "en_core_web_sm"]
    )
    for model_name in model_names:
        if not model_name:
            continue
        try:
            _SPACY_NLP_BY_MODE[language_mode] = spacy.load(model_name, disable=["lemmatizer"])
            return _SPACY_NLP_BY_MODE[language_mode]
        except OSError:
            try:
                model_module = importlib.import_module(model_name)
                if hasattr(model_module, "load"):
                    _SPACY_NLP_BY_MODE[language_mode] = model_module.load(disable=["lemmatizer"])
                    return _SPACY_NLP_BY_MODE[language_mode]
            except Exception:
                continue
    return None


def _get_yake_extractor(mode: str | None = None):
    language_mode = str(mode or get_language_mode() or "en").strip().lower()
    if language_mode in _YAKE_INIT_ATTEMPTED:
        return _YAKE_KEYWORD_EXTRACTOR_BY_MODE.get(language_mode)
    _YAKE_INIT_ATTEMPTED.add(language_mode)
    if _env_enabled("LEAF_DISABLE_YAKE"):
        return None
    try:
        import yake
    except ImportError:
        return None
    try:
        _YAKE_KEYWORD_EXTRACTOR_BY_MODE[language_mode] = yake.KeywordExtractor(
            lan="zh" if language_mode == "zh" else "en",
            n=2,
            dedupLim=0.85,
            dedupFunc="seqm",
            windowsSize=1,
            top=12,
        )
    except Exception:
        _YAKE_KEYWORD_EXTRACTOR_BY_MODE[language_mode] = None
    return _YAKE_KEYWORD_EXTRACTOR_BY_MODE[language_mode]


def _spacy_entity_candidates(text: str, mode: str | None = None) -> list[str]:
    nlp = _get_spacy_nlp(mode)
    if nlp is None:
        return []
    try:
        doc = nlp(text)
    except Exception:
        return []
    candidates: list[str] = []
    for ent in doc.ents:
        if ent.label_ in SPACY_ENTITY_LABELS:
            value = ent.text.strip()
            if value:
                candidates.append(value)
    if doc.has_annotation("DEP"):
        for chunk in doc.noun_chunks:
            text_value = chunk.text.strip()
            if not text_value:
                continue
            if any(token.pos_ in {"PROPN", "NOUN"} for token in chunk):
                candidates.append(text_value)
    return candidates


def _ensure_nltk_ready() -> bool:
    global _NLTK_READY
    if _NLTK_READY is not None:
        return _NLTK_READY
    try:
        import nltk
        for resource in [
            "tokenizers/punkt",
            "taggers/averaged_perceptron_tagger",
            "chunkers/maxent_ne_chunker",
            "corpora/words",
        ]:
            nltk.data.find(resource)
    except Exception:
        _NLTK_READY = False
        return _NLTK_READY
    _NLTK_READY = True
    return _NLTK_READY


def _nltk_entity_candidates(text: str, mode: str | None = None) -> list[str]:
    language_mode = str(mode or get_language_mode() or "en").strip().lower()
    if language_mode != "en":
        return []
    if _env_enabled("LEAF_DISABLE_NLTK"):
        return []
    if not _ensure_nltk_ready():
        return []
    try:
        import nltk
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(tagged, binary=False)
    except Exception:
        return []
    candidates: list[str] = []
    current_phrase: list[str] = []
    for node in tree:
        label = getattr(node, "label", None)
        if callable(label):
            node_label = label()
            if node_label in {"PERSON", "ORGANIZATION", "GPE", "LOCATION", "FACILITY"}:
                phrase = " ".join(token for token, _ in node.leaves()).strip()
                if phrase:
                    candidates.append(phrase)
                continue
        if isinstance(node, tuple):
            token, pos = node
            if pos in {"NNP", "NNPS"}:
                current_phrase.append(token)
                continue
        if current_phrase:
            candidates.append(" ".join(current_phrase))
            current_phrase = []
    if current_phrase:
        candidates.append(" ".join(current_phrase))
    return candidates


def _zh_phrase_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(value.strip() for value in re.findall(r"《([^》]{1,80})》", text))
    candidates.extend(value.strip() for value in re.findall(r"[“\"]([^”\"]{2,80})[”\"]", text))
    candidates.extend(value.strip() for value in re.findall(r"[「『]([^」』]{2,80})[」』]", text))
    candidates.extend(value.strip() for value in re.findall(r"\b[A-Za-z][A-Za-z0-9._+-]{1,}\b", text))
    candidates.extend(value.strip() for value in re.findall(r"\b[A-Z][a-zA-Z0-9_-]{1,}(?:\s+[A-Z][a-zA-Z0-9_-]{1,}){0,3}\b", text))
    for match in re.finditer(r"([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,8})的演唱会", text):
        candidates.append(match.group(1))
    for match in re.finditer(
        r"([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,12}(?:博物馆|展览|演唱会|电影|电视剧|歌曲|画展|摄影展|公园|餐厅|课程|比赛))",
        text,
    ):
        candidates.append(match.group(1))
    for match in re.finditer(
        r"(?:去了|参观了|看了|喜欢|最喜欢|学习|学了)([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,10})",
        text,
    ):
        phrase = match.group(1)
        if any(marker in phrase for marker in ("什么", "哪些", "哪个", "为什么", "怎么", "如何")):
            continue
        candidates.append(phrase)
    return candidates


def _clean_zh_candidate(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^(我最近|最近|另外我在|我在|我曾经|你最近|你曾经)", "", cleaned)
    cleaned = re.sub(r"^(去看了|看了|去了|参观了|学了|学习了|最喜欢|喜欢)", "", cleaned)
    cleaned = re.sub(r"^\d{1,2}月\d{1,2}[日号]?", "", cleaned)
    cleaned = re.sub(r"^(号|日)", "", cleaned)
    cleaned = re.sub(r"^(去看了|看了|去了|参观了|学了|学习了|最喜欢|喜欢)", "", cleaned)
    return cleaned.strip()


def extract_entities(text: str, mode: str | None = None) -> list[str]:
    language_mode = str(mode or get_language_mode() or "en").strip().lower()
    entities: list[str] = []
    entities.extend(_spacy_entity_candidates(text, mode=language_mode))
    entities.extend(_nltk_entity_candidates(text, mode=language_mode))
    if language_mode == "zh":
        entities.extend(_zh_phrase_candidates(text))
    else:
        quoted = re.findall(r'"([^"]{2,80})"', text)
        entities.extend(value.strip() for value in quoted)
        capitalized_phrases = re.findall(r"\b(?:[A-Z][a-zA-Z0-9_-]{1,}(?:\s+[A-Z][a-zA-Z0-9_-]{1,}){1,3})\b", text)
        entities.extend(capitalized_phrases)
        phrase_parts: set[str] = set()
        for phrase in capitalized_phrases:
            canonical_phrase = canonicalize_entities([phrase], limit=1)
            if canonical_phrase:
                phrase_parts.update(canonical_phrase[0].split())
        capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", text)
        for token in capitalized:
            canonical_token = ENTITY_RESOLVER.resolve(token.strip()).canonical
            if canonical_token and canonical_token in phrase_parts:
                continue
            entities.append(token)
        keywords = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", text.lower())
        counts = Counter(token for token in keywords if token not in STOPWORDS)
        entities.extend(token for token, count in counts.items() if count >= 2)
    deduped: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        candidate = _clean_zh_candidate(entity) if language_mode == "zh" else entity
        if not candidate:
            continue
        resolved = ENTITY_RESOLVER.resolve(candidate.strip())
        key = resolved.canonical
        if not key:
            continue
        if language_mode == "zh":
            key_tokens = [token for token in re.split(r"[^a-z0-9\u3400-\u9fff]+", key) if token]
            if not key_tokens:
                continue
            if all(token in ZH_ENTITY_NOISE_WORDS or token in ENTITY_NOISE_WORDS or token in STOPWORDS for token in key_tokens):
                continue
        else:
            key_tokens = [token for token in re.split(r"[^a-z0-9]+", key) if token]
            if not key_tokens:
                continue
            if all(token in ENTITY_NOISE_WORDS or token in STOPWORDS for token in key_tokens):
                continue
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(resolved.canonical)
    return deduped[:8]


def extract_semantic_references(text: str, mode: str | None = None) -> list[str]:
    language_mode = str(mode or get_language_mode() or "en").strip().lower()
    lowered = f" {str(text or '').lower()} "
    refs: list[str] = []
    seen: set[str] = set()
    for spec in SEMANTIC_LEXICON_BY_MODE.get(language_mode, SEMANTIC_LEXICON_BY_MODE["en"]):
        concept = str(spec["concept"])
        concept_hit = any(
            alias in str(text or "")
            if language_mode == "zh"
            else f" {alias} " in lowered
            for alias in spec["aliases"]
        )
        for alias, canonical in spec["items"].items():
            alias_hit = alias in str(text or "") if language_mode == "zh" else f" {alias} " in lowered
            if alias_hit and canonical not in seen:
                refs.append(canonical)
                seen.add(canonical)
                concept_hit = True
        if concept_hit and concept not in seen:
            refs.append(concept)
            seen.add(concept)
    return refs[:10]


def merge_memory_refs(texts: list[str], limit: int = 12) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for ref in extract_entities(text) + extract_semantic_references(text):
            canonical = ENTITY_RESOLVER.resolve(ref).canonical
            if not canonical or canonical in seen:
                continue
            refs.append(canonical)
            seen.add(canonical)
            if len(refs) >= limit:
                return refs
    return refs


def canonicalize_entities(entities: list[str], limit: int = 8) -> list[str]:
    canonical: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        normalized = canonicalize_entity(entity)
        if not normalized or normalized in seen:
            continue
        canonical.append(normalized)
        seen.add(normalized)
        if len(canonical) >= limit:
            break
    return canonical


def infer_memory_kind(atom_type: str, content: str) -> str:
    atom_type = str(atom_type or "").strip().lower()
    lowered = str(content or "").lower()
    if atom_type in {"preference"}:
        return "preference"
    if atom_type in {"goal", "decision", "open_loop"}:
        return "plan"
    if atom_type in {"relation"}:
        return "relation"
    if atom_type in {"profile", "identity", "state"}:
        return "state"
    if any(marker in lowered for marker in ["prefer", "favorite", "dislike", "love ", "hate "]):
        return "preference"
    if any(marker in lowered for marker in ["work as", "job", "works at", "live in", "lives in", "located in", "married", "dating"]):
        return "state"
    if any(marker in lowered for marker in ["plan", "goal", "want to", "need to", "will ", "should ", "must "]):
        return "plan"
    if any(marker in lowered for marker in ["because", "therefore", "related to", "with ", "between "]):
        return "relation"
    return "event"


def infer_atom_status(content: str) -> str:
    lowered = str(content or "").lower()
    if any(marker in lowered for marker in ["maybe", "might", "possibly", "unclear", "unsure"]):
        return "uncertain"
    if any(marker in lowered for marker in ["no longer", "used to", "previously", "formerly", "replaced", "instead of", "changed from", "stopped"]):
        return "superseded"
    return "active"


def _truncate_text(text: str, max_chars: int) -> str:
    stripped = re.sub(r"\s+", " ", str(text or "").strip())
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3].rstrip() + "..."


def _split_sentences(texts: list[str]) -> list[str]:
    sentences: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for sentence in _SENTENCE_SPLIT_RE.split(str(text or "").strip()):
            clean = re.sub(r"\s+", " ", sentence).strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            sentences.append(clean)
    return sentences


def _normalize_tag(text: str) -> str:
    lowered = re.sub(r"\s+", " ", str(text or "").strip().lower())
    lowered = re.sub(r"^[^a-z0-9\u3400-\u9fff]+|[^a-z0-9\u3400-\u9fff]+$", "", lowered)
    if not lowered:
        return ""
    if any(marker in lowered for marker in ("。", "，", ",", "?", "？", "!", "！", ":", "：", ";", "；", "\n")):
        return ""
    if len(lowered) > 20:
        return ""
    tokens = [token for token in re.split(r"[^a-z0-9\u3400-\u9fff]+", lowered) if token]
    if not tokens:
        return ""
    if all(token in STOPWORDS or token in ENTITY_NOISE_WORDS for token in tokens):
        return ""
    return lowered


@lru_cache(maxsize=4096)
def _score_tag_phrase(tag: str) -> tuple[int, int]:
    tokens = [token for token in re.split(r"[^a-z0-9\u3400-\u9fff]+", tag) if token]
    return (len(tokens), len(tag))


def build_text_tags(texts: list[str], max_tags: int = 5) -> list[str]:
    merged = re.sub(r"\s+", " ", " ".join(str(text or "").strip() for text in texts if str(text or "").strip())).strip()
    if not merged:
        return []
    candidates: list[str] = []
    extractor = _get_yake_extractor(get_language_mode())
    if extractor is not None:
        try:
            for keyword, _score in extractor.extract_keywords(merged):
                normalized = _normalize_tag(keyword)
                if normalized:
                    candidates.append(normalized)
        except Exception:
            pass
    candidates.extend(_normalize_tag(entity) for entity in extract_entities(merged))
    candidates.extend(_normalize_tag(ref) for ref in extract_semantic_references(merged))
    counts = Counter()
    ordered: list[str] = []
    for candidate in candidates:
        clean = str(candidate or "").strip()
        if not clean:
            continue
        counts[clean] += 1
        if clean not in ordered:
            ordered.append(clean)
    ranked = sorted(
        ordered,
        key=lambda item: (
            -counts[item],
            -_score_tag_phrase(item)[0],
            -_score_tag_phrase(item)[1],
            item,
        ),
    )
    return ranked[:max_tags]


def summarize_texts(texts: list[str], max_chars: int = 480) -> str:
    merged = " ".join(item.strip() for item in texts if item.strip())
    merged = re.sub(r"\s+", " ", merged).strip()
    if len(merged) <= max_chars:
        return merged
    sentences = _split_sentences(texts)
    if not sentences:
        return _truncate_text(merged, max_chars=max_chars)
    tags = build_text_tags(texts, max_tags=6)
    scored_sentences: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = max(0.0, 1.4 - 0.08 * index)
        for tag in tags:
            if tag and tag in lowered:
                score += 1.1
        sentence_entities = extract_entities(sentence)
        score += min(1.2, 0.18 * len(sentence_entities))
        sentence_refs = extract_semantic_references(sentence)
        score += min(0.8, 0.2 * len(sentence_refs))
        if len(sentence) <= max_chars * 0.7:
            score += 0.25
        scored_sentences.append((score, index, sentence))
    chosen: list[tuple[int, str]] = []
    char_count = 0
    for _score, index, sentence in sorted(scored_sentences, key=lambda item: (-item[0], item[1])):
        extra = len(sentence) + (1 if chosen else 0)
        if chosen and char_count + extra > max_chars:
            continue
        if not chosen and len(sentence) > max_chars:
            return _truncate_text(sentence, max_chars=max_chars)
        chosen.append((index, sentence))
        char_count += extra
        if char_count >= max_chars * 0.82:
            break
    if not chosen:
        return _truncate_text(merged, max_chars=max_chars)
    chosen.sort(key=lambda item: item[0])
    return _truncate_text(" ".join(sentence for _, sentence in chosen), max_chars=max_chars)


def make_synopsis(texts: list[str], max_chars: int = 180) -> str:
    return summarize_texts(texts, max_chars=max_chars)


def make_evidence_preview(texts: list[str], limit: int = 3, max_chars: int = 180) -> list[str]:
    previews: list[str] = []
    for text in texts:
        compact = summarize_texts([text], max_chars=max_chars)
        if compact and compact not in previews:
            previews.append(compact)
        if len(previews) >= limit:
            break
    return previews


class AtomExtractor:
    def __init__(self, llm: ChatClient | None = None):
        self.llm = llm
        self._runtime_metrics: dict[str, int] | None = None
        self._runtime_metrics_lock = threading.Lock()
        self._cache_dir = self._resolve_cache_dir()

    def reset_runtime_metrics(self) -> None:
        with self._runtime_metrics_lock:
            self._runtime_metrics = {
                "atom_prompt_tokens_total": 0,
                "atom_prompt_tokens_provider_usage_calls": 0,
                "atom_prompt_tokens_estimated_calls": 0,
            }

    def consume_runtime_metrics(self) -> dict[str, int]:
        with self._runtime_metrics_lock:
            metrics = dict(self._runtime_metrics or {})
            self._runtime_metrics = None
        return metrics

    def extract_atoms(self, span: RawSpan) -> list[MemoryAtom]:
        atoms = self._heuristic_atoms(span)
        llm_atoms = self._llm_atoms(span)
        merged = {(atom.atom_type, atom.content): atom for atom in atoms}
        for atom in llm_atoms:
            merged[(atom.atom_type, atom.content)] = atom
        return list(merged.values())

    def _heuristic_atoms(self, span: RawSpan) -> list[MemoryAtom]:
        text = span.text.strip()
        semantic_refs = extract_semantic_references(span_surface_text(span.speaker, text, span.metadata))
        entities = list(dict.fromkeys(extract_entities(text) + semantic_refs))
        canonical_entities = canonicalize_entities(entities)
        temporal_grounding = derive_temporal_grounding(text=text, timestamp=span.timestamp)
        atoms: list[MemoryAtom] = [
            MemoryAtom(
                atom_id=stable_id("atom", span.span_id, "observation", text),
                corpus_id=span.corpus_id,
                span_id=span.span_id,
                atom_type="observation",
                content=text,
                entities=entities,
                canonical_entities=canonical_entities,
                support_span_ids=[span.span_id],
                memory_kind=infer_memory_kind("observation", text),
                status=infer_atom_status(text),
                time_range=span.timestamp,
                confidence=0.4,
                metadata={"speaker": span.speaker, "temporal_grounding": temporal_grounding, "semantic_refs": semantic_refs},
            )
        ]
        caption = str(span.metadata.get("blip_caption") or "").strip()
        if caption:
            caption_refs = extract_semantic_references(caption)
            atoms.append(
                MemoryAtom(
                    atom_id=stable_id("atom", span.span_id, "image_observation", caption),
                    corpus_id=span.corpus_id,
                    span_id=span.span_id,
                    atom_type="observation",
                    content=f"Image evidence: {caption}",
                    entities=list(dict.fromkeys(extract_entities(caption) + caption_refs)),
                    canonical_entities=canonicalize_entities(list(dict.fromkeys(extract_entities(caption) + caption_refs))),
                    support_span_ids=[span.span_id],
                    memory_kind="event",
                    status="active",
                    time_range=span.timestamp,
                    confidence=0.45,
                    metadata={
                        "speaker": span.speaker,
                        "source": "blip_caption",
                        "temporal_grounding": temporal_grounding,
                        "semantic_refs": caption_refs,
                    },
                )
            )
        lowered = text.lower()
        if any(marker in lowered for marker in ["prefer", "like", "favorite", "favour", "dislike"]):
            atoms.append(self._make_atom(span, "preference", text, entities, 0.7, {"semantic_refs": semantic_refs}))
        if any(marker in lowered for marker in ["decide", "decision", "choose", "selected", "we will", "i will"]):
            atoms.append(self._make_atom(span, "decision", text, entities, 0.7, {"semantic_refs": semantic_refs}))
        if any(marker in lowered for marker in ["need to", "want to", "goal", "plan to", "should", "must"]):
            atoms.append(self._make_atom(span, "goal", text, entities, 0.6, {"semantic_refs": semantic_refs}))
        if any(marker in lowered for marker in ["todo", "follow up", "pending", "open question", "?"]):
            atoms.append(self._make_atom(span, "open_loop", text, entities, 0.6, {"semantic_refs": semantic_refs}))
        if any(marker in lowered for marker in ["error", "failed", "fix", "bug", "exception"]):
            atoms.append(self._make_atom(span, "error_fix", text, entities, 0.7, {"semantic_refs": semantic_refs}))
        return atoms

    def _llm_atoms(self, span: RawSpan) -> list[MemoryAtom]:
        if self.llm is None:
            return []
        cache_key = self._atom_cache_key(span)
        cached_atoms = self._load_cached_llm_atoms(span=span, cache_key=cache_key)
        if cached_atoms is not None:
            return cached_atoms
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract up to 5 memory atoms from a single interaction span. "
                    "Return JSON with key 'atoms'. Each atom must contain type, content, entities, confidence."
                ),
            },
            {
                "role": "user",
                "content": f"speaker={span.speaker}\ntext={span.text}",
            },
        ]
        prompt_tokens = estimate_message_tokens(messages)
        try:
            response = self.llm.chat(messages, max_tokens=400, temperature=0.0)
            provider_prompt_tokens = extract_prompt_tokens(response)
            if provider_prompt_tokens is not None:
                prompt_tokens = provider_prompt_tokens
            payload = extract_json_object(extract_chat_text(response))
        except (OpenAICompatError, ValueError):
            return []
        self._record_prompt_tokens(
            prompt_tokens=prompt_tokens,
            from_provider_usage=provider_prompt_tokens is not None,
        )
        atoms: list[MemoryAtom] = []
        for item in payload.get("atoms") or []:
            atom_type = str(item.get("type") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not atom_type or not content:
                continue
            entities = [str(entity) for entity in item.get("entities") or []]
            confidence = float(item.get("confidence", 0.6))
            atoms.append(self._make_atom(span, atom_type, content, entities, confidence))
        self._store_cached_llm_atoms(span=span, cache_key=cache_key, atoms=atoms)
        return atoms

    @staticmethod
    def _make_atom(
        span: RawSpan,
        atom_type: str,
        content: str,
        entities: list[str],
        confidence: float,
        extra_metadata: dict[str, object] | None = None,
    ) -> MemoryAtom:
        metadata = {"speaker": span.speaker}
        if extra_metadata:
            metadata.update(extra_metadata)
        return MemoryAtom(
            atom_id=stable_id("atom", span.span_id, atom_type, content),
            corpus_id=span.corpus_id,
            span_id=span.span_id,
            atom_type=atom_type,
            content=content,
            entities=entities[:8],
            canonical_entities=canonicalize_entities(entities),
            support_span_ids=[span.span_id],
            memory_kind=infer_memory_kind(atom_type, content),
            status=infer_atom_status(content),
            time_range=span.timestamp,
            confidence=confidence,
            metadata=metadata,
        )

    def _record_prompt_tokens(self, *, prompt_tokens: int, from_provider_usage: bool) -> None:
        with self._runtime_metrics_lock:
            if not isinstance(self._runtime_metrics, dict):
                return
            self._runtime_metrics["atom_prompt_tokens_total"] = int(
                self._runtime_metrics.get("atom_prompt_tokens_total", 0)
            ) + int(prompt_tokens)
            if from_provider_usage:
                self._runtime_metrics["atom_prompt_tokens_provider_usage_calls"] = int(
                    self._runtime_metrics.get("atom_prompt_tokens_provider_usage_calls", 0)
                ) + 1
            else:
                self._runtime_metrics["atom_prompt_tokens_estimated_calls"] = int(
                    self._runtime_metrics.get("atom_prompt_tokens_estimated_calls", 0)
                ) + 1

    def _resolve_cache_dir(self) -> Path | None:
        if _env_enabled("LEAF_DISABLE_ATOM_CACHE"):
            return None
        raw_dir = os.environ.get("LEAF_ATOM_CACHE_DIR", "").strip()
        if not raw_dir:
            raw_dir = os.path.expanduser("~/.cache/leaf/atom_extraction")
        try:
            path = Path(raw_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            return None

    def _atom_cache_key(self, span: RawSpan) -> str:
        language_mode = get_language_mode()
        model_name = str(getattr(getattr(self.llm, "config", None), "model_name", "") or "")
        base_url = str(getattr(getattr(self.llm, "config", None), "base_url", "") or "")
        prompt_revision = "atom_prompt_v1"
        digest = hashlib.sha1(
            "||".join(
                [
                    language_mode,
                    prompt_revision,
                    model_name,
                    base_url,
                    str(span.speaker or ""),
                    str(span.text or ""),
                ]
            ).encode("utf-8")
        ).hexdigest()
        return digest

    def _cache_file_path(self, cache_key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{cache_key}.json"

    def _load_cached_llm_atoms(self, *, span: RawSpan, cache_key: str) -> list[MemoryAtom] | None:
        cache_path = self._cache_file_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        items = payload.get("atoms")
        if not isinstance(items, list):
            return None
        atoms: list[MemoryAtom] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            atom_type = str(item.get("type") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not atom_type or not content:
                continue
            entities = [str(entity) for entity in item.get("entities") or []]
            try:
                confidence = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            atoms.append(self._make_atom(span, atom_type, content, entities, confidence))
        return atoms

    def _store_cached_llm_atoms(self, *, span: RawSpan, cache_key: str, atoms: list[MemoryAtom]) -> None:
        cache_path = self._cache_file_path(cache_key)
        if cache_path is None:
            return
        payload = {
            "atoms": [
                {
                    "type": atom.atom_type,
                    "content": atom.content,
                    "entities": list(atom.entities or []),
                    "confidence": atom.confidence,
                }
                for atom in atoms
            ]
        }
        temp_path = cache_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(cache_path)
        except OSError:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass


def build_graph_edges(span: RawSpan, atoms: list[MemoryAtom]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    entities = []
    for atom in atoms:
        entities.extend(atom.canonical_entities or atom.entities)
    entities = list(dict.fromkeys(entities))[:8]
    for entity in entities:
        canonical_entity = ENTITY_RESOLVER.resolve(entity).canonical
        if not canonical_entity:
            continue
        edges.append(
            GraphEdge(
                edge_id=stable_id("edge", span.span_id, span.speaker, canonical_entity, "mentions"),
                corpus_id=span.corpus_id,
                src=span.speaker,
                relation="mentions",
                dst=canonical_entity,
                valid_from=span.timestamp,
                confidence=0.5,
                provenance=span.span_id,
                metadata={"raw_entity": entity},
            )
        )
    canonical_entities = [ENTITY_RESOLVER.resolve(entity).canonical for entity in entities]
    canonical_entities = [entity for entity in canonical_entities if entity]
    canonical_entities = list(dict.fromkeys(canonical_entities))
    for left_index, left in enumerate(canonical_entities):
        for right in canonical_entities[left_index + 1:]:
            relation = "co_occurs_with"
            confidence = 0.4
            lowered_contents = " ".join(atom.content.lower() for atom in atoms)
            if any(marker in lowered_contents for marker in ["failed", "not ", "no longer", "instead of", "replaced", "contradict"]):
                relation = "contradicts"
                confidence = 0.5
            elif any(marker in lowered_contents for marker in ["support", "because", "therefore", "so that", "decision", "prefer"]):
                relation = "supports"
                confidence = 0.45
            edges.append(
                GraphEdge(
                    edge_id=stable_id("edge", span.span_id, left, right, relation),
                    corpus_id=span.corpus_id,
                    src=left,
                    relation=relation,
                    dst=right,
                    valid_from=span.timestamp,
                    confidence=confidence,
                    provenance=span.span_id,
                )
            )
    for atom in atoms:
        if not atom.entities:
            continue
        relation = "supports"
        lowered = atom.content.lower()
        if any(marker in lowered for marker in ["failed", "not ", "no longer", "contradict"]):
            relation = "contradicts"
        elif any(marker in lowered for marker in ["instead", "replace", "supersede", "updated", "changed"]):
            relation = "supersedes"
        for entity in atom.entities[:4]:
            canonical_entity = ENTITY_RESOLVER.resolve(entity).canonical
            if not canonical_entity:
                continue
            edges.append(
                GraphEdge(
                    edge_id=stable_id("edge", span.span_id, atom.atom_id, canonical_entity, relation),
                    corpus_id=span.corpus_id,
                    src=atom.atom_type,
                    relation=relation,
                    dst=canonical_entity,
                    valid_from=span.timestamp,
                    confidence=max(0.45, atom.confidence),
                    provenance=span.span_id,
                    metadata={
                        "atom_id": atom.atom_id,
                        "raw_entity": entity,
                        "memory_kind": atom.memory_kind,
                        "atom_status": atom.status,
                    },
                )
            )
    temporal_grounding = derive_temporal_grounding(text=span_surface_text(span.speaker, span.text, span.metadata), timestamp=span.timestamp)
    temporal_node = temporal_grounding.get("grounded_date") or temporal_grounding.get("grounded_month") or temporal_grounding.get("grounded_year")
    if temporal_node:
        edges.append(
            GraphEdge(
                edge_id=stable_id("edge", span.span_id, span.speaker, str(temporal_node), "occurred_on"),
                corpus_id=span.corpus_id,
                src=span.speaker,
                relation="occurred_on",
                dst=str(temporal_node),
                valid_from=span.timestamp,
                confidence=0.55,
                provenance=span.span_id,
                metadata={"precision": temporal_grounding.get("precision")},
            )
        )
    return edges
