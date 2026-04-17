from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

try:
    import regex as unicode_re
except ImportError:  # pragma: no cover - optional dependency fallback
    unicode_re = re

try:
    import jieba
except ImportError:  # pragma: no cover - optional dependency fallback
    jieba = None


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "by",
    "at", "from", "with", "without", "into", "over", "under", "after", "before",
}
ZH_STOPWORDS = {
    "的", "了", "呢", "吗", "啊", "呀", "吧", "和", "与", "及", "或", "而", "被",
    "在", "对", "把", "将", "向", "给", "跟", "还", "又", "也", "都", "就", "很",
    "太", "更", "最", "这", "那", "这些", "那些", "一个", "一下", "一种", "一些",
}
_CJK_RE = unicode_re.compile(r"[\p{Script=Han}]")
_LATIN_TOKEN_RE = unicode_re.compile(r"[\p{Latin}\d]+(?:[-_'][\p{Latin}\d]+)*")
_CJK_RUN_RE = unicode_re.compile(r"[\p{Script=Han}]+")
_EDGE_PUNCT_RE = unicode_re.compile(r"^[\p{P}\p{S}\s]+|[\p{P}\p{S}\s]+$")
_TITLE_SPAN_RE = unicode_re.compile(r"[《“\"「](.{1,80}?)[》”\"」]")


@lru_cache(maxsize=32768)
def normalize_surface_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@lru_cache(maxsize=32768)
def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(normalize_surface_text(text)))


@lru_cache(maxsize=32768)
def strip_edge_punctuation(text: str) -> str:
    normalized = normalize_surface_text(text)
    if not normalized:
        return ""
    stripped = _EDGE_PUNCT_RE.sub("", normalized)
    return stripped.strip()


def _tokenize_zh(text: str) -> list[str]:
    normalized = normalize_surface_text(text)
    if not normalized:
        return []
    if jieba is not None:
        tokens = [str(token).strip() for token in jieba.lcut(normalized, HMM=False)]
        return [token for token in tokens if token and strip_edge_punctuation(token)]
    return [match.group(0) for match in _CJK_RUN_RE.finditer(normalized)]


def _expand_cjk_subgrams(token: str, *, max_ngram: int) -> set[str]:
    expanded: set[str] = set()
    clean = strip_edge_punctuation(token)
    if len(clean) < 2:
        return expanded
    upper = min(max_ngram, len(clean))
    for size in range(2, upper + 1):
        for index in range(0, len(clean) - size + 1):
            expanded.add(clean[index : index + size])
    return expanded


@lru_cache(maxsize=32768)
def language_aware_terms(
    text: str,
    *,
    mode: str = "auto",
    include_cjk_subgrams: bool = False,
    max_cjk_ngram: int = 4,
) -> tuple[str, ...]:
    normalized = normalize_surface_text(text)
    if not normalized:
        return ()
    resolved_mode = str(mode or "auto").strip().lower()
    if resolved_mode == "auto":
        resolved_mode = "zh" if contains_cjk(normalized) else "en"
    tokens: set[str] = set()
    for token in _LATIN_TOKEN_RE.findall(normalized.lower()):
        if len(token) > 1:
            tokens.add(token)
    if resolved_mode == "zh":
        if include_cjk_subgrams:
            for run in _CJK_RUN_RE.findall(normalized):
                clean_run = strip_edge_punctuation(run)
                if clean_run and len(clean_run) >= 2:
                    tokens.update(_expand_cjk_subgrams(clean_run, max_ngram=max_cjk_ngram))
        for match in _TITLE_SPAN_RE.finditer(normalized):
            title = strip_edge_punctuation(match.group(1))
            if title and contains_cjk(title) and len(title) >= 2:
                tokens.add(title)
                if include_cjk_subgrams:
                    tokens.update(_expand_cjk_subgrams(title, max_ngram=max_cjk_ngram))
        for raw_token in _tokenize_zh(normalized):
            token = strip_edge_punctuation(raw_token)
            if not token or token in ZH_STOPWORDS:
                continue
            if contains_cjk(token):
                if len(token) >= 2:
                    tokens.add(token)
                if include_cjk_subgrams:
                    tokens.update(_expand_cjk_subgrams(token, max_ngram=max_cjk_ngram))
            elif len(token) > 1:
                tokens.add(token.lower())
    else:
        for token in re.findall(r"[a-z0-9]+", normalized.lower()):
            if len(token) > 2 and token not in STOPWORDS:
                tokens.add(token)
    return tuple(sorted(tokens))


@lru_cache(maxsize=32768)
def normalize_text(text: str) -> str:
    lowered = normalize_surface_text(text).lower()
    lowered = lowered.replace("_", " ")
    lowered = re.sub(r"\([^)]*\)", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\s-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


@lru_cache(maxsize=32768)
def canonicalize_entity(text: str) -> str:
    normalized = normalize_text(text)
    tokens = [token for token in normalized.split() if token not in STOPWORDS]
    return " ".join(tokens).strip()


@lru_cache(maxsize=32768)
def fuzzy_text_similarity(left: str, right: str) -> float:
    left_norm = canonicalize_entity(left)
    right_norm = canonicalize_entity(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if len(left_norm) < 4 or len(right_norm) < 4:
        return 0.0
    direct = SequenceMatcher(None, left_norm, right_norm).ratio()
    token_sorted_left = " ".join(sorted(left_norm.split()))
    token_sorted_right = " ".join(sorted(right_norm.split()))
    token_sorted = SequenceMatcher(None, token_sorted_left, token_sorted_right).ratio()
    return max(direct, token_sorted)


@lru_cache(maxsize=32768)
def generate_aliases(text: str) -> list[str]:
    normalized = normalize_text(text)
    canonical = canonicalize_entity(text)
    aliases = {normalized, canonical}
    words = [token for token in canonical.split() if token]
    if len(words) >= 2:
        aliases.add(words[-1])
        aliases.add(" ".join(words[:2]))
    compact = canonical.replace(" ", "")
    if compact:
        aliases.add(compact)
    aliases = {alias for alias in aliases if alias}
    return sorted(aliases)


@dataclass(slots=True)
class CanonicalEntity:
    raw: str
    canonical: str
    aliases: list[str]


class EntityResolver:
    @lru_cache(maxsize=32768)
    def resolve(self, text: str) -> CanonicalEntity:
        return CanonicalEntity(
            raw=text,
            canonical=canonicalize_entity(text),
            aliases=generate_aliases(text),
        )

    def overlap_score(self, left: str, right: str) -> float:
        left_resolved = self.resolve(left)
        right_resolved = self.resolve(right)
        if not left_resolved.canonical or not right_resolved.canonical:
            return 0.0
        if left_resolved.canonical == right_resolved.canonical:
            return 1.0
        if left_resolved.canonical in right_resolved.aliases or right_resolved.canonical in left_resolved.aliases:
            return 0.85
        left_tokens = set(left_resolved.canonical.split())
        right_tokens = set(right_resolved.canonical.split())
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens.intersection(right_tokens))
        if overlap == 0:
            return 0.0
        return overlap / max(len(left_tokens), len(right_tokens))
