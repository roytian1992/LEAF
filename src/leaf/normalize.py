from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "by",
    "at", "from", "with", "without", "into", "over", "under", "after", "before",
}


@lru_cache(maxsize=32768)
def normalize_text(text: str) -> str:
    lowered = str(text).strip().lower()
    lowered = lowered.replace("_", " ")
    lowered = re.sub(r"\([^)]*\)", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s-]", " ", lowered)
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
