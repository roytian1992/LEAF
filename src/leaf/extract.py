from __future__ import annotations

import hashlib
import importlib
import os
import re
from collections import Counter

from .normalize import EntityResolver, canonicalize_entity
from .clients import ChatClient, OpenAICompatError, extract_json_object
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

ENTITY_RESOLVER = EntityResolver()
SEMANTIC_LEXICON = [
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
]
SPACY_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT"}
_SPACY_NLP = None
_SPACY_INIT_ATTEMPTED = False
_NLTK_READY = None


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


def _get_spacy_nlp():
    global _SPACY_NLP, _SPACY_INIT_ATTEMPTED
    if _SPACY_INIT_ATTEMPTED:
        return _SPACY_NLP
    _SPACY_INIT_ATTEMPTED = True
    if _env_enabled("LEAF_DISABLE_SPACY"):
        return None
    try:
        import spacy
    except ImportError:
        return None
    model_names = [
        _env_first("LEAF_SPACY_MODEL"),
        "en_core_web_sm",
    ]
    for model_name in model_names:
        if not model_name:
            continue
        try:
            _SPACY_NLP = spacy.load(model_name, disable=["lemmatizer"])
            return _SPACY_NLP
        except OSError:
            try:
                model_module = importlib.import_module(model_name)
                if hasattr(model_module, "load"):
                    _SPACY_NLP = model_module.load(disable=["lemmatizer"])
                    return _SPACY_NLP
            except Exception:
                continue
    return None


def _spacy_entity_candidates(text: str) -> list[str]:
    nlp = _get_spacy_nlp()
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


def _nltk_entity_candidates(text: str) -> list[str]:
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


def extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    entities.extend(_spacy_entity_candidates(text))
    entities.extend(_nltk_entity_candidates(text))
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
        resolved = ENTITY_RESOLVER.resolve(entity.strip())
        key = resolved.canonical
        if not key:
            continue
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


def extract_semantic_references(text: str) -> list[str]:
    lowered = f" {str(text or '').lower()} "
    refs: list[str] = []
    seen: set[str] = set()
    for spec in SEMANTIC_LEXICON:
        concept = str(spec["concept"])
        concept_hit = any(f" {alias} " in lowered for alias in spec["aliases"])
        for alias, canonical in spec["items"].items():
            if f" {alias} " in lowered and canonical not in seen:
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


def summarize_texts(texts: list[str], max_chars: int = 480) -> str:
    merged = " ".join(item.strip() for item in texts if item.strip())
    merged = re.sub(r"\s+", " ", merged).strip()
    if len(merged) <= max_chars:
        return merged
    return merged[: max_chars - 3].rstrip() + "..."


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
        try:
            response = self.llm.text(messages, max_tokens=400, temperature=0.0)
            payload = extract_json_object(response)
        except (OpenAICompatError, ValueError):
            return []
        atoms: list[MemoryAtom] = []
        for item in payload.get("atoms") or []:
            atom_type = str(item.get("type") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not atom_type or not content:
                continue
            entities = [str(entity) for entity in item.get("entities") or []]
            confidence = float(item.get("confidence", 0.6))
            atoms.append(self._make_atom(span, atom_type, content, entities, confidence))
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
