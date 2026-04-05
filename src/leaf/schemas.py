from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RawSpan:
    span_id: str
    corpus_id: str
    session_id: str
    speaker: str
    text: str
    turn_index: int
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("embedding", None)
        return payload


@dataclass(slots=True)
class MemoryAtom:
    atom_id: str
    corpus_id: str
    span_id: str
    atom_type: str
    content: str
    entities: list[str] = field(default_factory=list)
    canonical_entities: list[str] = field(default_factory=list)
    support_span_ids: list[str] = field(default_factory=list)
    derived_from_atom_ids: list[str] = field(default_factory=list)
    memory_kind: str = "event"
    status: str = "active"
    time_range: str | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryPage:
    page_id: str
    corpus_id: str
    parent_id: str | None
    level: str
    title: str
    synopsis: str
    summary: str
    page_kind: str = "cluster"
    evidence_preview: list[str] = field(default_factory=list)
    entity_refs: list[str] = field(default_factory=list)
    canonical_entity_refs: list[str] = field(default_factory=list)
    raw_refs: list[str] = field(default_factory=list)
    anchor_span_ids: list[str] = field(default_factory=list)
    descendant_span_count: int = 0
    child_ids: list[str] = field(default_factory=list)
    time_range: str | None = None
    salience: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphEdge:
    edge_id: str
    corpus_id: str
    src: str
    relation: str
    dst: str
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = 0.5
    status: str = "active"
    provenance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidencePack:
    pages: list[MemoryPage]
    atoms: list[MemoryAtom]
    edges: list[GraphEdge]
    raw_spans: list[RawSpan]
    traversal_path: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
