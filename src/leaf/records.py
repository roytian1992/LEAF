from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryEventRecord:
    event_id: str
    corpus_id: str
    session_id: str
    speaker: str
    text: str
    turn_index: int
    timestamp: str | None = None
    raw_span_id: str | None = None
    entity_refs: list[str] = field(default_factory=list)
    canonical_entity_refs: list[str] = field(default_factory=list)
    atom_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryAtomRecord:
    atom_id: str
    event_id: str
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
class StateCandidate:
    candidate_id: str
    corpus_id: str
    event_id: str
    span_id: str
    atom_id: str
    subject: str
    slot: str
    value: str
    normalized_value: str
    memory_kind: str
    policy: str
    status: str
    confidence: float
    valid_from: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryObjectRecord:
    object_id: str
    corpus_id: str
    subject: str
    slot: str
    memory_kind: str
    policy: str
    latest_version_id: str | None = None
    status: str = "active"
    aliases: list[str] = field(default_factory=list)
    canonical_entities: list[str] = field(default_factory=list)
    created_at_event_id: str | None = None
    updated_at_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryObjectVersionRecord:
    version_id: str
    object_id: str
    corpus_id: str
    value: str
    normalized_value: str
    summary: str
    operation: str
    status: str
    confidence: float
    valid_from: str | None = None
    valid_to: str | None = None
    event_id: str | None = None
    atom_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryEvidenceLinkRecord:
    link_id: str
    corpus_id: str
    object_id: str
    version_id: str | None
    event_id: str
    span_id: str
    atom_id: str | None = None
    role: str = "support"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemorySnapshotRecord:
    snapshot_id: str
    corpus_id: str
    parent_id: str | None
    snapshot_kind: str
    scope_id: str
    title: str
    synopsis: str
    summary: str
    object_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    raw_refs: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    entity_refs: list[str] = field(default_factory=list)
    canonical_entity_refs: list[str] = field(default_factory=list)
    time_range: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
