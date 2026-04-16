from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ModelConfig:
    provider: str
    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 120


@dataclass(slots=True)
class IngestConfig:
    mode: str = "online"
    migration_refresh_derived: bool = True
    migration_build_entity_facets: bool = False
    migration_build_entity_bridges: bool = False
    migration_bridge_mode: str = "hybrid"


@dataclass(slots=True)
class LEAFConfig:
    llm: ModelConfig
    embedding: ModelConfig
    additional_llm: ModelConfig | None = None
    ingest: IngestConfig | None = None


def _load_model_config(section: dict) -> ModelConfig:
    return ModelConfig(
        provider=str(section.get("provider") or "openai"),
        model_name=str(section.get("model_name") or ""),
        api_key=str(section.get("api_key") or ""),
        base_url=str(section.get("base_url") or "").rstrip("/"),
        temperature=float(section.get("temperature", 0.0)),
        max_tokens=int(section.get("max_tokens", 1024)),
        timeout=int(section.get("timeout", 120)),
    )


def _load_ingest_config(section: dict) -> IngestConfig:
    payload = dict(section or {})
    mode = str(payload.get("mode") or "online").strip().lower()
    if mode not in {"online", "migration"}:
        raise ValueError(f"Unsupported ingest.mode: {mode}")
    bridge_mode = str(payload.get("migration_bridge_mode") or "hybrid").strip().lower()
    if bridge_mode not in {"hybrid", "embedding_cluster", "graph_lexical"}:
        raise ValueError(f"Unsupported ingest.migration_bridge_mode: {bridge_mode}")
    return IngestConfig(
        mode=mode,
        migration_refresh_derived=bool(payload.get("migration_refresh_derived", True)),
        migration_build_entity_facets=bool(payload.get("migration_build_entity_facets", False)),
        migration_build_entity_bridges=bool(payload.get("migration_build_entity_bridges", False)),
        migration_bridge_mode=bridge_mode,
    )


def load_config(path: str | Path) -> LEAFConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return LEAFConfig(
        llm=_load_model_config(payload.get("llm") or {}),
        embedding=_load_model_config(payload.get("embedding") or {}),
        additional_llm=_load_model_config(payload.get("additional_llm") or {}) if (payload.get("additional_llm") or {}) else None,
        ingest=_load_ingest_config(payload.get("ingest") or {}),
    )

