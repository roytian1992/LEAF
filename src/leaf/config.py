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
    top_p: float | None = None
    max_tokens: int = 1024
    timeout: int = 120
    seed: int | None = None


@dataclass(slots=True)
class IngestConfig:
    mode: str = "online"
    migration_refresh_derived: bool = True
    migration_build_entity_facets: bool = False
    migration_build_entity_bridges: bool = False
    migration_bridge_mode: str = "hybrid"
    online_evolution_enabled: bool = True
    online_evolution_turns_threshold: int = 50
    online_evolution_atoms_threshold: int = 40
    online_evolution_min_cluster_atoms: int = 3
    online_evolution_max_new_topics: int = 3
    online_evolution_max_depth: int = 4
    online_evolution_window_atom_limit: int = 80
    online_evolution_trigger_policy: str = "any"
    online_evolution_growth_strategy: str = "global_terms"
    online_evolution_secondary_assignment_enabled: bool = True
    online_evolution_secondary_max_assignments: int = 50
    online_evolution_secondary_min_score: float = 3.0
    online_evolution_secondary_min_term_overlap: int = 2
    online_evolution_secondary_min_embedding_score: float = 0.0
    online_evolution_secondary_text_mode: str = "content_entities"
    online_evolution_secondary_max_profile_terms: int = 0
    online_evolution_secondary_min_score_margin: float = 0.0
    online_evolution_secondary_min_score_ratio: float = 0.0
    online_evolution_evolved_primary_assignment_enabled: bool = True
    online_evolution_evolved_primary_assignment_mode: str = "all"


@dataclass(slots=True)
class LanguageConfig:
    mode: str = "en"


@dataclass(slots=True)
class LEAFConfig:
    llm: ModelConfig
    embedding: ModelConfig
    memory_llm: ModelConfig | None = None
    ingest: IngestConfig | None = None
    language: LanguageConfig | None = None


def _load_model_config(section: dict) -> ModelConfig:
    top_p_value = section.get("top_p")
    seed_value = section.get("seed")
    return ModelConfig(
        provider=str(section.get("provider") or "openai"),
        model_name=str(section.get("model_name") or ""),
        api_key=str(section.get("api_key") or ""),
        base_url=str(section.get("base_url") or "").rstrip("/"),
        temperature=float(section.get("temperature", 0.0)),
        top_p=float(top_p_value) if top_p_value is not None else None,
        max_tokens=int(section.get("max_tokens", 1024)),
        timeout=int(section.get("timeout", 120)),
        seed=int(seed_value) if seed_value is not None else None,
    )


def _load_ingest_config(section: dict) -> IngestConfig:
    payload = dict(section or {})
    mode = str(payload.get("mode") or "online").strip().lower()
    if mode not in {"online", "migration"}:
        raise ValueError(f"Unsupported ingest.mode: {mode}")
    bridge_mode = str(payload.get("migration_bridge_mode") or "hybrid").strip().lower()
    if bridge_mode not in {"hybrid", "embedding_cluster", "graph_lexical"}:
        raise ValueError(f"Unsupported ingest.migration_bridge_mode: {bridge_mode}")
    evolved_primary_assignment_enabled = bool(
        payload.get("online_evolution_evolved_primary_assignment_enabled", True)
    )
    evolved_primary_assignment_mode = _load_online_evolved_primary_assignment_mode(
        str(
            payload.get(
                "online_evolution_evolved_primary_assignment_mode",
                "all" if evolved_primary_assignment_enabled else "none",
            )
        )
    )
    return IngestConfig(
        mode=mode,
        migration_refresh_derived=bool(payload.get("migration_refresh_derived", True)),
        migration_build_entity_facets=bool(payload.get("migration_build_entity_facets", False)),
        migration_build_entity_bridges=bool(payload.get("migration_build_entity_bridges", False)),
        migration_bridge_mode=bridge_mode,
        online_evolution_enabled=bool(payload.get("online_evolution_enabled", True)),
        online_evolution_turns_threshold=max(1, int(payload.get("online_evolution_turns_threshold", 50))),
        online_evolution_atoms_threshold=max(1, int(payload.get("online_evolution_atoms_threshold", 40))),
        online_evolution_min_cluster_atoms=max(1, int(payload.get("online_evolution_min_cluster_atoms", 3))),
        online_evolution_max_new_topics=max(0, int(payload.get("online_evolution_max_new_topics", 3))),
        online_evolution_max_depth=max(1, int(payload.get("online_evolution_max_depth", 4))),
        online_evolution_window_atom_limit=max(1, int(payload.get("online_evolution_window_atom_limit", 80))),
        online_evolution_trigger_policy=_load_online_trigger_policy(
            str(payload.get("online_evolution_trigger_policy") or "any")
        ),
        online_evolution_growth_strategy=_load_online_growth_strategy(
            str(payload.get("online_evolution_growth_strategy") or "global_terms")
        ),
        online_evolution_secondary_assignment_enabled=bool(
            payload.get("online_evolution_secondary_assignment_enabled", True)
        ),
        online_evolution_secondary_max_assignments=max(
            0, int(payload.get("online_evolution_secondary_max_assignments", 50))
        ),
        online_evolution_secondary_min_score=float(payload.get("online_evolution_secondary_min_score", 3.0)),
        online_evolution_secondary_min_term_overlap=max(
            1, int(payload.get("online_evolution_secondary_min_term_overlap", 2))
        ),
        online_evolution_secondary_min_embedding_score=float(
            payload.get("online_evolution_secondary_min_embedding_score", 0.0)
        ),
        online_evolution_secondary_text_mode=_load_online_secondary_text_mode(
            str(payload.get("online_evolution_secondary_text_mode") or "content_entities")
        ),
        online_evolution_secondary_max_profile_terms=max(
            0, int(payload.get("online_evolution_secondary_max_profile_terms", 0))
        ),
        online_evolution_secondary_min_score_margin=float(
            payload.get("online_evolution_secondary_min_score_margin", 0.0)
        ),
        online_evolution_secondary_min_score_ratio=float(
            payload.get("online_evolution_secondary_min_score_ratio", 0.0)
        ),
        online_evolution_evolved_primary_assignment_enabled=evolved_primary_assignment_enabled,
        online_evolution_evolved_primary_assignment_mode=evolved_primary_assignment_mode,
    )


def _load_online_trigger_policy(value: str) -> str:
    policy = str(value or "any").strip().lower()
    if policy not in {"any", "all", "turns"}:
        raise ValueError(f"Unsupported ingest.online_evolution_trigger_policy: {policy}")
    return policy


def _load_online_growth_strategy(value: str) -> str:
    strategy = str(value or "global_terms").strip().lower()
    if strategy not in {"global_terms", "node_local"}:
        raise ValueError(f"Unsupported ingest.online_evolution_growth_strategy: {strategy}")
    return strategy


def _load_online_secondary_text_mode(value: str) -> str:
    mode = str(value or "content_entities").strip().lower()
    if mode not in {"content_entities", "content_only"}:
        raise ValueError(f"Unsupported ingest.online_evolution_secondary_text_mode: {mode}")
    return mode


def _load_online_evolved_primary_assignment_mode(value: str) -> str:
    mode = str(value or "all").strip().lower()
    if mode not in {"all", "none", "quality_v0", "quality_v1"}:
        raise ValueError(f"Unsupported ingest.online_evolution_evolved_primary_assignment_mode: {mode}")
    return mode


def _load_language_config(section: dict) -> LanguageConfig:
    payload = dict(section or {})
    mode = str(payload.get("mode") or "en").strip().lower()
    if mode not in {"en", "zh"}:
        raise ValueError(f"Unsupported language.mode: {mode}")
    return LanguageConfig(mode=mode)


def _load_optional_model_config(section: dict | None) -> ModelConfig | None:
    payload = dict(section or {})
    if not payload:
        return None
    return _load_model_config(payload)


def load_config(path: str | Path) -> LEAFConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    memory_section = payload.get("memory_llm")
    if not memory_section:
        memory_section = payload.get("additional_llm")
    return LEAFConfig(
        llm=_load_model_config(payload.get("llm") or {}),
        embedding=_load_model_config(payload.get("embedding") or {}),
        memory_llm=_load_optional_model_config(memory_section),
        ingest=_load_ingest_config(payload.get("ingest") or {}),
        language=_load_language_config(payload.get("language") or {}),
    )
