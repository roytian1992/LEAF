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
class LEAFConfig:
    llm: ModelConfig
    embedding: ModelConfig
    additional_llm: ModelConfig | None = None


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


def load_config(path: str | Path) -> LEAFConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return LEAFConfig(
        llm=_load_model_config(payload.get("llm") or {}),
        embedding=_load_model_config(payload.get("embedding") or {}),
        additional_llm=_load_model_config(payload.get("additional_llm") or {}) if (payload.get("additional_llm") or {}) else None,
    )


