from __future__ import annotations

import json
import math
import socket
import urllib.error
import urllib.request
from typing import Any

from .config import ModelConfig

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


class OpenAICompatError(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return safe_json_loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OpenAICompatError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise OpenAICompatError(f"URL error: {exc}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise OpenAICompatError(f"Timeout error: {exc}") from exc


class ChatClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> dict[str, Any]:
        payload = {
            "model": overrides.get("model", self.config.model_name),
            "messages": messages,
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
        }
        if "response_format" in overrides:
            payload["response_format"] = overrides["response_format"]
        return _post_json(
            url=f"{self.config.base_url}/chat/completions",
            payload=payload,
            api_key=self.config.api_key,
            timeout=int(overrides.get("timeout", self.config.timeout)),
        )

    def text(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        response = self.chat(messages, **overrides)
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatError(f"Unexpected chat response: {response}") from exc


class EmbeddingClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def embed(self, text: str) -> list[float]:
        payload = {
            "model": self.config.model_name,
            "input": text,
        }
        response = _post_json(
            url=f"{self.config.base_url}/embeddings",
            payload=payload,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )
        try:
            return list(response["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatError(f"Unexpected embedding response: {response}") from exc

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.config.model_name,
            "input": texts,
        }
        response = _post_json(
            url=f"{self.config.base_url}/embeddings",
            payload=payload,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )
        try:
            items = list(response["data"])
            embeddings: list[list[float] | None] = [None] * len(texts)
            for item in items:
                index = int(item["index"])
                embeddings[index] = list(item["embedding"])
            if any(embedding is None for embedding in embeddings):
                raise KeyError("missing_embedding")
            return [embedding for embedding in embeddings if embedding is not None]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise OpenAICompatError(f"Unexpected embedding response: {response}") from exc


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    if repair_json is not None:
        for start, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                repaired = repair_json(stripped[start:], skip_json_loads=True)
                obj = json.loads(repaired)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"Could not extract JSON object from text: {text[:200]}")


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if repair_json is None:
            raise
        repaired = repair_json(text, skip_json_loads=True)
        return json.loads(repaired)
