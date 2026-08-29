"""Task-agnostic semantic retrieval over Pandora's verified memory.

The paper uses ``BAAI/bge-large-en-v1.5`` to embed questions and retrieves the
top-K examples by cosine similarity without restricting the source task.  This
module implements that method directly and persists embeddings so repeated
experiments do not re-encode the complete memory.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import numpy as np


class TextEncoder(Protocol):
    def encode(self, sentences, **kwargs): ...


@dataclass(frozen=True)
class MemoryExample:
    dataset_id: str
    example_id: str
    question: str
    schema: str
    reasoning: str
    code: str
    source: str


def _extract_response(response: Any) -> tuple[str, str]:
    if isinstance(response, dict):
        return str(response.get("reasoning", "")), str(response.get("code", ""))

    text = str(response or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        parsed = json.loads(candidate)
        return str(parsed.get("reasoning", "")), str(parsed.get("code", ""))
    except (json.JSONDecodeError, AttributeError):
        return "", text


def load_verified_memory(paths: Iterable[Path]) -> list[MemoryExample]:
    """Load only examples that passed the stored execution comparison."""
    examples: list[MemoryExample] = []
    seen: set[tuple[str, str]] = set()

    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            verification = record.get("exec result", {})
            if verification and verification.get("res comp") is False:
                continue
            dataset_id = str(record.get("dataset_id", "unknown")).lower()
            example_id = str(record.get("id", ""))
            key = (dataset_id, example_id)
            if key in seen:
                continue
            question = str(record.get("question", "")).strip()
            if not question:
                continue
            reasoning, code = _extract_response(record.get("response", ""))
            if not code:
                continue
            seen.add(key)
            examples.append(MemoryExample(
                dataset_id=dataset_id,
                example_id=example_id,
                question=question,
                schema=str(record.get("schema", "")),
                reasoning=reasoning,
                code=code,
                source=str(path),
            ))
    return examples


class SemanticMemoryRetriever:
    """BGE-based cosine retriever with a deterministic on-disk index."""

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        examples: list[MemoryExample],
        model_name: str = "BAAI/bge-large-en-v1.5",
        cache_path: Optional[Path] = None,
        encoder: Optional[TextEncoder] = None,
    ):
        self.examples = examples
        self.model_name = model_name
        self.cache_path = cache_path
        self._encoder = encoder
        self._embeddings: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def _get_encoder(self) -> TextEncoder:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic retrieval requires sentence-transformers. "
                    "Install the project requirements before using shot_k > 0."
                ) from exc
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _fingerprint(self) -> str:
        digest = hashlib.sha256(self.model_name.encode())
        for example in self.examples:
            digest.update(example.dataset_id.encode())
            digest.update(b"\0")
            digest.update(example.example_id.encode())
            digest.update(b"\0")
            digest.update(example.question.encode())
        return digest.hexdigest()

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def _load_or_build_embeddings(self) -> np.ndarray:
        if self._embeddings is not None:
            return self._embeddings
        with self._lock:
            if self._embeddings is not None:
                return self._embeddings
            fingerprint = self._fingerprint()
            if self.cache_path and self.cache_path.exists():
                cached = np.load(self.cache_path, allow_pickle=False)
                if str(cached["fingerprint"].item()) == fingerprint:
                    self._embeddings = cached["embeddings"]
                    return self._embeddings

            encoder = self._get_encoder()
            embeddings = encoder.encode(
                [example.question for example in self.examples],
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            self._embeddings = self._normalize(embeddings)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    self.cache_path,
                    fingerprint=np.array(fingerprint),
                    embeddings=self._embeddings,
                )
            return self._embeddings

    def retrieve(
        self,
        question: str,
        top_k: int,
        target_dataset: str = "",
        mode: str = "cross_task",
    ) -> list[tuple[MemoryExample, float]]:
        if top_k <= 0 or not self.examples or mode == "disabled":
            return []
        embeddings = self._load_or_build_embeddings()
        query = self._get_encoder().encode(
            [self.QUERY_INSTRUCTION + question],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_vector = self._normalize(query)[0]
        scores = embeddings @ query_vector

        candidates = np.arange(len(self.examples))
        if mode == "same_dataset":
            target = target_dataset.lower()
            candidates = np.array([
                index for index, example in enumerate(self.examples)
                if example.dataset_id == target
            ], dtype=int)
        if not len(candidates):
            return []

        ranked = candidates[np.argsort(scores[candidates])[::-1][:top_k]]
        return [(self.examples[index], float(scores[index])) for index in ranked]


def format_demonstrations(retrieved: list[tuple[MemoryExample, float]]) -> str:
    """Render empty BOX schemas and verified code, excluding field values."""
    if not retrieved:
        return ""
    parts = ["## Retrieved Cross-Task Demonstrations"]
    for index, (example, score) in enumerate(retrieved, 1):
        parts.extend([
            f"### Demonstration {index} ({example.dataset_id}, similarity={score:.4f})",
            f"Question: {example.question}",
            "Empty BOX schema:",
            "```python",
            example.schema,
            "```",
            f"Reasoning: {example.reasoning}",
            "Verified Pandas code:",
            "```python",
            example.code,
            "```",
        ])
    return "\n".join(parts)
