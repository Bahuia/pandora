"""Manifest-driven heterogeneous source evaluation for Pandora."""

from __future__ import annotations

import json
import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from datasets.base import BaseDataset


class CrossSourceDataset(BaseDataset):
    """Load questions that require two or more BOX-compatible data sources.

    A manifest record contains ``sources`` entries of kind ``db``, ``table``,
    or ``kg``. Optional prefixes make table names collision-free.  The format
    is documented in ``data/cross_source/README.md``.
    """

    def __init__(self, data_root: str = "./data"):
        super().__init__(name="cross_source", data_root=data_root)
        self.root = Path(data_root) / "cross_source"

    def load_examples(self, stage: str = "test", qids: Optional[list] = None) -> list[dict]:
        path = self.root / f"cross_source.{stage}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Cross-source manifest not found at {path}. See data/cross_source/README.md."
            )
        with path.open(encoding="utf-8") as handle:
            examples = json.load(handle)
        if qids:
            selected = {str(qid) for qid in qids}
            examples = [example for example in examples if str(example.get("id")) in selected]
        return examples

    def preprocess(self, example: dict) -> dict:
        sources = [self._resolve_source(source) for source in example.get("sources", [])]
        if len(sources) < 2:
            raise ValueError("A cross-source example must declare at least two sources")

        schema_blocks: list[str] = []
        previews: list[str] = []
        foreign_keys: list[str] = []
        profiles: list[tuple[int, str, str, set[str]]] = []
        for source_index, source in enumerate(sources):
            blocks, source_previews, source_fks, source_profiles = self._inspect_source(
                source, source_index
            )
            schema_blocks.extend(blocks)
            previews.extend(source_previews)
            foreign_keys.extend(source_fks)
            profiles.extend(source_profiles)

        foreign_keys.extend(self._discover_cross_source_links(profiles))

        for link in example.get("links", []):
            foreign_keys.append(
                "CROSS SOURCE LINK "
                f"{link['left']} <-> {link['right']} "
                f"USING {link.get('method', 'normalized_surface')}"
            )

        return {
            "question": example.get("question", ""),
            "db_id": str(example.get("id", "unknown")),
            "example_id": str(example.get("id", "unknown")),
            "evidence": example.get("evidence", ""),
            "schema": {
                "box_schema": "\n\n".join(schema_blocks),
                "table_content": "\n\n".join(previews),
                "primary_keys": "",
                "foreign_keys": "\n".join(foreign_keys),
                "tables": {},
                "kb_type": "multi",
            },
            "context": {
                "kb_type": "multi",
                "sources": sources,
            },
        }

    def postprocess(self, exec_result: dict, processed: dict) -> dict:
        if not exec_result or not exec_result.get("success", False):
            return {"answer": [], "formatted": ""}
        answer = exec_result.get("result", [])
        return {"answer": answer, "formatted": str(answer)}

    def get_gold_answer(self, example: dict) -> list:
        answer = example.get("gold_answer", [])
        return [row if isinstance(row, list) else [row] for row in answer]

    def evaluate(self, predicted: list, gold: list) -> dict[str, Any]:
        normalize = lambda value: str(value).strip().casefold()
        pred_set = {
            tuple(normalize(value) for value in (row if isinstance(row, (list, tuple)) else [row]))
            for row in predicted
        }
        gold_set = {
            tuple(normalize(value) for value in (row if isinstance(row, (list, tuple)) else [row]))
            for row in gold
        }
        if not pred_set and not gold_set:
            return {"em": 1.0, "f1": 1.0, "correct": True}
        true_positive = len(pred_set & gold_set)
        precision = true_positive / len(pred_set) if pred_set else 0.0
        recall = true_positive / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        em = 1.0 if pred_set == gold_set else 0.0
        return {"em": em, "f1": f1, "correct": bool(em)}

    def _resolve_source(self, source: dict) -> dict:
        resolved = dict(source)
        path_key = "kg_dir" if source.get("kind") == "kg" else "path"
        raw_path = source.get(path_key, "")
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(self.data_root).parent / path
        resolved[path_key] = str(path.resolve())
        return resolved

    @staticmethod
    def _table_name(prefix: str, name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    def _inspect_source(
        self, source: dict, source_index: int
    ) -> tuple[list[str], list[str], list[str], list[tuple[int, str, str, set[str]]]]:
        kind = source.get("kind")
        prefix = source.get("prefix", "")
        frames: dict[str, pd.DataFrame] = {}
        foreign_keys: list[str] = []

        if kind == "db":
            connection = sqlite3.connect(source["path"])
            names = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )]
            for name in names:
                frames[self._table_name(prefix, name)] = pd.read_sql_query(
                    f'SELECT * FROM "{name}" LIMIT 50', connection
                )
                for row in connection.execute(f'PRAGMA foreign_key_list("{name}")'):
                    foreign_keys.append(
                        f"FOREIGN KEY {self._table_name(prefix, name)}['{row[3]}'] "
                        f"REFERENCES {self._table_name(prefix, row[2])}['{row[4]}']"
                    )
            connection.close()
        elif kind == "table":
            name = self._table_name(prefix, source.get("table_name", Path(source["path"]).stem))
            frames[name] = pd.read_csv(source["path"], nrows=50)
        elif kind == "kg":
            for path in sorted(Path(source["kg_dir"]).glob("*.csv")):
                name = self._table_name(prefix, path.stem.replace(".", "_"))
                frames[name] = pd.read_csv(path, dtype=str, nrows=50)
        else:
            raise ValueError(f"Unknown source kind: {kind}")

        schemas: list[str] = []
        previews: list[str] = []
        profiles: list[tuple[int, str, str, set[str]]] = []
        for name, frame in frames.items():
            lines = [f"{name} = pd.DataFrame({{"]
            for column in frame.columns:
                lines.append(f'    "{column}": [],  # ({frame[column].dtype})')
            lines.append("})")
            schemas.append("\n".join(lines))
            previews.append(
                f"TABLE `{name}` ({len(frame)} sampled rows):\n"
                f"{frame.head(3).to_string(index=False)}"
            )
            for column in frame.columns:
                values = {
                    self._normalize_surface(value)
                    for value in frame[column].dropna().astype(str).head(50)
                }
                values = {
                    value for value in values
                    if len(value) >= 3 and any(character.isalpha() for character in value)
                }
                if values:
                    profiles.append((source_index, name, str(column), values))
        return schemas, previews, foreign_keys, profiles

    @staticmethod
    def _normalize_surface(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()

    @classmethod
    def _discover_cross_source_links(
        cls, profiles: list[tuple[int, str, str, set[str]]]
    ) -> list[str]:
        """Infer high-confidence links from normalized entity surface forms."""
        links: list[str] = []
        for left_index, left in enumerate(profiles):
            for right in profiles[left_index + 1:]:
                if left[0] == right[0]:
                    continue
                exact = left[3] & right[3]
                method = "normalized_exact"
                evidence = sorted(exact)[:3]
                if not evidence:
                    fuzzy = next((
                        (left_value, right_value)
                        for left_value in sorted(left[3])
                        for right_value in sorted(right[3])
                        if min(len(left_value), len(right_value)) >= 4
                        and SequenceMatcher(None, left_value, right_value).ratio() >= 0.92
                    ), None)
                    if not fuzzy:
                        continue
                    method = "normalized_fuzzy"
                    evidence = [f"{fuzzy[0]} ~= {fuzzy[1]}"]
                links.append(
                    f"CROSS SOURCE LINK {left[1]}.{left[2]} <-> "
                    f"{right[1]}.{right[2]} USING {method}; evidence={evidence}"
                )
        return links
