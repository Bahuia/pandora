"""Paper-aligned KG-to-BOX conversion.

The builder performs bidirectional H-hop traversal from topic entities over a
question-relevant relation set, groups subjects by entity type (including CVT
nodes), and materializes sparse relation tables as CSV BOXes.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


TYPE_RELATIONS = {"type", "isa", "/type/object/type", "type.object.type"}


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str


def sanitize_identifier(value: str) -> str:
    value = value.strip().replace("/", ".").strip(".")
    value = re.sub(r"[^A-Za-z0-9_.]+", "_", value)
    value = value.replace(".", "_")
    if not value or value[0].isdigit():
        value = f"box_{value}"
    return value


def load_triples(path: Path) -> list[Triple]:
    """Load JSONL objects or three-column TSV/CSV triples."""
    triples: list[Triple] = []
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open(encoding="utf-8") as handle:
            content = handle.read().strip()
        records = json.loads(content) if content.startswith("[") else [
            json.loads(line) for line in content.splitlines() if line.strip()
        ]
        for record in records:
            triples.append(Triple(
                str(record.get("subject", record.get("s", ""))),
                str(record.get("predicate", record.get("p", ""))),
                str(record.get("object", record.get("o", ""))),
            ))
        return triples

    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if len(row) >= 3:
                triples.append(Triple(str(row[0]), str(row[1]), str(row[2])))
    return triples


class KGBoxBuilder:
    def __init__(self, triples: Iterable[Triple]):
        self.triples = list(triples)
        self.outgoing: dict[str, list[Triple]] = defaultdict(list)
        self.incoming: dict[str, list[Triple]] = defaultdict(list)
        self.entity_types: dict[str, set[str]] = defaultdict(set)
        for triple in self.triples:
            predicate_key = triple.predicate.casefold()
            if predicate_key in TYPE_RELATIONS:
                self.entity_types[triple.subject].add(triple.object)
                continue
            self.outgoing[triple.subject].append(triple)
            self.incoming[triple.object].append(triple)

    def select_relations(
        self,
        question: str,
        top_k: int,
        model_name: str = "BAAI/bge-large-en-v1.5",
        encoder=None,
    ) -> set[str]:
        relations = sorted({triple.predicate for triple in self.triples
                            if triple.predicate.casefold() not in TYPE_RELATIONS})
        if top_k <= 0 or top_k >= len(relations):
            return set(relations)
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Relation selection requires sentence-transformers") from exc
            encoder = SentenceTransformer(model_name)
        vectors = np.asarray(encoder.encode(
            [question] + [relation.replace(".", " ").replace("/", " ") for relation in relations],
            convert_to_numpy=True,
            show_progress_bar=False,
        ), dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        scores = vectors[1:] @ vectors[0]
        selected = np.argsort(scores)[::-1][:top_k]
        return {relations[index] for index in selected}

    def extract_subgraph(
        self, topic_entities: Iterable[str], relations: set[str], max_hops: int
    ) -> list[Triple]:
        """Run the paper's bidirectional, path-pruned depth-first search."""
        selected: set[Triple] = set()

        def visit(entity: str, depth: int, path: frozenset[str]) -> None:
            if depth >= max_hops:
                return
            for triple in self.outgoing.get(entity, []):
                if triple.predicate not in relations:
                    continue
                selected.add(triple)
                if triple.object not in path:
                    visit(triple.object, depth + 1, path | {triple.object})
            for triple in self.incoming.get(entity, []):
                if triple.predicate not in relations:
                    continue
                selected.add(triple)
                if triple.subject not in path:
                    visit(triple.subject, depth + 1, path | {triple.subject})

        for topic_entity in topic_entities:
            visit(topic_entity, 0, frozenset({topic_entity}))
        return sorted(selected, key=lambda item: (item.subject, item.predicate, item.object))

    def materialize(
        self, triples: Iterable[Triple], output_dir: Path
    ) -> tuple[str, list[list[str]]]:
        """Write one sparse BOX per entity type and return schema plus foreign keys."""
        output_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[Triple]] = defaultdict(list)
        for triple in triples:
            types = self.entity_types.get(triple.subject) or {"entity"}
            for entity_type in types:
                grouped[entity_type].append(triple)

        frames: dict[str, pd.DataFrame] = {}
        value_sets: dict[tuple[str, str], set[str]] = {}
        schema_blocks: list[str] = []
        for entity_type, box_triples in sorted(grouped.items()):
            table_name = sanitize_identifier(entity_type)
            subject_column = sanitize_identifier(entity_type.split(".")[-1].split("/")[-1])
            relation_columns = sorted({sanitize_identifier(item.predicate) for item in box_triples})
            rows: list[dict[str, Optional[str]]] = []
            for triple in box_triples:
                row: dict[str, Optional[str]] = {column: None for column in relation_columns}
                row[subject_column] = triple.subject
                row[sanitize_identifier(triple.predicate)] = triple.object
                rows.append(row)
            frame = pd.DataFrame(rows, columns=[subject_column] + relation_columns)
            frame = frame.drop_duplicates().reset_index(drop=True)
            frames[table_name] = frame
            frame.to_csv(output_dir / f"{table_name}.csv", index=False)

            lines = [f"{table_name} = pd.DataFrame({{"]
            for column in frame.columns:
                lines.append(f'    "{column}": [],  # (str)')
                value_sets[(table_name, column)] = set(frame[column].dropna().astype(str))
            lines.append("})")
            schema_blocks.append("\n".join(lines))

        foreign_keys: list[list[str]] = []
        columns = sorted(value_sets)
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1:]:
                if left[0] == right[0]:
                    continue
                if value_sets[left] & value_sets[right]:
                    foreign_keys.append([f"{left[0]}-{left[1]}", f"{right[0]}-{right[1]}"])
        with (output_dir / "foreign_key.json").open("w", encoding="utf-8") as handle:
            json.dump(foreign_keys, handle, ensure_ascii=False, indent=2)
        return "\n\n".join(schema_blocks), foreign_keys
