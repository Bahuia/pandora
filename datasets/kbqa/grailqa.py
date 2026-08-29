"""
GrailQA Dataset - KBQA Task

Each GrailQA question has a pre-computed subgraph stored as CSV tables:
  - box/test/<qid>/*.csv    (CSV tables)
  - box/box_schema.json     (box_schema strings per qid)
  - entity_link/...         (entity linking info)
  - box/test/<qid>/foreign_key.json (FK constraints per qid)
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from datasets.base import BaseDataset
from utils.file_utils import load_json
from utils.logger import setup_logger


class GrailQADataset(BaseDataset):
    """
    GrailQA dataset for KBQA tasks.

    Preprocess: Load KG subgraph schema + CSV table content + entity linking
    Postprocess: Format entity results as list of lists
    Evaluate: Set-based EM/F1 comparison of Freebase IDs
    """

    def __init__(self, data_root: str = "./data"):
        super().__init__(name="grailqa", data_root=data_root)
        self.logger = setup_logger("pandora.dataset.grailqa")
        self.kg_path = Path(data_root) / "grailqa" / "box"
        self.schema_path = self.kg_path / "box_schema.json"

        # Load box_schema cache
        self._box_schema_cache = {}
        if self.schema_path.exists():
            self._box_schema_cache = load_json(str(self.schema_path))
            self.logger.info(f"Loaded box_schema for {len(self._box_schema_cache)} questions")

        # Load entity linking cache
        self._entity_link_cache = {}
        self._load_entity_link_cache()

    def _load_entity_link_cache(self):
        """Pre-load entity linking info."""
        el_path = Path(self.data_root) / "grailqa" / "entity_link" / "grailqa.entity_link.test.json"
        if el_path.exists():
            try:
                with open(el_path, encoding="utf-8") as f:
                    self._entity_link_cache = json.load(f)
                self.logger.info(f"Loaded entity links for {len(self._entity_link_cache)} questions")
            except Exception as e:
                self.logger.warning(f"Failed to load entity link cache: {e}")

    def load_examples(self, stage: str = "test", qids: Optional[list] = None) -> list:
        """Load GrailQA examples."""
        data_file = Path(self.data_root) / "grailqa" / f"grailqa.{stage}.json"
        if not data_file.exists():
            data_file = Path(self.data_root) / "grailqa" / f"{stage}.json"

        if not data_file.exists():
            raise FileNotFoundError(f"GrailQA {stage} data not found at {data_file}")

        with open(data_file, encoding="utf-8") as f:
            examples = json.load(f)

        self.logger.info(f"Loaded {len(examples)} GrailQA {stage} examples")

        if qids:
            qid_set = set(str(q) for q in qids)
            examples = [ex for ex in examples if str(ex.get("qid", "")) in qid_set]
            self.logger.info(f"Filtered to {len(examples)} examples by qid")

        return examples

    def get_box_schema(self, qid: str) -> str:
        """Get box_schema string for a specific question."""
        return self._box_schema_cache.get(str(qid), "")

    def get_entity_links(self, qid: str) -> dict:
        """Get entity linking hints for a specific question."""
        return self._entity_link_cache.get(str(qid), {})

    def get_foreign_keys(self, qid: str) -> str:
        """Load foreign keys from the qid directory."""
        fk_path = self.kg_path / "test" / str(qid) / "foreign_key.json"
        if fk_path.exists():
            try:
                with open(fk_path) as f:
                    fks = json.load(f)
                lines = []
                for fk_pair in fks:
                    if len(fk_pair) >= 2:
                        src_parts = fk_pair[0].rsplit("-", 1)
                        tgt_parts = fk_pair[1].rsplit("-", 1)
                        if len(src_parts) == 2 and len(tgt_parts) == 2:
                            lines.append(
                                f"FOREIGN KEY {src_parts[0]}['{src_parts[1]}'] "
                                f"REFERENCES {tgt_parts[0]}['{tgt_parts[1]}']"
                            )
                return "\n".join(lines)
            except Exception as e:
                self.logger.warning(f"Failed to load FK for {qid}: {e}")
        return ""

    def _extract_primary_keys(self, box_schema: str) -> str:
        """
        Extract primary keys from box_schema.

        In KBQA, the first column of each table (same name as the table)
        is the primary key (the subject entity in the KG edge).

        Returns:
            Formatted string like:
            TABLE `table_name`: (col_name)
        """
        lines = []
        current_table = None
        for line in box_schema.split("\n"):
            table_match = re.match(r"^(\w+)\s*=\s*pd\.DataFrame\(\{", line.strip())
            if table_match:
                current_table = table_match.group(1)
                continue
            col_match = re.match(r'^\s*"([^"]+)"\s*:\s*\[\]', line.strip())
            if col_match and current_table:
                # The first column is the PK
                pk_col = col_match.group(1)
                lines.append(f"TABLE `{current_table}`: ({pk_col})")
                current_table = None  # Only capture the first column
        return "\n".join(lines)

    def _filter_empty_tables_from_schema(self, box_schema: str, csv_to_schema_map: dict) -> str:
        """
        Filter box_schema to only include tables that have non-empty CSV data.

        Removes table blocks from the schema where the corresponding CSV is empty
        or not found.

        Args:
            box_schema: Full box_schema string
            csv_to_schema_map: Mapping from CSV file path to schema table name

        Returns:
            Filtered box_schema with empty tables removed
        """
        if not csv_to_schema_map:
            return ""

        # Get set of schema table names that have data
        valid_tables = set(csv_to_schema_map.values())

        lines = box_schema.split('\n')
        result_lines = []
        current_table = None
        current_block = []
        brace_depth = 0

        for line in lines:
            stripped = line.strip()
            table_match = re.match(r'^(\w+)\s*=\s*pd\.DataFrame\(\{', stripped)
            if table_match:
                # Save previous block if it's a valid table
                if current_table and current_table.lower() in {t.lower() for t in valid_tables}:
                    result_lines.extend(current_block)

                current_table = table_match.group(1)
                current_block = [line]
                brace_depth = line.count('{') - line.count('}')
                continue

            if current_block:
                current_block.append(line)
                brace_depth += line.count('{') - line.count('}')

                # Check if we've closed the DataFrame
                if brace_depth == 0 and current_table:
                    # This block is complete
                    if current_table.lower() in {t.lower() for t in valid_tables}:
                        result_lines.extend(current_block)
                    current_table = None
                    current_block = []

        # Handle last block if not properly closed
        if current_block and current_table:
            if current_table.lower() in {t.lower() for t in valid_tables}:
                result_lines.extend(current_block)

        return '\n'.join(result_lines)

    def _match_csv_to_schema_tables(self, qid_dir: Path, box_schema: str) -> dict:
        """Match CSV files to box_schema table names by filename.

        CSV filename format: "namespace.table_name.csv"
        Table name is the part after the last dot: "namespace.table_name" → "table_name"

        Only includes non-empty CSVs (row count > 0).

        Returns:
            {csv_file_path: schema_table_name}
        """
        # Extract table names from box_schema
        schema_table_names = set()
        for line in box_schema.split("\n"):
            table_match = re.match(r"^(\w+)\s*=\s*pd\.DataFrame\(\{", line.strip())
            if table_match:
                schema_table_names.add(table_match.group(1).lower())

        if not qid_dir.exists():
            return {}

        csv_to_schema_map = {}

        for csv_file in qid_dir.glob("*.csv"):
            # Extract table name from filename: "computer.software_genre.csv" → "software_genre"
            stem = csv_file.stem  # "computer.software_genre"
            table_name = stem.rsplit(".", 1)[-1]  # "software_genre"

            # Check if this table name exists in the schema
            if table_name.lower() in schema_table_names:
                # Verify the CSV has data rows
                try:
                    row_count = sum(1 for _ in open(csv_file)) - 1  # Subtract header
                    if row_count > 0:
                        csv_to_schema_map[str(csv_file)] = table_name
                except Exception:
                    pass

        return csv_to_schema_map

    def _build_table_content(self, qid_dir: Path, csv_to_schema_map: dict, entity_ids: Optional[list] = None) -> str:
        """Build table_content string from CSV files using schema table names.

        If entity_ids are provided, ensures rows containing those IDs are included
        in the sample, so the LLM can see which columns contain mentioned entities.
        """
        table_content_parts = []
        if not qid_dir.exists():
            return ""

        entity_ids = entity_ids or []
        entity_id_set = set(eid for eid in entity_ids if eid and eid.strip())

        # Group CSV files by their mapped schema table name
        for csv_path_str, schema_table_name in csv_to_schema_map.items():
            csv_path = Path(csv_path_str)
            try:
                df = pd.read_csv(csv_path, dtype=str)
                if df.empty:
                    # Skip empty tables — don't include in table_content
                    continue
                sample_rows = df.head(3)

                # If entity IDs are provided, find rows that contain them
                entity_matched_rows = []
                entity_matched_cols = {}  # col_name -> set of matched entity IDs

                if entity_id_set:
                    for col in df.columns:
                        col_values = set(df[col].dropna().unique())
                        matched = col_values & entity_id_set
                        if matched:
                            entity_matched_cols[col] = matched
                            # Get rows where this column contains the matched entity ID
                            for eid in matched:
                                matching = df[df[col] == eid]
                                entity_matched_rows.append(matching.head(2))

                if entity_matched_cols:
                    # Build enriched sample: head(3) + entity-matched rows
                    matched_df = pd.concat(entity_matched_rows, ignore_index=True) if entity_matched_rows else pd.DataFrame()
                    combined = pd.concat([sample_rows, matched_df], ignore_index=True).drop_duplicates()

                    # Add annotation about which columns contain entity IDs
                    col_notes = []
                    for col, matched_ids in entity_matched_cols.items():
                        ids_str = ', '.join(f'`{eid}`' for eid in sorted(matched_ids))
                        col_notes.append(f"  ⭐ Column '{col}' contains mentioned entity ID(s): {ids_str}")

                    sample_text = combined.head(6).to_string(index=False)
                    note_text = "\n".join(col_notes)
                    table_content_parts.append(
                        f"TABLE `{schema_table_name}` ({len(df)} rows):\n{sample_text}\n{note_text}"
                    )
                else:
                    sample_text = sample_rows.to_string(index=False)
                    table_content_parts.append(
                        f"TABLE `{schema_table_name}` ({len(df)} rows):\n{sample_text}"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to read {csv_path}: {e}")

        return "\n\n".join(table_content_parts)

    def _filter_entity_links(self, example: dict, entity_links: dict) -> dict:
        """
        Filter entity_links to only include entities that appear in the example's
        schema field.

        Schema format: "entity_name: entity_id | entity_name: entity_id | relation1 relation2 ..."
        The part before '|' contains the entities that should be used in the query.

        This prevents the schema linking prompt from showing irrelevant entities
        that may confuse the LLM.

        Args:
            example: The raw example dict (contains 'schema' field)
            entity_links: Full entity linking dict from entity_link cache

        Returns:
            Filtered entity_links dict containing only entities from schema.
        """
        schema_field = example.get("schema", "")
        if not schema_field or not schema_field.strip():
            return entity_links

        # Parse schema field to extract entity names and IDs
        parts = schema_field.split(" | ", 1)
        entity_part = parts[0].strip()

        if not entity_part:
            # No entities in schema, return empty dict
            return {}

        # Parse "name1: id1 name2: id2" format
        # Entity names can contain colons and spaces (e.g., "king arthur: the role-playing wargame")
        # IDs are Freebase-style: m.XXXXX, g.XXXXX, cvt.XXXXX
        # Strategy: find all Freebase IDs and take the text before each as the entity name
        import re
        schema_entities = {}
        # Match Freebase IDs followed by space or end of string
        id_pattern = re.compile(r'((?:cvt\.|m\.|g\.)\S+)(\s|$)')
        remaining = entity_part
        while remaining:
            m = id_pattern.search(remaining)
            if not m:
                break
            eid = m.group(1)
            # Text before the ID is the entity name (strip trailing whitespace and colons)
            name = remaining[:m.start()].strip().rstrip(':').strip()
            if name:
                schema_entities[eid] = name
            remaining = remaining[m.end():]

        if not schema_entities:
            return {}

        # Filter entity_links: keep only entities whose ID appears in schema
        filtered = {}
        for entity_name, entity_id in entity_links.items():
            if entity_id in schema_entities:
                # Use the entity name from schema (may differ slightly from entity_link)
                filtered[schema_entities[entity_id]] = entity_id

        # Log if filtering removed entities
        removed_count = len(entity_links) - len(filtered)
        if removed_count > 0:
            removed_names = [n for n, eid in entity_links.items() if eid not in schema_entities]
            self.logger.debug(
                f"qid={example.get('qid')}: filtered {removed_count} entity(ies) "
                f"not in schema: {removed_names}"
            )

        return filtered

    def preprocess(self, example: dict) -> dict:
        """
        Preprocess GrailQA example into standardized format.

        Returns:
            {
                "question": str,
                "schema": dict,           # box_schema, table_content, foreign_keys
                "context": dict,          # qid, kg_dir, entity_links, csv_to_schema_map
                "hints": list[str],       # entity linking hints as strings
                "example_id": str,
                "evidence": str,
            }
        """
        qid = str(example.get("qid", "unknown"))
        question = example.get("question", "")
        evidence = example.get("evidence", "")

        # Get box_schema
        box_schema = self.get_box_schema(qid)
        if not box_schema:
            self.logger.warning(f"No box_schema found for qid={qid}")

        # Get KG directory
        qid_dir = self.kg_path / "test" / qid

        # Match CSV files to schema table names — skip empty CSVs
        csv_to_schema_map = self._match_csv_to_schema_tables(qid_dir, box_schema)

        # Filter box_schema to only include tables that have non-empty CSV data
        box_schema = self._filter_empty_tables_from_schema(box_schema, csv_to_schema_map)

        # Get entity linking hints — needed BEFORE _build_table_content for enriched sampling
        entity_links = self.get_entity_links(qid)

        # Filter entity_links to only include entities that appear in struct_in
        entity_links = self._filter_entity_links(example, entity_links)

        # Build table content using matched CSVs — pass entity IDs for enriched sampling
        table_content = self._build_table_content(
            qid_dir, csv_to_schema_map, entity_ids=list(entity_links.values())
        )

        # Load foreign keys
        foreign_keys = self.get_foreign_keys(qid)

        # Generate primary keys from box_schema (after filtering empty tables)
        # In KBQA, the first column of each table (same name as the table) is the primary key
        primary_keys = self._extract_primary_keys(box_schema)

        hints = []
        for entity_name, entity_id in entity_links.items():
            hints.append(f'Entity "{entity_name}" → Freebase ID: `{entity_id}`')

        # Build schema dict
        schema = {
            "box_schema": box_schema,
            "table_content": table_content,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "tables": {},
            "kb_type": "kg",
        }

        # Build context — executor needs kg_dir + csv_to_schema_map
        context = {
            "qid": qid,
            "kg_dir": str(qid_dir),
            "entity_links": entity_links,
            "kb_type": "kg",
            "csv_to_schema_map": csv_to_schema_map,
            "box_schema": box_schema,  # For executor reference
        }

        return {
            "question": question,
            "schema": schema,
            "context": context,
            "hints": hints,
            "example_id": qid,
            "evidence": evidence,
        }

    def postprocess(self, exec_result: dict, processed: dict) -> dict:
        """
        Postprocess KG query execution result.

        For KBQA:
        1. Normalize to list of lists
        2. Remove rows where all values are null-like (None/NaN)
        3. Remove duplicate rows
        """
        if not exec_result.get("success", False):
            return {"answer": [], "formatted": f"Execution failed: {exec_result.get('error', '')}"}

        result = exec_result.get("result", [])

        if not result:
            return {"answer": [], "formatted": "[]"}

        # Normalize to list of lists
        if isinstance(result, list):
            answer = [list(row) if isinstance(row, (list, tuple)) else [row] for row in result]
        else:
            answer = [[result]] if result is not None else []

        # Remove rows where all values are null-like
        def _is_null(v):
            if v is None:
                return True
            if isinstance(v, float):
                import math
                return math.isnan(v)
            return False

        filtered = []
        for row in answer:
            if all(_is_null(v) for v in row):
                continue
            filtered.append(row)

        # Remove duplicate rows (preserve order)
        seen = set()
        deduped = []
        for row in filtered:
            key = tuple(v if not _is_null(v) else None for v in row)
            if key not in seen:
                seen.add(key)
                deduped.append(row)

        return {
            "answer": deduped,
            "formatted": str(deduped),
        }

    def evaluate(self, predicted: list, gold: list) -> dict[str, Any]:
        """
        Evaluate KBQA results using set-based comparison.

        GrailQA answers are Freebase IDs or entity values — order doesn't matter.
        """
        if not predicted and not gold:
            return {"em": 1.0, "f1": 1.0, "hit_1": 1.0, "correct": True}

        if not predicted or not gold:
            return {"em": 0.0, "f1": 0.0, "hit_1": 0.0, "correct": False}

        # Flatten and normalize
        pred_set = set()
        for row in predicted:
            if isinstance(row, (list, tuple)):
                for val in row:
                    pred_set.add(self._normalize_value(val))
            else:
                pred_set.add(self._normalize_value(row))

        gold_set = set()
        for row in gold:
            if isinstance(row, (list, tuple)):
                for val in row:
                    gold_set.add(self._normalize_value(val))
            else:
                gold_set.add(self._normalize_value(row))

        # Remove empty strings
        pred_set.discard("")
        gold_set.discard("")

        # Exact Match
        em = 1.0 if pred_set == gold_set else 0.0

        # F1 Score
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        first_prediction = None
        if predicted:
            first_row = predicted[0]
            first_prediction = first_row[0] if isinstance(first_row, (list, tuple)) else first_row
        hit_1 = 1.0 if self._normalize_value(first_prediction) in gold_set else 0.0

        return {
            "em": em,
            "f1": f1,
            "hit_1": hit_1,
            "correct": em == 1.0,
        }

    @staticmethod
    def _normalize_value(val) -> str:
        """Normalize a value for comparison."""
        import numpy as np

        if val is None:
            return ""
        if isinstance(val, (np.integer,)):
            return str(int(val))
        if isinstance(val, (np.floating,)):
            v = float(val)
            if v == int(v):
                return str(int(v))
            return str(round(v, 10))
        s = str(val).strip()
        # Normalize numbers in strings
        try:
            num = float(s)
            if num == int(num):
                return str(int(num))
            return str(round(num, 10))
        except (ValueError, OverflowError):
            pass
        # Lowercase for case-insensitive comparison
        return s.lower()

    def get_gold_answer(self, example: dict) -> list:
        """
        Extract gold answer from GrailQA example.

        Uses answer_argument (Freebase ID) as the primary gold standard.
        Returns list of lists: [[answer1], [answer2], ...]
        """
        ans = example.get("answer", {})
        if isinstance(ans, dict):
            args = ans.get("answer_argument", [])
            if args:
                return [[a] for a in args]

            # Fallback to entity_name
            names = ans.get("entity_name", [])
            if names and any(n.strip() for n in names):
                return [[n] for n in names if n.strip()]
        return []
