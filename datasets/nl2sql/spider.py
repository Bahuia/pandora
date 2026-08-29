"""
Spider Dataset - NL2SQL Task

Implementation of the Spider dataset for text-to-SQL tasks.
"""

from pathlib import Path
from typing import Any, Optional
import json


from datasets.base import BaseDataset
from utils.file_utils import load_json
from utils.logger import setup_logger


class SpiderDataset(BaseDataset):
    """
    Spider dataset for NL2SQL tasks.

    Preprocess: Load database schema + foreign keys
    Postprocess: Format SQL execution results
    Evaluate: Compare predicted vs gold SQL results
    """

    def __init__(self, data_root: str = "./data"):
        super().__init__(name="spider", data_root=data_root)
        self.logger = setup_logger("pandora.dataset.spider")
        self.db_path = Path(data_root) / "spider" / "dev_database"
        self.schema_path = Path(data_root) / "spider" / "dev_tables.json"

    def load_examples(self, stage: str, qids: Optional[list] = None) -> list[dict]:
        """Load Spider examples for a specific stage."""
        self.stage = stage
        root = Path(self.data_root) / "spider"
        candidates = [root / f"spider.{stage}.json", root / f"{stage}.json"]
        data_file = next((path for path in candidates if path.exists()), None)

        if data_file is None:
            raise FileNotFoundError(f"Spider {stage} data not found; searched: {candidates}")

        examples = load_json(data_file)
        if qids:
            qid_set = {str(qid) for qid in qids}
            examples = [
                example for example in examples
                if str(example.get("question_id", example.get("id", ""))) in qid_set
            ]
        self.logger.info(f"Loaded {len(examples)} Spider {stage} examples")

        return examples

    def preprocess(self, example: dict) -> dict:
        """
        Preprocess Spider example.

        Returns:
            {
                "question": str,
                "schema": dict,
                "context": dict,
                "hints": list[str],
                "example_id": str,
            }
        """
        db_id = example.get("db_id", "unknown")

        # Load database schema
        schema = self._load_schema(db_id)

        # Build context
        context = {
            "db_id": db_id,
            "db_path": str(self.db_path / db_id / f"{db_id}.sqlite"),
        }

        # Get hints/evidence
        hints = example.get("evidence", [])
        if isinstance(hints, str):
            hints = [hints]

        return {
            "question": example.get("question", ""),
            "schema": schema,
            "context": context,
            "hints": hints,
            "example_id": example.get("question_id", db_id),
        }

    def postprocess(self, exec_result: dict, processed: dict) -> dict:
        """
        Postprocess SQL execution result.

        Returns:
            {
                "answer": list[list],
                "formatted": str,
            }
        """
        if not exec_result["success"]:
            return {"answer": [], "formatted": f"Execution failed: {exec_result.get('error')}"}

        rows = exec_result.get("result", [])

        # Normalize result format — also clean numpy types for JSON-friendly output
        import numpy as np

        if isinstance(rows, list):
            raw = [list(row) if isinstance(row, (list, tuple)) else [row] for row in rows]
        else:
            raw = [[rows]] if rows is not None else []

        # Clean numpy types to native Python types
        answer = []
        for row in raw:
            clean_row = []
            for v in row:
                if isinstance(v, (np.integer,)):
                    clean_row.append(int(v))
                elif isinstance(v, (np.floating,)):
                    clean_row.append(float(v))
                elif isinstance(v, np.ndarray):
                    clean_row.append(v.tolist())
                else:
                    clean_row.append(v)
            answer.append(clean_row)

        return {
            "answer": answer,
            "formatted": str(answer),
        }

    def _normalize_value(self, val: Any) -> Any:
        """
        Normalize a single value for comparison.

        - Convert None/null to empty string
        - Convert numpy types to native Python types
        - Normalize numbers: int/float equivalence (309445 == 309445.0)
        - Strip whitespace from strings
        """
        import math
        import numpy as np

        if val is None:
            return ""

        # Convert numpy types to native Python types FIRST
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, (np.integer,)):
            val = int(val)
        elif isinstance(val, (np.floating,)):
            val = float(val)

        # Normalize numbers
        if isinstance(val, (int, float)):
            if isinstance(val, float):
                # NaN check
                if math.isnan(val):
                    return ""
                # If it's a whole number, treat as int for comparison
                if val == int(val) and abs(val) < 1e15:
                    return int(val)
                return round(val, 10)
            return val

        # String: strip whitespace
        val_str = str(val).strip()

        # Try to parse as number for cross-type comparison
        try:
            num = float(val_str)
            if num == int(num) and abs(num) < 1e15:
                return int(num)
            return round(num, 10)
        except (ValueError, OverflowError):
            pass

        return val_str

    def evaluate(self, predicted: list, gold: list) -> dict[str, Any]:
        """
        Evaluate SQL execution results.

        Uses set-based comparison (order doesn't matter).
        Values are normalized so that int/float equivalence works:
        309445 == 309445.0
        """
        if not predicted and not gold:
            return {"em": 1.0, "exec_acc": 1.0, "f1": 1.0, "correct": True}

        if not predicted or not gold:
            return {"em": 0.0, "exec_acc": 0.0, "f1": 0.0, "correct": False}

        # Normalize and convert to sets of tuples
        pred_set = set(
            tuple(self._normalize_value(x) for x in row)
            for row in predicted
        )
        gold_set = set(
            tuple(self._normalize_value(x) for x in row)
            for row in gold
        )

        # Exact match
        em = 1.0 if pred_set == gold_set else 0.0

        # F1 score
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "em": em,
            "exec_acc": em,
            "f1": f1,
            "correct": em == 1.0,
        }

    def _load_schema(self, db_id: str) -> dict:
        """Load database schema for a specific db_id."""
        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Spider schema file not found: {self.schema_path}. "
                "See DATASETS.md for the official download and expected layout."
            )

        with self.schema_path.open(encoding="utf-8") as handle:
            schemas = json.load(handle)
        schema = next((item for item in schemas if item.get("db_id") == db_id), None)
        if schema is None:
            raise KeyError(f"Database {db_id!r} is absent from {self.schema_path}")

        table_names = schema.get("table_names_original") or schema.get("table_names", [])
        columns = schema.get("column_names_original") or schema.get("column_names", [])
        column_types = schema.get("column_types", [])
        tables: dict[str, list[dict[str, str]]] = {name: [] for name in table_names}
        for index, column in enumerate(columns):
            table_index, column_name = column
            if table_index < 0:  # Spider's synthetic '*' column.
                continue
            tables[table_names[table_index]].append({
                "name": column_name,
                "type": column_types[index] if index < len(column_types) else "text",
            })

        def resolve_column(column_index: int) -> dict[str, str]:
            table_index, column_name = columns[column_index]
            return {"table": table_names[table_index], "column": column_name}

        return {
            "db_id": db_id,
            "tables": tables,
            "primary_keys": [resolve_column(index) for index in schema.get("primary_keys", [])],
            "foreign_keys": [
                {"from": resolve_column(source), "to": resolve_column(target)}
                for source, target in schema.get("foreign_keys", [])
            ],
        }
