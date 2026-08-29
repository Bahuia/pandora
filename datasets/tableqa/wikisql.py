"""
WikiSQL Dataset - TableQA Task

Each example has an embedded table (header + rows) in the JSON.
The agent generates Python Pandas code to query these tables.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from datasets.base import BaseDataset
from utils.logger import setup_logger


class WikiSQLDataset(BaseDataset):
    """
    WikiSQL dataset for TableQA tasks.

    Tables are embedded in the JSON (header + rows).
    Each example also has a table_id pointing to CSV files.
    """

    def __init__(self, data_root: str = "./data"):
        super().__init__(name="wikisql", data_root=data_root)
        self.logger = setup_logger("pandora.dataset.wikisql")
        self.csv_path = Path(data_root) / "wikisql" / "csv"

        # Cache for loaded DataFrames
        self._table_cache = {}

    def _load_csv(self, table_id: str) -> pd.DataFrame:
        """
        Load table from CSV file.

        table_id format: e.g., "1-10015132-16" or path within csv structure
        """
        # Try different possible paths
        possible_paths = [
            self.csv_path / f"{table_id}.csv",
        ]

        # Search in subdirectories if not found directly
        if not any(p.exists() for p in possible_paths):
            for csv_file in self.csv_path.rglob(f"{table_id}.csv"):
                possible_paths.append(csv_file)
                break

        for path in possible_paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    df.columns = [c.strip().replace("\n", " ").replace("\\n", " ") for c in df.columns]
                    return df
                except Exception as e:
                    self.logger.warning(f"Failed to load CSV {path}: {e}")

        return pd.DataFrame()

    def load_examples(self, stage: str = "test", qids: Optional[list] = None) -> list:
        """Load WikiSQL examples."""
        data_file = Path(self.data_root) / "wikisql" / f"wikisql.{stage}.json"
        if not data_file.exists():
            data_file = Path(self.data_root) / "wikisql" / f"{stage}.json"

        if not data_file.exists():
            raise FileNotFoundError(f"WikiSQL {stage} data not found at {data_file}")

        with open(data_file, encoding="utf-8") as f:
            examples = json.load(f)

        self.logger.info(f"Loaded {len(examples)} WikiSQL {stage} examples")

        if qids:
            qid_set = set(str(q) for q in qids)
            examples = [ex for ex in examples if str(ex.get("id", ex.get("table", {}).get("id", ""))) in qid_set]
            self.logger.info(f"Filtered to {len(examples)} examples by id")

        return examples

    def _build_dataframe(self, example: dict) -> pd.DataFrame:
        """
        Build a DataFrame from the embedded table in the example.

        Table format: {"header": [...], "rows": [[...], [...]]}
        """
        table = example.get("table", {})
        if not table:
            return pd.DataFrame()

        header = table.get("header", [])
        rows = table.get("rows", [])

        if not header or not rows:
            return pd.DataFrame()

        # Clean column names
        clean_header = [h.replace("\\n", " ").strip() for h in header]

        df = pd.DataFrame(rows, columns=clean_header)
        return df

    def _build_box_schema(self, df: pd.DataFrame, table_name: str = "table") -> str:
        """Build a box_schema string from a DataFrame."""
        lines = [f'{table_name} = pd.DataFrame({{']
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(v) for v in sample[:3])
            if sample_str:
                desc = f"({dtype}), sample: {sample_str}"
            else:
                desc = f"({dtype})"
            lines.append(f'    "{col}": [],  # {desc}')
        lines.append("})")
        return "\n".join(lines)

    def _build_table_content(self, df: pd.DataFrame, table_name: str = "table") -> str:
        """Build table content string (first 5 rows)."""
        if df.empty:
            return ""
        preview = df.head(5).to_string(index=False)
        return f"TABLE `{table_name}` ({len(df)} rows):\n{preview}"

    def preprocess(self, example: dict) -> dict:
        """
        Preprocess WikiSQL example.
        """
        table = example.get("table", {})
        table_id = table.get("id", str(example.get("id", "unknown")))
        qid = str(example.get("id", table_id))
        question = example.get("question", "")

        # Try to load from CSV first
        df = self._load_csv(table_id)
        if df.empty:
            # Fallback to embedded table
            df = self._build_dataframe(example)

        # Cache the DataFrame
        self._table_cache[qid] = df

        # Build schema
        table_name = "table"
        box_schema = self._build_box_schema(df, table_name)
        table_content = self._build_table_content(df, table_name)

        # Build primary keys — first column is the PK
        first_col = df.columns[0] if len(df.columns) > 0 else "unknown"
        primary_keys = f"TABLE `{table_name}`: ({first_col})"

        schema = {
            "box_schema": box_schema,
            "table_content": table_content,
            "primary_keys": primary_keys,
            "foreign_keys": "",
            "tables": {table_name: df},
            "kb_type": "table",
        }

        context = {
            "qid": qid,
            "table_id": table_id,
            "table_df": df,
            "kb_type": "table",
        }

        return {
            "question": question,
            "schema": schema,
            "context": context,
            "hints": [],
            "example_id": qid,
            "evidence": "",
        }

    def postprocess(self, exec_result: dict, processed: dict) -> dict:
        """Postprocess table query execution result."""
        if not exec_result.get("success", False):
            return {
                "answer": [],
                "formatted": f"Execution failed: {exec_result.get('error', '')}",
            }

        result = exec_result.get("result", [])

        if not result:
            return {"answer": [], "formatted": "[]"}

        # Normalize to list of lists
        if isinstance(result, list):
            answer = []
            for row in result:
                if isinstance(row, (list, tuple)):
                    answer.append([row[0]])
                else:
                    answer.append([row])
        else:
            answer = [[result]] if result is not None else []

        return {
            "answer": answer,
            "formatted": str(answer),
        }

    def evaluate(self, predicted: list, gold: list) -> dict[str, Any]:
        """
        Evaluate TableQA results using set-based comparison.
        """
        if not predicted and not gold:
            return {"em": 1.0, "denotation_accuracy": 1.0, "f1": 1.0, "correct": True}

        if not predicted or not gold:
            return {"em": 0.0, "denotation_accuracy": 0.0, "f1": 0.0, "correct": False}

        pred_set = set()
        for row in predicted:
            if isinstance(row, (list, tuple)):
                val = row[0] if row else ""
                pred_set.add(self._normalize_value(val))
            else:
                pred_set.add(self._normalize_value(row))

        gold_set = set()
        for row in gold:
            if isinstance(row, (list, tuple)):
                val = row[0] if row else ""
                gold_set.add(self._normalize_value(val))
            else:
                gold_set.add(self._normalize_value(row))

        pred_set.discard("")
        gold_set.discard("")

        em = 1.0 if pred_set == gold_set else 0.0

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "em": em,
            "denotation_accuracy": em,
            "f1": f1,
            "correct": em == 1.0,
        }

    def get_gold_answer(self, example: dict) -> list:
        """
        Extract gold answer from WikiSQL example.

        WikiSQL gold answers are in the 'answer' or 'sql' field.
        """
        # Try answer_text first (preprocessed format)
        ans = example.get("answer_text", [])
        if isinstance(ans, list) and ans:
            return [[str(a).strip().lower()] for a in ans if str(a).strip()]

        # Try answer field
        ans = example.get("answer", [])
        if isinstance(ans, list) and ans:
            return [[str(a).strip().lower()] for a in ans if str(a).strip()]

        return []

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
        s = str(val).strip().lower()
        try:
            cleaned = s.replace(",", "").replace("$", "").replace("%", "")
            num = float(cleaned)
            if num == int(num):
                return str(int(num))
            return str(round(num, 10))
        except (ValueError, OverflowError):
            pass
        return s
