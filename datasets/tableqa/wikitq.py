"""
WikiTableQuestions Dataset - TableQA Task

Each example has an embedded table (header + rows) in the JSON.
The agent generates Python Pandas code to query these tables.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from datasets.base import BaseDataset
from utils.file_utils import load_json
from utils.logger import setup_logger


class WikiTQDataset(BaseDataset):
    """
    WikiTableQuestions dataset for TableQA tasks.

    Tables are embedded in the JSON (not separate CSV files).
    Each table has a 'header' (column names) and 'rows' (list of lists).
    """

    def __init__(self, data_root: str = "./data"):
        super().__init__(name="wikitq", data_root=data_root)
        self.logger = setup_logger("pandora.dataset.wikitq")
        self.csv_path = Path(data_root) / "wikitq" / "csv"

        # Cache for loaded DataFrames
        self._table_cache = {}

    def _load_csv(self, table_id: str) -> pd.DataFrame:
        """
        Load table from CSV file.

        table_id format: "csv/203-csv/733.csv"
        csv_path: data/wikitq/csv/
        Full path: data/wikitq/csv/203-csv/733.csv
        """
        # table_id is relative to wikitq root, e.g., "csv/203-csv/733.csv"
        # csv_path is already data/wikitq/csv/
        # So we need the part after "csv/" from table_id
        clean_id = table_id
        if clean_id.startswith("csv/"):
            clean_id = clean_id[4:]  # Remove "csv/" prefix
        elif clean_id.startswith("csv\\"):
            clean_id = clean_id[4:]

        csv_file = self.csv_path / clean_id
        if not csv_file.exists():
            self.logger.warning(f"CSV file not found: {csv_file}")
            return pd.DataFrame()

        try:
            # Use python engine for better error handling
            df = pd.read_csv(
                csv_file,
                encoding='utf-8',
                on_bad_lines='skip',
                engine='python',
            )
            # Clean column names
            df.columns = [c.strip().replace("\n", " ").replace("\\n", " ") for c in df.columns]
            return df
        except pd.errors.ParserError as e:
            self.logger.warning(f"CSV parse error for {csv_file}: {e}")
            # Fallback: try reading with different options
            try:
                df = pd.read_csv(
                    csv_file,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    error_bad_lines=False,
                    warn_bad_lines=True,
                )
                df.columns = [c.strip().replace("\n", " ").replace("\\n", " ") for c in df.columns]
                return df
            except Exception as e2:
                self.logger.warning(f"Failed to load CSV with fallback: {e2}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"Failed to load CSV {csv_file}: {e}")
            return pd.DataFrame()

    def load_examples(self, stage: str = "test", qids: Optional[list] = None) -> list:
        """Load WikiTQ examples."""
        data_file = Path(self.data_root) / "wikitq" / f"wikitq.{stage}.json"
        if not data_file.exists():
            data_file = Path(self.data_root) / "wikitq" / f"{stage}.json"

        if not data_file.exists():
            raise FileNotFoundError(f"WikiTQ {stage} data not found at {data_file}")

        with open(data_file, encoding="utf-8") as f:
            examples = json.load(f)

        self.logger.info(f"Loaded {len(examples)} WikiTQ {stage} examples")

        if qids:
            qid_set = set(str(q) for q in qids)
            examples = [ex for ex in examples if str(ex.get("id", "")) in qid_set]
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
        """
        Build a box_schema string from a DataFrame.

        Format matches the style used by nl2sql and kbqa:
        table = pd.DataFrame({
            "col1": [],  # (dtype), description
        })
        """
        lines = [f'{table_name} = pd.DataFrame({{']
        for col in df.columns:
            dtype = str(df[col].dtype)
            # Get sample values for description
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
        Preprocess WikiTQ example.

        Loads the table from CSV file and builds schema information.

        Returns:
            {
                "question": str,
                "schema": dict,
                "context": dict,
                "hints": list[str],
                "example_id": str,
                "evidence": str,
            }
        """
        qid = str(example.get("id", "unknown"))
        question = example.get("question", "")
        table_id = example.get("table_id", "")

        # Load table from CSV
        if table_id:
            df = self._load_csv(table_id)
        else:
            # Fallback: build from embedded table
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
        """
        Postprocess table query execution result.

        WikiTQ questions typically ask for a single value.
        If the result has multiple columns per row, extract only the first column.
        """
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
                    # WikiTQ typically asks for a single value — take only the first column
                    # This handles cases where LLM incorrectly returns all columns
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

        WikiTQ answers are typically single values. Each predicted/gold answer
        is compared as a set of single values (first element of each row).

        Comparison is case-insensitive (both predicted and gold are lowercased).
        """
        if not predicted and not gold:
            return {"em": 1.0, "denotation_accuracy": 1.0, "f1": 1.0, "correct": True}

        if not predicted or not gold:
            return {"em": 0.0, "denotation_accuracy": 0.0, "f1": 0.0, "correct": False}

        # For WikiTQ: each row should be a single value.
        # If the LLM returned multiple columns, take only the first column.
        # Gold answers are already lowercased in get_gold_answer().
        # _normalize_value() lowercases predicted values for case-insensitive comparison.
        pred_set = set()
        for row in predicted:
            if isinstance(row, (list, tuple)):
                # Take only the first element (single-value answer)
                val = row[0] if row else ""
                pred_set.add(self._normalize_value(val))
            else:
                pred_set.add(self._normalize_value(row))

        gold_set = set()
        for row in gold:
            if isinstance(row, (list, tuple)):
                # Take only the first element (single-value answer)
                val = row[0] if row else ""
                gold_set.add(self._normalize_value(val))
            else:
                gold_set.add(self._normalize_value(row))

        # Remove empty strings
        pred_set.discard("")
        gold_set.discard("")

        # Exact Match: sets must be equal
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
            "denotation_accuracy": em,
            "f1": f1,
            "correct": em == 1.0,
        }

    def get_gold_answer(self, example: dict) -> list:
        """
        Extract gold answer from WikiTQ example.

        WikiTQ gold answers are in 'answer_text' (list of strings) or 'label'.
        """
        # Try answer_text first
        ans = example.get("answer_text", [])
        if isinstance(ans, list) and ans:
            return [[str(a).strip().lower()] for a in ans if str(a).strip()]

        # Try label
        label = example.get("label", "")
        if isinstance(label, str) and label.strip():
            return [[label.strip().lower()]]

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
        # Try to normalize numbers
        try:
            cleaned = s.replace(",", "").replace("$", "").replace("%", "")
            num = float(cleaned)
            if num == int(num):
                return str(int(num))
            return str(round(num, 10))
        except (ValueError, OverflowError):
            pass
        return s
