"""BIRD Dataset - NL2SQL Task (Simplified)"""

import re
import json
from pathlib import Path
from typing import Any, Optional

from datasets.base import BaseDataset
from utils.file_utils import load_json
from utils.logger import setup_logger

logger = setup_logger("pandora.datasets.bird")


class BirdDataset(BaseDataset):
    """Simplified BIRD dataset - schema loading moved to PandoraAgent._prepare_db_info()."""

    def __init__(self, data_root: str = "./data"):
        super().__init__(name="bird", data_root=data_root)
        self.data_dir = Path(data_root) / "bird"

    def load_examples(self, stage: str, qids: Optional[list] = None) -> list:
        """Load BIRD examples."""
        self.stage = stage
        candidates = [
            self.data_dir / f"bird.{stage}.json",
            self.data_dir / f"{stage}.json",
        ]
        dev_file = next((path for path in candidates if path.exists()), None)
        if dev_file is None:
            raise FileNotFoundError(
                f"BIRD {stage} data not found; searched: {candidates}"
            )

        logger.info(f"Loading BIRD from {dev_file}")
        with open(dev_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)

        if qids:
            qid_set = {str(qid) for qid in qids}
            examples = [
                example for example in examples
                if str(example.get("question_id", example.get("id", ""))) in qid_set
            ]

        # Evidence is already embedded in the JSON file (bird.dev.json / dev_california.json)
        # No need to load from a separate evidence file
        with_evidence = sum(1 for ex in examples if ex.get('evidence'))
        logger.info(f"Loaded {len(examples)} BIRD examples ({with_evidence} with evidence)")
        return examples

    def preprocess(self, example: dict) -> dict:
        """Minimal preprocess - schema loading handled by PandoraAgent."""
        return {
            "question": example.get('question', ''),
            "evidence": example.get('evidence', ''),
            "db_id": example.get('db_id', 'california_schools'),
        }

    def postprocess(self, exec_result: dict, processed: dict) -> dict:
        """Postprocess execution result."""
        result = exec_result.result if hasattr(exec_result, 'result') else exec_result.get('result', [])
        return {"answer": result, "formatted": str(result)}

    def _normalize_value(self, val: Any) -> Any:
        """
        Normalize a single value for comparison.

        - Keep None as None (for [[null]] comparison)
        - Convert NaN to None
        - Convert numpy types to native Python types
        - Normalize numbers: convert float and int to a canonical form
          (e.g., 507.0 → 507, 3.10 → 3.1) so that int/float comparisons work
        - Normalize date/datetime formats:
          '2011-02-15 15:22:25.0' → '2011-02-15 15:22:25'
          '2010-07-19 19:16:14.0' → '2010-07-19 19:16:14'
          '1983-02-07T00:00:00' → '1983-02-07'
        """
        import numpy as np
        import math

        # Keep None as None (was previously converted to "" which caused issues
        # with frozenset comparison for [[null]] answers)
        if val is None:
            return None

        # Handle NaN floats (from NaN-producing operations like mean() on empty set)
        if isinstance(val, float) and math.isnan(val):
            return None

        # Convert numpy types to native Python types FIRST
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, (np.integer,)):
            val = int(val)
        if isinstance(val, (np.floating,)):
            if math.isnan(val):
                return None
            val = float(val)

        # Handle numbers: normalize float vs int equivalence
        if isinstance(val, (int, float)):
            # If it's a float with .0 suffix, treat as int
            if isinstance(val, float) and val == int(val) and not (val == 0.0 and str(val).startswith('-')):
                return int(val)
            # Otherwise normalize float to consistent precision
            if isinstance(val, float):
                return round(val, 10)
            return val

        # String: strip whitespace
        val_str = str(val).strip()

        # Date/datetime normalization
        val_str = self._normalize_datetime_string(val_str)

        # Also check if the string represents a whole number (e.g., "507.0")
        try:
            num = float(val_str)
            if num == int(num):
                return int(num)
            return round(num, 10)
        except (ValueError, OverflowError):
            pass

        return val_str

    @staticmethod
    def _normalize_datetime_string(s: str) -> str:
        """
        Normalize date/datetime string formats for comparison.

        - '2011-02-15 15:22:25.0' → '2011-02-15 15:22:25'  (trailing .0 on seconds)
        - '2010-07-19T19:16:14.0' → '2010-07-19 19:16:14'  (T separator + trailing .0)
        - '1983-02-07T00:00:00' → '1983-02-07'             (midnight → date only)
        - '2010-07-19 19:16:14' → '2010-07-19 19:16:14'    (already normalized)
        """
        # Pattern: date with T separator and time with trailing .0
        # e.g., '2010-07-19T19:16:14.0'
        m = re.match(
            r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})\.0$', s
        )
        if m:
            date_part, time_part = m.group(1), m.group(2)
            if time_part == '00:00:00':
                return date_part
            return f'{date_part} {time_part}'

        # Pattern: date with space separator and time with trailing .0
        # e.g., '2011-02-15 15:22:25.0'
        m = re.match(
            r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\.0$', s
        )
        if m:
            date_part, time_part = m.group(1), m.group(2)
            if time_part == '00:00:00':
                return date_part
            return f'{date_part} {time_part}'

        # Pattern: date with T separator and midnight time
        # e.g., '1983-02-07T00:00:00'
        m = re.match(
            r'^(\d{4}-\d{2}-\d{2})T00:00:00$', s
        )
        if m:
            return m.group(1)

        # Pattern: date with midnight time (space separator)
        # e.g., '1983-02-07 00:00:00'
        m = re.match(
            r'^(\d{4}-\d{2}-\d{2})\s+00:00:00$', s
        )
        if m:
            return m.group(1)

        return s

    def _normalize_row(self, row: tuple) -> frozenset:
        """Normalize a row into an unordered frozenset for comparison.

        Column order within a row is ignored — ('A', 'B') == ('B', 'A').
        None values are preserved as None (not converted to empty string)
        so that [[null]] can match correctly.
        """
        normalized_values = []
        for v in row:
            nv = self._normalize_value(v)
            # Keep None as None in the frozenset (don't convert to "")
            # This ensures [[None]] produces frozenset({None}), not frozenset()
            normalized_values.append(nv)
        return frozenset(normalized_values)

    def evaluate(self, predicted: list, gold: list) -> dict:
        """
        Evaluate prediction against gold using execution-based comparison.

        For BIRD dataset, we compare the execution results, not the SQL strings.
        Uses set-based comparison where:
        - Row order doesn't matter (set comparison)
        - Column order within each row doesn't matter (frozenset per row)
        - Values are normalized (float/int equivalence, datetime formats, etc.)
        """
        # Handle empty results
        if not predicted and not gold:
            return {"em": 1.0, "exec_acc": 1.0, "f1": 1.0, "correct": True}

        if not predicted or not gold:
            return {"em": 0.0, "exec_acc": 0.0, "f1": 0.0, "correct": False}

        # Normalize and convert to sets of frozensets (unordered rows)
        try:
            pred_set = set()
            for row in predicted:
                if not row:
                    continue
                # Normalize each value; keep None as None (don't filter it out)
                normalized_values = [self._normalize_value(x) for x in row]
                if normalized_values:
                    pred_set.add(frozenset(normalized_values))

            gold_set = set()
            for row in gold:
                if not row:
                    continue
                normalized_values = [self._normalize_value(x) for x in row]
                if normalized_values:
                    gold_set.add(frozenset(normalized_values))

        except Exception as e:
            logger.warning(f"Error normalizing results: {e}")
            return {"em": 0.0, "exec_acc": 0.0, "f1": 0.0, "correct": False}

        # Exact Match (set-based, row order and column order both don't matter)
        em = 1.0 if pred_set == gold_set else 0.0

        # F1 Score
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
