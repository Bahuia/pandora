"""
Answer Post-processing Rules

Generic, rule-based post-processing for normalized comparison
between predicted answers and gold answers.

All rules are deterministic — no LLM calls needed.

Rules applied (in order):
0. Convert numpy types to native Python types (np.float64 → float, np.int64 → int, np.bool_ → bool)
1. Strip T00:00:00 from date-only datetime strings
2. Strip trailing .0 from time strings (HH:MM:SS.0 → HH:MM:SS)
3. Filter out rows where all values are None
4. Drop all-None columns from the entire answer
5. Normalize float↔int equivalence (88.0 → 88)
"""

import re
import numpy as np
from typing import Any, List


def normalize_wikitq_value(val: Any) -> Any:
    """
    Normalize a single WikiTQ answer value.

    - Converts numpy types to native types.
    - Normalizes floats to ints for whole numbers (11.0 → 11).
    - Normalizes numeric strings ('11.0' → '11').
    """
    if val is None:
        return None

    # Convert numpy types
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        val = float(val)

    # Normalize floats to ints for whole numbers
    if isinstance(val, float):
        # val == val checks for NaN
        if val == int(val) and val == val:
            return int(val)
        return val

    # Normalize numeric strings
    if isinstance(val, str):
        try:
            num = float(val)
            # Check for NaN and whole number
            if num == int(num) and num == num:
                return str(int(num))
            return val
        except (ValueError, OverflowError):
            return val

    return val


def postprocess_wikitq_answer(answer: Any) -> Any:
    """
    WikiTQ-specific answer post-processing.

    WikiTQ answers should be in the format [[val1], [val2], ...].
    This function ensures the answer is properly formatted without
    applying the generic postprocess_answer which can incorrectly
    split string answers into characters.

    Rules:
    1. If answer is a single value (not a list), wrap it: [[value]]
    2. If answer is a flat list of values, wrap each: [[v1], [v2], ...]
    3. If answer is already [[v1], [v2], ...], keep as is
    4. Normalize float to int for whole numbers (11.0 → 11)
    5. Normalize numeric strings ('11.0' → '11')
    """
    if answer is None:
        return []

    if not isinstance(answer, list):
        # Single value: 11.0 → [[11.0]]
        return [[normalize_wikitq_value(answer)]]

    if not answer:
        return []

    # Check if it's already in [[v1], [v2], ...] format
    if all(isinstance(row, (list, tuple)) for row in answer):
        # Already properly formatted — normalize values
        normalized = []
        for row in answer:
            norm_row = [normalize_wikitq_value(v) for v in row]
            normalized.append(norm_row)
        return normalized

    # Flat list: [v1, v2, ...] → [[v1], [v2], ...]
    return [[normalize_wikitq_value(v)] for v in answer]


def postprocess_answer(answer: Any) -> Any:
    """
    Normalize an answer for fair comparison with gold answers.

    Applied rules:
    1. **Datetime normalization**: Remove midnight time component from
       date-only strings (e.g., '1983-02-07T00:00:00' → '1983-02-07').
    2. **Time precision normalization**: Remove trailing .0 from time
       strings (e.g., '20:18:28.0' → '20:18:28').
    3. **None row filtering**: Remove rows where every value is None,
       which can occur when Pandas fails to translate SQL IS NOT NULL.
    4. **All-None column removal**: If a column position is None across
       ALL rows, drop that column entirely. This happens when the agent
       selects extra columns that happen to be all-null.
    5. **Float→Int normalization**: Convert float values like 88.0 to
       int 88, and strip trailing .0 from numeric strings like '507.0'.

    Args:
        answer: Raw answer from code execution (typically list of lists).

    Returns:
        Normalized answer with consistent formatting.
    """
    if not answer:
        return answer

    # ── Per-row normalization (Rules 0, 1, 2, 3) ──
    normalized = []
    for row in answer:
        if not isinstance(row, (list, tuple)):
            normalized.append(row)
            continue

        # Rule 3: Normalize null-like values to None but KEEP the row
        # (Previously these rows were dropped, causing false negatives when
        #  the gold answer is [[null]] — e.g., AVG() over empty set returns NULL)
        if all(_is_null_like(v) for v in row):
            # Keep as [None] instead of dropping
            normalized.append([None])
            continue

        new_row = []
        for val in row:
            # Skip null-like values (None, NaN)
            if _is_null_like(val):
                new_row.append(None)
                continue

            # Rule 0: Convert numpy types to native Python types
            # This ensures formatted_answer and downstream comparison work correctly.
            # np.float64(375.0) → 375.0, np.int64(0) → 0, np.True_ → True
            if isinstance(val, (np.bool_,)):
                new_row.append(bool(val))
                continue
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)

            if isinstance(val, str):
                # Rule 1: Strip T00:00:00 from date-only strings
                val = re.sub(r'T00:00:00$', '', val)
                # Rule 2: Strip trailing .0 from time strings
                val = re.sub(r'(\d{2}:\d{2}:\d{2})\.0$', r'\1', val)
                # Rule 5: Normalize numeric strings (e.g., '507.0' → '507')
                val = _normalize_numeric_string(val)
            # Rule 5: Float→Int normalization
            elif isinstance(val, float) and val == int(val) and not (_is_negative_zero(val)):
                val = int(val)
            new_row.append(val)

        # Rule 3 (re-check): After normalization, if row is all-None, keep it
        # (same reasoning as above — [[None]] is a valid result, not an error)
        if not new_row or all(v is None for v in new_row):
            normalized.append([None])
            continue

        normalized.append(new_row)

    if not normalized:
        return normalized

    # ── Cross-row normalization (Rule 4: drop all-None columns) ──
    normalized = _drop_all_none_columns(normalized)

    return normalized


def _normalize_numeric_string(s: str) -> str:
    """
    If a string represents a number, normalize it:
    - '507.0' → '507' (whole number)
    - '3.10' → '3.1' (remove trailing zeros)
    - Leave non-numeric strings unchanged.
    """
    try:
        num = float(s)
        if num == int(num) and not (num == 0.0 and s.startswith('-')):
            return str(int(num))
        # Normalize float: remove unnecessary trailing zeros
        return f'{num:g}'
    except (ValueError, OverflowError):
        return s


def _is_negative_zero(val: float) -> bool:
    """Check if float is negative zero (-0.0)."""
    return val == 0.0 and str(val).startswith('-')


def _is_null_like(val: Any) -> bool:
    """Check if a value is None or NaN (null-like)."""
    if val is None:
        return True
    if isinstance(val, float) and val != val:  # NaN check
        return True
    return False


def _drop_all_none_columns(rows: list[list]) -> list[list]:
    """
    Rule 4: Remove columns (positions) where ALL rows have None.

    This handles the case where the agent selects extra columns that
    happen to be all-null (e.g., AdmFName2/3 when only AdmFName1 has data).

    Exception: If ALL rows are entirely None (i.e., every row is [None]),
    do NOT drop columns — this represents a valid [[null]] result, not
    extra columns to be removed.

    Example:
        Input:  [['A', 'B', None, None], ['C', 'D', None, None]]
        Output: [['A', 'B'], ['C', 'D']]

        Input:  [[None], [None]]
        Output: [[None], [None]]  (not [[]] — preserve [[null]] semantics)
    """
    if not rows or not rows[0]:
        return rows

    # Check if ALL rows are entirely None — if so, don't drop anything
    all_rows_null = all(
        isinstance(row, (list, tuple)) and all(v is None for v in row)
        for row in rows
    )
    if all_rows_null:
        return rows

    num_cols = max(len(row) for row in rows)

    # Find column indices where ALL rows have None
    cols_to_drop = set()
    for col_idx in range(num_cols):
        all_none = True
        for row in rows:
            if col_idx < len(row) and row[col_idx] is not None:
                all_none = False
                break
        if all_none:
            cols_to_drop.add(col_idx)

    if not cols_to_drop:
        return rows

    # Remove the all-None columns from each row
    result = []
    for row in rows:
        new_row = [
            val for idx, val in enumerate(row)
            if idx not in cols_to_drop
        ]
        result.append(new_row)

    return result
