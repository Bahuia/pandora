"""
ValueCorrector — 过滤值验证与修正

在代码融合后、执行前，用 DB 实际数据验证并修正 Pandas 过滤值。

修复类型：
- Type A: 拼写错误 (TR00_1_2 → TR000_1_2)
- Type B: 自然语言 vs 代码格式 (Czech koruna → CZK)
- Type C: 值不完整/多余 (Engineering → Engineering Department)
- Type D: 精确匹配 vs 模糊匹配 (== → str.contains)
- Type E: 单字符错误 (LAM → KAM)

纯规则驱动，无需 LLM。
"""

from utils.logger import setup_logger

logger = setup_logger("pandora.value_corrector")

import re
import sqlite3
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Optional
from pathlib import Path


def correct_filter_values(data: dict, db_path: str) -> dict:
    """
    Validate all filter values in merged_code against actual DB values.
    Auto-correct mismatches using fuzzy matching and pattern analysis.

    Args:
        data: Agent data dict containing merged_code, schema_and_value_linking, accumulated_code, etc.
        db_path: Path to the SQLite database file.

    Returns:
        Updated data dict with corrected merged_code.
    """
    if not db_path or not Path(db_path).exists():
        return data

    merged_code = data.get("merged_code", "")
    if not merged_code or not merged_code.strip():
        return data

    clean_code = _extract_pure_code(merged_code)
    if not clean_code.strip():
        return data

    conn = sqlite3.connect(db_path)
    corrections = []

    # Collect execution state hints from accumulated_code (contains actual DB values)
    accumulated_code = data.get("accumulated_code", "")
    exec_hints = _extract_all_sample_values(accumulated_code)

    # ── Strategy 1: Validate df['col'] == 'value' patterns ──
    corrections.extend(
        _validate_equality_filters(clean_code, conn, data, exec_hints=exec_hints)
    )

    # ── Strategy 2: Validate df['col'].str.contains('value') patterns ──
    corrections.extend(
        _validate_contains_filters(clean_code, conn, data, exec_hints=exec_hints)
    )

    # ── Strategy 3: Validate df['col'] != 'value' patterns ──
    corrections.extend(
        _validate_neq_filters(clean_code, conn, data, exec_hints=exec_hints)
    )

    # ── Strategy 4: Validate df['col'].isin([...]) patterns ──
    corrections.extend(
        _validate_isin_filters(clean_code, conn, data, exec_hints=exec_hints)
    )

    # ── Strategy 5: Check Value Linking extracted values that may be missing ──
    corrections.extend(
        _check_missing_value_linking(clean_code, conn, data)
    )

    conn.close()


    if not corrections:
        logger.debug("No filter corrections needed")
        return data

    # Filter out non-value-correction findings before cross-validation
    value_corrections = [c for c in corrections if "old" in c and "new" in c]
    non_value_findings = [c for c in corrections if "old" not in c or "new" not in c]

    # Cross-validate corrections: when multiple values share a common prefix pattern
    # (e.g., both 'TR00_1_2' and 'TR00_1' are missing a digit), prefer corrections
    # that map to the SAME parent entity (e.g., same molecule_id).
    if len(value_corrections) >= 2:
        value_corrections = _cross_validate_corrections(value_corrections, db_path, clean_code)

    corrections = value_corrections + non_value_findings

    # Apply all corrections to the code
    corrected_code = clean_code
    applied = set()  # Track applied corrections to avoid duplicates


    for correction in corrections:
        # Skip non-value-correction findings (e.g., missing_value_linking)
        if "old" not in correction or "new" not in correction:
            continue

        old_str = correction["old"]
        new_str = correction["new"]

        # Skip if this exact replacement was already applied
        key = (old_str, new_str)
        if key in applied:
            continue

        if old_str in corrected_code:
            corrected_code = corrected_code.replace(old_str, new_str, 1)
            applied.add(key)
            correction["applied"] = True
        else:
            correction["applied"] = False


    # Update the data
    data["merged_code"] = corrected_code
    data["value_corrections"] = [c for c in corrections if c.get("applied")]
    data["value_findings"] = corrections
    logger.info(f"ValueCorrector: applied {len(data['value_corrections'])} correction(s), {len(data['value_findings'])} total finding(s)")

    return data


def _extract_pure_code(merged_code: str) -> str:
    """Extract pure Python code from potentially markdown-wrapped merged_code."""
    # Remove markdown code blocks
    code = re.sub(r"```python\s*", "", merged_code)
    code = re.sub(r"```\s*", "", code)
    return code.strip()


def _validate_equality_filters(code: str, conn: sqlite3.Connection, data: dict, exec_hints: dict = None) -> list:
    """
    Strategy 1: Find df['col'] == 'value' and validate 'value' against DB.

    Fixes:
    - Type A: Spelling errors → fuzzy match finds closest DB value
    - Type B: NL vs code format → semantic similarity finds match
    - Type C: Incomplete values → fuzzy match finds complete value
    - Type E: Single-char typos → fuzzy match with high cutoff
    """
    findings = []

    # Match: df['col'] == 'value' or df["col"] == 'value'
    pattern = r"""(\w+)\[['"](\w+)['"]\]\s*==\s*['"]([^'"]+)['"]"""

    for match in re.finditer(pattern, code):
        df_name = match.group(1)
        col_name = match.group(2)
        value = match.group(3)

        if len(value) < 2:  # Skip very short values
            continue

        # Infer table name from df variable name
        table_name = _infer_table_name(df_name, col_name, code, data, conn)
        if not table_name:
            continue

        # Query DB for distinct values
        db_values = _get_distinct_values(conn, table_name, col_name)
        if not db_values:
            continue

        # Check exact match
        if value in db_values:
            continue  # Value is correct, no correction needed

        # Attempt fuzzy match
        correction = _attempt_fuzzy_correction(
            col_name, value, db_values, code, match,
            match_type="==",
        )
        if correction:
            findings.append(correction)

        # If no fuzzy match, try pattern-based correction
        elif not correction:
            correction = _attempt_pattern_correction(
                col_name, value, db_values, code, match,
                match_type="==",
            )
            if correction:
                findings.append(correction)

    return findings


def _validate_contains_filters(code: str, conn: sqlite3.Connection, data: dict, exec_hints: dict = None) -> list:
    """
    Strategy 2: Validate df['col'].str.contains('value') patterns.

    If the value doesn't match any DB content even with contains,
    try fuzzy match to find a better value.
    """
    findings = []

    pattern = r"""(\w+)\[['"](\w+)['"]\]\.str\.contains\(\s*['"]([^'"]+)['"]"""

    for match in re.finditer(pattern, code):
        df_name = match.group(1)
        col_name = match.group(2)
        value = match.group(3)

        if len(value) < 2:
            continue

        table_name = _infer_table_name(df_name, col_name, code, data, conn)
        if not table_name:
            continue

        db_values = _get_distinct_values(conn, table_name, col_name)
        if not db_values:
            continue

        # Check if any DB value contains this search term
        has_match = any(value.lower() in str(db_val).lower() for db_val in db_values)
        if has_match:
            continue

        # Try fuzzy match
        correction = _attempt_fuzzy_correction(
            col_name, value, db_values, code, match,
            match_type="contains",
        )
        if correction:
            findings.append(correction)

    return findings


def _validate_neq_filters(code: str, conn: sqlite3.Connection, data: dict, exec_hints: dict = None) -> list:
    """Strategy 3: Validate df['col'] != 'value' patterns."""
    findings = []

    pattern = r"""(\w+)\[['"](\w+)['"]\]\s*!=\s*['"]([^'"]+)['"]"""

    for match in re.finditer(pattern, code):
        df_name = match.group(1)
        col_name = match.group(2)
        value = match.group(3)

        if len(value) < 2:
            continue

        table_name = _infer_table_name(df_name, col_name, code, data, conn)
        if not table_name:
            continue

        db_values = _get_distinct_values(conn, table_name, col_name)
        if not db_values:
            continue

        if value in db_values:
            continue

        correction = _attempt_fuzzy_correction(
            col_name, value, db_values, code, match,
            match_type="!=",
        )
        if correction:
            findings.append(correction)

    return findings


def _validate_isin_filters(code: str, conn: sqlite3.Connection, data: dict, exec_hints: dict = None) -> list:
    """Strategy 4: Validate df['col'].isin([values]) patterns."""
    findings = []

    # Match: .isin(['val1', 'val2', ...])
    pattern = r"""(\w+)\[['"](\w+)['"]\]\.isin\(\s*\[([^\]]+)\]\s*\)"""

    for match in re.finditer(pattern, code):
        df_name = match.group(1)
        col_name = match.group(2)
        values_str = match.group(3)

        # Parse individual values
        values = re.findall(r"['\"]([^'\"]+)['\"]", values_str)

        table_name = _infer_table_name(df_name, col_name, code, data, conn)
        if not table_name:
            continue

        db_values = _get_distinct_values(conn, table_name, col_name)
        if not db_values:
            continue

        for value in values:
            if len(value) < 2:
                continue
            if value in db_values:
                continue

            correction = _attempt_fuzzy_correction(
                col_name, value, db_values, code, match,
                match_type="isin",
            )
            if correction:
                findings.append(correction)

    return findings


def _check_missing_value_linking(code: str, conn: sqlite3.Connection, data: dict) -> list:
    """
    Strategy 5: Check if Value Linking extracted values that are NOT
    present anywhere in the fused code. This catches cases where the
    LLM correctly extracted a value in Step 2a but forgot to use it.
    """
    findings = []

    value_linking = data.get("schema_and_value_linking", {}).get("value_linking", [])
    if not value_linking:
        return findings

    for vl_value in value_linking:
        vl_value = str(vl_value).strip()
        if len(vl_value) < 2:
            continue

        # Check if this value appears in the code
        if vl_value in code:
            continue

        # Value was extracted but not used — log as a finding
        findings.append({
            "type": "missing_value_linking",
            "value": vl_value,
            "note": "Value extracted by Value Linking but not found in fused code",
            "applied": False,
        })

    return findings


def _attempt_fuzzy_correction(
    col_name: str,
    value: str,
    db_values: list,
    code: str,
    match: re.Match,
    match_type: str = "==",
) -> Optional[dict]:
    """
    Attempt fuzzy matching to find the correct DB value.

    Uses multiple similarity strategies (in priority order):
    1. Substring containment (value is embedded in DB value) — HIGHEST PRIORITY
    2. Execution state hint extraction (from df_vars comments in code)
    3. SequenceMatcher (difflib) with cutoff 0.7, with substring tiebreaker
    4. Case-insensitive matching
    5. Token-level similarity (for space-separated values)
    """
    # ── Priority 1: Substring containment ──
    # If the extracted value is a substring of a DB value, that DB value
    # is likely the intended one (e.g., 'TR00_1_2' → 'TR000_1_2').
    # This handles "missing digit/character" errors where the LLM truncated a value.
    substring_matches = []
    for db_val in db_values:
        db_val_str = str(db_val)
        if len(value) >= 3 and value in db_val_str:
            # Value is embedded in DB value — very strong signal
            similarity = len(value) / len(db_val_str)  # Coverage ratio
            substring_matches.append((similarity, db_val_str))

    if substring_matches:
        # Sort by coverage ratio (higher = more of the DB value is covered)
        substring_matches.sort(reverse=True)
        best_cov, best_val = substring_matches[0]
        return _build_correction(
            col_name, value, best_val, code, match, match_type,
            method="substring_containment",
            similarity=round(best_cov, 3),
        )

    # ── Priority 2: Execution state hint extraction ──
    # Parse df_vars comments from the code to find actual sample values.
    # Format: "# bond_id (object): ['TR000_1_2', 'TR000_1_2', 'TR000_1_2']"
    exec_hints = _extract_sample_values_from_comments(code, col_name)
    if exec_hints:
        # Check if any hint is a fuzzy match for the extracted value
        for hint in exec_hints:
            if SequenceMatcher(None, value, hint).ratio() >= 0.6:
                return _build_correction(
                    col_name, value, hint, code, match, match_type,
                    method="execution_state_hint",
                    similarity=round(SequenceMatcher(None, value, hint).ratio(), 3),
                )

    # ── Priority 3: SequenceMatcher with structural tiebreaker ──
    # Get ALL matches above cutoff, then sort by:
    #   1. Similarity (primary)
    #   2. Structural consistency: prefer matches with common prefix/suffix patterns
    all_matches = get_close_matches(value, db_values, n=20, cutoff=0.7)
    if all_matches:
        scored = []
        for db_val in all_matches:
            base_sim = SequenceMatcher(None, value, db_val).ratio()

            # Tiebreaker 1: Common prefix length bonus
            # (handles cases where LLM truncated a value: TR00 → TR000)
            common_prefix = 0
            for a, b in zip(value, str(db_val)):
                if a == b:
                    common_prefix += 1
                else:
                    break
            prefix_bonus = 0.02 * common_prefix

            # Tiebreaker 2: Prefer shorter DB values (Occam's razor for insertions)
            length_penalty = -0.01 * abs(len(str(db_val)) - len(value))

            scored.append((base_sim + prefix_bonus + length_penalty, db_val))

        scored.sort(reverse=True)
        best_score, best_val = scored[0]
        return _build_correction(
            col_name, value, best_val, code, match, match_type,
            method="fuzzy_match",
            similarity=round(SequenceMatcher(None, value, best_val).ratio(), 3),
        )

    # ── Priority 4: Case-insensitive exact match ──
    value_lower = value.lower()
    for db_val in db_values:
        if str(db_val).lower() == value_lower:
            return _build_correction(
                col_name, value, str(db_val), code, match, match_type,
                method="case_insensitive_match",
                similarity=1.0,
            )

    # ── Priority 5: Token-level similarity ──
    value_tokens = set(value.lower().split())
    if len(value_tokens) >= 2:
        best_score = 0
        best_db_val = None
        for db_val in db_values:
            db_tokens = set(str(db_val).lower().split())
            if not db_tokens:
                continue
            overlap = len(value_tokens & db_tokens)
            score = overlap / max(len(value_tokens), len(db_tokens))
            if score > best_score and score >= 0.6:
                best_score = score
                best_db_val = str(db_val)

        if best_db_val:
            return _build_correction(
                col_name, value, best_db_val, code, match, match_type,
                method="token_overlap",
                similarity=best_score,
            )

    return None


def _attempt_pattern_correction(
    col_name: str,
    value: str,
    db_values: list,
    code: str,
    match: re.Match,
    match_type: str = "==",
) -> Optional[dict]:
    """
    Attempt pattern-based correction when fuzzy matching fails.

    Handles:
    - Type D: Exact match should be pattern match (LIKE)
    - Numeric format differences ('0:01:40' vs '1:40%')
    """
    # Check if any DB value contains parts of the search value
    value_parts = re.split(r"[\s\-_:]", value)
    long_parts = [p for p in value_parts if len(p) >= 3]

    if not long_parts:
        return None

    # Find DB values that contain the most long parts
    best_matches = []
    for db_val in db_values:
        db_val_str = str(db_val).lower()
        matched_parts = sum(1 for p in long_parts if p.lower() in db_val_str)
        if matched_parts > 0:
            best_matches.append((matched_parts, db_val_str))

    if not best_matches:
        return None

    # Sort by number of matched parts
    best_matches.sort(reverse=True)
    best_count, best_val = best_matches[0]

    # If most parts match, suggest replacing == with .str.contains()
    if best_count >= len(long_parts) * 0.5:
        old_pattern = f"['{col_name}'] == '{value}'"
        new_pattern = f"['{col_name}'].str.contains('{value}', regex=False)"

        if old_pattern in code:
            return {
                "type": "pattern_correction",
                "column": col_name,
                "old": old_pattern,
                "new": new_pattern,
                "method": "exact_to_contains",
                "note": f"No exact match for '{value}', converted to str.contains()",
                "applied": False,
            }

    return None


def _cross_validate_corrections(corrections: list, db_path: str, code: str) -> list:
    """
    Cross-validate multiple corrections to find consistent entity mappings.

    When multiple filter values share a common truncation/typo pattern
    (e.g., both 'TR00_1_2' and 'TR00_1' are missing a digit), this function
    finds the consistent pair that maps to the same parent entity.

    Algorithm:
    1. Group corrections by their shared prefix pattern.
    2. For each group, find DB records where ALL corrected values coexist.
    3. Prefer the group with the highest total similarity score.

    This handles cases like ID=296 where independent fuzzy matching
    can't disambiguate between TR000/TR100/TR200/etc.
    """
    if len(corrections) < 2:
        return corrections

    # Group corrections by their common prefix (first N chars of old value)
    groups = {}
    for c in corrections:
        old_val = c.get("old_value", "")
        # Extract common prefix (e.g., 'TR' from 'TR00_1_2')
        prefix = ""
        for ch in old_val:
            if ch.isalpha():
                prefix += ch
            else:
                break
        if prefix:
            groups.setdefault(prefix, []).append(c)

    # For groups with 2+ corrections, try cross-validation
    for prefix, group in groups.items():
        if len(group) < 2:
            continue

        # Extract old values and their candidate new values
        old_values = [c["old_value"] for c in group]
        all_candidates = []
        for c in group:
            candidates = [c["new_value"]]
            # Also add other close matches from DB
            col = c.get("column", "")
            # Get table name from correction context
            table_match = re.search(rf"(\w+)\['{col}'\]", code)
            if table_match:
                df_name = table_match.group(1)
                # Try to infer table (simplified)
                table_name = df_name
                try:
                    conn2 = sqlite3.connect(db_path)
                    cursor = conn2.cursor()
                    cursor.execute(f'SELECT DISTINCT "{col}" FROM "{table_name}" LIMIT 5000')
                    all_db_vals = [str(r[0]) for r in cursor.fetchall() if r[0] is not None]
                    from difflib import get_close_matches as gcm
                    candidates = gcm(c["old_value"], all_db_vals, n=5, cutoff=0.8)
                    conn2.close()
                except:
                    pass
            all_candidates.append((c, candidates))

        # Find consistent pairs: for each combination of candidates,
        # check if they coexist in the same DB record
        best_score = 0
        best_assignments = None

        for combo in _generate_combinations(all_candidates):
            # Check if these values coexist (via SQL JOIN)
            if _values_coexist(combo, db_path, code):
                # Calculate total similarity score
                total_sim = sum(
                    SequenceMatcher(None, c["old_value"], new_val).ratio()
                    for c, new_val in combo
                )
                if total_sim > best_score:
                    best_score = total_sim
                    best_assignments = combo

        if best_assignments:
            # Update corrections with the consistent assignments
            for c, new_val in best_assignments:
                c["new_value"] = new_val
                c["old"] = c["old"].replace(c["old_value"], new_val, 1)
                c["new"] = c["new"].replace(c["new_value"], new_val, 1)
                c["method"] = "cross_validated"
                c["similarity"] = round(
                    SequenceMatcher(None, c["old_value"], new_val).ratio(), 3
                )

    return corrections


def _generate_combinations(all_candidates: list) -> list:
    """Generate all combinations of (correction, candidate) assignments."""
    if not all_candidates:
        yield []
        return
    first, rest = all_candidates[0], all_candidates[1:]
    for candidate in first[1]:  # candidates list
        for combo in _generate_combinations(rest):
            yield [(first[0], candidate)] + combo


def _values_coexist(combo: list, db_path: str, code: str) -> bool:
    """
    Check if the given (column, value) pairs can coexist in the same DB record.

    For simple cases: check if there's a molecule_id where both bond_id and
    atom_id match the given values.
    """
    if len(combo) < 2:
        return True

    # Try to find a common join key
    # Heuristic: look for 'molecule_id' as a common FK
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Build a query that checks coexistence
        # This assumes the columns belong to tables that can be joined via molecule_id
        tables_used = set()
        for c, val in combo:
            col = c.get("column", "")
            table_match = re.search(rf"(\w+)\['{col}'\]", code)
            if table_match:
                tables_used.add(table_match.group(1))

        if len(tables_used) < 2:
            conn.close()
            return True  # Same table, no join needed

        tables = list(tables_used)
        # Try joining on molecule_id
        try:
            conditions = " AND ".join(
                f'{t}."{c[0]["column"]}" = ?' for t, c in combo
            )
            values = [v for _, v in combo]
            query = f'''
                SELECT COUNT(*) FROM {tables[0]}
                JOIN {tables[1]} ON {tables[0]}.molecule_id = {tables[1]}.molecule_id
                WHERE {conditions}
            '''
            cursor.execute(query, values)
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except:
            conn.close()
            return True  # Can't verify, assume OK

    except:
        return True


def _build_correction(
    col_name: str,
    old_value: str,
    new_value: str,
    code: str,
    match: re.Match,
    match_type: str,
    method: str,
    similarity: float,
) -> dict:
    """Build a correction dict with old and new code patterns."""
    if match_type == "==":
        old_pattern = f"['{col_name}'] == '{old_value}'"
        new_pattern = f"['{col_name}'] == '{new_value}'"
    elif match_type == "!=":
        old_pattern = f"['{col_name}'] != '{old_value}'"
        new_pattern = f"['{col_name}'] != '{new_value}'"
    elif match_type == "contains":
        old_pattern = f"['{col_name}'].str.contains('{old_value}'"
        new_pattern = f"['{col_name}'].str.contains('{new_value}'"
    elif match_type == "isin":
        old_pattern = f"'{old_value}'"
        new_pattern = f"'{new_value}'"
    else:
        old_pattern = old_value
        new_pattern = new_value

    return {
        "type": "value_correction",
        "column": col_name,
        "old_value": old_value,
        "new_value": new_value,
        "old": old_pattern,
        "new": new_pattern,
        "method": method,
        "similarity": round(similarity, 3),
        "note": f"Corrected '{old_value}' → '{new_value}' ({method}, similarity={similarity:.3f})",
        "applied": False,
    }


def _get_distinct_values(
    conn: sqlite3.Connection,
    table_name: str,
    col_name: str,
    limit: int = 5000,
) -> list:
    """Get all distinct values for a column, with safety limit."""
    try:
        # Verify table exists
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        if not cursor.fetchone():
            return []

        cursor.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL LIMIT {limit}'
        )
        return [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
    except Exception:
        return []


def _extract_all_sample_values(accumulated_code: str) -> dict:
    """
    Extract all column sample values from execution state comments in accumulated_code.

    Parses comments like:
        # bond_id (object): ['TR000_1_2', 'TR000_1_2', 'TR000_1_2']
        # atom_id (object): ['TR000_1', 'TR000_2', 'TR000_3']

    Returns:
        {col_name: [list of unique sample values]}
    """
    result = {}
    pattern = r"#\s*(\w+)\s*\([^)]+\):\s*\[([^\]]*)\]"
    for m in re.finditer(pattern, accumulated_code):
        col_name = m.group(1)
        vals_str = m.group(2)
        values = []
        for val_match in re.finditer(r"'([^']*)'", vals_str):
            val = val_match.group(1)
            if val and val not in values:
                values.append(val)
        if values:
            # Merge with existing values for this column
            if col_name not in result:
                result[col_name] = []
            for v in values:
                if v not in result[col_name]:
                    result[col_name].append(v)
    return result


def _extract_sample_values_from_comments(code: str, col_name: str) -> list:
    """
    Extract actual sample values from execution state comments in the code.

    The execution state comments contain lines like:
        # bond_id (object): ['TR000_1_2', 'TR000_1_2', 'TR000_1_2']

    This provides a strong hint for which DB values are actually present
    in the execution context.

    Args:
        code: The fused Python code (may contain execution state comments)
        col_name: The column name to look for

    Returns:
        List of unique sample values for the column, or empty list.
    """
    values = []
    # Match pattern: "# col_name (type): ['val1', 'val2', ...]"
    pattern = rf"#\s*{re.escape(col_name)}\s*\([^)]+\):\s*\[([^\]]*)\]"
    for m in re.finditer(pattern, code):
        vals_str = m.group(1)
        # Parse the values from the list
        for val_match in re.finditer(r"'([^']*)'", vals_str):
            val = val_match.group(1)
            if val and val not in values:
                values.append(val)
    return values


def _build_merge_map(code: str) -> dict:
    """
    Build a mapping from DataFrame variable names to their source tables.

    Parses .merge() patterns:
        df = t1.merge(t2, on='col')
        df = t1.merge(t2, left_on='a', right_on='b')

    Returns:
        {'df': ['t1', 't2'], ...}
    """
    merge_map = {}
    # Match: var = table.merge(other_table, ...)
    pattern = r"(\w+)\s*=\s*(\w+)\s*\.merge\(\s*(\w+)"
    for m in re.finditer(pattern, code):
        df_name = m.group(1)
        left_table = m.group(2)
        right_table = m.group(3)
        merge_map[df_name] = [left_table, right_table]
    return merge_map


def _infer_table_name(df_name: str, col_name: str, code: str, data: dict, conn: sqlite3.Connection) -> Optional[str]:
    """
    Infer the database table name from the Pandas DataFrame variable name.

    Uses multiple strategies:
    1. Direct match: df variable name matches table name
    2. Merge source analysis: df is a merge result → find which source table has the column
    3. Schema linking match: cross-reference with linked tables
    4. Enriched schema match: check against known table names
    5. Fallback: assume df name is the table name
    """
    linked_tables = data.get("linked_tables")

    # Strategy 1: Direct match with linked tables
    if linked_tables:
        for table in linked_tables:
            if table.lower() == df_name.lower():
                return table

    # Strategy 2: Merge source analysis
    merge_map = _build_merge_map(code)
    if df_name in merge_map:
        source_tables = merge_map[df_name]
        # Check which source table has the column
        for src_table in source_tables:
            if _column_exists_in_table(conn, src_table, col_name):
                return src_table
        # Column not found in either source — return first source table as fallback
        return source_tables[0] if source_tables else df_name

    # Strategy 3: Match with enriched schema
    enriched = data.get("enriched_schema", {})
    for table_name in enriched:
        if table_name.lower() == df_name.lower():
            return table_name

    # Strategy 4: Match with schema box_schema
    box_schema = data.get("schema", {}).get("box_schema", "")
    table_pattern = rf"(\w+)\s*=\s*pd\.DataFrame"
    for m in re.finditer(table_pattern, box_schema):
        if m.group(1).lower() == df_name.lower():
            return m.group(1)

    # Fallback: assume df name is the table name
    return df_name


def _column_exists_in_table(conn: sqlite3.Connection, table_name: str, col_name: str) -> bool:
    """Check if a column exists in a table."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        if not cursor.fetchone():
            return False
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        cols = [row[1] for row in cursor.fetchall()]
        return col_name in cols
    except Exception:
        return False
