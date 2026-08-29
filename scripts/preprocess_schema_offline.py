#!/usr/bin/env python3
"""
Offline Schema Analysis for BIRD Database

Pre-enriches column meanings and analyzes schema risks for ALL databases,
without any question dependency. Results are cached and reused during inference.

Output: data/bird/box/{db_id}.json containing:
  - box_schema: the enriched schema as pd.DataFrame code
  - risk_hints: formatted risk hints string
  - optimized_schema: {table: {col: meaning}}
  - risks: raw risk list

Usage:
    python scripts/preprocess_schema_offline.py --model gpt-4o-mini --api-key YOUR_KEY
    python scripts/preprocess_schema_offline.py --model deepseek-chat --api-key YOUR_KEY
"""

import argparse
import json
import os
import sys
import re
import time
import sqlite3
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

from models.registry import ModelRegistry
from utils.logger import setup_logger
from utils.token_counter import count_prompt_tokens


# ==============================================================================
# Metadata extraction (reused from SchemaMetadataExtractor)
# ==============================================================================

def extract_metadata_from_db(db_path: str, top_k: int = 5) -> dict:
    """Extract rich metadata from a SQLite database."""
    if not Path(db_path).exists():
        return {}

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logger.error(f"Cannot connect to DB: {db_path}: {e}")
        return {}

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]

        metadata: Dict[str, Any] = {"tables": {}, "foreign_keys": [], "primary_keys": {}}

        for table_name in table_names:
            table_meta = _extract_table_metadata(cursor, table_name, top_k)
            metadata["tables"][table_name] = table_meta

            cursor.execute(f'PRAGMA table_info("{table_name}")')
            cols = cursor.fetchall()
            pk_cols = [c[1] for c in cols if c[5]]
            if pk_cols:
                metadata["primary_keys"][table_name] = pk_cols

            cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
            fks = cursor.fetchall()
            for fk in fks:
                metadata["foreign_keys"].append({
                    "from_table": table_name,
                    "from_column": fk[3],
                    "to_table": fk[2],
                    "to_column": fk[4],
                })

        return metadata
    finally:
        conn.close()


def _extract_table_metadata(cursor, table_name: str, top_k: int) -> dict:
    """Extract metadata for a single table."""
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    cols = cursor.fetchall()

    table_info = {"columns": {}}

    for col in cols:
        col_name = col[1]
        declared_type = col[2] or "text"

        cursor.execute(f'SELECT "{col_name}" FROM "{table_name}" LIMIT {top_k}')
        sample_vals = [row[0] for row in cursor.fetchall()]

        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        total_rows = cursor.fetchone()[0]

        cursor.execute(f'SELECT COUNT("{col_name}") FROM "{table_name}"')
        non_null_count = cursor.fetchone()[0]

        null_ratio = (total_rows - non_null_count) / total_rows if total_rows > 0 else 1.0

        cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}"')
        distinct_count = cursor.fetchone()[0]

        patterns = _detect_patterns(sample_vals, declared_type)

        table_info["columns"][col_name] = {
            "declared_type": declared_type,
            "sample_values": sample_vals,
            "null_ratio": round(null_ratio, 2),
            "distinct_count": distinct_count,
            "total_rows": total_rows,
            "patterns": patterns,
        }

    return table_info


def _detect_patterns(values: list, declared_type: str) -> dict:
    """Detect value patterns in a column."""
    if not values:
        return {}

    patterns = {}
    date_count = sum(1 for v in values if isinstance(v, str) and re.match(r'^\d{4}-\d{2}-\d{2}', v))
    time_count = sum(1 for v in values if isinstance(v, str) and re.match(r'^\d{2}:\d{2}', v))
    id_count = sum(1 for v in values if isinstance(v, str) and re.match(r'^(m|g)\.\w+', v))

    if date_count > len(values) * 0.5:
        patterns["type"] = "date"
    elif time_count > len(values) * 0.5:
        patterns["type"] = "time"
    elif id_count > len(values) * 0.5:
        patterns["type"] = "freebase_id"

    return patterns


def format_metadata_for_llm(metadata: dict) -> str:
    """Format metadata as a string for LLM consumption."""
    if not metadata or "tables" not in metadata:
        return "No schema metadata available."

    lines = []
    for table_name, table_info in metadata.get("tables", {}).items():
        lines.append(f"Table: {table_name}")
        for col_name, col_info in table_info.get("columns", {}).items():
            flags = _build_flags(col_info.get("patterns", {}), col_info.get("null_ratio", 0))
            flag_str = " ".join(flags)
            samples = col_info.get("sample_values", [])
            samples_str = ", ".join(str(v) for v in samples[:3])

            line = f"  - {col_name} ({col_info.get('declared_type', 'unknown')})"
            if flag_str:
                line += f" {flag_str}"
            line += f" | Sample: {samples_str}"
            lines.append(line)
        lines.append("")

    return "\n".join(lines)


def _build_flags(patterns: dict, null_ratio: float) -> list:
    """Build emoji flags for columns."""
    flags = []
    p_type = patterns.get("type")
    if p_type == "date":
        flags.append("📅 DATE")
    elif p_type == "time":
        flags.append("📅 TIME")
    elif p_type == "freebase_id":
        flags.append("🆔 ID")

    if null_ratio > 0.5:
        flags.append("⚠️ HIGH-NULL")

    return flags


# ==============================================================================
# Offline schema analysis
# ==============================================================================

def _extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    json_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    start_idx = response.find('{')
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(response[start_idx: i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


def _format_risk_hints(risks: list[dict]) -> str:
    """Format risk hints for injection into prompts."""
    if not risks:
        return ""
    lines = [
        "",
        "### ⚠️ Schema Analysis: Potential Pitfalls",
        "",
        "Based on the actual data patterns in this database, watch out for these issues:",
        "",
    ]
    for risk in risks[:8]:
        severity = risk.get("severity", "warning")
        icon = "🔴" if severity == "critical" else "🟡"
        risk_type = risk.get("risk_type", "risk").replace("_", " ").title()
        message = risk.get("message", "")
        wrong_desc = risk.get("wrong_description", "")

        lines.append(f"{icon} **{risk_type}** ({severity})")
        if message:
            lines.append(f"- {message}")
        if wrong_desc:
            lines.append(f"  ⛔ AVOID: {wrong_desc}")
        lines.append("")
    return "\n".join(lines)


def _build_enriched_box_schema(optimized_schema: dict, metadata: dict) -> str:
    """Build enriched box_schema from optimized_schema + original metadata."""
    lines = []
    for table_name, columns in optimized_schema.items():
        lines.append(f"{table_name} = pd.DataFrame({{")
        for col_name, meaning in columns.items():
            # Get original type from metadata
            col_type = "unknown"
            if table_name in metadata.get("tables", {}):
                col_meta = metadata["tables"][table_name]["columns"].get(col_name, {})
                col_type = col_meta.get("declared_type", "unknown")

            if col_type and col_type != "unknown":
                comment = f"  # ({col_type}), {meaning}"
            else:
                comment = f"  # {meaning}"
            lines.append(f'    "{col_name}": [],{comment}')
        lines.append("})")
        lines.append("")
    return "\n".join(lines)


def analyze_db_offline(model, db_path: str, template_dir: str) -> dict:
    """
    Run offline schema analysis for a single database.

    Returns:
        {
            "optimized_schema": {table: {col: meaning}},
            "risks": [...],
            "risk_hints": formatted string,
            "box_schema": enriched box_schema string,
            "db_path": original db path,
        }
    """
    # Step 1: Extract metadata
    metadata = extract_metadata_from_db(db_path, top_k=5)
    if not metadata.get("tables"):
        logger.warning(f"No tables found in {db_path}")
        return {}

    schema_metadata_text = format_metadata_for_llm(metadata)

    # Step 2: Load offline template
    template_path = Path(template_dir) / "schema_analysis_offline.txt"
    if not template_path.exists():
        logger.error(f"Offline template not found: {template_path}")
        return {}

    template = template_path.read_text(encoding="utf-8")
    prompt = template.replace("{{schema_metadata}}", schema_metadata_text)
    print(prompt)
    exit()

    # Step 3: Call LLM
    logger.info(f"Calling LLM for offline schema analysis: {db_path}")
    try:
        response = model.generate(prompt)
    except Exception as e:
        logger.error(f"LLM call failed for {db_path}: {e}")
        return {}

    # Step 4: Parse response
    parsed = _extract_json_from_response(response)
    optimized_schema = parsed.get("optimized_schema", {})
    risks = parsed.get("risks", [])

    if not optimized_schema:
        logger.warning(f"LLM returned empty optimized_schema for {db_path}")
        return {}

    # Step 5: Build outputs
    risk_hints = _format_risk_hints(risks)
    box_schema = _build_enriched_box_schema(optimized_schema, metadata)

    logger.info(f"Offline analysis complete for {db_path}: {len(optimized_schema)} tables, {len(risks)} risks")

    return {
        "optimized_schema": optimized_schema,
        "risks": risks,
        "risk_hints": risk_hints,
        "box_schema": box_schema,
        "db_path": db_path,
    }


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Offline schema analysis for BIRD databases")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key (falls back to env var)")
    parser.add_argument("--base-url", default=None, help="Explicit OpenAI-compatible API base URL")
    parser.add_argument("--data-dir", type=str, default="./data/bird/dev_database",
                        help="Directory containing SQLite databases")
    parser.add_argument("--output-dir", type=str, default="./data/bird/box",
                        help="Output directory for enriched schemas")
    parser.add_argument("--prompt-dir", type=str, default="./prompts/tasks/nl2sql",
                        help="Directory containing prompt templates")
    parser.add_argument("--top-k", type=int, default=5, help="Sample rows for metadata extraction")
    parser.add_argument("--db-ids", type=str, nargs="+", default=None,
                        help="Process only these database IDs (default: all)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed DBs")
    parser.add_argument("--retry-failed", action="store_true", help="Re-process failed DBs")
    return parser.parse_args()


def main():
    global logger
    args = parse_args()
    logger = setup_logger("schema_offline")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create model
    model = ModelRegistry.create(
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    logger.info(f"Model: {args.model}")

    # Find all databases
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    db_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if args.db_ids:
        db_dirs = [d for d in db_dirs if d.name in args.db_ids]

    total = len(db_dirs)
    logger.info(f"Found {total} databases to process")

    results = {"total": total, "success": 0, "skipped": 0, "failed": 0, "details": {}}

    for i, db_dir in enumerate(db_dirs, 1):
        db_id = db_dir.name
        db_path = str(db_dir / f"{db_id}.sqlite")
        output_file = output_dir / f"{db_id}.json"

        logger.info(f"[{i}/{total}] Processing: {db_id}")

        # Skip if already processed (unless retry-failed)
        if args.resume and output_file.exists():
            try:
                existing = json.loads(output_file.read_text())
                if existing.get("box_schema"):
                    logger.info(f"  SKIP (already processed)")
                    results["skipped"] += 1
                    continue
            except Exception:
                pass  # Corrupted file, re-process

        # Check if it's a failed file
        if args.retry_failed and output_file.exists():
            try:
                existing = json.loads(output_file.read_text())
                if existing.get("box_schema"):
                    logger.info(f"  SKIP (not failed)")
                    results["skipped"] += 1
                    continue
            except Exception:
                pass  # Corrupted, re-process

        # Run analysis
        start_time = time.time()
        result = analyze_db_offline(model, db_path, args.prompt_dir)
        elapsed = time.time() - start_time

        if result and result.get("box_schema"):
            # Save result
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            results["success"] += 1
            results["details"][db_id] = {
                "status": "success",
                "time_sec": round(elapsed, 2),
                "tables": len(result.get("optimized_schema", {})),
                "risks": len(result.get("risks", [])),
            }
            logger.info(f"  ✓ Saved to {output_file} ({elapsed:.1f}s)")
        else:
            results["failed"] += 1
            results["details"][db_id] = {
                "status": "failed",
                "time_sec": round(elapsed, 2),
                "error": "empty result",
            }
            logger.warning(f"  ✗ Failed for {db_id}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Offline Schema Analysis Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Total: {results['total']}")
    logger.info(f"Success: {results['success']}")
    logger.info(f"Skipped: {results['skipped']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
