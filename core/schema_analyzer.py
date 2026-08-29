"""
Unified Schema Analyzer for Pandora

Handles schema analysis for all task types (NL2SQL, KBQA, TableQA):
1. Extracts metadata from SQLite DBs (NL2SQL) or existing box_schema (KBQA/TableQA)
2. Uses LLM to optimize column meanings and identify query-specific risks
3. Returns structured enriched_schema and risk_hints

Usage:
    # For NL2SQL (SQLite)
    analyzer = SchemaAnalyzer(model=your_model, db_path="data/bird/dev/db.sqlite")
    result = analyzer.analyze(question="...", box_schema="...")

    # For KBQA/TableQA (No SQLite, uses box_schema/table data)
    analyzer = SchemaAnalyzer(model=your_model)
    result = analyzer.analyze(question="...", schema_metadata="...")
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional, List, Dict
from collections import defaultdict
from utils.token_counter import count_prompt_tokens, count_messages_tokens

from utils.logger import setup_logger


# ==============================================================================
# Part 1: Metadata Extraction (NL2SQL path)
# ==============================================================================

class SchemaMetadataExtractor:
    """
    Extract rich metadata from a SQLite database for NL2SQL tasks.
    """

    def __init__(self, db_path: str, top_k: int = 5):
        self.db_path = db_path
        self.top_k = top_k
        self.logger = setup_logger("pandora.schema_metadata")

    def extract_all(self) -> dict:
        """Extract metadata for all tables in the database."""
        if not Path(self.db_path).exists():
            return {}

        try:
            conn = sqlite3.connect(self.db_path)
        except Exception as e:
            self.logger.error(f"Cannot connect to DB: {e}")
            return {}

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]

            metadata: Dict[str, Any] = {
                "tables": {},
                "foreign_keys": [],
                "primary_keys": {},
            }

            for table_name in table_names:
                table_meta = self._extract_table_metadata(cursor, table_name)
                metadata["tables"][table_name] = table_meta

                # Extract PK
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                cols = cursor.fetchall()
                pk_cols = [c[1] for c in cols if c[5]]
                if pk_cols:
                    metadata["primary_keys"][table_name] = pk_cols

                # Extract FKs
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

    def _extract_table_metadata(self, cursor, table_name: str) -> dict:
        """Extract metadata for a single table."""
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        cols = cursor.fetchall()

        table_info = {"columns": {}}

        for col in cols:
            col_name = col[1]
            declared_type = col[2] or "text"

            cursor.execute(f'SELECT "{col_name}" FROM "{table_name}" LIMIT {self.top_k}')
            sample_vals = [row[0] for row in cursor.fetchall()]

            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            total_rows = cursor.fetchone()[0]

            cursor.execute(f'SELECT COUNT("{col_name}") FROM "{table_name}"')
            non_null_count = cursor.fetchone()[0]

            null_ratio = (total_rows - non_null_count) / total_rows if total_rows > 0 else 1.0

            cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}"')
            distinct_count = cursor.fetchone()[0]

            patterns = self._detect_patterns(sample_vals, declared_type)

            table_info["columns"][col_name] = {
                "declared_type": declared_type,
                "sample_values": sample_vals,
                "null_ratio": round(null_ratio, 2),
                "distinct_count": distinct_count,
                "total_rows": total_rows,
                "patterns": patterns,
            }

        return table_info

    def _detect_patterns(self, values: list, declared_type: str) -> dict:
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

    def format_for_llm(self, metadata: dict) -> str:
        """Format metadata as a string for LLM consumption."""
        if not metadata or "tables" not in metadata:
            return "No schema metadata available."

        lines = []
        for table_name, table_info in metadata.get("tables", {}).items():
            lines.append(f"Table: {table_name}")
            for col_name, col_info in table_info.get("columns", {}).items():
                flags = self._build_flags(col_info.get("patterns", {}), col_info.get("null_ratio", 0))
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

    def _build_flags(self, patterns: dict, null_ratio: float) -> List[str]:
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
# Part 2: LLM Analysis (Shared by all task types)
# ==============================================================================

class LLMSchemaAnalyzer:
    """
    Uses LLM to analyze schema metadata, optimize column meanings,
    and identify query-specific risks.
    """

    def __init__(self, model, template_dir: str = "", generate_fn=None):
        self.model = model
        self.generate_fn = generate_fn or model.generate
        self.template_dir = Path(template_dir)
        self.template_name = "schema_analysis.txt"
        self.logger = setup_logger("pandora.llm_risk_analyzer")
        self._template_content: Optional[str] = None

    def _load_template(self) -> str:
        if self._template_content is not None:
            return self._template_content
        template_path = self.template_dir / self.template_name
        if not template_path.exists():
            self.logger.error(f"Template not found: {template_path}")
            return ""
        self._template_content = template_path.read_text(encoding="utf-8")
        return self._template_content

    def analyze_with_optimized_schema(
        self,
        schema_metadata_text: str,
        question: str,
        evidence: str = "",
        few_shot_examples: str = "",
    ) -> dict:
        if not schema_metadata_text or not schema_metadata_text.strip():
            return {"optimized_schema": {}, "risk_hints": ""}

        template = self._load_template()
        if not template:
            return {"optimized_schema": {}, "risk_hints": ""}

        try:
            prompt = template.replace("{{question}}", question)
            prompt = prompt.replace("{{evidence}}", evidence or "(none)")
            prompt = prompt.replace("{{schema_metadata}}", schema_metadata_text)
            prompt = prompt.replace("{{few_shot_examples}}", few_shot_examples)

            self.logger.info(f"Calling LLM for schema analysis (question: {question[:80]}...)")
            response = self.generate_fn(prompt)

            # Extract JSON
            parsed = self._extract_json(response)
            optimized_schema = parsed.get("optimized_schema", {})
            risks = parsed.get("risks", [])

            # Format risk hints
            risk_hints = self._format_risk_hints(risks)

            if risk_hints:
                self.logger.info(f"Analysis complete: {len(risks)} risk(s) identified")

            return {
                "optimized_schema": optimized_schema,
                "risk_hints": risk_hints,
            }

        except Exception as e:
            self.logger.warning(f"Schema analysis failed: {e}")
            return {"optimized_schema": {}, "risk_hints": ""}

    def _extract_json(self, response: str) -> dict:
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
                if response[i] == '{': brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(response[start_idx : i + 1])
                        except json.JSONDecodeError:
                            break
        return {}

    def _format_risk_hints(self, risks: list[dict]) -> str:
        if not risks:
            return ""
        lines = [
            "",
            "### ⚠️ Schema Analysis: Potential Pitfalls for This Query",
            "",
            "Based on the actual data patterns in this database, watch out for these issues:",
            "",
        ]
        for risk in risks[:5]:
            severity = risk.get("severity", "warning")
            icon = "🔴" if severity == "critical" else "🟡"
            risk_type = risk.get("risk_type", "risk").replace("_", " ").title()
            message = risk.get("message", "")
            wrong_desc = risk.get("wrong_description", "")

            lines.append(f"{icon} **{risk_type}** ({severity})")
            if message: lines.append(f"- {message}")
            if wrong_desc: lines.append(f"  ⛔ AVOID: {wrong_desc}")
            lines.append("")
        return "\n".join(lines)


# ==============================================================================
# Part 3: Unified Schema Analyzer
# ==============================================================================

class SchemaAnalyzer:
    """
    Unified schema analyzer for all task types (NL2SQL, KBQA, TableQA).

    - For NL2SQL: Extracts metadata from SQLite DB and analyzes it.
    - For KBQA/TableQA: Reads CSV tables, extracts sample values per column,
      replaces entity IDs with names (KBQA), and formats for LLM analysis.
    """

    def __init__(
        self,
        model,
        db_path: str = None,
        top_k: int = 5,
        template_dir: str = "./prompts/tasks/nl2sql",
        entity_id_to_name: dict = None,
        generate_fn=None,
    ):
        self.model = model
        self.db_path = db_path
        self.top_k = top_k
        self.template_dir = template_dir
        self.entity_id_to_name = entity_id_to_name or {}
        self.logger = setup_logger("pandora.schema_analyzer")

        self.metadata_extractor = SchemaMetadataExtractor(db_path, top_k) if db_path else None
        self.risk_analyzer = LLMSchemaAnalyzer(model, template_dir, generate_fn=generate_fn)

    def _extract_kbqa_metadata(self, kg_dir: str, box_schema: str = "") -> str:
        """
        Extract metadata from KBQA CSV tables with sample values per column.
        Replaces entity IDs with entity names in sample values for LLM readability.
        Detects CVT/blank node columns (columns where values are Freebase IDs
        that have NO entity name in the map).
        """
        import pandas as pd

        kg_path = Path(kg_dir)
        if not kg_path.exists():
            return box_schema or "No schema metadata available."

        # Parse table names from box_schema
        schema_tables = set()
        for line in box_schema.split('\n'):
            m = re.match(r'^(\w+)\s*=\s*pd\.DataFrame\(\{', line.strip())
            if m:
                schema_tables.add(m.group(1).lower())

        # Build entity replacement map (ID -> Name), sorted by length desc
        replacements = []
        known_ids = set()
        if self.entity_id_to_name:
            replacements = sorted(
                self.entity_id_to_name.items(),
                key=lambda x: -len(x[0])
            )
            known_ids = set(self.entity_id_to_name.keys())

        def _replace_ids(text: str) -> str:
            if not replacements:
                return text
            for eid, name in replacements:
                text = text.replace(eid, name)
            return text

        # Regex for Freebase IDs: m.XXX or g.XXX
        freebase_id_re = re.compile(r'^[mg]\.\w+$')

        lines = []
        cvt_columns = []  # Collect CVT column info for annotation

        csv_files = sorted(kg_path.glob("*.csv"))

        for csv_file in csv_files:
            table_name = csv_file.stem  # e.g., "medicine.medical_trial"
            simple_name = table_name.rsplit(".", 1)[-1]  # e.g., "medical_trial"

            # Skip if table doesn't match box_schema
            if schema_tables and simple_name.lower() not in schema_tables:
                continue

            try:
                df = pd.read_csv(csv_file, dtype=str)
                if df.empty:
                    continue

                lines.append(f"Table: {table_name}")

                for col in df.columns:
                    # Get raw sample values BEFORE replacement (for CVT detection)
                    raw_vals = df[col].dropna().head(self.top_k).tolist()
                    raw_str_vals = [str(v).strip() for v in raw_vals if str(v).strip()]

                    # Also check ALL values (not just samples) for CVT detection
                    all_raw_vals = df[col].dropna().unique()[:50]
                    all_str_vals = [str(v).strip() for v in all_raw_vals if str(v).strip()]

                    # Detect CVT/blank node: values are Freebase IDs NOT in entity map
                    freebase_in_all = sum(1 for v in all_str_vals if freebase_id_re.match(v))
                    unknown_in_all = sum(1 for v in all_str_vals if freebase_id_re.match(v) and v not in known_ids)

                    is_cvt = False
                    if len(all_str_vals) > 0 and freebase_in_all > 0:
                        unknown_ratio = unknown_in_all / len(all_str_vals)
                        freebase_ratio = freebase_in_all / len(all_str_vals)
                        # If most values are Freebase IDs and most of those have no name → CVT column
                        if freebase_ratio > 0.5 and unknown_ratio > 0.5:
                            is_cvt = True

                    # Get sample values WITH replacement for display
                    sample_vals = [_replace_ids(str(v)) for v in raw_str_vals]

                    total_rows = len(df)
                    non_null = df[col].notna().sum()
                    null_ratio = round((total_rows - non_null) / total_rows, 2) if total_rows > 0 else 1.0
                    distinct_count = df[col].nunique()

                    patterns = self._detect_kbqa_patterns(sample_vals)
                    flags = self._build_flags(patterns, null_ratio)

                    if is_cvt:
                        flags.append("🔗 CVT-BLANK-NODE")
                        # Collect FK info for this column
                        fk_targets = self._find_fk_target(col, simple_name, kg_path)
                        cvt_columns.append({
                            "table": simple_name,
                            "column": col,
                            "fk_targets": fk_targets,
                        })

                    flag_str = " ".join(flags)
                    samples_str = ", ".join(sample_vals[:3]) if sample_vals else "(empty)"

                    line = f"  - {col} (text)"
                    if flag_str:
                        line += f" {flag_str}"
                    line += f" | Sample: {samples_str}"
                    lines.append(line)

                lines.append("")

            except Exception as e:
                self.logger.warning(f"Failed to read {csv_file}: {e}")

        # Append CVT column analysis section
        if cvt_columns:
            lines.append("")
            lines.append("## 🔗 CVT / Blank Node Columns (CRITICAL)")
            lines.append("")
            lines.append("The following columns contain mostly Freebase IDs with NO corresponding entity name.")
            lines.append("These are **blank nodes** (compound value types / CVT nodes) — they serve as **intermediate")
            lines.append("bridging tables** to connect multi-ary relations. **You CANNOT query these columns directly.**")
            lines.append("You MUST **merge (join)** through them using foreign keys to reach the actual entity.")
            lines.append("")
            for cvt in cvt_columns:
                fk_text = ""
                if cvt["fk_targets"]:
                    fk_text = " — Possible join paths: " + ", ".join(
                        f"merge with `{t}` table" for t in cvt["fk_targets"]
                    )
                lines.append(f"- **`{cvt['table']}.{cvt['column']}`**: This is a blank node column.")
                lines.append(f"  ⛔ WRONG: Directly filter or return this column — it only contains meaningless IDs.")
                lines.append(f"  ✅ CORRECT: Use `df.merge(other_table, left_on='{cvt['column']}', right_on='other_col')` "
                             f"to connect to the actual entity data.{fk_text}")
            lines.append("")

        result = "\n".join(lines)
        return result if result.strip() else (box_schema or "No schema metadata available.")

    @staticmethod
    def _find_fk_target(column: str, table_name: str, kg_path: Path) -> list[str]:
        """Find which other tables might FK to this column by scanning foreign_key.json files."""
        targets = []
        try:
            # Check if there's a foreign_key.json in the same directory
            fk_file = kg_path / "foreign_key.json"
            if fk_file.exists():
                import json
                with open(fk_file) as f:
                    fk_data = json.load(f)
                if isinstance(fk_data, list):
                    for fk in fk_data:
                        if isinstance(fk, list) and len(fk) >= 2:
                            # FK format: ["table.col", "target_table.target_col"]
                            src_parts = fk[0].rsplit("-", 1) if "-" in fk[0] else None
                            if src_parts and src_parts[1] == column:
                                tgt_parts = fk[1].rsplit("-", 1) if "-" in fk[1] else None
                                if tgt_parts:
                                    targets.append(tgt_parts[0])
        except Exception:
            pass
        return list(set(targets))

    @staticmethod
    def _detect_kbqa_patterns(values: list) -> dict:
        """Detect patterns in KBQA column values."""
        if not values:
            return {}
        patterns = {}
        date_count = sum(1 for v in values if isinstance(v, str) and re.match(r'^\d{4}-\d{2}-\d{2}', v))
        time_count = sum(1 for v in values if isinstance(v, str) and re.match(r'^\d{2}:\d{2}', v))

        if date_count > len(values) * 0.5:
            patterns["type"] = "date"
        elif time_count > len(values) * 0.5:
            patterns["type"] = "time"

        return patterns

    @staticmethod
    def _build_flags(patterns: dict, null_ratio: float) -> List[str]:
        """Build emoji flags for columns."""
        flags = []
        p_type = patterns.get("type")
        if p_type == "date":
            flags.append("📅 DATE")
        elif p_type == "time":
            flags.append("📅 TIME")
        if null_ratio > 0.5:
            flags.append("⚠️ HIGH-NULL")
        return flags

    def analyze(
        self,
        question: str,
        evidence: str = "",
        box_schema: str = "",
        kg_dir: str = None,
        few_shot_examples: str = "",
    ) -> dict:
        """
        Main entry point. Analyzes schema and returns enriched_schema + risk_hints.
        """
        # Step 1: Get schema metadata
        if self.metadata_extractor:
            # NL2SQL path: extract from SQLite
            self.logger.info("Extracting metadata from SQLite DB")
            metadata = self.metadata_extractor.extract_all()
            schema_metadata_text = self.metadata_extractor.format_for_llm(metadata)
        elif kg_dir:
            # KBQA/TableQA path: extract from CSV tables with sample values
            self.logger.info("Extracting metadata from KBQA CSV tables")
            schema_metadata_text = self._extract_kbqa_metadata(kg_dir, box_schema)
        else:
            # Fallback: use box_schema directly
            self.logger.info("Using provided box_schema for analysis")
            schema_metadata_text = box_schema or "No schema metadata available."

        # Step 2: LLM Analysis (Optimize meanings + Identify risks)
        self.logger.info(f"Analyzing schema for question: {question[:80]}...")
        result = self.risk_analyzer.analyze_with_optimized_schema(
            schema_metadata_text=schema_metadata_text,
            question=question,
            evidence=evidence,
            few_shot_examples=few_shot_examples,
        )

        # Step 3: Build structured enriched_schema
        optimized_schema = result.get("optimized_schema", {})
        if optimized_schema:
            enriched = self._parse_optimized_schema(optimized_schema)
        else:
            enriched = self._parse_box_schema_to_structured(box_schema)

        return {
            "enriched_schema": enriched,
            "risk_hints": result.get("risk_hints", ""),
        }

    def _parse_optimized_schema(self, optimized_schema: dict) -> dict:
        """Convert LLM optimized_schema to enriched_schema format."""
        enriched = {}
        for table_name, columns in optimized_schema.items():
            enriched[table_name] = {}
            for col_name, meaning in columns.items():
                enriched[table_name][col_name] = {
                    "type": "object",
                    "meaning": meaning,
                }
        return enriched

    def _parse_box_schema_to_structured(self, box_schema: str) -> dict:
        """Parse raw box_schema into structured format (fallback)."""
        enriched = {}
        current_table = None
        for line in box_schema.split('\n'):
            stripped = line.strip()
            table_match = re.match(r'^(\w+)\s*=\s*pd\.DataFrame\(\{', stripped)
            if table_match:
                current_table = table_match.group(1)
                enriched[current_table] = {}
                continue

            col_match = re.match(r'^\s*"([^"]+)":\s*\[\](?:,\s*)?(#.*)?$', line)
            if col_match and current_table:
                col_name = col_match.group(2)
                comment = (col_match.group(3) or "").strip().lstrip('#').strip()

                type_match = re.match(r'^\(([^)]+)\)\s*,?\s*(.*)', comment)
                if type_match:
                    col_type = type_match.group(1)
                    meaning = type_match.group(2).strip()
                else:
                    col_type = "unknown"
                    meaning = comment

                enriched[current_table][col_name] = {
                    "type": col_type,
                    "meaning": meaning if meaning else col_name,
                }
        return enriched
