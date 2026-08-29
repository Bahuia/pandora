"""
Pandora Unified Agent

Flow:
1. _prepare_example: Load KB info and build schema context
1.5. _analyze_and_enrich_schema: LLM-optimized column meanings + column types + risks
   → Stores enriched_schema: {table: {col: {type, meaning}}}
2. link_schema: Schema + value linking (uses FULL box_schema)
3. _build_box_schema: Dynamically builds box_schema from enriched_schema
   - Before schema_linking: all tables
   - After schema_linking: only linked tables
4. task_decomposition: Task breakdown (uses linked tables only)
5. For each subtask:
   a. code_reasoning: Generate Pandas code for subtask
   b. execute_and_repair_subtask: Execute and fix errors IMMEDIATELY
6. merge_subtask_codes: LLM fusion of all subtask codes
7. run_merged_code: Execute merged code with full data
8. _retry_failed_execution: If execution fails, LLM-based repair loop
9. evaluation: Compare with gold answer
"""

import json
import re
import sqlite3
import os
import copy
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, List, Dict
from pathlib import Path

import pandas as pd

from datasets.base import BaseDataset
from models.base import BaseModel
from utils.code_executor import CodeExecutor, ExecutionResult
from utils.config import ConfigView
from utils.logger import setup_logger
from prompts.base import TemplateEngine
from core.voting import VoteAggregator
from core.schema_analyzer import SchemaAnalyzer
from utils.file_utils import load_json
from utils.answer_postprocess import postprocess_answer, postprocess_wikitq_answer
from utils.value_corrector import correct_filter_values
from utils.schema_utils import (
    load_column_descriptions,
    prepare_kb_info,
)
from utils.memory_retriever import (
    SemanticMemoryRetriever,
    format_demonstrations,
    load_verified_memory,
)


# ==============================================================================
# Stage Profiler - tracks time + prompt token length per pipeline stage
# ==============================================================================

class _StageProfiler:
    """Tracks per-stage execution time and LLM prompt token lengths."""

    def __init__(self):
        self.stages: list[dict] = []
        self._current: Optional[dict] = None

    def start(self, name: str):
        self._current = {
            "stage": name,
            "start_time": time.time(),
            "llm_calls": [],
        }

    def end(self):
        if self._current is None:
            return
        self._current["time_sec"] = round(time.time() - self._current["start_time"], 3)
        self._current["total_llm_calls"] = len(self._current["llm_calls"])
        self._current["total_prompt_tokens_est"] = sum(
            c.get("prompt_tokens_est", 0) for c in self._current["llm_calls"]
        )
        self._current.pop("start_time")
        self.stages.append(self._current)
        self._current = None

    def record_llm_call(self, prompt: str, stage_note: str = ""):
        if self._current is None:
            return
        # Token estimation: ~1 token per ~3.5 chars (rough heuristic for English/mixed)
        self._current["llm_calls"].append({
            "note": stage_note,
            "prompt_length_chars": len(prompt),
            "prompt_tokens_est": len(prompt) // 3 + 1,
        })

    def summary(self) -> dict:
        total_time = sum(s.get("time_sec", 0) for s in self.stages)
        total_llm = sum(s.get("total_llm_calls", 0) for s in self.stages)
        total_tokens = sum(s.get("total_prompt_tokens_est", 0) for s in self.stages)
        return {
            "total_time_sec": round(total_time, 3),
            "total_llm_calls": total_llm,
            "total_prompt_tokens_est": total_tokens,
            "per_stage": [
                {
                    "stage": s["stage"],
                    "time_sec": s.get("time_sec", 0),
                    "llm_calls": s.get("total_llm_calls", 0),
                    "prompt_tokens_est": s.get("total_prompt_tokens_est", 0),
                }
                for s in self.stages
            ],
        }


class PandoraAgent:
    """
    Unified Pandora Agent.
    """

    def __init__(self, dataset: BaseDataset, model: BaseModel, executor: Optional[CodeExecutor] = None, config: Optional[dict] = None):
        self.dataset = dataset
        self.model = model
        self.config = ConfigView(config)
        self.executor = executor or CodeExecutor(
            timeout=self.config.get("execution.timeout", 30),
            max_memory_mb=self.config.get("execution.max_memory_mb", 512),
            max_cpu_seconds=self.config.get("execution.max_cpu_seconds", 30),
        )
        self.logger = setup_logger("pandora.agent")
        self.voting = VoteAggregator()
        prompt_root = self.config.get("paths.prompt_root", "./prompts")
        self.template_engine = TemplateEngine(template_dir=prompt_root)
        self._profile_local = threading.local()
        self._profiler: Optional[_StageProfiler] = None  # stage profiling

        # Configuration
        self.max_tries = self.config.get("inference.max_tries", 3)
        self.max_revision_rounds = self.config.get("inference.max_revision_rounds", 3)
        self.do_execution_guidance = self.config.get("inference.do_execution_guidance", True)
        self.max_workers = self.config.get("inference.max_workers", 4)
        self.shot_k = self.config.get("inference.shot_k", 10)
        self.do_code_merge = self.config.get("inference.do_code_merge", True)
        self.do_decomposition = self.config.get("inference.do_decomposition", True)
        self.do_final_execution_repair = self.config.get(
            "inference.do_final_execution_repair", True
        )
        self.retrieval_mode = self.config.get("retrieval.mode", "cross_task")

        # Schema Risk Analyzer (LLM-driven, per-sample)
        self.schema_analyzer: Optional[SchemaAnalyzer] = None
        # Knowledge source type (determined from dataset name)
        self.kb_type = self._detect_kb_type()

        # Task type: determines which prompt templates and pipeline steps to use
        self.task_type = self._detect_task_type()

        # Paper-consistent, task-agnostic semantic demonstration retrieval.
        self.memory_retriever: Optional[SemanticMemoryRetriever] = None
        self._initialize_memory_retriever()

        # Pre-loaded knowledge base info
        self.kb_info = {}
        self.column_descriptions = {}
        self.column_types_map = {}  # {db_id: {table_name: {col_name: col_type}}}

        if self.config.get("inference.preload_kb_info", False):
            self._prepare_kb_info()
        self._load_column_types()

        # Offline enriched schema cache (precomputed by preprocess_schema_offline.py)
        # Maps db_id -> {optimized_schema, risks, risk_hints, box_schema}
        self.offline_schema_cache: Dict[str, dict] = {}
        # Offline hints cache: Maps db_id -> list of hint strings
        self.hints_cache: Dict[str, list[str]] = {}
        self._load_offline_schema_cache()

        # KBQA Entity Abstraction Maps (ID <-> Name)
        # Used to hide IDs from LLM and restore them for execution
        self.entity_id_to_name = {}
        self.entity_name_to_id = {}
        if self.task_type == "kbqa":
            self._load_entity_maps()

    @property
    def _profiler(self) -> Optional[_StageProfiler]:
        """Keep profiling state isolated across concurrent samples and votes."""
        return getattr(self._profile_local, "profiler", None)

    @_profiler.setter
    def _profiler(self, value: Optional[_StageProfiler]) -> None:
        self._profile_local.profiler = value

    def _initialize_memory_retriever(self) -> None:
        """Load verified examples without loading the embedding model eagerly."""
        if not self.config.get("retrieval.enabled", True):
            return
        data_root = Path(self.config.get("paths.data_root", "./data"))
        memory_files = self.config.get("retrieval.memory_files", [
            "pandora.memory.db.json",
            "pandora.memory.table.json",
            "pandora.memory.kg.json",
        ])
        paths = [data_root / filename for filename in memory_files]
        examples = load_verified_memory(paths)
        if not examples:
            self.logger.warning("No verified memory examples were found")
            return

        cache_root = Path(self.config.get("paths.cache_root", "./cache"))
        self.memory_retriever = SemanticMemoryRetriever(
            examples=examples,
            model_name=self.config.get(
                "retrieval.model_name", "BAAI/bge-large-en-v1.5"
            ),
            cache_path=cache_root / "bge_memory_index.npz",
        )
        self.logger.info(
            "Loaded %d verified examples for task-agnostic retrieval", len(examples)
        )

    def _get_retrieved_demonstrations(self, data: dict) -> str:
        """Retrieve once per inference run and reuse the same top-K memory items."""
        shot_k = int(data.get("shot_k", self.shot_k) or 0)
        if shot_k <= 0 or self.memory_retriever is None:
            return ""
        context = data.setdefault("context", {})
        cache_key = f"{shot_k}:{self.retrieval_mode}"
        cached = context.get("retrieved_demonstrations")
        if cached and cached.get("key") == cache_key:
            return cached["formatted"]

        retrieved = self.memory_retriever.retrieve(
            question=data.get("question", ""),
            top_k=shot_k,
            target_dataset=self.dataset.name,
            mode=self.retrieval_mode,
        )
        formatted = format_demonstrations(retrieved)
        context["retrieved_demonstrations"] = {
            "key": cache_key,
            "formatted": formatted,
            "examples": [
                {
                    "dataset_id": example.dataset_id,
                    "example_id": example.example_id,
                    "similarity": round(score, 6),
                }
                for example, score in retrieved
            ],
        }
        return formatted

    def _llm_generate(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """
        Wrapper around model.generate() that records prompt length for profiling.
        """
        if self._profiler is not None:
            self._profiler.record_llm_call(prompt, stage_note=kwargs.pop("note", ""))
        return self.model.generate(prompt, system_message=system_message, **kwargs)

    def _load_entity_maps(self):
        """Load ID <-> Name mappings for KBQA entity abstraction.

        Uses dataset-specific entity names file:
        - GrailQA: data/grailqa/entity_names.json
        - WebQSP:  data/webqsp/entity_names.json
        """
        dataset_name = self.dataset.name.lower()
        data_root = Path(self.config.get("paths.data_root", "./data"))
        path = data_root / dataset_name / "entity_names.json"

        if path.exists():
            try:
                with open(path) as f:
                    # Load ID -> Name map
                    self.entity_id_to_name = json.load(f)

                    # Build Name -> ID map for code restoration
                    # If duplicate names exist, the last one wins (usually fine)
                    self.entity_name_to_id = {v: k for k, v in self.entity_id_to_name.items()}

                self.logger.info(f"Loaded {len(self.entity_id_to_name)} entity name mappings")
            except Exception as e:
                self.logger.warning(f"Failed to load entity names: {e}")
        else:
            self.logger.warning(f"Entity names file not found at {path}")

    def _transform_context_for_llm(self, data):
        """
        Replace Entity IDs with Names in context data for LLM visibility.
        Also populates 'entity_map' in context for code restoration.

        Logic:
        1. entity_links: {Mention: ID} -> {Name: Name}.
           Both Key and Value are the Canonical Entity Name.
           We discard the Mention to ensure consistency.
        2. entity_map: {Canonical Name: Entity ID}.
           Strict mapping used for restoring IDs in generated code.
        """
        if not self.entity_id_to_name:
            return data

        context = data.setdefault('context', {})

        # Initialize a fresh entity map for this specific example
        # STRICTLY maps {Canonical Name: Entity ID}
        entity_map = {}

        # 1. Transform entity_links
        # Input format: { "Mention_1": "m.id1", ... }
        original_links = context.get('entity_links', {})
        if original_links:
            for mention, eid in original_links.items():
                # Get the canonical name from our global dictionary
                canonical_name = self.entity_id_to_name.get(eid, eid)

                # For Code Restoration, map Canonical Name -> ID
                # Do NOT store the mention here
                entity_map[canonical_name] = eid

        # 2. Transform table_content
        # Replace IDs with Names in the sample rows text so LLM sees Names
        if 'table_content' in data.get('schema', {}):
            data['schema']['table_content'] = self._replace_ids_with_names(data['schema']['table_content'])

        # Save map back to context
        context['entity_map'] = entity_map
        return data

    def _replace_ids_with_names(self, text):
        """Replace entity IDs with Names in text (e.g., table_content, df_vars)."""
        if not self.entity_id_to_name:
            return text
        # Regex for Freebase IDs: m.XXX or g.XXX
        id_pattern = re.compile(r'\b(m\.[a-zA-Z0-9_]+|g\.[a-zA-Z0-9_]+)\b')

        def replacer(match):
            eid = match.group(0)
            return self.entity_id_to_name.get(eid, eid)

        return id_pattern.sub(replacer, text)

    def _restore_entities_in_code(self, code, data):
        """
        Replace Entity Names with IDs in code before execution.
        Uses the local context map (entity_map) to ensure only relevant entities are replaced.
        """
        # Get the entity map specific to this question from context
        entity_map = data.get('context', {}).get('entity_map', {})

        if not entity_map:
            return code

        # Sort by length descending to replace longer names first
        # This avoids partial replacements (e.g., replacing "King" inside "King Arthur")
        sorted_entities = sorted(entity_map.items(), key=lambda x: -len(x[0]))

        for name, eid in sorted_entities:
            # Simple string replacement is robust enough here because we only have relevant entities
            code = code.replace(name, eid)

        return code

    def _detect_kb_type(self) -> str:
        """Detect knowledge base type from dataset name."""
        dataset_name = self.dataset.name.lower()

        if dataset_name in ["bird", "spider", "spider-syn", "wikisql"]:
            return "db"
        elif dataset_name in ["grailqa", "webqsp", "cwq"]:
            return "kg"
        elif dataset_name in ["wikitq", "wikitablequestions", "tableqa"]:
            return "table"
        elif dataset_name in ["cross_source", "cross-source"]:
            return "multi"
        else:
            self.logger.warning(f"Unknown dataset type: {dataset_name}, defaulting to 'db'")
            return "db"

    def _detect_task_type(self) -> str:
        """Detect task type from dataset name.

        Determines which prompt templates and pipeline steps to use.
        - 'nl2sql': BIRD, Spider - uses DBAnalyzer, ValueCorrector, SQL-based eval
        - 'kbqa': GrailQA, WebQSP - skips DBAnalyzer/ValueCorrector, entity-based eval
        - 'tableqa': WikiTQ, WikiSQL - table-based reasoning
        """
        dataset_name = self.dataset.name.lower()
        if dataset_name in ["grailqa", "webqsp", "cwq"]:
            return "kbqa"
        elif dataset_name in ["wikitq", "wikitablequestions", "wikisql"]:
            return "tableqa"
        else:
            return "nl2sql"

    def _detect_sql_patterns(self, text: str) -> set:
        """Detect SQL/Python patterns present in text (question, code, or SQL)."""
        text_upper = text.upper()
        patterns = set()

        # SQL-level patterns
        if re.search(r'CASE\s+WHEN', text, re.IGNORECASE):
            if 'SUM' in text_upper or 'COUNT' in text_upper:
                patterns.add('conditional_agg')
            else:
                patterns.add('value_mapping')
        if 'IIF' in text_upper:
            patterns.add('value_mapping')

        # Python-level patterns
        if re.search(r'if\s+.+?\s+else\s+', text, re.IGNORECASE):
            # Python if-else expression (ternary or inline)
            if 'sum(' in text.lower() or 'count' in text.lower() or 'len(' in text.lower():
                patterns.add('conditional_agg')
            else:
                patterns.add('value_mapping')

        if 'COUNT' in text_upper and 'DISTINCT' in text_upper:
            patterns.add('count_distinct')
        if '.nunique()' in text:
            patterns.add('count_distinct')
        if 'LIKE' in text_upper:
            patterns.add('like_pattern')
        if '.startswith(' in text or '.str.contains(' in text:
            patterns.add('like_pattern')
        if 'BETWEEN' in text_upper:
            patterns.add('between')
        if re.search(r'\bIN\s*\(', text, re.IGNORECASE):
            patterns.add('subquery')
        if re.search(r'(?:SUM|COUNT)\s*\(.*\)\s*\*?\s*100', text, re.IGNORECASE) or 'percentage' in text.lower():
            patterns.add('percentage')
        if 'GROUP BY' in text_upper and 'ORDER BY' in text_upper and 'LIMIT' in text_upper:
            patterns.add('group_order_limit')
        if '.groupby(' in text and '.sort_values(' in text and '.head(' in text:
            patterns.add('group_order_limit')
        if 'HAVING' in text_upper:
            patterns.add('having')
        if 'UNION' in text_upper:
            patterns.add('union')
        if re.search(r'(?:STRFTIME|JULIANDAY|SUBSTR|SUBSTRING)', text, re.IGNORECASE):
            patterns.add('string_date')
        if '.dt.' in text or 'pd.to_datetime' in text:
            patterns.add('string_date')

        # JOIN detection
        jc = len(re.findall(r'\.merge\(', text)) + len(re.findall(r'\bJOIN\b', text_upper))
        if jc >= 3:
            patterns.add('multi_join')
        elif jc >= 1:
            patterns.add('join')

        # ORDER BY / LIMIT
        if ('ORDER BY' in text_upper or '.sort_values(' in text) and '.head(' in text:
            if 'GROUP BY' not in text_upper and '.groupby(' not in text:
                patterns.add('order_limit')
            elif 'ORDER BY' in text_upper or '.sort_values(' in text:
                patterns.add('group_order_limit')

        # NULL checks
        if 'IS NULL' in text_upper or 'IS NOT NULL' in text_upper or '.notna()' in text or '.isna()' in text:
            patterns.add('null_check')

        # Aggregation
        if '.mean()' in text or 'AVG' in text_upper:
            patterns.add('simple_agg')

        if not patterns:
            patterns.add('basic')

        return patterns

    def _prepare_kb_info(self):
        """Prepare and cache knowledge base information."""
        data_root = Path(self.config.get("paths.data_root", "./data"))

        # Dataset-specific paths for DB type
        dataset_name = self.dataset.name.lower()
        split = "train" if getattr(self.dataset, "stage", "dev") == "train" else "dev"

        # Load column descriptions for DB type
        if self.kb_type == "db":
            if dataset_name in ("spider", "spider-syn"):
                db_dir = data_root / "spider" / f"{split}_database"
            else:
                db_dir = data_root / "bird" / f"{split}_database"
            self.column_descriptions = load_column_descriptions(str(db_dir))
            self.logger.info(f"Loaded {len(self.column_descriptions)} column descriptions")

        # Get schema file path - dataset-specific
        if dataset_name in ("spider", "spider-syn"):
            schema_file = data_root / "spider" / f"spider.tables.{split}.json"
        else:
            schema_file = data_root / "bird" / f"bird.tables.{split}.json"
        schema_data = None
        if schema_file.exists():
            schema_data = {x['db_id']: x for x in load_json(schema_file)}

        # Determine data directory
        if self.kb_type == "db":
            if dataset_name in ("spider", "spider-syn"):
                kb_dir = data_root / "spider" / f"{split}_database"
            else:
                kb_dir = data_root / "bird" / f"{split}_database"
        elif self.kb_type == "kg":
            kb_dir = data_root / "kg"
        else:
            kb_dir = data_root / "tables"

        if not kb_dir.exists():
            self.logger.warning(f"KB directory not found: {kb_dir}")
            return

        self.logger.info(f"Preparing {self.kb_type} info...")

        for kb_id in kb_dir.iterdir():
            if not kb_id.is_dir() and self.kb_type != "table":
                continue

            try:
                if self.kb_type == "db":
                    kb_schema = schema_data.get(kb_id.name) if schema_data else None
                    kb_info = prepare_kb_info(
                        kb_type="db",
                        kb_id=kb_id.name,
                        data_dir=str(kb_dir),
                        schema_data=kb_schema,
                        column_descriptions=self.column_descriptions,
                        top_k_row=3
                    )
                elif self.kb_type == "kg":
                    kb_info = prepare_kb_info(
                        kb_type="kg",
                        kb_id=kb_id.name,
                        data_dir=str(kb_dir),
                        top_k_row=3
                    )
                else:
                    continue

                self.kb_info[kb_id.name] = kb_info

            except Exception as e:
                self.logger.warning(f"Failed to prepare kb_info for {kb_id}: {e}")

        self.logger.info(f"Prepared kb_info for {len(self.kb_info)} knowledge bases")

    def _load_column_types(self):
        """Load column types from dataset-specific schema file.

        Builds a mapping: {db_id: {table_name: {col_name: col_type}}}
        """
        data_root = Path(self.config.get("paths.data_root", "./data"))
        dataset_name = self.dataset.name.lower()
        split = "train" if getattr(self.dataset, "stage", "dev") == "train" else "dev"
        if dataset_name in ("spider", "spider-syn"):
            schema_file = data_root / "spider" / f"spider.tables.{split}.json"
        else:
            schema_file = data_root / "bird" / f"bird.tables.{split}.json"

        if not schema_file.exists():
            self.logger.warning(f"Schema file not found: {schema_file}")
            return

        try:
            schema_list = load_json(schema_file)
            for db_entry in schema_list:
                db_id = db_entry["db_id"]
                table_names = db_entry.get("table_names_original", [])
                column_names_original = db_entry.get("column_names_original", [])
                column_types = db_entry.get("column_types", [])

                db_types = {}
                for idx, (table_idx, col_name) in enumerate(column_names_original):
                    if table_idx < 0:  # Skip wildcard column [-1, "*"]
                        continue
                    if table_idx >= len(table_names):
                        continue
                    table_name = table_names[table_idx]
                    col_type = column_types[idx] if idx < len(column_types) else "unknown"

                    if table_name not in db_types:
                        db_types[table_name] = {}
                    db_types[table_name][col_name] = col_type

                self.column_types_map[db_id] = db_types

            total_tables = sum(len(v) for v in self.column_types_map.values())
            self.logger.info(f"Loaded column types for {len(self.column_types_map)} DBs, {total_tables} tables")
        except Exception as e:
            self.logger.warning(f"Failed to load column types: {e}")

    def _get_column_types_for_db(self, db_id: str) -> dict:
        """Get column types mapping for a specific database."""
        if db_id in self.column_types_map:
            return self.column_types_map[db_id]
        return {}

    def _build_schema_metadata_from_tables(self, processed: dict) -> str:
        """
        Build schema metadata string from actual table data for KBQA/TableQA.
        """
        context = processed.get('context', {})
        table_df = context.get('table_df')

        if table_df is not None and not table_df.empty:
            # TableQA (WikiTQ): single embedded table
            lines = []
            lines.append("table = pd.DataFrame({")
            for col in table_df.columns:
                dtype = str(table_df[col].dtype)
                sample = table_df[col].dropna().head(3).tolist()
                sample_str = ", ".join(str(v) for v in sample[:3])
                lines.append(f'    "{col}": [],  # ({dtype}), sample: {sample_str}')
            lines.append("})")
            return "\n".join(lines)

        # KBQA (GrailQA): use box_schema
        box_schema = processed.get('schema', {}).get('box_schema', '')
        return box_schema

    # ── Offline Schema Cache ──────────────────────────────────────────────

    def _load_offline_schema_cache(self):
        """
        Load precomputed optimized_schema from data/{dataset}/box/ directory.
        Only uses optimized_schema — risks are not loaded.
        """
        if self.task_type != "nl2sql":
            return

        data_root = Path(self.config.get("paths.data_root", "./data"))
        dataset_name = self.dataset.name.lower()

        # Determine box directory: data/bird/box/ or data/spider/box/
        if dataset_name == "bird":
            box_dir = data_root / "bird" / "box"
        elif dataset_name in ("spider", "spider-syn"):
            box_dir = data_root / "spider" / "box"
        else:
            return

        if not box_dir.exists():
            self.logger.info(f"[Offline schema] Box directory not found: {box_dir}")
            return

        json_files = sorted(box_dir.glob("*.json"))
        if not json_files:
            self.logger.info(f"[Offline schema] No precomputed schemas in {box_dir}")
            return

        for jf in json_files:
            db_id = jf.stem
            try:
                cached = load_json(jf)
                opt_schema = cached.get("optimized_schema", {})
                if opt_schema:
                    self.offline_schema_cache[db_id] = opt_schema
                    n_tables = len(opt_schema)
                    n_cols = sum(len(cols) for cols in opt_schema.values())
                    self.logger.debug(f"[Offline schema] Loaded {db_id}: {n_tables} tables, {n_cols} cols")

                # Load hints if present
                hints = cached.get("hints", [])
                if hints:
                    self.hints_cache[db_id] = hints
                    self.logger.debug(f"[Offline schema] Loaded {len(hints)} hint(s) for {db_id}")
            except Exception as e:
                self.logger.warning(f"[Offline schema] Failed to load {jf}: {e}")

        self.logger.info(
            f"[Offline schema] Loaded {len(self.offline_schema_cache)} precomputed schemas "
            f"from {box_dir}"
        )

    # ── Schema Analysis ───────────────────────────────────────────────────

    def _analyze_and_enrich_schema(self, processed: dict, db_id: str) -> dict:
        """
        Step 1.5: Schema enrichment for all task types.

        By default, schema analysis is question-specific for every task. An
        offline NL2SQL schema cache can be enabled explicitly for legacy runs.

        Produces:
        - enriched_schema: {table_name: {col_name: {"type": str, "meaning": str}}}
        - risk_hints: Always empty string (risk analysis removed)
        """
        use_offline_cache = self.config.get("schema_analysis.use_offline_cache", False)
        if use_offline_cache and self.task_type == "nl2sql" and db_id in self.offline_schema_cache:
            opt_schema = self.offline_schema_cache[db_id]
            self.logger.info(f"[Step 1.5/6] Using OFFLINE optimized_schema for {db_id} "
                             f"({len(opt_schema)} tables, {sum(len(v) for v in opt_schema.values())} cols)")

            enriched = {}
            for table_name, columns in opt_schema.items():
                enriched[table_name] = {}
                for col_name, meaning in columns.items():
                    enriched[table_name][col_name] = {
                        "type": "unknown",  # type will be inferred from box_schema comments
                        "meaning": meaning,
                    }

            return {
                "enriched_schema": enriched,
                "risk_hints": "",  # risk analysis removed
            }

        # ── Fallback: online LLM analysis (KBQA/TableQA or missing offline cache) ──
        question = processed.get("question", "")
        evidence = processed.get("evidence", "")
        db_path = processed.get("context", {}).get("db_path", "")
        box_schema = processed.get("schema", {}).get("box_schema", "")

        if not db_path or not Path(db_path).exists():
            box_schema = self._build_schema_metadata_from_tables(processed) or box_schema

        prompt_root = self.config.get("paths.prompt_root", "./prompts")
        task = self.task_type if self.task_type in ("kbqa", "tableqa") else "nl2sql"
        template_dir = str(Path(prompt_root) / "tasks" / task)

        # Analyzer instances are per-run because samples may execute concurrently.
        schema_analyzer = SchemaAnalyzer(
            model=self.model,
            db_path=db_path if db_path and Path(db_path).exists() else None,
            top_k=2,
            template_dir=template_dir,
            entity_id_to_name=self.entity_id_to_name if self.task_type == "kbqa" else None,
            generate_fn=self._llm_generate,
        )

        try:
            self.logger.info(f"[Step 1.5/6] Analyzing schema ONLINE (task_type={self.task_type})")

            # For KBQA, pass kg_dir so SchemaAnalyzer can read CSV sample data
            kg_dir = None
            if self.task_type == "kbqa":
                kg_dir = processed.get('context', {}).get('kg_dir')

            result = schema_analyzer.analyze(
                question=question,
                evidence=evidence,
                box_schema=box_schema,
                kg_dir=kg_dir,
                few_shot_examples=self._get_retrieved_demonstrations(processed),
            )

            enriched = result.get("enriched_schema", {})
            risk_hints = result.get("risk_hints", "")

            # Fallback to parsing box_schema if LLM returned nothing
            if not enriched:
                enriched = self._parse_schema_to_structured(box_schema)

            return {
                "enriched_schema": enriched,
                "risk_hints": risk_hints,
            }

        except Exception as e:
            self.logger.warning(f"Schema analysis failed: {e}")
            enriched = self._parse_schema_to_structured(box_schema)
            return {"enriched_schema": enriched, "risk_hints": ""}

    def _parse_schema_to_structured(self, box_schema: str) -> dict:
        """Parse original box_schema into structured format (fallback when analysis fails).

        Extracts column type and meaning from comments: "# (TYPE), meaning"

        Returns:
            {table_name: {col_name: {"type": str, "meaning": str}}}
        """
        enriched = {}
        current_table = None
        for line in box_schema.split('\n'):
            stripped = line.strip()
            table_match = re.match(r'^(\w+)\s*=\s*pd\.DataFrame\(\{', stripped)
            if table_match:
                current_table = table_match.group(1)
                enriched[current_table] = {}
                continue

            col_match = re.match(r'^(\s*)"([^"]+)":\s*\[\](?:,\s*)?(#.*)?$', line)
            if col_match and current_table:
                col_name = col_match.group(2)
                comment = (col_match.group(3) or "").lstrip(',').lstrip(' ').strip()

                # Extract type and meaning from comment: "# (TYPE), meaning"
                type_match = re.match(r'^#\s*\(([^)]+)\)\s*,?\s*(.*)', comment)
                if type_match:
                    col_type = type_match.group(1)
                    meaning = type_match.group(2).strip() if type_match.group(2).strip() else col_name
                else:
                    col_type = "unknown"
                    meaning = comment.lstrip('#').strip() if comment else col_name

                enriched[current_table][col_name] = {
                    "type": col_type,
                    "meaning": meaning,
                }

        return enriched

    def _parse_optimized_box_schema(self, optimized_box_schema: str) -> dict:
        """Parse optimized box_schema string into structured enriched_schema.

        The optimized box_schema already contains both column type and LLM-optimized
        meaning in comments: "col": [],  # (TYPE), optimized_meaning

        Returns:
            {table_name: {col_name: {"type": str, "meaning": str}}}
        """
        enriched = {}
        current_table = None
        for line in optimized_box_schema.split('\n'):
            stripped = line.strip()
            table_match = re.match(r'^(\w+)\s*=\s*pd\.DataFrame\(\{', stripped)
            if table_match:
                current_table = table_match.group(1)
                enriched[current_table] = {}
                continue

            col_match = re.match(r'^(\s*)"([^"]+)":\s*\[\](?:,\s*)?(#.*)?$', line)
            if col_match and current_table:
                col_name = col_match.group(2)
                comment = (col_match.group(3) or "").lstrip(',').lstrip(' ').strip()

                # Extract type and meaning from comment: "# (TYPE), meaning"
                type_match = re.match(r'^#\s*\(([^)]+)\)\s*,?\s*(.*)', comment)
                if type_match:
                    col_type = type_match.group(1)
                    meaning = type_match.group(2).strip() if type_match.group(2).strip() else col_name
                else:
                    col_type = "unknown"
                    meaning = comment.lstrip('#').strip() if comment else col_name

                enriched[current_table][col_name] = {
                    "type": col_type,
                    "meaning": meaning,
                }

        return enriched

    def _build_box_schema(self, enriched_schema: dict, tables: Optional[set[str]] = None) -> str:
        """Build box_schema string from structured enriched schema.

        Args:
            enriched_schema: {table_name: {col_name: {"type": str, "meaning": str}}}
            tables: Set of table names to include (None = all tables)

        Returns:
            Formatted box_schema string.
        """

        lines = []
        for table_name, columns in enriched_schema.items():
            if tables is not None and table_name.lower() not in tables:
                continue
            lines.append(f"{table_name} = pd.DataFrame({{")
            for col_name, info in columns.items():
                col_type = info.get("type", "unknown")
                meaning = info.get("meaning", col_name)
                if col_type and col_type != "unknown":
                    comment = f"  # ({col_type}), {meaning}"
                else:
                    comment = f"  # {meaning}"
                lines.append(f'    "{col_name}": [],{comment}')
            lines.append("})")
            lines.append("")
        return '\n'.join(lines)

    def _load_kb_info_on_demand(self, kb_id: str):
        """Load KB info on-demand if not in cache."""
        data_root = Path(self.config.get("paths.data_root", "./data"))
        dataset_name = self.dataset.name.lower()
        split = "train" if getattr(self.dataset, "stage", "dev") == "train" else "dev"

        if self.kb_type == "db":
            if dataset_name in ("spider", "spider-syn"):
                kb_dir = data_root / "spider" / f"{split}_database"
                schema_file = data_root / "spider" / f"spider.tables.{split}.json"
            else:
                kb_dir = data_root / "bird" / f"{split}_database"
                schema_file = data_root / "bird" / f"bird.tables.{split}.json"
            kb_schema = None
            if schema_file.exists():
                schema_data = {x['db_id']: x for x in load_json(schema_file)}
                kb_schema = schema_data.get(kb_id)

            kb_info = prepare_kb_info(
                kb_type="db",
                kb_id=kb_id,
                data_dir=str(kb_dir),
                schema_data=kb_schema,
                column_descriptions=self.column_descriptions,
                top_k_row=3
            )
        elif self.kb_type == "kg":
            kb_dir = data_root / "kg"
            kb_info = prepare_kb_info(
                kb_type="kg",
                kb_id=kb_id,
                data_dir=str(kb_dir),
                top_k_row=3
            )
        else:
            return

        self.kb_info[kb_id] = kb_info

    def _fallback_direct_llm_answer_tableqa(self, data: dict, processed: dict) -> dict:
        """
        WikiTQ Fallback: When code generation and repair both fail,
        directly ask the LLM to answer the question based on the table content.

        Uses Chain-of-Thought (CoT) reasoning before extracting the answer.

        Returns:
            Updated data dict with parsed answer.
        """
        question = processed.get('question', '')

        # Get FULL table content for fallback (critical for LLM to answer correctly)
        # Standard table_content only has first 5 rows, which is insufficient for many questions.
        df = processed.get('context', {}).get('table_df')
        if df is not None and not df.empty:
            table_content = df.to_string(index=False)
        else:
            table_content = processed.get('schema', {}).get('table_content', '')

        if not question or not table_content:
            self.logger.warning("No question or table content for direct LLM fallback")
            return data

        # Load prompt template
        prompt_root = self.config.get("paths.prompt_root", "./prompts")
        template_path = Path(prompt_root) / "tasks" / "tableqa" / "direct_answer.txt"

        if template_path.exists():
            template = template_path.read_text(encoding="utf-8")
            prompt = template.replace("{{question}}", question).replace("{{table_content}}", table_content)
        else:
            self.logger.warning("Direct answer template not found, using fallback")
            return data

        try:
            self.logger.info("Calling LLM for direct answer fallback (CoT)")
            response = self._llm_generate(prompt)

            # Parse JSON response (same pattern as final_execution_feedback and other LLM parsing)
            parsed = self._extract_json_from_response(response)

            thinking = parsed.get("thinking", "")
            raw_answer = parsed.get("answer", "")

            # Parse answer into standard WikiTQ format
            answer_list = self._parse_fallback_answer(raw_answer)

            self.logger.info(f"CoT thinking: {thinking[:100]}...")
            self.logger.info(f"Raw answer: {raw_answer}")
            self.logger.info(f"Parsed answer: {answer_list}")

            # Update data with the fallback result
            data['python_results'] = {
                'success': True,
                'result': answer_list,
                'is_empty': len(answer_list) == 0,
            }
            data['python_exception'] = False
            data['fallback_llm_thinking'] = thinking
            data['fallback_llm_raw_answer'] = raw_answer
            data['fallback_llm_parsed'] = answer_list

        except Exception as e:
            self.logger.warning(f"Direct LLM fallback failed: {e}")
            data['fallback_llm_error'] = str(e)

        return data

    def _fallback_direct_llm_answer_kbqa(self, data: dict, processed: dict) -> dict:
        """
        KBQA Fallback: When code generation and repair both fail,
        directly ask the LLM to answer the question based on the raw CSV tables.

        Hides Entity IDs in tables (replaces with Names), asks LLM for Names,
        then restores IDs in the final answer.

        Returns:
            Updated data dict with parsed answer.
        """
        question = processed.get('question', '')
        kg_dir = processed.get('context', {}).get('kg_dir', '')

        if not question or not kg_dir:
            self.logger.warning("No question or KG dir for KBQA direct LLM fallback")
            return data

        kg_path = Path(kg_dir)
        if not kg_path.exists():
            self.logger.warning(f"KG directory not found: {kg_dir}")
            return data

        # Get entity map (Name -> ID) for restoring answers later
        entity_map = processed.get('context', {}).get('entity_map', {})
        # Build ID -> Name map for hiding IDs in tables
        id_to_name = {v: k for k, v in entity_map.items()}
        # Add global map as fallback
        if hasattr(self, 'entity_id_to_name'):
            id_to_name.update(self.entity_id_to_name)

        # Load and truncate CSVs, hiding IDs
        tables_content = []
        csv_files = sorted(kg_path.glob("*.csv"))
        for csv_file in csv_files:
            table_name = csv_file.stem
            try:
                # Read first 100 rows
                df = pd.read_csv(csv_file, dtype=str).head(100)
                content_str = df.to_string(index=False)
                # Hide IDs: replace ID with Name###ID
                # Sort by length descending to replace longer IDs first
                sorted_ids = sorted(id_to_name.items(), key=lambda x: -len(x[0]))
                for eid, name in sorted_ids:
                    content_str = content_str.replace(eid, f"{name}###{eid}")
                tables_content.append(f"### Table: {table_name}\n```\n{content_str}\n```")
            except Exception as e:
                self.logger.warning(f"Failed to load {csv_file}: {e}")

        if not tables_content:
            self.logger.warning("No CSV tables found for fallback")
            return data

        tables_text = "\n\n".join(tables_content)

        # Prepare Entity Links text (Names only)
        el_text = ""
        if entity_map:
            el_text = "## Relevant Entities\n"
            for name in entity_map.keys():
                el_text += f"- **{name}**\n"
            el_text += "\n"

        # Load prompt template
        prompt_root = self.config.get("paths.prompt_root", "./prompts")
        template_path = Path(prompt_root) / "tasks" / "kbqa" / "direct_answer.txt"

        if template_path.exists():
            template = template_path.read_text(encoding="utf-8")
            prompt = template.replace("{{question}}", question) \
                             .replace("{{entity_links_section}}", el_text) \
                             .replace("{{tables_content}}", tables_text)
        else:
            self.logger.warning("KBQA Direct answer template not found")
            return data

        try:
            self.logger.info("Calling LLM for KBQA direct answer fallback (Name###ID)")
            response = self._llm_generate(prompt)

            # Parse JSON response
            parsed = self._extract_json_from_response(response)

            thinking = parsed.get("thinking", "")
            raw_answer = parsed.get("answer", "")

            # Robust parsing: raw_answer might be a parsed list, or a string representation
            # e.g. "[['Apple###m.0123'], ...]" or "[['Apple###m.0123'], ...]"
            import ast
            answer_list = []

            if isinstance(raw_answer, list):
                answer_list = raw_answer
            elif isinstance(raw_answer, str):
                try:
                    parsed_obj = json.loads(raw_answer)
                    if isinstance(parsed_obj, list):
                        answer_list = parsed_obj
                    else:
                        answer_list = [[parsed_obj]]
                except json.JSONDecodeError:
                    try:
                        if raw_answer.strip().startswith('['):
                            parsed_obj = ast.literal_eval(raw_answer.strip())
                            if isinstance(parsed_obj, list):
                                answer_list = parsed_obj
                            else:
                                answer_list = [[parsed_obj]]
                        else:
                            answer_list = [[raw_answer]]
                    except:
                        answer_list = [[raw_answer]]

            self.logger.info(f"Parsed answer structure: {answer_list}")

            # Restore IDs from Name###ID format
            final_answer_list = []
            for item in answer_list:
                restored_row = []

                # Ensure item is a list
                if not isinstance(item, list):
                    item = [item]

                for val in item:
                    val_str = str(val)
                    # Clean up quotes/brackets that might have leaked from parsing
                    # Remove [, ], ', ", whitespace
                    val_str = re.sub(r"[\[\]'\"]", "", val_str).strip()

                    if '###' in val_str:
                        eid = val_str.split('###')[-1]
                        eid = re.sub(r"[\[\]'\"]", "", eid).strip()
                        if eid:
                            restored_row.append(eid)
                    else:
                        # If no ###, check if it's already an ID
                        if re.match(r'^[mg]\.\w+', val_str):
                            restored_row.append(val_str)
                        else:
                            eid = entity_map.get(val_str, val_str)
                            restored_row.append(eid)

                if restored_row:
                    final_answer_list.append(restored_row)

            self.logger.info(f"Final answer (IDs): {final_answer_list}")

            # Update data with the fallback result
            data['python_results'] = {
                'success': True,
                'result': final_answer_list,
                'is_empty': len(final_answer_list) == 0,
            }
            data['python_exception'] = False
            data['fallback_llm_thinking'] = thinking
            data['fallback_llm_raw_answer'] = raw_answer
            data['fallback_llm_parsed'] = final_answer_list

        except Exception as e:
            self.logger.warning(f"KBQA Direct LLM fallback failed: {e}")
            data['fallback_llm_error'] = str(e)

        return data

    def _validate_tableqa_answer(self, data: dict, processed: dict) -> dict:
        """
        WikiTQ Answer Validation: Sanity check the Python-generated answer using LLM.
        If invalid, triggers direct LLM fallback.
        """
        question = processed.get('question', '')
        py_result = data.get('python_results', {}).get('result')
        if not py_result:
            return data  # Let fallback handle empty later

        answer_str = str(py_result)

        prompt_root = self.config.get("paths.prompt_root", "./prompts")
        template_path = Path(prompt_root) / "tasks" / "tableqa" / "answer_validation.txt"
        if not template_path.exists():
            return data

        prompt = template_path.read_text(encoding="utf-8").replace("{{question}}", question).replace("{{answer}}", answer_str)

        try:
            self.logger.info("Validating WikiTQ answer with LLM...")
            response = self._llm_generate(prompt)
            parsed = self._extract_json_from_response(response)

            if not parsed.get("valid", True):
                self.logger.info(f"WikiTQ Validation Failed: {parsed.get('reason')}")
                return self._fallback_direct_llm_answer_tableqa(data, processed)
        except Exception as e:
            self.logger.warning(f"Answer validation failed: {e}")

        return data

    @staticmethod
    def _extract_json_from_response(response: str) -> dict:
        """
        Extract JSON from LLM response text.
        Same pattern used in final_execution_feedback and other LLM parsing methods.
        """
        # Strategy 1: JSON code block
        json_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 2: Direct JSON object by brace matching
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
                            return json.loads(response[start_idx : i + 1])
                        except json.JSONDecodeError:
                            break

        return {}

    @staticmethod
    def _parse_fallback_answer(raw_answer: str) -> list:
        """
        Parse the raw answer string into the standard WikiTQ answer format.

        Input can be:
        - A single value: "Brazil" → [["Brazil"]]
        - A JSON array: '["Brazil", "France"]' → [["Brazil"], ["France"]]
        - A comma-separated list: "Brazil, France" → [["Brazil"], ["France"]]
        - Already a list: ["Brazil"] → [["Brazil"]]

        Returns:
            List of single-value lists: [[val1], [val2], ...]
        """
        if not raw_answer:
            return []

        # Handle case where answer is already a list
        if isinstance(raw_answer, list):
            return [[str(v)] for v in raw_answer if v]

        # Handle case where answer is not a string (shouldn't happen but be safe)
        if not isinstance(raw_answer, str):
            return [[str(raw_answer)]]

        raw_answer = raw_answer.strip()
        if not raw_answer:
            return []

        # Try to parse as JSON array first
        if raw_answer.startswith('['):
            try:
                parsed = json.loads(raw_answer)
                if isinstance(parsed, list):
                    return [[str(v)] for v in parsed if v]
                elif parsed:
                    return [[str(parsed)]]
            except json.JSONDecodeError:
                pass

        # Comma-separated list
        if ',' in raw_answer:
            return [[v.strip()] for v in raw_answer.split(',') if v.strip()]

        # Single value
        return [[raw_answer]]

    def _execute_gold_sql(self, gold_sql: str, context: dict) -> list:
        """Execute gold SQL to get the actual answer for evaluation."""
        db_path = context.get("db_path", "")
        if not db_path or not os.path.exists(db_path):
            return []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(gold_sql)
            rows = cursor.fetchall()
            conn.close()
            return [list(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to execute gold SQL: {e}")
            return []

    def run(self, example: dict, shot_k: int = 10) -> dict[str, Any]:
        """
        Complete inference flow.

        Pipeline:
        Phase A  — prepare + schema analysis (Steps 1, 1.5)
        Phase B  — schema linking + table resolution (Steps 2a, 2b)
        Phase C  — task decomposition (Step 2c)
        Phase D  — subtask code reasoning + execution feedback (Step 3)
        Phase E  — LLM code merge with self-check (Step 4)
        Phase F  — merged code execution + LLM repair (Step 5)
        Phase G  — task-specific fallbacks
        Phase H  — post-process + evaluate
        """
        example_id = example.get('qid', example.get('id', example.get('question_id', 'unknown')))
        self.logger.info(f"Running inference for example {example_id}")

        # ── Stage Profiling init ──
        self._profiler = _StageProfiler()

        # Phase A: Prepare + Schema Analysis
        processed, data = self._prepare_and_analyze(example, example_id, shot_k)

        # Propagate shot_k for downstream few-shot formatting
        data['shot_k'] = shot_k

        # Phase B: Schema Linking + Linked Table Resolution
        data = self._schema_linking(data)

        # Phase C: Task Decomposition
        data = self._task_decomposition(data)

        # Phase D: Subtask Code Reasoning + Execution Feedback
        data = self._subtask_pipeline(data)

        # Phase E: LLM Code Merge with Self-Check
        data = self._code_merge_and_self_check(data)

        # Phase F: Execute Merged Code + LLM Repair
        data = self._execute_and_retry(data)

        # Phase G: Task-Specific LLM Fallbacks
        data = self._task_fallbacks(data, processed)

        # Phase H: Post-Process + Evaluate
        output, gold_answer, gold_sql, metrics = self._evaluate(data, processed, example)

        return self._build_result(example_id, data, output, metrics, gold_answer, example, self._profiler.summary(), gold_sql)

    # ──────────────────────────────────────────────────────────────
    # Pipeline Phases
    # ──────────────────────────────────────────────────────────────

    def _prepare_and_analyze(self, example: dict, example_id: str, shot_k: int) -> tuple:
        """Phase A: Prepare example + schema enrichment (Steps 1, 1.5).

        Returns:
            (processed, data) — raw processed dict and the working data dict
        """
        # Step 1: Prepare example
        self._profiler.start("prepare")
        self.logger.info(f"[Step 1/6] Preparing example {example_id}")
        processed = self._prepare_example(example)
        processed["shot_k"] = shot_k
        self._profiler.end()

        # Step 1.5: Schema Analysis
        self._profiler.start("schema_analysis")
        if not self.config.get("schema_analysis.enabled", True):
            schema_analysis = {"enriched_schema": {}, "risk_hints": ""}
            self.logger.info("[Step 1.5/6] SchemaAnalyzer disabled by configuration")
        else:
            self.logger.info(f"[Step 1.5/6] Analyzing and enriching schema")
            db_id = processed.get('db_id', '')
            schema_analysis = self._analyze_and_enrich_schema(processed, db_id)
        self._profiler.end()

        data = processed.copy()
        data['enriched_schema'] = schema_analysis.get('enriched_schema', {})
        data['context']['schema_risks'] = schema_analysis.get('risk_hints', '')

        return processed, data

    def _schema_linking(self, data: dict) -> dict:
        """Phase B: Schema linking + linked table resolution (Steps 2a, 2b).

        - Runs schema_linking to identify relevant tables/columns/values
        - Transforms KBQA entity IDs → Names for LLM readability
        - Unifies risks from DB analysis and schema linking
        - Resolves bridge tables via FK graph traversal
        """
        # Step 2a: Schema Linking + Value Linking
        self._profiler.start("schema_linking")
        self.logger.info(f"[Step 2a/6] Schema linking and value linking")
        data = self.schema_linking(data)
        self._profiler.end()

        # KBQA Entity Abstraction: Transform remaining IDs → Names
        if self.task_type == "kbqa":
            self._transform_context_for_llm(data)

        # Step 2a.1: Unify risks
        self._unify_risks(data)

        # Step 2b: Resolve linked + bridge tables
        self._profiler.start("bridge_tables")
        schema_and_value_linking = data.get('schema_and_value_linking', {})
        schema_linking = schema_and_value_linking.get('schema_linking', [])
        linked_tables = self._extract_linked_tables(schema_linking)

        if linked_tables and schema_and_value_linking.get('schema_linking'):
            self.logger.info(f"[Step 2b/6] Using linked tables: {linked_tables}")
            foreign_keys = data.get('schema', {}).get('foreign_keys', '')
            bridge_tables = self._find_bridge_tables(linked_tables, foreign_keys)
            if bridge_tables:
                linked_tables = linked_tables | bridge_tables
                self.logger.info(f"[Step 2b/6] Added bridge tables: {bridge_tables}")
            data['linked_tables'] = linked_tables
        else:
            self.logger.warning(f"[Step 2b/6] Schema linking failed or empty, using full schema")
            data['linked_tables'] = None
        self._profiler.end()

        return data

    def _task_decomposition(self, data: dict) -> dict:
        """Phase C: Task decomposition (Step 2c).

        Breaks the question into subtasks using linked schema context.
        """
        self._profiler.start("task_decomposition")
        if not self.do_decomposition:
            question = data.get("question", "")
            data["task_decomposition"] = {
                "task_decomposition": [
                    "Solve the complete question in one code-generation step: "
                    f"{question} # FINAL SUBTASK: assign the complete answer to `result`."
                ],
                "thinking": "Step-wise decomposition disabled by ablation setting.",
            }
            data["accumulated_code"] = ""
            self._profiler.end()
            return data
        self.logger.info(f"[Step 2c/6] Task decomposition")
        data = self.task_decomposition(data)
        self._profiler.end()
        return data

    def _subtask_pipeline(self, data: dict) -> dict:
        """Phase D: Subtask code reasoning + execution feedback (Step 3).

        For each subtask:
        1. code_reasoning — LLM generates Pandas code
        2. execution_feedback — execute + auto-repair on error (if enabled)
        """
        n_subtasks = len(data.get('task_decomposition', {}).get('task_decomposition', []))
        self.logger.info(f"[Step 3/6] Generating code for {n_subtasks} subtasks with execution feedback")

        for i in range(n_subtasks):
            self._profiler.start(f"code_reasoning_subtask_{i+1}")
            data = self.code_reasoning(task_id=i, data=data)
            self._profiler.end()

            if self.do_execution_guidance:
                self._profiler.start(f"execution_feedback_subtask_{i+1}")
                data = self.execution_feedback(task_id=i, data=data, accumulated_code=data['accumulated_code'])
                self._profiler.end()

        return data

    def _code_merge_and_self_check(self, data: dict) -> dict:
        """Phase E: LLM-based code merge with self-check against known failure patterns.

        Takes accumulated_code from subtask pipeline, asks LLM to:
        1. Fuse subtask codes into a coherent merged program
        2. Self-check against 8 known failure patterns (numeric string comparison,
           missing aggregation, date format, LIKE fuzzy match, etc.)
        3. Fix any issues found
        """
        self._profiler.start("code_merge")
        if not self.do_code_merge:
            data["merged_code"] = self._extract_pure_code(
                data.get("accumulated_code", "")
            )
            data["merged_thinking"] = (
                "Subtask code was concatenated without LLM merging (ablation)."
            )
            self._profiler.end()
            return data
        self.logger.info(f"[Step 4a/6] LLM code merge with self-check")
        data = self.code_merge(data)
        self._profiler.end()
        return data

    def _execute_and_retry(self, data: dict) -> dict:
        """Phase F: Execute merged code on full data.

        1. Try merged_code (from code_merge) directly on full data
        2. If success → use it
        3. If failure → repair with traceback feedback for at most L rounds
        """
        merged_code = data.get('merged_code', '')
        if not merged_code or not merged_code.strip():
            # Fallback: if no merged_code, try accumulated_code directly
            accumulated_code = data.get('accumulated_code', '')
            merged_code = self._extract_pure_code(accumulated_code)
            data['merged_code'] = merged_code
            data['merged_thinking'] = data.get('merged_thinking', '') or "Used accumulated_code directly (no code_merge)"
            data['merged_from_accumulated'] = True

        pure_code = merged_code.strip()

        self._profiler.start("merged_try")
        self.logger.info(f"[Step 5/6] Trying merged_code directly ({len(pure_code)} chars)")
        context = data.get("context", {})
        code_to_execute = self._restore_entities_in_code(pure_code, data) if self.task_type == "kbqa" else pure_code
        direct_result = self.executor.execute(code_to_execute, context, use_full_code=True, top_k_row=None)
        self._profiler.end()

        if direct_result.success:
            self.logger.info(f"[Step 5/6] Merged code succeeded on full data")
            data['merged_code'] = pure_code
            data['python_results'] = direct_result.to_dict()
            data['python_exception'] = False
        else:
            reason = f"error: {direct_result.error}" if direct_result.error else "empty result"
            self.logger.warning(f"[Step 5/6] Merged code failed ({reason})")

            data['python_exception'] = True
            data['python_exception_msg'] = direct_result.error or "empty result"
            data['python_results'] = direct_result.to_dict()
            if self.do_final_execution_repair:
                self._profiler.start("final_execution_repair")
                data = self.final_execution_feedback(data)
                self._profiler.end()

        return data

    def _task_fallbacks(self, data: dict, processed: dict) -> dict:
        """Phase F: Task-specific LLM fallbacks.

        - WikiTQ: Validate answer → fallback to direct LLM if code fails
        - KBQA: Fallback to direct LLM on CSV tables if code fails
        """
        # WikiTQ Answer Validation (LLM sanity check)
        if self.task_type == "tableqa":
            self._profiler.start("answer_validation")
            data = self._validate_tableqa_answer(data, processed)
            self._profiler.end()

        # WikiTQ Fallback
        if self.task_type == "tableqa":
            exec_ok = data.get('python_results', {}).get('success', False)
            has_answer = bool(data.get('python_results', {}).get('result'))
            if not exec_ok or not has_answer:
                self._profiler.start("wikitq_llm_fallback")
                self.logger.info(f"[WikiTQ Fallback] Code execution failed/empty, using LLM direct answer")
                data = self._fallback_direct_llm_answer_tableqa(data, processed)
                self._profiler.end()

        # KBQA Fallback
        if self.task_type == "kbqa":
            exec_ok = data.get('python_results', {}).get('success', False)
            has_answer = bool(data.get('python_results', {}).get('result'))
            if not exec_ok or not has_answer:
                self._profiler.start("kbqa_llm_fallback")
                self.logger.info(f"[KBQA Fallback] Code execution failed/empty, using LLM direct answer on CSVs")
                data = self._fallback_direct_llm_answer_kbqa(data, processed)
                self._profiler.end()

        return data

    def _evaluate(self, data: dict, processed: dict, example: dict) -> tuple:
        """Phase G: Post-process + evaluate (Step 6).

        Returns:
            (output, gold_answer, gold_sql, metrics)
        """
        # Postprocess
        output = self.dataset.postprocess(data.get('python_results', []), processed)

        # Dataset-specific answer post-processing
        if self.task_type == "tableqa" and self.dataset.name == "wikitq":
            output["answer"] = postprocess_wikitq_answer(output["answer"])
        else:
            output["answer"] = postprocess_answer(output["answer"])

        # Evaluate
        self._profiler.start("evaluation")
        gold_sql = ""
        if self.task_type in ("kbqa", "tableqa") or self.kb_type == "multi":
            self.logger.info(f"[Step 6/6] Evaluating ({self.task_type}: set-based EM/F1)")
            gold_answer = self.dataset.get_gold_answer(example) if hasattr(self.dataset, 'get_gold_answer') else []
            if gold_answer:
                self.logger.info(f"Gold answer: {len(gold_answer)} entries")
        else:
            self.logger.info(f"[Step 6/6] Evaluating (NL2SQL: gold SQL execution)")
            gold_sql = example.get("SQL", "") or example.get("query", "") or example.get("label", "")
            gold_answer = []
            if gold_sql:
                try:
                    gold_answer = self._execute_gold_sql(gold_sql, processed.get("context", {}))
                    self.logger.info(f"Gold SQL executed: {len(gold_answer)} rows")
                except Exception as e:
                    self.logger.warning(f"Failed to execute gold SQL: {e}")
                    gold_answer = []

        metrics = self.dataset.evaluate(output["answer"], gold_answer)
        self._profiler.end()

        return output, gold_answer, gold_sql, metrics

    def _prepare_example(self, example: dict) -> dict:
        """Preprocess example using dataset-specific logic.

        For KBQA (GrailQA): routes to dataset.preprocess() which handles
        subgraph CSV loading, box_schema, entity linking, etc.
        For NL2SQL/TableQA: uses cached kb_info from _prepare_kb_info().
        """
        # KBQA/TableQA: use dataset.preprocess() which handles embedded tables
        if self.task_type in ("kbqa", "tableqa"):
            processed = self.dataset.preprocess(example)
            # Ensure required keys exist for pipeline compatibility
            processed.setdefault('db_id', processed.get('example_id', 'unknown'))
            processed.setdefault('evidence', '')

            return processed

        # NL2SQL/TableQA: use cached kb_info
        kb_id = example.get('db_id', example.get('kg_id', example.get('table_id', 'california_schools')))

        if kb_id not in self.kb_info:
            self.logger.warning(f"kb_info not found for kb_id={kb_id}, loading on-demand")
            self._load_kb_info_on_demand(kb_id)

        kb_info = self.kb_info.get(kb_id, {})

        schema = {
            "box_schema": kb_info.get("box_schema", ""),
            "table_content": kb_info.get("table_content", ""),
            "primary_keys": kb_info.get("primary_keys", ""),
            "foreign_keys": kb_info.get("foreign_keys", ""),
            "tables": kb_info.get("table_df", {}),
            "kb_type": kb_info.get("kb_type", self.kb_type),
        }

        return {
            "question": example.get('question', ''),
            "schema": schema,
            "context": {
                "db_path": kb_info.get("db_path", ""),
                "kb_type": self.kb_type,
            },
            "evidence": example.get('evidence', ''),
            "db_id": kb_id,
        }

    @staticmethod
    def _normalize_prompt_whitespace(prompt: str) -> str:
        """
        Normalize whitespace in a rendered prompt.

        Rules:
        - No more than 2 consecutive blank lines between blocks
        - Strip trailing whitespace from each line
        - Strip leading/trailing whitespace from the whole prompt
        """
        import re
        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in prompt.split('\n')]
        prompt = '\n'.join(lines)
        # Collapse 3+ consecutive newlines to exactly 2
        prompt = re.sub(r'\n{3,}', '\n\n', prompt)
        return prompt.strip()

    def _build_prompt(
        self,
        template,
        **render_kwargs,
    ) -> str:
        """
        Build a rendered prompt. schema_risks is passed as a template variable.
        Templates should use {{schema_risks}} to render risks.
        """
        return self._normalize_prompt_whitespace(template.render(**render_kwargs))

    def _unify_risks(self, data: dict) -> None:
        """
        Unify risks from DB analysis and schema linking into one place.

        Combines:
        - data['context']['schema_risks']: risks from DB analysis (formatted risk hints)
        - data['schema_and_value_linking']['ambiguities_and_risks']: ambiguities from schema linking

        Stores the unified result back into data['context']['schema_risks'], which is
        passed to templates as {{schema_risks}} variable via _build_prompt_with_risks().
        """
        db_risks = data.get('context', {}).get('schema_risks', '')
        sl_risks = data.get('schema_and_value_linking', {}).get('ambiguities_and_risks', [])

        # Filter out placeholder values
        sl_risks = [r for r in sl_risks if r and str(r).strip() and str(r).strip().lower() != 'none']

        if not sl_risks:
            # No schema linking risks, keep DB risks as-is
            return

        # Build schema linking risks section
        sl_section = "\n### Schema Linking Ambiguities & Risks\n\n"
        sl_section += "The following ambiguities or risks were identified during schema linking:\n\n"
        for i, risk in enumerate(sl_risks, 1):
            sl_section += f"- **{risk}**\n"

        # Combine: DB risks first, then schema linking risks
        if db_risks:
            data['context']['schema_risks'] = db_risks + "\n" + sl_section
        else:
            data['context']['schema_risks'] = sl_section

        self.logger.info(f"Unified risks: {len(sl_risks)} from schema linking + DB analysis")

    def _get_box_schema(self, data: dict) -> str:
        """Build box_schema from enriched structured schema.

        Before schema_linking: includes all tables
        After schema_linking:  includes only linked tables
        If schema_linking failed: falls back to all tables
        """
        enriched = data.get('enriched_schema', {})
        linked_tables = data.get('linked_tables')  # None or set of table names

        if enriched:
            return self._build_box_schema(enriched, tables=linked_tables)

        # Fallback: parse original box_schema and filter by linked_tables if available
        raw_box_schema = data.get("schema", {}).get("box_schema", "")
        if not linked_tables:
            return raw_box_schema

        return self._filter_box_schema_by_tables(raw_box_schema, linked_tables)

    @staticmethod
    def _filter_box_schema_by_tables(box_schema: str, linked_tables: set[str]) -> str:
        """Filter raw box_schema string to only include specified linked tables.

        Parses the box_schema and extracts only the table blocks that match
        linked_tables (case-insensitive). Preserves the original formatting
        of kept tables.
        """
        if not box_schema or not linked_tables:
            return box_schema

        lines = box_schema.split('\n')
        result_parts = []
        current_table = None
        current_block = []
        brace_depth = 0

        for line in lines:
            stripped = line.strip()
            # Detect table start: table_name = pd.DataFrame({
            table_match = re.match(r'^(\w+)\s*=\s*pd\.DataFrame\(\{', stripped)
            if table_match:
                # Save previous block if it was a linked table
                if current_table and current_table.lower() in linked_tables:
                    result_parts.append('\n'.join(current_block))

                current_table = table_match.group(1)
                current_block = [line]
                # Count braces to find matching close
                brace_depth = line.count('{') - line.count('}')
                continue

            if current_table is not None:
                current_block.append(line)
                brace_depth += line.count('{') - line.count('}')
                if brace_depth <= 0:
                    # Block ended, check if it's a linked table
                    if current_table.lower() in linked_tables:
                        result_parts.append('\n'.join(current_block))
                    current_table = None
                    current_block = []

        # Handle case where last block wasn't properly closed
        if current_table and current_table.lower() in linked_tables:
            result_parts.append('\n'.join(current_block))

        if not result_parts:
            # No tables matched - return original with a warning comment
            return box_schema

        return '\n\n'.join(result_parts)

    def _filter_empty_tables(self, box_schema: str, table_content: str,
                              primary_keys: str, foreign_keys: str) -> tuple:
        """
        Filter out tables that have no data rows from schema components.

        Parses table_content to find tables with actual data (row count > 0),
        then removes empty tables from box_schema, primary_keys, and foreign_keys.

        Returns:
            (filtered_box_schema, filtered_table_content, filtered_primary_keys, filtered_foreign_keys)
        """
        # Extract tables with data from table_content
        # Format: TABLE `table_name` (N rows):
        data_tables = set()
        for m in re.finditer(r"TABLE `(\w+)` \((\d+) rows\):", table_content):
            table_name = m.group(1)
            row_count = int(m.group(2))
            if row_count > 0:
                data_tables.add(table_name.lower())

        if not data_tables:
            # No tables with data found - return empty strings
            return "", "", "", ""

        # Filter box_schema to only include tables with data
        filtered_box_schema = self._filter_box_schema_by_tables(box_schema, data_tables)

        # Filter table_content to only include tables with data
        filtered_content_parts = []
        current_table = None
        current_block = []
        for line in table_content.split('\n'):
            m = re.match(r"TABLE `(\w+)` \((\d+) rows\):", line)
            if m:
                # Save previous block if it had data
                if current_table and current_table.lower() in data_tables:
                    filtered_content_parts.append('\n'.join(current_block))
                current_table = m.group(1)
                current_block = [line]
            elif current_table:
                current_block.append(line)

        # Don't forget the last block
        if current_block and current_table and current_table.lower() in data_tables:
            filtered_content_parts.append('\n'.join(current_block))

        filtered_table_content = '\n\n'.join(filtered_content_parts)

        # Filter primary_keys to only include tables with data
        # Format: TABLE `table_name`: (col_name)
        filtered_pk_parts = []
        for line in primary_keys.split('\n'):
            m = re.match(r"TABLE `(\w+)`:", line)
            if m and m.group(1).lower() in data_tables:
                filtered_pk_parts.append(line)
        filtered_primary_keys = '\n'.join(filtered_pk_parts)

        # Filter foreign_keys to only include FKs where both tables have data
        # Format: FOREIGN KEY table_a['col'] REFERENCES table_b['col']
        filtered_fk_parts = []
        for line in foreign_keys.split('\n'):
            m = re.match(r"FOREIGN KEY (\w+)\[.*REFERENCES (\w+)\[", line)
            if m:
                src_table = m.group(1).lower()
                ref_table = m.group(2).lower()
                if src_table in data_tables and ref_table in data_tables:
                    filtered_fk_parts.append(line)
        filtered_foreign_keys = '\n'.join(filtered_fk_parts)

        return filtered_box_schema, filtered_table_content, filtered_primary_keys, filtered_foreign_keys

    def schema_linking(self, data: dict) -> dict:
        """
        Step 1: Schema linking + value linking (uses full schema).
        Identifies relevant tables, columns, and values from the question.
        """
        question = data["question"]
        schema = data.get("schema", {})

        try:
            task = self.task_type
            template = self.template_engine.get_template("schema_linking", task)
            schema_risks = data.get('context', {}).get('schema_risks', '')

            # Get hints for this db_id (if available)
            db_id = data.get('db_id', '')
            db_hints = self.hints_cache.get(db_id, [])
            hints_prompt = self._format_hints_prompt(db_hints)

            # Filter out empty tables before building prompt
            box_schema = self._get_box_schema(data)
            table_content = schema.get("table_content", "")
            primary_keys = schema.get("primary_keys", "")
            foreign_keys = schema.get("foreign_keys", "")

            if task == 'kbqa':
                box_schema, table_content, primary_keys, foreign_keys = self._filter_empty_tables(
                    box_schema, table_content, primary_keys, foreign_keys
                )

            demonstrations = self._get_retrieved_demonstrations(data)

            prompt = self._build_prompt(
                template,
                schema_risks=schema_risks,
                box_schema=box_schema,
                table_content=table_content,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                question=question,
                evidence=data.get("evidence", ""),
                value_linking=json.dumps(data.get("context", {}).get("value_linking", []), indent=2),
                hints=hints_prompt,
                few_shot_examples=demonstrations,
                schema_linking_fewshot_examples=demonstrations,
                entity_links=data.get("context", {}).get("entity_links", {}),
            )
            response = self._llm_generate(prompt)
            result = self._extract_json(response)

            # Store schema linking results
            value_linking = result.get('value_linking', [])
            # Strip whitespace from each value to avoid matching errors (e.g., ' = ' → '=')
            # Values may be int/float from LLM response, convert to str first
            value_linking = [str(v).strip() for v in value_linking if v is not None and str(v).strip()]
            data['schema_and_value_linking'] = {
                'schema_linking': result.get('schema_linking', []),
                'value_linking': value_linking,
                'thinking': result.get('thinking', ''),
                'ambiguities_and_risks': result.get('ambiguities_and_risks', []),
            }

            # ── Value Resolution: Query DB for actual cell values ──
            data = self._resolve_value_linking(data)
            return data
        except Exception as e:
            self.logger.warning(f"Schema linking failed: {e}")
            data['schema_and_value_linking'] = {
                "schema_linking": [],
                "value_linking": [],  # already empty, no strip needed
                "thinking": f"Error: {e}",
                "ambiguities_and_risks": ["Failed to parse LLM response"]
            }
            return data

    # ── Value Resolution ──────────────────────────────────────────────

    def _resolve_value_linking(self, data: dict) -> dict:
        """Full-DB value resolution: scan ALL tables and ALL columns.

        For each value in value_linking, scan every column of every table in the DB
        to find the best fuzzy match. If a table has at least one match, it is
        auto-added to schema_linking → linked_tables for downstream pipeline.

        Detailed match results (which table, which column, which value, similarity)
        are saved to data['context']['value_resolution_details'] for prompt injection.

        Args:
            data: Data dict with schema_and_value_linking populated.

        Returns:
            Updated data dict.
        """
        from difflib import SequenceMatcher

        VALUE_SIMILARITY_THRESHOLD = 0.75
        TABLE_BLACKLIST = ['Examination']  # Tables to exclude from value resolution

        value_linking = data.get('schema_and_value_linking', {}).get('value_linking', [])
        if not value_linking:
            self.logger.info("[Value Resolution] No values to resolve")
            return data

        db_path = data.get('context', {}).get('db_path', '')
        if not db_path or not Path(db_path).exists():
            self.logger.warning(f"[Value Resolution] DB not found: {db_path}, skipping")
            return data

        self.logger.info(f"[Value Resolution] Full-DB scan for {len(value_linking)} values: {value_linking}")

        # ── Phase 0: Pre-filter pure numeric values (skip DB scan) ──
        import re
        numeric_pattern = re.compile(r'^-?\d+(\.\d+)?$')
        skip_indices = set()
        for idx, val in enumerate(value_linking):
            if val and numeric_pattern.match(str(val).strip()):
                skip_indices.add(idx)
                self.logger.info(f"  ⏭️ Skipping numeric value '{val}' at index {idx}")

        if len(skip_indices) == len(value_linking):
            self.logger.info("[Value Resolution] All values are numeric, skipping DB scan entirely")
            # Keep original value_linking unchanged
            return data

        # ── Phase 1: Scan ALL tables and ALL columns ──
        # Structure: { search_value_idx: [ {table, column, matched_value, score}, ... ] }
        match_results: dict[int, list[dict]] = {i: [] for i in range(len(value_linking)) if i not in skip_indices}

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get ALL table names in DB
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]

            total_scanned = 0
            for tbl_name in all_tables:
                # Skip blacklisted tables
                if tbl_name in TABLE_BLACKLIST:
                    continue

                # Get all columns for this table
                cursor.execute(f"PRAGMA table_info(\"{tbl_name}\")")
                columns_info = cursor.fetchall()
                if not columns_info:
                    continue

                col_names = [c[1] for c in columns_info]

                for col_name in col_names:
                    # Skip primary key columns named 'id'
                    if col_name.lower() == 'id':
                        continue
                    # Get ALL distinct values (limit for performance)
                    try:
                        cursor.execute(
                            f"SELECT DISTINCT \"{col_name}\" FROM \"{tbl_name}\" "
                            f"WHERE \"{col_name}\" IS NOT NULL"
                        )
                        rows = cursor.fetchall()
                    except Exception:
                        continue

                    total_scanned += len(rows)

                    for (cell_val,) in rows:
                        if cell_val is None:
                            continue
                        cell_str = str(cell_val).strip()
                        if not cell_str:
                            continue

                        cell_lower = cell_str.lower()

                        # Check against each value_linking value
                        for idx, raw_val in enumerate(value_linking):
                            if not raw_val:
                                continue
                            search_val = str(raw_val).strip()
                            search_lower = search_val.lower()

                            # Exact match
                            if cell_lower == search_lower:
                                score = 1.0
                            else:
                                score = SequenceMatcher(None, search_lower, cell_lower).ratio()

                            if score >= VALUE_SIMILARITY_THRESHOLD:
                                match_results[idx].append({
                                    'table': tbl_name,
                                    'column': col_name,
                                    'matched_value': cell_str,
                                    'score': round(score, 4),
                                })

            conn.close()
            self.logger.info(f"[Value Resolution] Scanned {len(all_tables)} tables, "
                             f"{total_scanned} total cells, {sum(len(v) for v in match_results.values())} matches")

        except Exception as e:
            self.logger.error(f"[Value Resolution] Full-DB scan failed: {e}")
            return data

        # ── Phase 2: Build resolved values + auto-add matched tables ──
        resolved_values = []
        resolution_details = []  # For downstream prompt injection
        matched_tables = set()   # Tables to auto-add to schema_linking

        for idx, raw_val in enumerate(value_linking):
            # Pure numeric values: keep original, no DB scan
            if idx in skip_indices:
                resolved_values.append(raw_val)
                resolution_details.append({
                    'search_value': str(raw_val),
                    'matched': 'numeric_skip',
                    'reason': 'Pure numeric value, skipped DB scan',
                })
                continue

            matches = match_results.get(idx, [])
            # Filter out blacklisted table matches (safety net)
            matches = [m for m in matches if m['table'] not in TABLE_BLACKLIST]

            if not matches:
                resolved_values.append(None)
                resolution_details.append({
                    'search_value': str(raw_val),
                    'matched': False,
                    'matches': [],
                })
                continue

            # Pick best match (highest score)
            best = max(matches, key=lambda m: m['score'])
            resolved_values.append(best['matched_value'])
            matched_tables.add(best['table'])

            resolution_details.append({
                'search_value': str(raw_val),
                'matched': True,
                'best_match': best,
                'all_matches': sorted(matches, key=lambda m: -m['score'])[:5],
            })

            self.logger.info(f"  ✓ '{raw_val}' → '{best['matched_value']}' "
                             f"(score={best['score']:.3f}, {best['table']}.{best['column']})")

        # Update value_linking with resolved values
        data['schema_and_value_linking']['value_linking'] = resolved_values
        data['context']['value_resolution_details'] = resolution_details

        # ── Phase 3: Auto-add matched tables to schema_linking ──
        if matched_tables:
            schema_linking = data.get('schema_and_value_linking', {}).get('schema_linking', [])

            # Get columns already linked for each table
            existing_table_cols: dict[str, set[str]] = {}
            for entry in schema_linking:
                for col_path in entry.get('columns', []):
                    if '.' in col_path:
                        tbl_part = col_path.split('.')[0].lower()
                        col_part = col_path.split('.')[1]
                        existing_table_cols.setdefault(tbl_part, set()).add(col_part)

            # For each matched table, ensure it's in schema_linking with matched columns
            for tbl_name in matched_tables:
                tbl_lower = tbl_name.lower()
                matched_cols = set()
                for detail in resolution_details:
                    if not detail.get('matched'):
                        continue
                    for m in detail.get('all_matches', []):
                        if m['table'].lower() == tbl_lower:
                            matched_cols.add(m['column'])

                # Check if table already exists in schema_linking
                existing_tables_in_linking = set()
                for entry in schema_linking:
                    for t in entry.get('tables', []):
                        existing_tables_in_linking.add(t.lower())

                if tbl_lower not in existing_tables_in_linking:
                    # Auto-add this table to schema_linking
                    col_paths = [f"{tbl_name}.{c}" for c in sorted(matched_cols)]
                    new_entry = {
                        'phrase': f"(auto-added by value resolution: {', '.join(sorted(matched_cols))})",
                        'tables': [tbl_name],
                        'columns': col_paths,
                    }
                    schema_linking.append(new_entry)
                    self.logger.info(f"  + Auto-added table '{tbl_name}' to schema_linking "
                                     f"(columns: {sorted(matched_cols)})")

            data['schema_and_value_linking']['schema_linking'] = schema_linking
            self.logger.info(f"[Value Resolution] Auto-added {len(matched_tables)} tables to schema_linking: {sorted(matched_tables)}")

        n_resolved = sum(1 for v in resolved_values if v is not None)
        self.logger.info(f"[Value Resolution] Complete: {n_resolved}/{len(value_linking)} resolved, "
                         f"{len(matched_tables)} tables auto-added")

        return data

    @staticmethod
    def _format_value_resolution_prompt(value_resolution_details: list) -> str:
        """Format value resolution details for prompt injection.

        Returns a human-readable section describing which values matched which
        DB tables/columns, to be injected into downstream prompts.
        """
        matched_entries = [d for d in value_resolution_details if d.get('matched')]
        if not matched_entries:
            return ""

        lines = []
        for detail in matched_entries:
            search_val = detail.get('search_value', '?')
            best = detail.get('best_match', {})
            matched_val = best.get('matched_value', '?')
            table = best.get('table', '?')
            column = best.get('column', '?')
            score = best.get('score', 0)
            lines.append(f"- **'{search_val}'** → found in **`{table}.{column}`** as **`{matched_val}`** (similarity: {score:.0%})")

        return '\n'.join(lines)

    @staticmethod
    def _format_hints_prompt(hints: list[str]) -> str:
        """Format database-specific hints for prompt injection.

        Returns the hints joined as a plain list, or empty string if none.
        The template wraps this in a {% if hints %} conditional section.
        """
        if not hints:
            return ""
        return '\n'.join(f'- {h}' for h in hints)

    def task_decomposition(self, data: dict) -> dict:
        """Task decomposition (uses linked_schema - filtered to only linked tables)."""
        question = data["question"]
        linked_tables = data.get('linked_tables')
        schema_and_value_linking = data.get('schema_and_value_linking', {})

        try:
            task = self.task_type if self.task_type in ("kbqa", "tableqa") else "nl2sql"
            template = self.template_engine.get_template("task_decomposition", task)
            schema_risks = data.get('context', {}).get('schema_risks', '')

            # Value resolution details from DB scan
            value_res = data.get('context', {}).get('value_resolution_details', [])
            value_resolution_prompt = self._format_value_resolution_prompt(value_res)

            # Get hints for this db_id (if available)
            db_id = data.get('db_id', '')
            db_hints = self.hints_cache.get(db_id, [])
            hints_prompt = self._format_hints_prompt(db_hints)

            # Extract available table names from schema_linking for explicit constraint
            available_tables = sorted(linked_tables) if linked_tables else []
            available_str = ', '.join(f"'{t}'" for t in available_tables) if available_tables else 'NONE'
            demonstrations = self._get_retrieved_demonstrations(data)

            prompt = self._build_prompt(
                template,
                schema_risks=schema_risks,
                box_schema=self._get_box_schema(data),
                table_content=self._get_filtered_table_content(data.get('schema', {}), linked_tables),
                primary_keys=self._get_filtered_primary_keys(data.get('schema', {}), linked_tables),
                foreign_keys=self._get_filtered_foreign_keys(data.get('schema', {}), linked_tables),
                question=question,
                evidence=data.get("evidence", ""),
                schema_linking=json.dumps(schema_and_value_linking.get('schema_linking', []), indent=2),
                value_linking=json.dumps(schema_and_value_linking.get('value_linking', []), indent=2),
                value_resolution=value_resolution_prompt,
                hints=hints_prompt,
                available_tables=available_str,
                few_shot_examples=demonstrations,
                task_decomp_fewshot_examples=demonstrations,
                entity_links=data.get("context", {}).get("entity_links", {}),
            )
            response = self._llm_generate(prompt)
            result = self._extract_json(response)

            raw_subtasks = result.get('task_decomposition', [f"Subtask 1: {question}"])
            total_subtasks = len(raw_subtasks)
            # Append hard constraints (as # comments) to each subtask so downstream code_reasoning enforces them
            constrained_subtasks = []
            for i, sub in enumerate(raw_subtasks):
                constraints = ["  # ⛔ HARD CONSTRAINTS:"]
                constraints.append("# You MUST NOT use `if` / `else` / `elif` or any conditional branching in your code. Write purely declarative pandas operations.")
                if i >= 1:
                    constraints.append("# You MUST NOT use `pd.merge()`, `.merge()`, `.join()`, or any table join. Only operate on DataFrames already produced by earlier subtasks.")
                if i == total_subtasks - 1:
                    constraints.append("# FINAL SUBTASK: You MUST assign the final query output to a variable named `result`. The last line of your code MUST be: result = list(final_df[['columns']].itertuples(index=False, name=None))")
                else:
                    constraints.append("# **NEVER use generic names:** `result`, `subtask_out`, `output`, `df`, `temp`, `data`, `final`.")
                constrained_subtasks.append(sub + " ".join(constraints))

            data['task_decomposition'] = {
                'task_decomposition': constrained_subtasks,
                'thinking': result.get('thinking', ''),
            }
            data['accumulated_code'] = ""
            return data
        except Exception as e:
            self.logger.warning(f"Task decomposition failed: {e}")
            raw_subtasks = [f"Subtask 1: {question}"]
            total_subtasks = len(raw_subtasks)
            constrained_subtasks = []
            for i, sub in enumerate(raw_subtasks):
                constraints = ["  # ⛔ HARD CONSTRAINTS:"]
                constraints.append("# You MUST NOT use `if` / `else` / `elif` or any conditional branching in your code. Write purely declarative pandas operations.")
                if i >= 1:
                    constraints.append("# You MUST NOT use `pd.merge()`, `.merge()`, `.join()`, or any table join. Only operate on DataFrames already produced by earlier subtasks.")
                if i == total_subtasks - 1:
                    constraints.append("# FINAL SUBTASK: You MUST assign the final query output to a variable named `result`. The last line of your code MUST be: result = list(final_df[['columns']].itertuples(index=False, name=None))")
                else:
                    constraints.append("# **NEVER use generic names:** `result`, `subtask_out`, `output`, `df`, `temp`, `data`, `final`.")
                constrained_subtasks.append(sub + " ".join(constraints))
            data['task_decomposition'] = {
                'task_decomposition': constrained_subtasks,
                'thinking': f"Error: {e}",
            }
            data['accumulated_code'] = ""
            return data

    def code_reasoning(self, task_id: int, data: dict) -> dict:
        """Generate code for a single subtask."""
        td = data.get('task_decomposition', {})
        schema_and_value_linking = data.get('schema_and_value_linking', {})
        schema = data.get("schema", {})
        linked_tables = data.get('linked_tables')
        task_decomposition_list = td.get('task_decomposition', [])
        is_last_subtask = (task_id == len(task_decomposition_list) - 1)

        try:
            task = self.task_type if self.task_type in ("kbqa", "tableqa") else "nl2sql"
            template = self.template_engine.get_template("code_reasoning", task)
            subtask_description = td['task_decomposition'][task_id]

            schema_risks = data.get('context', {}).get('schema_risks', '')

            # Value resolution details from DB scan
            value_res = data.get('context', {}).get('value_resolution_details', [])
            value_resolution_prompt = self._format_value_resolution_prompt(value_res)
            demonstrations = self._get_retrieved_demonstrations(data)

            prompt = self._build_prompt(
                template,
                schema_risks=schema_risks,
                box_schema=self._get_box_schema(data),
                table_content=self._get_filtered_table_content(schema, linked_tables),
                primary_keys=self._get_filtered_primary_keys(schema, linked_tables),
                foreign_keys=self._get_filtered_foreign_keys(schema, linked_tables),
                evidence=data.get("evidence", ""),
                potential_linked_schema=json.dumps(schema_and_value_linking.get('schema_linking', []), indent=2),
                value_linking=json.dumps(schema_and_value_linking.get('value_linking', []), indent=2),
                value_resolution=value_resolution_prompt,
                code_for_earlier_subtasks=data.get('accumulated_code', ''),
                current_subtasks=subtask_description,
                few_shot_examples=demonstrations,
                entity_links=data.get("context", {}).get("entity_links", {}),
                is_last_subtask=is_last_subtask,
            )
            response = self._llm_generate(prompt)
            result = self._extract_json(response)

            thinking = result.get('thinking', '')
            code = result.get('code', '')

            # print(json.dumps(result.get('code_raw', ''), indent=2))
            # print()
            # print(json.dumps(result.get('checklist', ''), indent=2))
            # print()
            # print(json.dumps(result.get('code', ''), indent=2))
            # print()
            # print('--------------------------------------------------------------------')

            if 'accumulated_code' not in data:
                data['accumulated_code'] = ""

            data['previous_accumulated_code'] = copy.deepcopy(data['accumulated_code'])
            data['accumulated_code'] += (
                f"\n### {subtask_description}\n\n"
                f"#### Thinking:\n#### {thinking}\n\n"
                f"#### Code:\n{code}\n"
            )
            data['current_subtask_code'] = code
            return data
        except Exception as e:
            self.logger.warning(f"Code generation failed for task {task_id}: {e}")
            return data

    def execution_feedback(self, task_id: int, data: dict, accumulated_code: str, use_full_code: bool=False) -> dict:
        """
        Execute code and fix errors immediately.
        After execution, captures all newly defined DataFrame variables and adds
        their first 3 rows as comments to data['accumulated_code'], so the next
        code_reasoning call can understand the current execution state.

        NEW: If execution succeeds but result is empty (is_empty=True), treats it
        as an error and triggers the same repair loop - because empty results in
        KBQA subtasks usually mean the query/filter is wrong.
        """
        context = data.get("context", {})
        clean_code = self._extract_pure_code(accumulated_code)
        # KBQA Entity Restoration: Replace Names with IDs in code for execution
        code_to_execute = self._restore_entities_in_code(clean_code, data)

        # Execute the code (use sampled rows for speed during intermediate steps)
        exec_result = self.executor.execute(
            code_to_execute, context, use_full_code=use_full_code, top_k_row=50
        )

        # KBQA Entity Abstraction: Replace IDs with Names in execution state for LLM visibility
        if exec_result.df_vars:
            exec_result.df_vars = self._replace_ids_with_names(exec_result.df_vars)

        self.logger.info(f"Execution result: success={exec_result.success}, is_empty={exec_result.is_empty}, df_vars={'yes' if exec_result.df_vars else 'none'}")
        self.logger.debug(f"Execution output: {exec_result.to_dict()}")

        # Always inject DataFrame state comments from initial execution (success or failure)
        if exec_result.df_vars:
            data['accumulated_code'] = self._inject_state_comments(data['accumulated_code'], exec_result.df_vars)

        # If execution succeeded AND result is NOT empty, do logic reflection check
        if exec_result.success and (not exec_result.is_empty or (exec_result.is_empty and self.dataset.name == 'bird')):
            data['execution_feedback'] = False
            # do_reflection = self.config.get("inference.do_execution_reflection", True)
            # if do_reflection:
            #     data = self.self_reflection(task_id=task_id, data=data)
            return data

        # Execution failed OR result is empty - start revision loop
        data['execution_feedback'] = True
        data['revision_history'] = []

        td = data.get('task_decomposition', {})
        subtask_description = td['task_decomposition'][task_id]

        # Build error message: use actual traceback if available, or empty-result hint
        if exec_result.is_empty:
            error_msg = (
                "RESULT IS EMPTY - your query returned 0 rows.\n"
                "This usually means your filter conditions, table selection, or column references are wrong.\n"
                "Check the Box Schema for correct table/column names. "
                "Make sure you're filtering on the right column and the value actually exists in the data.\n"
                "Common causes: (1) wrong table name, (2) wrong column name, "
                "(3) filter value mismatch (case sensitivity, whitespace), "
                "(4) confused subject/reverse relation direction."
            )
        else:
            error_msg = exec_result.error or "Unknown error"


        buggy_code = data.get('current_subtask_code', '')

        for attempt in range(self.max_revision_rounds):
            repair_result = self._repair_single_subtask_code(
                buggy_code=buggy_code,
                traceback=error_msg,
                data=data,
                task_id=task_id
            )

            if not repair_result:
                break

            revised_code, repair_thinking = repair_result

            data['revision_history'].append({
                'buggy_code': buggy_code,
                'revised_code': revised_code,
                'thinking': repair_thinking,
            })

            candidate_accumulated_code = (
                data['previous_accumulated_code'] +
                f"\n### {subtask_description}\n\n"
                f"#### Thinking:\n#### {repair_thinking}\n\n"
                f"#### Code:\n{revised_code}\n"
            )

            exec_result = self.executor.execute(
                candidate_accumulated_code, context
            )

            # Always inject state comments after each revision attempt, regardless of success/failure
            if exec_result.df_vars:
                candidate_accumulated_code = self._inject_state_comments(candidate_accumulated_code, exec_result.df_vars)

            # Success AND non-empty → accept the fix
            if exec_result.success and not exec_result.is_empty:
                data['accumulated_code'] = candidate_accumulated_code
                data['current_subtask_code'] = revised_code
                break

            buggy_code = revised_code
            # Even on failure, update accumulated_code so next subtask's code_reasoning sees the latest state
            data['accumulated_code'] = candidate_accumulated_code

        return data

    def _repair_single_subtask_code(self, buggy_code: str, traceback: str, data: dict, task_id: int) -> Optional[tuple]:
        """Ask LLM to fix subtask code based on execution error.

        Returns: (revised_code, thinking) or None
        """
        try:
            task = self.task_type if self.task_type in ("kbqa", "tableqa") else "nl2sql"
            template = self.template_engine.get_template("execution_feedback", task)
            schema = data.get("schema", {})
            linked_tables = data.get('linked_tables')
            schema_and_value_linking = data.get('schema_and_value_linking', {})
            td = data.get('task_decomposition', {})

            schema_risks = data.get('context', {}).get('schema_risks', '')

            # Value resolution details from DB scan
            value_res = data.get('context', {}).get('value_resolution_details', [])
            value_resolution_prompt = self._format_value_resolution_prompt(value_res)

            # KBQA Entity Abstraction: Replace IDs with Names in traceback for LLM consistency
            llm_traceback = self._replace_ids_with_names(traceback) if self.task_type == "kbqa" else traceback
            demonstrations = self._get_retrieved_demonstrations(data)

            prompt = self._build_prompt(
                template,
                schema_risks=schema_risks,
                question=data.get("question", ""),
                evidence=data.get("evidence", ""),
                current_subtasks=td['task_decomposition'][task_id],
                box_schema=self._get_box_schema(data),
                table_content=self._get_filtered_table_content(schema, linked_tables),
                primary_keys=self._get_filtered_primary_keys(schema, linked_tables),
                foreign_keys=self._get_filtered_foreign_keys(schema, linked_tables),
                potential_linked_schema=json.dumps(schema_and_value_linking.get('schema_linking', []), indent=2),
                code_for_earlier_subtasks=data.get('previous_accumulated_code', ''),
                execution_state=self._extract_execution_state(data.get('accumulated_code', '')),
                value_resolution=value_resolution_prompt,
                buggy_code=buggy_code,
                traceback=llm_traceback,
                few_shot_examples=demonstrations,
                entity_links=data.get("context", {}).get("entity_links", {}),
            )
            response = self._llm_generate(prompt)
            result = self._extract_json(response)
            revised_code = result.get('revised_code')
            thinking = result.get('thinking', '')
            if revised_code:
                return (revised_code, thinking)
            return None
        except Exception as e:
            self.logger.error(f"Code repair failed: {e}")
            return None

    def self_reflection(self, task_id: int, data: dict) -> dict:
        """
        Post-execution reflection: even when code runs without errors,
        check for logical issues that could produce wrong results.

        Checks performed:
        1. JOIN/FK compliance - are merge conditions aligned with schema FK?
        2. Column selection - does the output have the right number of columns?
        3. Filter validity - are filter values likely to match DB content?
        4. DISTINCT / deduplication - should results be deduplicated?
        5. ORDER BY / LIMIT semantics - does the code match "most/least/top" intent?
        6. Empty result risk - could overly strict filters produce empty results?
        """
        question = data.get("question", "")
        schema = data.get("schema", {})
        linked_tables = data.get('linked_tables')
        schema_and_value_linking = data.get('schema_and_value_linking', {})
        fk = schema.get("foreign_keys", "")
        pk = schema.get("primary_keys", "")
        current_code = data.get('current_subtask_code', '')

        try:
            task = self.task_type if self.task_type in ("kbqa", "tableqa") else "nl2sql"
            template = self.template_engine.get_template("self_reflection", task)
            schema_risks = data.get('context', {}).get('schema_risks', '')

            # Detect semantic hints from question
            q_lower = question.lower()
            needs_distinct = any(w in q_lower for w in [
                'distinct', 'unique', 'different', 'list all the',
                'all the members', 'all the names', 'list all'
            ])
            needs_order = any(w in q_lower for w in [
                'most', 'least', 'top ', 'first', 'last', 'recent',
                'best', 'worst', 'highest', 'lowest', 'largest',
                'smallest', 'more than', 'than in', 'the most',
                'the least', 'the first', 'the last'
            ])
            needs_limit = any(w in q_lower for w in [
                'limit 1', 'top 1', 'the most', 'the least',
                'the first', 'the last', 'the latest', 'the earliest'
            ])

            prompt = self._build_prompt(
                template,
                schema_risks=schema_risks,
                question=question,
                evidence=data.get("evidence", ""),
                current_subtasks=data.get('task_decomposition', {}).get('task_decomposition', [data.get("question", "")])[task_id],
                box_schema=self._get_box_schema(data),
                table_content=self._get_filtered_table_content(schema, linked_tables),
                primary_keys=pk,
                foreign_keys=fk,
                potential_linked_schema=json.dumps(schema_and_value_linking.get('schema_linking', []), indent=2),
                previous_accumulated_code=data.get('accumulated_code', ''),
                current_subtask_code=current_code,
                execution_state=self._extract_execution_state(data.get('accumulated_code', '')),
                needs_distinct=str(needs_distinct),
                needs_order=str(needs_order),
                needs_limit=str(needs_limit),
                entity_links=json.dumps(data.get("context", {}).get("entity_links", {}), indent=2),
            )

            response = self._llm_generate(prompt)
            result = self._extract_json(response)

            issues = result.get('issues', [])
            revised_code = result.get('revised_code', '')

            if issues and revised_code and revised_code.strip():
                # Apply the revised code
                data['current_subtask_code'] = revised_code
                td = data.get('task_decomposition', {})
                subtask_description = td['task_decomposition'][task_id]

                # Extract existing state blocks from current accumulated_code
                # (injected by the initial execution, must be preserved)
                state_match = re.search(
                    r'(\n\s*# ===== Execution State:.*?# ===== End of Execution State =====)',
                    data['accumulated_code'],
                    re.DOTALL
                )
                existing_state_block = state_match.group(1) if state_match else ""

                # Re-execute with revised code
                accumulated_code = (
                    data['previous_accumulated_code'] +
                    f"\n### {subtask_description}\n\n"
                    f"#### Thinking:\n#### Post-execution reflection fixed: {', '.join(issues)}\n\n"
                    f"#### Code:\n{revised_code}\n"
                    f"{existing_state_block}"
                )

                context = data.get("context", {})
                new_exec = self.executor.execute(accumulated_code, context)

                if new_exec.success:
                    data['accumulated_code'] = accumulated_code
                    if new_exec.df_vars:
                        data['accumulated_code'] = self._inject_state_comments(data['accumulated_code'], new_exec.df_vars)
                    data['reflection_fixed'] = True
                    data['reflection_issues'] = issues
                    self.logger.info(f"Reflection fixed {len(issues)} issues: {issues}")
                else:
                    self.logger.warning(f"Reflection revised code failed: {new_exec.error}")
                    data['reflection_issues'] = issues
            elif issues:
                self.logger.info(f"Reflection found issues but no fix provided: {issues}")
                data['reflection_issues'] = issues
            else:
                self.logger.info("Reflection: no issues found")

            return data

        except Exception as e:
            self.logger.warning(f"Reflection failed: {e}")
            return data

    def _extract_execution_state(self, accumulated_code: str) -> str:
        """
        Extract FULL execution state from accumulated_code comments.

        Unlike _extract_df_shapes_from_comments which only extracts shape lines,
        this extracts column names, dtypes, AND sample values for every DataFrame,
        as well as Series and scalar variables.

        Returns formatted string like:
            DataFrame 'df_name' - shape (100, 5)
              columns:
                col1 (object): ['val1', 'val2', 'val3']
                col2 (int64): [1, 2, 3]
              head(3):
                 col1 col2
              0  val1    1
            Scalar 'percentage' = 34.78260869565217
        """
        state_lines = []
        in_state = False
        for line in accumulated_code.split('\n'):
            stripped = line.strip()
            if '===== Execution State:' in stripped and '=====' in stripped:
                in_state = True
                continue
            if '===== End of Execution State =====' in stripped:
                in_state = False
                continue
            if in_state and stripped:
                state_lines.append(stripped)
        if not state_lines:
            return "No execution state available from prior subtasks."
        return '\n'.join(state_lines)

    def _extract_df_shapes_from_comments(self, accumulated_code: str) -> str:
        """Extract DataFrame shape info from execution state comments."""
        shapes = []
        in_state = False
        for line in accumulated_code.split('\n'):
            stripped = line.strip()
            if '===== Execution State:' in stripped and '=====' in stripped:
                in_state = True
                continue
            if '===== End of Execution State =====' in stripped:
                in_state = False
                continue
            if in_state:
                if 'shape (' in stripped or stripped.startswith('# DataFrame ') or stripped.startswith('DataFrame '):
                    shapes.append(stripped.lstrip('#').strip())
        return '\n'.join(shapes) if shapes else "No execution state available"

    def code_merge(self, data: dict) -> dict:
        """Fuse subtask codes using LLM."""
        question = data["question"]
        schema = data.get("schema", {})
        linked_tables = data.get('linked_tables')
        schema_and_value_linking = data.get('schema_and_value_linking', {})

        try:
            task = self.task_type
            template = self.template_engine.get_template("code_merge", task)
            schema_risks = data.get('context', {}).get('schema_risks', '')
            demonstrations = self._get_retrieved_demonstrations(data)
            prompt = self._build_prompt(
                template,
                schema_risks=schema_risks,
                question=question,
                evidence=data.get("evidence", ""),
                box_schema=self._get_box_schema(data),
                table_content=self._get_filtered_table_content(schema, linked_tables),
                primary_keys=self._get_filtered_primary_keys(schema, linked_tables),
                foreign_keys=self._get_filtered_foreign_keys(schema, linked_tables),
                potential_linked_schema=json.dumps(schema_and_value_linking.get('schema_linking', []), indent=2),
                code_reasoning_steps=data.get('accumulated_code', ''),
                few_shot_examples=demonstrations,
                merge_fewshot_examples=demonstrations,
                entity_links=data.get("context", {}).get("entity_links", {}),
            )
            response = self._llm_generate(prompt)
            result = self._extract_json(response)

            data['merged_thinking'] = result.get('thinking', '')
            data['merged_code'] = result.get('code', '')

            if not data['merged_code'] or not data['merged_code'].strip():
                self.logger.warning("LLM returned empty merged code, falling back to extraction from accumulated_code")
                data['merged_code'] = self._fallback_merge_from_accumulated(data)
                data['merged_thinking'] += " [Fallback: LLM returned empty code, extracted from accumulated_code]"

            self.logger.info(f"Code fusion complete")
            return data
        except Exception as e:
            self.logger.warning(f"Code fusion failed: {e}")
            data['merged_code'] = self._fallback_merge_from_accumulated(data)
            data['merged_thinking'] = f"Fallback due to error: {e}"
            return data

    def _correct_entity_ids(self, data: dict) -> dict:
        """KBQA post-processing: Replace entity names with Freebase IDs in merged code.

        The LLM tends to use natural language names (e.g., 'Gran Turismo 4 Kicks')
        instead of Freebase IDs (e.g., 'm.01fc1gk'). This method:
        1. Collects all entity→ID mappings from entity_links
        2. Finds string literals in the merged code (case-insensitive)
        3. Replaces entity names with their Freebase IDs

        This is critical because subgraph CSV tables store Freebase IDs, not names.
        """
        import re as _re

        merged_code = data.get('merged_code', '')
        if not merged_code or not merged_code.strip():
            return data

        # Collect entity→ID mappings from all sources
        entity_map = {}

        # Source 1: entity_links from context
        entity_links = data.get('context', {}).get('entity_links', {})
        if isinstance(entity_links, dict):
            entity_map.update(entity_links)

        if not entity_map:
            self.logger.info("[Step 4.3] No entity links available for ID correction")
            return data

        replacements = []
        corrected_code = merged_code

        # Sort by length descending - replace longer names first to avoid partial matches
        # e.g., replace "Gran Turismo 4 Kicks" before "Gran Turismo 4"
        sorted_entities = sorted(entity_map.items(), key=lambda x: -len(x[0]))

        for entity_name, fb_id in sorted_entities:
            # Case-insensitive search for the entity name in string literals
            # Pattern: single or double quoted string matching the entity name (case-insensitive)
            pattern_single = _re.compile(
                r"'(" + _re.escape(entity_name) + r")'",
                _re.IGNORECASE
            )
            pattern_double = _re.compile(
                r'"(' + _re.escape(entity_name) + r')"',
                _re.IGNORECASE
            )

            if pattern_single.search(corrected_code) or pattern_double.search(corrected_code):
                corrected_code = pattern_single.sub(f"'{fb_id}'", corrected_code)
                corrected_code = pattern_double.sub(f'"{fb_id}"', corrected_code)
                replacements.append(f"'{entity_name}' → '{fb_id}'")
                corrected_code = corrected_code.replace(f'"{entity_name}"', f'"{fb_id}"')
                replacements.append(f"'{entity_name}' → '{fb_id}'")

        if replacements:
            data['merged_code'] = corrected_code
            data['entity_id_corrections'] = replacements
            self.logger.info(f"[Step 4.3] Entity ID corrections: {replacements}")
        else:
            self.logger.info(f"[Step 4.3] No entity name replacements needed")

        return data

    def _fallback_merge_from_accumulated(self, data: dict) -> str:
        """
        Enhanced fallback: extract ALL subtask codes from accumulated_code.

        Previous fallback only used current_subtask_code (the last subtask),
        which discarded all previous subtask code. This method extracts
        every subtask's code from accumulated_code and concatenates them.
        """
        accumulated_code = data.get('accumulated_code', '')
        if not accumulated_code or not accumulated_code.strip():
            # If accumulated_code is also empty, try current_subtask_code as last resort
            last_resort = data.get('current_subtask_code', '')
            if last_resort and last_resort.strip():
                self.logger.info("Fallback: using current_subtask_code as last resort")
                return last_resort
            self.logger.warning("Fallback: no code available, returning empty")
            return ""

        clean_code = self._extract_pure_code(accumulated_code)
        if clean_code and clean_code.strip():
            self.logger.info(f"Fallback: extracted {len(clean_code.splitlines())} lines from accumulated_code")
            return clean_code

        # Last resort: current_subtask_code
        last_resort = data.get('current_subtask_code', '')
        self.logger.info("Fallback: using current_subtask_code as last resort")
        return last_resort

    def _extract_pure_code(self, accumulated_code: str) -> str:
        """
        Extract pure Python code from accumulated_code which may contain markdown headers.
        """
        lines = accumulated_code.split('\n')
        clean_lines = []

        for line in lines:
            if line.startswith('#') or line.strip() == "":
                continue
            clean_lines.append(line.rstrip())

        code = '\n'.join(clean_lines)
        code = re.sub(r'```python|```', '', code)
        return code

    def run_merged_code(self, data: dict, merged_code: str) -> dict:
        """
        Final execution of merged code.
        Uses ALL rows (top_k_row=None) for accurate results.
        """
        context = data.get("context", {})

        # KBQA Entity Restoration: Replace Names with IDs in code for execution
        code_to_execute = self._restore_entities_in_code(merged_code, data)

        python_results = self.executor.execute(
            code_to_execute, context, use_full_code=True, top_k_row=None
        ).to_dict()

        data['python_results'] = python_results
        data['python_exception'] = not python_results['success'] if python_results.get('success') is not None else False
        return data

    def final_execution_feedback(self, data: dict) -> dict:
        """
        Final Execution Retry: If the merged code failed to execute, send the code +
        error to LLM for repair, then re-execute. Retries up to max_revision_rounds.
        Uses full rows (top_k_row=None) for accurate results.
        """
        context = data.get("context", {})
        merged_code = data.get("merged_code", "")

        max_retries = self.max_revision_rounds
        retry_history = []

        for attempt in range(max_retries):
            # KBQA Entity Restoration: Replace Names with IDs in code for execution
            code_to_execute = self._restore_entities_in_code(merged_code, data)

            exec_result = self.executor.execute(code_to_execute, context, use_full_code=True, top_k_row=None)

            if exec_result.success:
                data['merged_code'] = merged_code
                data['python_results'] = exec_result.to_dict()
                data['python_exception'] = False
                data['code_repair_success'] = True
                data['code_repair_attempts'] = attempt
                data['code_repair_history'] = retry_history
                self.logger.info(f"Final execution retry succeeded on attempt {attempt + 1}")
                return data

            error = exec_result.error or "Unknown error"
            retry_history.append({
                'attempt': attempt + 1,
                'error': error,
                'code': merged_code,
            })

            corrected_code = self._repair_merged_code(merged_code, error, data)

            if not corrected_code or corrected_code.strip() == merged_code.strip():
                self.logger.warning(f"Final retry attempt {attempt + 1}: LLM returned no fix")
                break

            merged_code = corrected_code

        data['code_repair_history'] = retry_history
        data['code_repair_success'] = False
        data['code_repair_attempts'] = len(retry_history)
        self.logger.warning(f"Final execution retry exhausted after {len(retry_history)} attempts")

        data['python_results'] = exec_result.to_dict()
        data['python_exception'] = True

        return data

    def _repair_merged_code(self, buggy_code: str, traceback: str, data: dict) -> Optional[str]:
        """Ask LLM to fix merged code based on execution error."""
        try:
            task = self.task_type if self.task_type in ("kbqa", "tableqa") else "nl2sql"
            template = self.template_engine.get_template("final_execution_feedback", task)
            question = data.get("question", "")
            schema = data.get("schema", {})
            linked_tables = data.get('linked_tables')
            schema_and_value_linking = data.get('schema_and_value_linking', {})

            schema_risks = data.get('context', {}).get('schema_risks', '')

            # Value resolution details from DB scan
            value_res = data.get('context', {}).get('value_resolution_details', [])
            value_resolution_prompt = self._format_value_resolution_prompt(value_res)

            # KBQA Entity Abstraction: Replace IDs with Names in traceback for LLM consistency
            llm_traceback = self._replace_ids_with_names(traceback) if self.task_type == "kbqa" else traceback
            demonstrations = self._get_retrieved_demonstrations(data)

            prompt = self._build_prompt(
                template,
                schema_risks=schema_risks,
                question=question,
                box_schema=self._get_box_schema(data),
                table_content=self._get_filtered_table_content(schema, linked_tables),
                primary_keys=self._get_filtered_primary_keys(schema, linked_tables),
                foreign_keys=self._get_filtered_foreign_keys(schema, linked_tables),
                evidence=data.get("evidence", ""),
                potential_linked_schema=json.dumps(schema_and_value_linking.get('schema_linking', []), indent=2),
                value_resolution=value_resolution_prompt,
                buggy_code=buggy_code,
                traceback=llm_traceback,
                few_shot_examples=demonstrations,
                entity_links=data.get("context", {}).get("entity_links", {}),
            )
            response = self._llm_generate(prompt)
            result = self._extract_json(response)
            return result.get('corrected_code')
        except Exception as e:
            self.logger.error(f"Fused code repair failed: {e}")
            return None

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from response."""
        json_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
            try:
                return json.loads(json_str)
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
                        json_str = response[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                        break

        return {}

    def _extract_linked_tables(self, schema_linking: list) -> set[str]:
        """Extract unique table names from schema_linking results."""
        tables = set()
        for item in schema_linking:
            for t in item.get('tables', []):
                tables.add(t.lower())
        # Also check columns for table.column format
        for item in schema_linking:
            for col in item.get('columns', []):
                if '.' in col:
                    tables.add(col.split('.')[0].lower())
        return tables

    def _find_bridge_tables(self, linked_tables: set[str], foreign_keys: str) -> set[str]:
        """
        Find bridge tables needed to connect all linked tables via FK relationships.

        If the LLM-selected tables can only be connected through intermediate tables
        (not directly selected), this method adds those bridge tables so the generated
        code has all necessary tables for JOIN operations.

        Args:
            linked_tables: Set of table names selected by schema linking (lowercase).
            foreign_keys: FK constraint string, e.g.:
                "FOREIGN KEY badges['UserId'] REFERENCES users['Id']\n..."

        Returns:
            Set of bridge table names (lowercase) that should be added to linked_tables.
        """
        if len(linked_tables) <= 1:
            return set()

        # Parse FK constraints to build adjacency graph
        adj: dict[str, set[str]] = {}

        for line in foreign_keys.split('\n'):
            line = line.strip()
            if not line or not line.upper().startswith('FOREIGN KEY'):
                continue

            match = re.match(
                r"FOREIGN\s+KEY\s+(\w+)\[.*?\]\s+REFERENCES\s+(\w+)\[.*?\]",
                line,
                re.IGNORECASE,
            )
            if match:
                t1 = match.group(1).lower()
                t2 = match.group(2).lower()
                adj.setdefault(t1, set()).add(t2)
                adj.setdefault(t2, set()).add(t1)

        # For every pair of linked tables, find the shortest path between them.
        # Any intermediate table on any path is a bridge table that should be added.
        bridge_tables: set[str] = set()
        linked_list = sorted(linked_tables)

        for i in range(len(linked_list)):
            for j in range(i + 1, len(linked_list)):
                t1 = linked_list[i]
                t2 = linked_list[j]
                # Find shortest path from t1 to t2
                path = self._bfs_shortest_path(t1, t2, adj)
                if path is not None:
                    # Add all intermediate tables (excluding endpoints)
                    for table in path:
                        if table not in linked_tables:
                            bridge_tables.add(table)

        return bridge_tables

    @staticmethod
    def _bfs_shortest_path(
        start: str, end: str, adj: dict[str, set[str]]
    ) -> Optional[list[str]]:
        """
        BFS to find the shortest path from start to end.
        Returns the list of tables on the path (including start and end),
        or None if no path exists.
        """
        from collections import deque

        if start == end:
            return [start]

        visited: dict[str, str] = {start: ""}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node == end:
                # Reconstruct path
                path = []
                current = end
                while current:
                    path.append(current)
                    current = visited.get(current, "")
                return list(reversed(path))

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = node
                    queue.append(neighbor)

        return None  # No path found

    # ── Helpers: filter raw schema fields by linked tables ──

    @staticmethod
    def _get_filtered_table_content(schema: dict, linked_tables: Optional[set[str]]) -> str:
        """Return table_content, filtered to linked tables if specified."""
        table_content = schema.get("table_content", "")
        if not linked_tables or not table_content:
            return table_content
        sections = []
        for section in table_content.split('\n\n'):
            section = section.strip()
            if not section:
                continue
            for line in section.split('\n'):
                if line.startswith('TABLE `'):
                    sec_table = line.split('`')[1].lower() if '`' in line else None
                    break
            else:
                continue
            if sec_table in linked_tables:
                sections.append(section)
        return '\n\n'.join(sections)

    @staticmethod
    def _get_filtered_primary_keys(schema: dict, linked_tables: Optional[set[str]]) -> str:
        """Return primary_keys, filtered to linked tables if specified."""
        pks = schema.get("primary_keys", "")
        if not linked_tables or not pks:
            return pks
        filtered = []
        for line in pks.split('\n'):
            line = line.strip()
            if not line or not line.startswith('TABLE `'):
                continue
            pk_table = line.split('`')[1].lower() if '`' in line else None
            if pk_table in linked_tables:
                filtered.append(line)
        return '\n'.join(filtered)

    @staticmethod
    def _get_filtered_foreign_keys(schema: dict, linked_tables: Optional[set[str]]) -> str:
        """Return foreign_keys, filtered to linked tables if specified."""
        fks = schema.get("foreign_keys", "")
        if not linked_tables or not fks:
            return fks
        filtered = []
        for line in fks.split('\n'):
            line = line.strip()
            if not line or not line.startswith('FOREIGN KEY'):
                continue
            # Extract table name - for KBQA, table names may have namespace prefix
            # like 'cvg.computer_game_engine', so take the part after the last dot.
            # For nl2sql/tableqa, names have no dot and are used as-is.
            fk_raw = line.split('[')[0].replace('FOREIGN KEY ', '').strip().lower()
            fk_table = fk_raw.rsplit('.', 1)[-1] if '.' in fk_raw else fk_raw
            ref_raw = line.split('REFERENCES')[1].split('[')[0].strip().lower() if 'REFERENCES' in line else ''
            ref_table = ref_raw.rsplit('.', 1)[-1] if '.' in ref_raw else ref_raw
            if fk_table in linked_tables and ref_table in linked_tables:
                filtered.append(line)
        return '\n'.join(filtered)

    def run_with_voting(self, example: dict, n_votes: int = 5, shot_k: int = 10) -> dict[str, Any]:
        """Multi-vote inference with parallel execution."""
        example_id = example.get('qid', example.get('id', example.get('question_id', 'unknown')))
        self.logger.info(f"Running {n_votes}-vote inference for example {example_id}")

        votes = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_vote = {executor.submit(self.run, example, shot_k): i + 1 for i in range(n_votes)}
            for future in as_completed(future_to_vote):
                vote_id = future_to_vote[future]
                try:
                    result = future.result()
                    result["vote_id"] = vote_id
                    votes.append(result)
                    self.logger.info(f"Vote {vote_id}/{n_votes} completed")
                except Exception as e:
                    self.logger.error(f"Vote {vote_id} failed: {e}")
                    votes.append({
                        "vote_id": vote_id,
                        "success": False,
                        "error": str(e),
                        "answer": None
                    })

        if not votes:
            return {"example_id": example_id, "success": False, "error": "All votes failed", "n_votes": n_votes}

        aggregator = VoteAggregator()
        aggregated = aggregator.aggregate(votes)
        aggregated["example_id"] = example_id
        aggregated["n_votes"] = n_votes
        aggregated["all_votes"] = votes

        # KBQA: Add sparql_query and s_expression from raw example to top-level result
        sparql = example.get("sparql_query", "")
        s_expr = example.get("s_expression", "")
        if sparql:
            aggregated["sparql_query"] = sparql
        if s_expr:
            aggregated["s_expression"] = s_expr

        # WebQSP: Extract gold_sparql from 'label' field
        if self.task_type == "kbqa" and not sparql:
            gold_sparql = example.get("label", "")
            if gold_sparql:
                aggregated["gold_sparql"] = gold_sparql

        # NL2SQL (spider-syn): Extract gold SQL from 'label' field
        if self.task_type == "nl2sql" and not example.get("SQL"):
            gold_sql = example.get("SQL", "") or example.get("query", "") or example.get("label", "")
            if gold_sql:
                aggregated["gold_sql"] = gold_sql

        self.logger.info(f"Voting complete: {aggregated.get('vote_count', 0)}/{n_votes} votes")

        return aggregated

    def _build_result(self, example_id: str, data: dict, output: dict, metrics: Optional[dict] = None, gold_answer: Optional[list] = None, example: Optional[dict] = None, profile: Optional[dict] = None, gold_sql: str = "") -> dict:
        """Build result dictionary.

        Note: 'schema_risks' contains unified risks from both DB analysis
        and schema linking ambiguities, stored in data['context']['schema_risks'].
        The original 'ambiguities_and_risks' from schema linking is still available
        inside data['schema_and_value_linking'] for reference.

        For KBQA (GrailQA), also extracts sparql_query and s_expression from the raw example.
        For WikiTQ, includes fallback LLM answer if code-based approach failed.
        """
        result = {
            "example_id": example_id,
            "question": data.get("question", ""),
            "evidence": data.get("evidence", ""),
            "db_id": example.get('db_id', '') if example else "",
            "schema_risks": data.get("context", {}).get("schema_risks", ""),
            "linked_tables": list(self._extract_linked_tables(
                data.get("schema_and_value_linking", {}).get("schema_linking", [])
            )),
            "schema_and_value_linking": data.get("schema_and_value_linking", {}),
            "task_decomposition": data.get("task_decomposition", {}),
            "accumulated_code": data.get("accumulated_code", ""),
            "merged_code": data.get("merged_code", ""),
            "merged_thinking": data.get("merged_thinking", ""),
            "code_repair_history": data.get("code_repair_history", []),
            "code_repair_success": data.get("code_repair_success", False),
            "code_repair_attempts": data.get("code_repair_attempts", 0),
            "answer": output["answer"],
            "formatted_answer": output.get("formatted"),
            "python_results": data.get('python_results'),
            "python_exception": data.get('python_exception', False),
            "metrics": metrics,
            "gold_answer": gold_answer[:10] if gold_answer else [],
            "gold_sql": gold_sql,
            "revision_history": data.get("revision_history", []),
            "execution_feedback": data.get("execution_feedback", False),
            "value_corrections": data.get("value_corrections", []),
            "value_findings": data.get("value_findings", []),
            # WikiTQ Fallback LLM answer
            "fallback_llm_thinking": data.get("fallback_llm_thinking", ""),
            "fallback_llm_raw_answer": data.get("fallback_llm_raw_answer", ""),
            "fallback_llm_parsed": data.get("fallback_llm_parsed", []),
        }

        # KBQA: Add sparql_query and s_expression from raw example
        if example:
            sparql = example.get("sparql_query", "")
            s_expr = example.get("s_expression", "")
            if sparql:
                result["sparql_query"] = sparql
            if s_expr:
                result["s_expression"] = s_expr

            # WebQSP: Extract gold_sparql from 'label' field
            if self.task_type == "kbqa" and not sparql:
                gold_sparql = example.get("label", "")
                if gold_sparql:
                    result["gold_sparql"] = gold_sparql

            # NL2SQL (spider-syn): gold_sql already passed as parameter

        # Profiling summary (time + token length per stage)
        if profile:
            result["profile"] = profile

        return result

    def _inject_state_comments(self, code: str, df_vars: str) -> str:
        """
        Append ONLY NEWLY defined variable state as comments at the END of code.

        The executor captures ALL variables in globals() (DataFrames, Series, scalars),
        including ones from previous subtasks. We compare with existing state blocks in
        the accumulated_code to filter out already-documented variables.

        For each NEW DataFrame, the state includes:
        - shape (rows, cols)
        - column names + dtypes + sample values
        - head(3): first 3 rows printed as a table

        Old state blocks are preserved as-is - only new ones are appended.
        Each variable appears exactly once, under its own subtask.

        If no NEW variables are created AND there is already an execution state block
        at the end of the code, do NOT add a new placeholder block.
        """
        # Extract all variable names from EXISTING state blocks
        existing_names = set()
        existing_blocks = list(re.finditer(
            r'# ===== Execution State:.*?# ===== End of Execution State =====',
            code,
            re.DOTALL
        ))

        for m in existing_blocks:
            # Match DataFrame, Series, and Scalar entries
            for n in re.finditer(r"(?:DataFrame|Series|Scalar) '(\w+)'", m.group()):
                existing_names.add(n.group(1))

        # If no new variables exist AND there's already a state block at the end,
        # do NOT add a new placeholder block
        if existing_blocks:
            last_block = existing_blocks[-1]
            # Check if the last block is at or near the end of the code
            after_last = code[last_block.end():].strip()
            has_no_new_vars = not df_vars or not df_vars.strip()
            if has_no_new_vars and not after_last:
                # Nothing new to add, and there's already a state block at the end
                return code

        comment_lines = ["", "", "# ===== Execution State: Variables ====="]

        if df_vars and df_vars.strip():
            # Extract only the NEW variables from df_vars
            # df_vars format: "DataFrame 'name' - shape ...\n---\nSeries 'name2' ...\n---\nScalar 'name3' ..."
            new_blocks = []
            for block in df_vars.split("\n---\n"):
                name_match = re.match(r"(?:DataFrame|Series|Scalar) '(\w+)'", block.strip())
                if name_match and name_match.group(1) not in existing_names:
                    new_blocks.append(block)

            if new_blocks:
                new_content = "\n---\n".join(new_blocks)
                for line in new_content.split("\n"):
                    comment_lines.append(f"# {line}")
            else:
                comment_lines.append("# (No new variables in this subtask)")
        else:
            comment_lines.append("# (No new variables in this subtask)")

        comment_lines.append("# ===== End of Execution State =====")
        return code + "\n".join(comment_lines)
