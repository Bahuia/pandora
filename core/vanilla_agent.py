"""
Pandora Vanilla Agent

Simple one-shot LLM baseline for NL2SQL, KBQA, and TableQA.

Unlike PandoraAgent which uses code reasoning + execution feedback + code fusion,
VanillaAgent directly asks the LLM to produce the final answer in one call:
- NL2SQL  → Generate SQL directly
- KBQA    → Generate SPARQL directly
- TableQA → Generate Python code + answer directly

Reuse: model interface, template engine, dataset preprocess/postprocess/evaluate, config.
"""

import json
import re
import os
import time
import sqlite3
from pathlib import Path
from typing import Any, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets.base import BaseDataset
from models.base import BaseModel
from prompts.base import TemplateEngine
from utils.config import ConfigView
from utils.logger import setup_logger
from utils.answer_postprocess import postprocess_answer, postprocess_wikitq_answer
from core.voting import VoteAggregator


class VanillaAgent:
    """
    Simple one-shot LLM baseline agent.

    Flow per task type:
    - NL2SQL:  preprocess → prompt(LM) → parse SQL → execute SQL → evaluate
    - KBQA:    preprocess → prompt(LM) → parse SPARQL → evaluate (string match / gold comparison)
    - TableQA: preprocess → prompt(LM) → parse code → execute code → evaluate
    """

    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        config: Optional[dict] = None,
    ):
        self.dataset = dataset
        self.model = model
        self.config = ConfigView(config)
        self.logger = setup_logger("pandora.vanilla")
        prompt_root = self.config.get("paths.prompt_root", "./prompts")
        self.template_engine = TemplateEngine(template_dir=prompt_root)
        self.voting = VoteAggregator()

        # Configuration
        self.max_tries = self.config.get("inference.max_tries", 3)
        self.max_workers = self.config.get("inference.max_workers", 4)
        self.do_voting = self.config.get("inference.do_voting", False)

        # Detect task type
        self.task_type = self._detect_task_type()

        # Cache KB info (reuse same pattern as PandoraAgent)
        self.kb_info: Dict[str, Any] = {}
        if self.config.get("inference.preload_kb_info", False):
            self._prepare_kb_info()

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def run(self, example: dict, shot_k: int = 0) -> dict[str, Any]:
        """
        Single-inference run.

        Args:
            example: Raw example dict from dataset.
            shot_k: Number of few-shot examples (currently not injected; reserved).

        Returns:
            Result dict with answer, metrics, and intermediate data.
        """
        example_id = example.get("qid", example.get("id", example.get("question_id", "unknown")))
        self.logger.info(f"[Vanilla] Running inference for example {example_id}")

        start_time = time.time()

        # Step 1: Prepare example
        processed = self._prepare_example(example)
        question = processed.get("question", "")
        self.logger.info(f"[Vanilla] Task={self.task_type}, Question={question[:100]}...")

        # Step 2: Build & render prompt
        prompt = self._build_prompt(processed)

        # Step 3: Call LLM
        llm_start = time.time()
        raw_response = self._llm_call_with_retry(prompt)
        llm_time = time.time() - llm_start
        self.logger.info(f"[Vanilla] LLM response ({llm_time:.1f}s): {raw_response[:200]}...")

        # Step 4: Parse LLM response (task-specific)
        parsed = self._parse_response(raw_response)

        # Step 5: Execute / evaluate (task-specific)
        if self.task_type == "nl2sql":
            data = self._execute_and_evaluate_nl2sql(parsed, processed, example)
        elif self.task_type == "kbqa":
            data = self._execute_and_evaluate_kbqa(parsed, processed, example)
        elif self.task_type == "tableqa":
            data = self._execute_and_evaluate_tableqa(parsed, processed, example)
        else:
            self.logger.warning(f"Unknown task type: {self.task_type}")
            data = {"question": question, "raw_response": raw_response}

        # Build result
        total_time = time.time() - start_time
        result = self._build_result(example_id, data, parsed, total_time)
        return result

    def run_with_voting(
        self,
        example: dict,
        n_votes: int = 5,
        shot_k: int = 0,
    ) -> dict[str, Any]:
        """Multi-vote inference with parallel execution."""
        example_id = example.get("qid", example.get("id", example.get("question_id", "unknown")))
        self.logger.info(f"[Vanilla] Running {n_votes}-vote inference for example {example_id}")

        votes = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_vote = {
                executor.submit(self.run, example, shot_k): i + 1 for i in range(n_votes)
            }
            for future in as_completed(future_to_vote):
                vote_id = future_to_vote[future]
                try:
                    result = future.result()
                    result["vote_id"] = vote_id
                    votes.append(result)
                    self.logger.info(f"[Vanilla] Vote {vote_id}/{n_votes} completed")
                except Exception as e:
                    self.logger.error(f"[Vanilla] Vote {vote_id} failed: {e}")
                    votes.append({
                        "vote_id": vote_id,
                        "success": False,
                        "error": str(e),
                        "answer": None,
                    })

        if not votes:
            return {"example_id": example_id, "success": False, "error": "All votes failed", "n_votes": n_votes}

        aggregated = self.voting.aggregate(votes)
        aggregated["example_id"] = example_id
        aggregated["n_votes"] = n_votes
        aggregated["all_votes"] = sorted(votes, key=lambda vote: int(vote.get("vote_id", 0)))
        return aggregated

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _detect_task_type(self) -> str:
        """Detect task type from dataset name."""
        dataset_name = self.dataset.name.lower()
        if dataset_name in ["grailqa", "webqsp", "cwq"]:
            return "kbqa"
        elif dataset_name in ["wikitq", "wikitablequestions", "wikisql"]:
            return "tableqa"
        else:
            return "nl2sql"

    def _prepare_kb_info(self):
        """Pre-cache KB info (same pattern as PandoraAgent, simplified)."""
        try:
            from utils.schema_utils import prepare_kb_info, load_column_descriptions
            from utils.file_utils import load_json

            data_root = Path(self.config.get("paths.data_root", "./data"))
            dataset_name = self.dataset.name.lower()

            column_descriptions = {}
            schema_data = None

            if dataset_name in ("spider", "spider-syn"):
                db_dir = data_root / "spider" / "dev_database"
                schema_file = data_root / "spider" / "spider.tables.dev.json"
            else:
                db_dir = data_root / "bird" / "dev_database"
                schema_file = data_root / "bird" / "bird.tables.dev.json"

            if dataset_name in ("spider", "spider-syn", "bird"):
                column_descriptions = load_column_descriptions(str(db_dir))
                if schema_file.exists():
                    schema_data = {x["db_id"]: x for x in load_json(schema_file)}

            kb_dir = db_dir if dataset_name in ("spider", "spider-syn", "bird") else None
            if kb_dir and kb_dir.exists():
                for kb_id in kb_dir.iterdir():
                    if not kb_id.is_dir():
                        continue
                    try:
                        kb_schema = schema_data.get(kb_id.name) if schema_data else None
                        kb_info = prepare_kb_info(
                            kb_type="db",
                            kb_id=kb_id.name,
                            data_dir=str(kb_dir),
                            schema_data=kb_schema,
                            column_descriptions=column_descriptions,
                            top_k_row=3,
                        )
                        self.kb_info[kb_id.name] = kb_info
                    except Exception as e:
                        self.logger.warning(f"Failed to prepare kb_info for {kb_id}: {e}")

            self.logger.info(f"[Vanilla] Prepared kb_info for {len(self.kb_info)} KBs")
        except Exception as e:
            self.logger.warning(f"[Vanilla] KB info preparation failed: {e}")

    def _prepare_example(self, example: dict) -> dict:
        """Preprocess example using dataset-specific logic."""
        # For NL2SQL tasks, use cached kb_info
        if self.task_type == "nl2sql":
            kb_id = example.get("db_id", "california_schools")
            if kb_id not in self.kb_info:
                self.logger.warning(f"kb_info not found for {kb_id}, loading on-demand")
                self._load_kb_info_on_demand(kb_id)

            kb_info = self.kb_info.get(kb_id, {})
            return {
                "question": example.get("question", ""),
                "schema": {
                    "box_schema": kb_info.get("box_schema", ""),
                    "table_content": kb_info.get("table_content", ""),
                    "primary_keys": kb_info.get("primary_keys", ""),
                    "foreign_keys": kb_info.get("foreign_keys", ""),
                },
                "context": {
                    "db_path": kb_info.get("db_path", ""),
                },
                "evidence": example.get("evidence", ""),
                "db_id": kb_id,
            }

        # For KBQA/TableQA, use dataset.preprocess()
        return self.dataset.preprocess(example)

    def _load_kb_info_on_demand(self, kb_id: str):
        """Load KB info on-demand if not cached."""
        try:
            from utils.schema_utils import prepare_kb_info
            from utils.file_utils import load_json

            data_root = Path(self.config.get("paths.data_root", "./data"))
            dataset_name = self.dataset.name.lower()

            if dataset_name in ("spider", "spider-syn"):
                kb_dir = data_root / "spider" / "dev_database"
                schema_file = data_root / "spider" / "spider.tables.dev.json"
            else:
                kb_dir = data_root / "bird" / "dev_database"
                schema_file = data_root / "bird" / "bird.tables.dev.json"

            schema_data = None
            if schema_file.exists():
                schema_data = {x["db_id"]: x for x in load_json(schema_file)}

            kb_schema = schema_data.get(kb_id) if schema_data else None
            kb_info = prepare_kb_info(
                kb_type="db",
                kb_id=kb_id,
                data_dir=str(kb_dir),
                schema_data=kb_schema,
                column_descriptions={},
                top_k_row=3,
            )
            self.kb_info[kb_id] = kb_info
        except Exception as e:
            self.logger.warning(f"Failed to load kb_info on-demand for {kb_id}: {e}")

    def _build_prompt(self, processed: dict) -> str:
        """Build task-specific vanilla prompt."""
        try:
            template = self.template_engine.get_template("vanilla", self.task_type)
        except FileNotFoundError:
            self.logger.warning(f"Vanilla template not found for task={self.task_type}, using fallback")
            return self._fallback_prompt(processed)

        schema = processed.get("schema", {})
        context = processed.get("context", {})

        variables = {
            "question": processed.get("question", ""),
            "evidence": processed.get("evidence", ""),
            "box_schema": schema.get("box_schema", ""),
            "table_content": schema.get("table_content", ""),
            "primary_keys": schema.get("primary_keys", ""),
            "foreign_keys": schema.get("foreign_keys", ""),
        }

        # KBQA: add entity links
        if self.task_type == "kbqa":
            variables["entity_links"] = json.dumps(
                context.get("entity_links", {}), indent=2, ensure_ascii=False
            )

        return template.render(**variables)

    def _fallback_prompt(self, processed: dict) -> str:
        """Fallback prompt when template is missing."""
        question = processed.get("question", "")
        schema = processed.get("schema", {})
        task_label = {
            "nl2sql": "SQL query",
            "kbqa": "SPARQL query",
            "tableqa": "Python code",
        }.get(self.task_type, "answer")

        return (
            f"Given the following schema and question, generate a {task_label} to answer the question.\n\n"
            f"Schema:\n{schema.get('box_schema', '')}\n\n"
            f"Question: {question}\n\n"
            f"Return your answer as JSON: {{\"thinking\": \"...\", \"answer\": \"...\"}}"
        )

    def _llm_call_with_retry(self, prompt: str) -> str:
        """Call LLM with retry on transient failures."""
        last_error = None
        for attempt in range(self.max_tries):
            try:
                response = self.model.generate(prompt)
                return response
            except Exception as e:
                last_error = e
                self.logger.warning(f"[Vanilla] LLM call attempt {attempt + 1}/{self.max_tries} failed: {e}")
                time.sleep(1 * (attempt + 1))

        raise RuntimeError(f"LLM call failed after {self.max_tries} attempts: {last_error}")

    @staticmethod
    def _extract_json_from_response(response: str) -> dict:
        """Extract JSON from LLM response text."""
        # Strategy 1: JSON code block
        json_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 2: Brace matching
        start_idx = response.find("{")
        if start_idx != -1:
            brace_count = 0
            for i in range(start_idx, len(response)):
                if response[i] == "{":
                    brace_count += 1
                elif response[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(response[start_idx: i + 1])
                        except json.JSONDecodeError:
                            break

        return {}

    def _parse_response(self, raw_response: str) -> dict:
        """Parse LLM response into structured dict."""
        parsed = self._extract_json_from_response(raw_response)

        if self.task_type == "nl2sql":
            return {
                "thinking": parsed.get("thinking", ""),
                "sql": parsed.get("sql", parsed.get("answer", "")),
            }
        elif self.task_type == "kbqa":
            return {
                "thinking": parsed.get("thinking", ""),
                "sparql": parsed.get("sparql", parsed.get("answer", "")),
            }
        elif self.task_type == "tableqa":
            return {
                "thinking": parsed.get("thinking", ""),
                "code": parsed.get("code", parsed.get("answer", "")),
            }

        return {"thinking": parsed.get("thinking", ""), "raw": raw_response}

    # ── Task-specific execution & evaluation ──

    def _execute_and_evaluate_nl2sql(
        self, parsed: dict, processed: dict, example: dict
    ) -> dict:
        """Execute predicted SQL, get gold SQL result, compare."""
        sql = parsed.get("sql", "")
        db_path = processed.get("context", {}).get("db_path", "")

        data = {
            "question": processed.get("question", ""),
            "predicted_sql": sql,
            "thinking": parsed.get("thinking", ""),
            "gold_answer": [],
            "predicted_answer": [],
        }

        # Execute predicted SQL
        if sql and db_path:
            try:
                data["predicted_answer"] = self._execute_sql(sql, db_path)
                self.logger.info(f"[Vanilla-NL2SQL] Predicted SQL executed: {len(data['predicted_answer'])} rows")
            except Exception as e:
                self.logger.warning(f"[Vanilla-NL2SQL] Predicted SQL execution failed: {e}")
                data["predicted_answer"] = []
                data["sql_error"] = str(e)

        # Execute gold SQL
        gold_sql = example.get("SQL", "") or example.get("query", "") or example.get("label", "")
        if gold_sql and db_path:
            try:
                data["gold_answer"] = self._execute_sql(gold_sql, db_path)
                self.logger.info(f"[Vanilla-NL2SQL] Gold SQL executed: {len(data['gold_answer'])} rows")
            except Exception as e:
                self.logger.warning(f"[Vanilla-NL2SQL] Gold SQL execution failed: {e}")
                data["gold_answer"] = []

        # Evaluate
        metrics = self.dataset.evaluate(data["predicted_answer"], data["gold_answer"])
        data["metrics"] = metrics
        return data

    def _execute_and_evaluate_kbqa(
        self, parsed: dict, processed: dict, example: dict
    ) -> dict:
        """For KBQA, compare predicted SPARQL with gold SPARQL."""
        sparql = parsed.get("sparql", "")

        data = {
            "question": processed.get("question", ""),
            "predicted_sparql": sparql,
            "thinking": parsed.get("thinking", ""),
        }

        # Get gold SPARQL
        gold_sparql = (
            example.get("sparql_query", "")
            or example.get("label", "")
            or example.get("s_expression", "")
        )
        data["gold_sparql"] = gold_sparql

        # For KBQA, use set-based EM/F1 on gold answer
        gold_answer = []
        if hasattr(self.dataset, "get_gold_answer"):
            gold_answer = self.dataset.get_gold_answer(example) or []

        # Postprocess: use dataset's postprocess with a dummy exec_result
        # Since vanilla doesn't execute code, we need to parse the SPARQL or
        # directly use the gold answer for evaluation
        predicted_list = self._parse_kbqa_answer(sparql)

        metrics = self.dataset.evaluate(predicted_list, gold_answer)
        data["metrics"] = metrics
        data["predicted_answer"] = predicted_list
        data["gold_answer"] = gold_answer
        return data

    def _execute_and_evaluate_tableqa(
        self, parsed: dict, processed: dict, example: dict
    ) -> dict:
        """Execute predicted Python code on table, compare with gold answer."""
        code = parsed.get("code", "")
        table_df = processed.get("context", {}).get("table_df")

        data = {
            "question": processed.get("question", ""),
            "predicted_code": code,
            "thinking": parsed.get("thinking", ""),
            "gold_answer": [],
            "predicted_answer": [],
        }

        # Execute predicted code
        if code and table_df is not None:
            try:
                exec_env = {"pd": __import__("pandas"), "table": table_df.copy()}
                exec(code, exec_env)
                answer = exec_env.get("answer", None)
                data["predicted_answer"] = self._normalize_tableqa_answer(answer)
                self.logger.info(f"[Vanilla-TableQA] Code executed, answer: {data['predicted_answer']}")
            except Exception as e:
                self.logger.warning(f"[Vanilla-TableQA] Code execution failed: {e}")
                data["predicted_answer"] = []
                data["code_error"] = str(e)

        # Get gold answer
        if hasattr(self.dataset, "get_gold_answer"):
            data["gold_answer"] = self.dataset.get_gold_answer(example) or []

        # Evaluate
        metrics = self.dataset.evaluate(data["predicted_answer"], data["gold_answer"])
        data["metrics"] = metrics
        return data

    # ── Utility methods ──

    @staticmethod
    def _execute_sql(sql: str, db_path: str) -> list:
        """Execute SQL query against SQLite database."""
        if not db_path or not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [list(row) for row in rows]
        finally:
            conn.close()

    @staticmethod
    def _parse_kbqa_answer(sparql: str) -> list:
        """Try to extract entity values from SPARQL for evaluation."""
        if not sparql:
            return []
        # Fallback: return SPARQL as a single-item answer for string comparison
        # Real KBQA evaluation would need a SPARQL endpoint
        return [[sparql.strip()]]

    @staticmethod
    def _normalize_tableqa_answer(answer) -> list:
        """Normalize TableQA answer to list-of-lists format."""
        if answer is None:
            return []
        if isinstance(answer, list):
            return [[str(v)] for v in answer] if answer and not isinstance(answer[0], list) else answer
        return [[str(answer)]]

    def _build_result(
        self,
        example_id: str,
        data: dict,
        parsed: dict,
        total_time: float,
    ) -> dict:
        """Build standardized result dictionary."""
        return {
            "example_id": example_id,
            "question": data.get("question", ""),
            "thinking": parsed.get("thinking", ""),
            "metrics": data.get("metrics", {"em": 0.0, "f1": 0.0, "correct": False}),
            "answer": data.get("predicted_answer", []),
            "gold_answer": data.get("gold_answer", []),
            # Task-specific
            "predicted_sql": data.get("predicted_sql", ""),
            "predicted_sparql": data.get("predicted_sparql", ""),
            "predicted_code": data.get("predicted_code", ""),
            "gold_sparql": data.get("gold_sparql", ""),
            "raw_response": data.get("raw_response", ""),
            # Timing
            "total_time_sec": round(total_time, 2),
            "agent": "vanilla",
        }
