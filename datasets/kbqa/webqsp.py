"""
WebQSP Dataset - KBQA Task

Implementation of the WebQuestionsSP dataset for knowledge base question answering.
Uses the same subgraph CSV table format as GrailQA.
"""

import json
from pathlib import Path
from typing import Any, Optional

from .grailqa import GrailQADataset
from utils.file_utils import load_json
from utils.logger import setup_logger


class WebQSPDataset(GrailQADataset):
    """
    WebQSP dataset for KBQA tasks.

    Extends GrailQADataset with WebQSP-specific format:
    - Question comes from raw_data.RawQuestion or raw_data.ProcessedQuestion
    - id field uses QuestionId format (e.g., WebQTest-0)
    - Entity linking from webqsp.entity_link.test.json
    - Gold answers from raw_data.Parses
    """

    def __init__(self, data_root: str = "./data"):
        # Skip GrailQADataset.__init__ — we override everything
        from datasets.base import BaseDataset
        BaseDataset.__init__(self, name="webqsp", data_root=data_root)
        self.kg_path = Path(data_root) / "webqsp" / "box"
        self.schema_path = self.kg_path / "box_schema.json"
        self.logger = setup_logger("pandora.dataset.webqsp")

        # Load box_schema cache
        self._box_schema_cache = {}
        if self.schema_path.exists():
            from utils.file_utils import load_json
            self._box_schema_cache = load_json(str(self.schema_path))
            self.logger.info(f"Loaded box_schema for {len(self._box_schema_cache)} questions")

        # Load entity linking cache
        self._entity_link_cache = {}
        self._load_entity_link_cache()

    def _load_entity_link_cache(self):
        """Pre-load entity linking info."""
        el_path = Path(self.data_root) / "webqsp" / "entity_link" / "webqsp.entity_link.test.json"
        if el_path.exists():
            try:
                with open(el_path, encoding="utf-8") as f:
                    self._entity_link_cache = json.load(f)
                self.logger.info(f"Loaded entity links for {len(self._entity_link_cache)} questions")
            except Exception as e:
                self.logger.warning(f"Failed to load entity link cache: {e}")

    def get_entity_links(self, qid: str) -> dict:
        """Get entity linking hints for a specific question."""
        return self._entity_link_cache.get(str(qid), {})

    def load_examples(self, stage: str = "test", qids: Optional[list] = None) -> list:
        """Load WebQSP examples."""
        data_file = Path(self.data_root) / "webqsp" / f"webqsp.{stage}.json"
        if not data_file.exists():
            data_file = Path(self.data_root) / "webqsp" / f"{stage}.json"

        if not data_file.exists():
            raise FileNotFoundError(f"WebQSP {stage} data not found at {data_file}")

        with open(data_file, encoding="utf-8") as f:
            examples = json.load(f)

        self.logger.info(f"Loaded {len(examples)} WebQSP {stage} examples")

        if qids:
            qid_set = set(str(q) for q in qids)
            examples = [ex for ex in examples if str(ex.get("id", ex.get("qid", ""))) in qid_set]
            self.logger.info(f"Filtered to {len(examples)} examples by qid")

        return examples

    def get_gold_answer(self, example: dict) -> list:
        """
        Extract gold answer from WebQSP example.

        Gold answers are in raw_data.Parses[].Answers[].AnswerArgument (Freebase IDs).
        Collects all unique answers across all parses.
        Returns list of lists: [[answer1], [answer2], ...]
        """
        gold_ids = set()

        raw = example.get("raw_data", {})
        parses = raw.get("Parses", [])
        for parse in parses:
            answers = parse.get("Answers", [])
            for ans in answers:
                # AnswerArgument contains the Freebase ID (e.g., "m.01428y")
                eid = ans.get("AnswerArgument", "")
                if eid and eid.strip():
                    gold_ids.add(eid.strip())

        if gold_ids:
            return [[eid] for eid in sorted(gold_ids)]

        return []

    def preprocess(self, example: dict) -> dict:
        """
        Preprocess WebQSP example into standardized format.

        Key difference from GrailQA: the question comes from raw_data.
        """
        qid = str(example.get("id", example.get("qid", "unknown")))
        raw = example.get("raw_data", {})

        # Get question from raw_data
        question = raw.get("ProcessedQuestion", raw.get("RawQuestion", ""))
        if not question:
            question = example.get("question", "")

        evidence = example.get("evidence", "")

        # Get box_schema
        box_schema = self.get_box_schema(qid)
        if not box_schema:
            self.logger.warning(f"No box_schema found for qid={qid}")

        # Get KG directory
        qid_dir = self.kg_path / "test" / qid

        # Match CSV files to schema table names
        csv_to_schema_map = self._match_csv_to_schema_tables(qid_dir, box_schema)

        # Get entity linking hints
        entity_links = self.get_entity_links(qid)

        # Filter entity_links using the schema field
        entity_links = self._filter_entity_links(example, entity_links)

        # Build table content using matched CSVs
        table_content = self._build_table_content(
            qid_dir, csv_to_schema_map, entity_ids=list(entity_links.values())
        )

        # Load foreign keys
        foreign_keys = self.get_foreign_keys(qid)

        # Generate primary keys from box_schema
        primary_keys = self._extract_primary_keys(box_schema)

        hints = []
        for entity_name, entity_id in entity_links.items():
            hints.append(f'Entity "{entity_name}" → Freebase ID: `{entity_id}`')

        schema = {
            "box_schema": box_schema,
            "table_content": table_content,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "tables": {},
            "kb_type": "kg",
        }

        context = {
            "qid": qid,
            "kg_dir": str(qid_dir),
            "entity_links": entity_links,
            "kb_type": "kg",
            "csv_to_schema_map": csv_to_schema_map,
            "box_schema": box_schema,
        }

        return {
            "question": question,
            "schema": schema,
            "context": context,
            "hints": hints,
            "example_id": qid,
            "evidence": evidence,
        }
