"""
Spider-Syn Dataset - NL2SQL Task

Implementation of the Spider-Syn dataset for text-to-SQL tasks.
Spider-Syn shares the same databases as the original Spider dataset.
"""

from pathlib import Path
from typing import Any

from datasets.nl2sql.spider import SpiderDataset
from utils.file_utils import load_json
from utils.logger import setup_logger


class SpiderSynDataset(SpiderDataset):
    """
    Spider-Syn dataset for NL2SQL tasks.
    Reuses Spider's database schemas and paths.
    """

    def __init__(self, data_root: str = "./data"):
        super().__init__(data_root=data_root)
        # Override name
        self.name = "spider-syn"
        self.logger = setup_logger("pandora.dataset.spider_syn")
        # Re-use Spider's DB path
        # self.db_path is already set by super() to data/spider/database

    def load_examples(self, stage: str = "test", qids: list = None) -> list[dict]:
        """Load Spider-Syn examples."""
        # Spider-Syn usually has a test file
        data_file = Path(self.data_root) / "spider-syn" / f"spider-syn.{stage}.json"

        if not data_file.exists():
            raise FileNotFoundError(f"Spider-Syn {stage} data not found at {data_file}")

        examples = load_json(data_file)
        self.logger.info(f"Loaded {len(examples)} Spider-Syn {stage} examples")

        if qids:
            qid_set = set(str(q) for q in qids)
            examples = [ex for ex in examples if str(ex.get("id", "")) in qid_set]

        return examples
