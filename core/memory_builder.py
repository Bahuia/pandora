"""Progressive construction of verified Pandas demonstration memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from core.agent import PandoraAgent
from datasets.base import BaseDataset


class ProgressiveMemoryBuilder:
    """Generate demonstrations and retain only execution-verified examples."""

    def __init__(self, agent: PandoraAgent, dataset: BaseDataset, output_path: Path):
        self.agent = agent
        self.dataset = dataset
        self.output_path = output_path

    def build(
        self,
        examples: Iterable[dict],
        target_count: int,
        shot_k: int,
        resume: bool = True,
    ) -> list[dict]:
        records = self._load_existing() if resume else []
        completed = {(str(item.get("dataset_id")), str(item.get("id"))) for item in records}

        for example in examples:
            if len(records) >= target_count:
                break
            example_id = self._example_id(example)
            key = (self.dataset.name, example_id)
            if key in completed:
                continue

            result = self.agent.run(example, shot_k=shot_k)
            metrics = result.get("metrics") or {}
            if not metrics.get("correct", False):
                continue

            processed = self.agent._prepare_example(example)
            schema = processed.get("schema", {}).get("box_schema", "")
            record = {
                "dataset_id": self.dataset.name,
                "id": example_id,
                "question": result.get("question", processed.get("question", "")),
                "schema": schema,
                "response": json.dumps({
                    "reasoning": result.get("merged_thinking", ""),
                    "code": result.get("merged_code", ""),
                }, ensure_ascii=False),
                "exec result": {
                    "predicted": result.get("answer", []),
                    "gold": result.get("gold_answer", []),
                    "metrics": metrics,
                    "res comp": True,
                },
                "verification": {
                    "execution_success": bool(
                        (result.get("python_results") or {}).get("success", False)
                    ),
                    "code_repair_attempts": result.get("code_repair_attempts", 0),
                },
            }
            records.append(record)
            completed.add(key)
            self._save(records)
        return records

    def _load_existing(self) -> list[dict]:
        if not self.output_path.exists():
            return []
        with self.output_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _save(self, records: list[dict]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.output_path)

    @staticmethod
    def _example_id(example: dict) -> str:
        return str(example.get(
            "qid", example.get("id", example.get("question_id", example.get("db_id", "unknown")))
        ))
