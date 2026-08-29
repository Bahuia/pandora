"""
Pandora Base Dataset Interface

Abstract base class for all dataset types (NL2SQL, KBQA, TableQA).

All datasets implement this interface, providing task-specific:
- preprocess(): Convert raw example to standardized format
- postprocess(): Convert execution result to standardized answer
- evaluate(): Compare predicted answer with gold answer

The core inference flow is unified in PandoraAgent.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.

    Each task (NL2SQL/KBQA/TableQA) implements this class,
    providing differentiated preprocess, postprocess, and evaluate methods,
    while the core inference flow is uniformly handled by PandoraAgent.
    """

    def __init__(self, name: str, data_root: str):
        """
        Initialize base dataset.

        Args:
            name: Dataset name (e.g., "spider", "grailqa", "wikitq")
            data_root: Root directory for dataset files
        """
        self.name = name
        self.data_root = data_root

    @abstractmethod
    def load_examples(self, stage: str) -> list[dict]:
        """
        Load examples for a specific stage.

        Args:
            stage: Data stage ("train", "dev", "test", etc.)

        Returns:
            List of raw example dictionaries
        """
        pass

    @abstractmethod
    def preprocess(self, example: dict) -> dict:
        """
        Preprocess a raw example into standardized format.

        This is task-specific - each dataset type structures its
        schema and context differently.

        Args:
            example: Raw example dictionary from load_examples()

        Returns:
            Standardized format:
            {
                "question": str,           # The natural language question
                "schema": dict,            # Database/KG/Table schema
                "context": dict,           # Execution context (db_path, kg_dir, table_df, etc.)
                "hints": list[str],        # Optional hints/evidence
                "example_id": str,         # Unique example identifier
            }
        """
        pass

    @abstractmethod
    def postprocess(self, exec_result: dict, processed: dict) -> dict:
        """
        Postprocess execution result into standardized answer.

        This is task-specific - each dataset type formats its
        output differently.

        Args:
            exec_result: Execution result from code executor:
                {
                    "success": bool,
                    "result": Any,
                    "error": str | None,
                    "execution_time": float,
                }
            processed: Preprocessed example (from preprocess())

        Returns:
            {
                "answer": list[list],      # Standardized answer format
                "formatted": str,          # Optional formatted string
            }
        """
        pass

    @abstractmethod
    def evaluate(self, predicted: list, gold: list) -> dict[str, Any]:
        """
        Evaluate predicted answer against gold answer.

        This is task-specific - each dataset type uses different
        metrics and comparison logic.

        Args:
            predicted: Predicted answer (list of lists)
            gold: Gold answer (list of lists)

        Returns:
            {
                "em": float,               # Exact Match score
                "f1": float,               # F1 Score
                "correct": bool,           # Whether prediction is correct
            }
        """
        pass

    def get_schema(self, example_id: str) -> dict:
        """
        Get schema for a specific example.

        Optional method - some datasets may need to load
        schema on-demand rather than during preprocess.

        Args:
            example_id: Example identifier

        Returns:
            Schema dictionary
        """
        raise NotImplementedError("get_schema not implemented for this dataset")
