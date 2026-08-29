from pathlib import Path

import pytest

from run import create_dataset
from utils.config import Config


DATA_ROOT = Path(__import__("os").environ.get("PANDORA_DATA_ROOT", "data"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATA_ROOT.exists(), reason="benchmark data is not installed"),
]


@pytest.mark.parametrize(
    "task,name,stage,expected",
    [
        ("nl2sql", "spider", "dev", 1034),
        ("nl2sql", "spider-syn", "test", 1034),
        ("nl2sql", "bird", "dev", 1534),
        ("tableqa", "wikitq", "test", 4344),
        ("tableqa", "wikisql", "test", 15878),
        ("kbqa", "grailqa", "test", 6463),
        ("kbqa", "webqsp", "test", 1616),
    ],
)
def test_paper_dataset_adapters_load(task, name, stage, expected):
    config = Config(task_name=task)
    config._config["paths"]["data_root"] = str(DATA_ROOT.resolve())
    dataset = create_dataset(task, name, config)
    assert len(dataset.load_examples(stage)) == expected
