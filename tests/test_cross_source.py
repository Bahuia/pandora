import json
import sqlite3

import pandas as pd

from datasets.cross_source import CrossSourceDataset
from utils.code_executor import CodeExecutor


def test_cross_source_alignment_and_execution(tmp_path):
    database = tmp_path / "people.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE people (name TEXT, role TEXT)")
    connection.execute("INSERT INTO people VALUES ('Alice Smith', 'director')")
    connection.commit()
    connection.close()
    awards = tmp_path / "awards.csv"
    pd.DataFrame({"person": ["Alice Smith"], "award": ["Gold"]}).to_csv(awards, index=False)

    data_root = tmp_path / "data"
    manifest_root = data_root / "cross_source"
    manifest_root.mkdir(parents=True)
    manifest = [{
        "id": "x1",
        "question": "Which award did the director receive?",
        "sources": [
            {"kind": "db", "path": str(database), "prefix": "db_"},
            {"kind": "table", "path": str(awards), "prefix": "csv_"},
        ],
        "gold_answer": [["Gold"]],
    }]
    (manifest_root / "cross_source.test.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = CrossSourceDataset(str(data_root))
    processed = dataset.preprocess(dataset.load_examples("test")[0])
    assert "normalized_exact" in processed["schema"]["foreign_keys"]
    result = CodeExecutor(timeout=10).execute(
        "joined = db_people.merge(csv_awards, left_on='name', right_on='person')\n"
        "result = list(joined[['award']].itertuples(index=False, name=None))",
        processed["context"],
    )
    assert result.success
    assert result.result == [("Gold",)]
