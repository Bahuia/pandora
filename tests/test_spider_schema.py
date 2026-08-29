import json

from datasets.nl2sql.spider import SpiderDataset


def test_spider_schema_loader_builds_tables_and_keys(tmp_path):
    root = tmp_path / "spider"
    root.mkdir()
    schema = [{
        "db_id": "school",
        "table_names_original": ["student", "class"],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "class_id"], [1, "id"]],
        "column_types": ["text", "number", "number", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[2, 3]],
    }]
    (root / "dev_tables.json").write_text(json.dumps(schema), encoding="utf-8")

    loaded = SpiderDataset(str(tmp_path))._load_schema("school")
    assert loaded["tables"]["student"][0] == {"name": "id", "type": "number"}
    assert loaded["primary_keys"][1] == {"table": "class", "column": "id"}
    assert loaded["foreign_keys"][0]["from"] == {"table": "student", "column": "class_id"}
