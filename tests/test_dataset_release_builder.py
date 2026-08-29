from scripts.build_dataset_release import transform
from datasets.tableqa.wikitq import WikiTQDataset


def test_wikitq_release_record_drops_training_wrappers():
    record = {
        "id": "q1",
        "question": "Question?",
        "table_id": "table-1",
        "table": {"header": ["x"], "rows": [[1]]},
        "answer_text": ["1"],
        "instruction": "wrapper",
        "output": "historical output",
    }
    assert transform("wikitq", record) == {
        "id": "q1",
        "question": "Question?",
        "table_id": "table-1",
        "table": {"header": ["x"], "rows": [[1]]},
        "answer_text": ["1"],
    }


def test_wikisql_release_record_compacts_table_metadata():
    record = {
        "question": "Question?",
        "table": {
            "id": "t1",
            "header": ["x"],
            "rows": [[1]],
            "types": ["real"],
            "page_title": "unused",
        },
        "answer_text": ["1"],
        "sql": {"sel": 0},
        "response": "unused",
    }
    transformed = transform("wikisql", record)
    assert transformed["id"] == "t1"
    assert "page_title" not in transformed["table"]
    assert "response" not in transformed


def test_wikitq_uses_embedded_table_when_csv_is_not_materialized(tmp_path):
    dataset = WikiTQDataset(str(tmp_path))
    processed = dataset.preprocess(
        {
            "id": "q1",
            "question": "What is x?",
            "table_id": "csv/missing.csv",
            "table": {"header": ["x"], "rows": [[1]]},
        }
    )
    assert processed["context"]["table_df"].iloc[0]["x"] == 1


def test_grailqa_release_record_drops_training_wrappers():
    record = {
        "qid": "1",
        "question": "Question?",
        "answer": {"answer_argument": ["m.1"]},
        "schema": "topic: m.2 | relation",
        "level": "compositional",
        "s_expression": "(JOIN relation m.2)",
        "sparql_query": "SELECT ?x WHERE {}",
        "instruction": "training wrapper",
        "formatted_input": "unused",
        "output": "historical output",
    }
    transformed = transform("grailqa", record)
    assert transformed["qid"] == "1"
    assert transformed["answer"] == {"answer_argument": ["m.1"]}
    assert "instruction" not in transformed
    assert "formatted_input" not in transformed
    assert "output" not in transformed
