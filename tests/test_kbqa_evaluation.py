import json

from datasets.kbqa.grailqa import GrailQADataset
from datasets.kbqa.webqsp import WebQSPDataset


def test_empty_gold_is_not_counted_as_correct():
    dataset = GrailQADataset.__new__(GrailQADataset)

    metrics = dataset.evaluate([], [])

    assert metrics == {
        "em": 0.0,
        "f1": 0.0,
        "hit_1": 0.0,
        "correct": False,
        "evaluable": False,
    }


def test_prepared_grailqa_subset_excludes_missing_box(tmp_path):
    dataset_root = tmp_path / "grailqa"
    (dataset_root / "box" / "test" / "kept").mkdir(parents=True)
    (dataset_root / "box" / "box_schema.json").write_text(
        json.dumps({"kept": "table = pd.DataFrame({})"}), encoding="utf-8"
    )
    examples = [
        {"qid": "kept", "question": "kept", "answer": {"answer_argument": ["m.1"]}},
        {"qid": "missing", "question": "missing", "answer": {"answer_argument": ["m.2"]}},
    ]
    (dataset_root / "grailqa.test.json").write_text(json.dumps(examples), encoding="utf-8")

    dataset = GrailQADataset(str(tmp_path))

    assert [example["qid"] for example in dataset.load_examples("test")] == ["kept"]


def test_prepared_webqsp_subset_excludes_empty_gold(tmp_path):
    dataset_root = tmp_path / "webqsp"
    for qid in ("kept", "empty-gold"):
        (dataset_root / "box" / "test" / qid).mkdir(parents=True, exist_ok=True)
    (dataset_root / "box" / "box_schema.json").write_text(
        json.dumps({"kept": "table = pd.DataFrame({})", "empty-gold": "table = pd.DataFrame({})"}),
        encoding="utf-8",
    )
    examples = [
        {
            "id": "kept",
            "raw_data": {"Parses": [{"Answers": [{"AnswerArgument": "m.1"}]}]},
        },
        {"id": "empty-gold", "raw_data": {"Parses": [{"Answers": []}]}},
    ]
    (dataset_root / "webqsp.test.json").write_text(json.dumps(examples), encoding="utf-8")

    dataset = WebQSPDataset(str(tmp_path))

    assert [example["id"] for example in dataset.load_examples("test")] == ["kept"]
