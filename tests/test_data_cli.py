import json
from pathlib import Path

import requests

import pandora_data.cli as data_cli
from pandora_data.cli import (
    _jsonl_to_json,
    _manifest,
    _materialize_webqsp,
    command_prepare,
    verify_dataset,
)


def _spider_source(root: Path) -> Path:
    source = root / "official-spider"
    source.mkdir()
    examples = [{"question_id": index, "db_id": f"db{index % 20}"} for index in range(1034)]
    (source / "dev.json").write_text(json.dumps(examples), encoding="utf-8")
    (source / "tables.json").write_text("[]", encoding="utf-8")
    databases = source / "database"
    for index in range(20):
        database = databases / f"db{index}"
        database.mkdir(parents=True)
        (database / f"db{index}.sqlite").write_bytes(b"")
    return source


def test_prepare_imports_official_spider_layout(tmp_path):
    manifest = _manifest()
    data_root = tmp_path / "data"
    result = command_prepare(
        ["spider"], data_root, tmp_path / "cache", _spider_source(tmp_path), manifest
    )
    assert result == 0
    ready, issues = verify_dataset("spider", data_root, manifest)
    assert ready, issues
    assert (data_root / "spider" / "spider.tables.dev.json").exists()


def test_jsonl_materialization_is_count_checked(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    target = tmp_path / "records.json"
    _jsonl_to_json(source, target, expected_count=2)
    assert json.loads(target.read_text(encoding="utf-8")) == [{"id": 1}, {"id": 2}]


def test_spider_syn_reports_spider_dependency(tmp_path):
    manifest = _manifest()
    annotation = tmp_path / "spider-syn" / "spider-syn.test.json"
    annotation.parent.mkdir()
    annotation.write_text(json.dumps([{}] * 1034), encoding="utf-8")
    ready, issues = verify_dataset("spider-syn", tmp_path, manifest)
    assert not ready
    assert any("dependency spider" in issue for issue in issues)


def test_manifest_artifacts_materialize_their_own_dataset():
    manifest = _manifest()
    for name in ("spider-syn", "bird", "wikitq", "wikisql", "grailqa", "webqsp"):
        for artifact in manifest["datasets"][name]["artifacts"]:
            assert artifact["target"] == "." or artifact["target"].startswith(f"{name}/")


def test_webqsp_official_conversion_uses_published_subset(tmp_path):
    dataset_root = tmp_path / "webqsp"
    dataset_root.mkdir()
    (dataset_root / "subset.test.json").write_text(
        json.dumps({"qids": ["WebQTest-0"]}), encoding="utf-8"
    )
    source = tmp_path / "WebQSP.test.json"
    source.write_text(
        json.dumps(
            {
                "Questions": [
                    {
                        "QuestionId": "WebQTest-0",
                        "RawQuestion": "Who?",
                        "ProcessedQuestion": "who",
                        "Parses": [
                            {
                                "TopicEntityName": "Topic",
                                "TopicEntityMid": "m.topic",
                                "Answers": [{"AnswerArgument": "m.answer", "EntityName": "Answer"}],
                            }
                        ],
                    },
                    {"QuestionId": "WebQTest-1", "Parses": []},
                ]
            }
        ),
        encoding="utf-8",
    )

    _materialize_webqsp(source, dataset_root)

    records = json.loads((dataset_root / "webqsp.test.json").read_text(encoding="utf-8"))
    links = json.loads(
        (dataset_root / "entity_link" / "webqsp.entity_link.test.json").read_text(encoding="utf-8")
    )
    assert [record["id"] for record in records] == ["WebQTest-0"]
    assert records[0]["schema"] == "Topic: m.topic |"
    assert links == {"WebQTest-0": {"Topic": "m.topic", "Answer": "m.answer"}}


def test_cli_download_error_explains_proxy_and_source_options(monkeypatch, capsys, tmp_path):
    def fail_prepare(*_args, **_kwargs):
        raise requests.exceptions.SSLError("untrusted proxy certificate")

    monkeypatch.setattr(data_cli, "command_prepare", fail_prepare)
    result = data_cli.cli(["prepare", "--dataset", "webqsp", "--root", str(tmp_path / "data")])

    assert result == 2
    message = capsys.readouterr().err
    assert "REQUESTS_CA_BUNDLE" in message
    assert "--source" in message
    assert "microsoft.com" in message
