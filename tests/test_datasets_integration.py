from pathlib import Path
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    ],
)
def test_paper_dataset_adapters_load(task, name, stage, expected):
    config = Config(task_name=task)
    config._config["paths"]["data_root"] = str(DATA_ROOT.resolve())
    dataset = create_dataset(task, name, config)
    assert len(dataset.load_examples(stage)) == expected


@pytest.mark.parametrize(
    "task,name,stage",
    [
        ("nl2sql", "spider", "dev"),
        ("nl2sql", "spider-syn", "test"),
        ("nl2sql", "bird", "dev"),
        ("tableqa", "wikitq", "test"),
        ("tableqa", "wikisql", "test"),
    ],
)
def test_benchmark_first_example_preprocesses(task, name, stage):
    config = Config(task_name=task)
    config._config["paths"]["data_root"] = str(DATA_ROOT.resolve())
    dataset = create_dataset(task, name, config)
    example = dataset.load_examples(stage)[0]
    processed = dataset.preprocess(example)
    assert processed["question"]
    if task == "nl2sql" and name != "bird":
        assert Path(processed["context"]["db_path"]).is_file()
    if task == "tableqa":
        assert not processed["context"]["table_df"].empty


class _SmokeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        prompt = payload["messages"][-1]["content"]
        if "natural language to SQL" in prompt:
            content = json.dumps({"thinking": "smoke", "sql": "SELECT 1"})
        else:
            content = json.dumps({"thinking": "smoke", "code": "answer = []"})
        response = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        return


@pytest.fixture(scope="module")
def compatible_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SmokeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "task,name,stage",
    [
        ("nl2sql", "spider", "dev"),
        ("nl2sql", "spider-syn", "test"),
        ("nl2sql", "bird", "dev"),
        ("tableqa", "wikitq", "test"),
        ("tableqa", "wikisql", "test"),
    ],
)
def test_benchmark_cli_http_smoke(task, name, stage, compatible_endpoint, tmp_path):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "run.py"),
        "--mode", "vanilla",
        "--task", task,
        "--dataset", name,
        "--stage", stage,
        "--model", "pandora-smoke",
        "--provider", "openai-compatible",
        "--base-url", compatible_endpoint,
        "--data-root", str(DATA_ROOT.resolve()),
        "--output-dir", str(tmp_path),
        "--shot-k", "0",
        "--retrieval-mode", "disabled",
        "--num-samples", "1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    outputs = list(tmp_path.glob(f"{name}_{stage}_*.json"))
    assert len(outputs) == 1
    result = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert result["accuracy_metrics"]["total_samples"] == 1
