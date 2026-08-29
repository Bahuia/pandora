from run import parse_args


def test_cli_accepts_external_roots():
    args = parse_args([
        "--task", "nl2sql", "--dataset", "spider",
        "--data-root", "/tmp/data", "--config-dir", "/tmp/configs",
        "--base-url", "https://example.invalid/v1",
    ])
    assert args.data_root == "/tmp/data"
    assert args.config_dir == "/tmp/configs"
    assert args.base_url == "https://example.invalid/v1"


def test_cli_accepts_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("PANDORA_BASE_URL", "https://gateway.example.invalid/v1")
    args = parse_args([
        "--task", "tableqa", "--dataset", "wikitq",
        "--provider", "openai-compatible", "--shot-k", "0",
        "--retrieval-mode", "disabled",
    ])
    assert args.provider == "openai-compatible"
    assert args.base_url == "https://gateway.example.invalid/v1"
    assert args.shot_k == 0
    assert args.retrieval_mode == "disabled"
