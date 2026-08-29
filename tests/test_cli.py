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
