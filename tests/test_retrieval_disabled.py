from unittest.mock import Mock, patch

from core.agent import PandoraAgent


def test_disabled_retrieval_does_not_scan_example_store(tmp_path):
    dataset = Mock(name="dataset")
    dataset.name = "spider"
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "prompt_root": str(tmp_path / "prompts"),
            "cache_root": str(tmp_path / "cache"),
        },
        "inference": {"shot_k": 0},
        "retrieval": {"enabled": True, "mode": "disabled"},
    }
    with patch("core.agent.load_verified_memory", side_effect=AssertionError("unexpected scan")):
        agent = PandoraAgent(dataset=dataset, model=Mock(), config=config)
    assert agent.memory_retriever is None
