from pathlib import Path

from utils.config import Config, ConfigView


def test_config_view_reads_dotted_keys():
    view = ConfigView({"inference": {"shot_k": 10}})
    assert view.get("inference.shot_k") == 10
    assert view.get("inference.missing", 3) == 3


def test_config_paths_are_anchored_to_project_root():
    config = Config()
    assert Path(config.get("paths.data_root")).is_absolute()
    assert Path(config.get("paths.data_root")).name == "data"
