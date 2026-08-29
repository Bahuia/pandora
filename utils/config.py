"""
Pandora Configuration Module

Loads and merges configuration from YAML files.
"""

import os
from importlib.resources import files
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterator, Optional

from .file_utils import load_yaml


class Config:
    """
    Configuration manager for Pandora.

    Loads default config and merges with task-specific overrides.
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        """
        Initialize configuration.

        Args:
            config_dir: Directory containing config files (default: ./configs)
            task_name: Optional task name to load task-specific config
        """
        self.config_dir = (
            Path(config_dir).expanduser().resolve()
            if config_dir
            else Path(str(files("configs"))).resolve()
        )
        self.task_name = task_name

        # Load default config
        self._config = self._load_config("default.yaml")

        # Load task-specific config if provided
        if task_name:
            task_config = self._load_config(f"tasks/{task_name}.yaml")
            self._config = self._merge_configs(self._config, task_config)
        self._resolve_project_paths()

    def _resolve_project_paths(self) -> None:
        """Resolve resource and writable paths consistently after installation."""
        project_root = self.config_dir.parent.resolve()
        paths = self._config.setdefault("paths", {})
        for key, raw_value in list(paths.items()):
            if not isinstance(raw_value, str):
                continue
            path = Path(raw_value).expanduser()
            if path.is_absolute():
                paths[key] = str(path)
            elif key == "prompt_root":
                paths[key] = str((project_root / "prompts").resolve())
            else:
                paths[key] = str((Path.cwd() / path).resolve())

        data_root = os.environ.get("PANDORA_DATA_ROOT")
        if data_root:
            paths["data_root"] = str(Path(data_root).expanduser().resolve())

    def _load_config(self, filename: str) -> dict:
        """Load a config file by name."""
        path = self.config_dir / filename
        if path.exists():
            return load_yaml(path)
        return {}

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge two config dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "model.name", "inference.shot_k")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get config value using bracket notation."""
        return self.get(key)

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.get("model.name", "gpt-4o-mini")

    @property
    def shot_k(self) -> int:
        """Get number of few-shot examples."""
        return self.get("inference.shot_k", 10)

    @property
    def max_tries(self) -> int:
        """Get maximum retry attempts."""
        return self.get("inference.max_tries", 3)

    @property
    def do_execution_guidance(self) -> bool:
        """Check if execution guidance is enabled."""
        return self.get("inference.do_execution_guidance", True)

    @property
    def n_votes(self) -> int:
        """Get number of votes for majority voting."""
        return self.get("inference.n_votes", 5)

    def to_dict(self) -> dict:
        """Return full configuration as dictionary."""
        import copy
        return copy.deepcopy(self._config)


class ConfigView(Mapping):
    """Read nested configuration dictionaries using dotted keys.

    Agents receive plain dictionaries from ``Config.to_dict()``.  A normal
    ``dict.get('inference.shot_k')`` cannot read ``config['inference']['shot_k']``;
    this lightweight view keeps the public dictionary contract while making
    dotted access consistent throughout the runtime.
    """

    def __init__(self, config: Optional[Any] = None):
        if isinstance(config, ConfigView):
            self._data = config._data
        elif isinstance(config, Config):
            self._data = config.to_dict()
        elif isinstance(config, Mapping):
            self._data = dict(config)
        elif config is None:
            self._data = {}
        else:
            raise TypeError(f"Unsupported configuration type: {type(config).__name__}")

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._data
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    def __getitem__(self, key: str) -> Any:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict:
        return dict(self._data)
