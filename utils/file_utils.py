"""
Pandora File Utilities

Common file operations for JSON, YAML, and directory management.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path], encoding: str = "utf-8") -> Any:
    """
    Load JSON from file.

    Args:
        path: Path to JSON file
        encoding: File encoding

    Returns:
        Parsed JSON data
    """
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def save_json(
    data: Any,
    path: Union[str, Path],
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    mode: str = "w",
    default: callable = None,
) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to serialize
        path: Path to output file
        encoding: File encoding
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters
        mode: File open mode ("w" for write, "a" for append)
        default: Custom JSON encoder default handler
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, mode, encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=default)


def append_json(
    data: Any,
    path: Union[str, Path],
    encoding: str = "utf-8",
    default: callable = None,
) -> None:
    """
    Append a single JSON object to a JSONL file.

    Args:
        data: Data to append
        path: Path to JSONL file
        encoding: File encoding
        default: Custom JSON encoder default handler
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "a", encoding=encoding) as f:
        f.write(json.dumps(data, ensure_ascii=False, default=default) + "\n")


def load_yaml(path: Union[str, Path], encoding: str = "utf-8") -> dict:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML file
        encoding: File encoding

    Returns:
        Parsed YAML data as dictionary
    """
    with open(path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_jsonl(path: Union[str, Path], encoding: str = "utf-8") -> list[dict]:
    """
    Load JSONL file (one JSON object per line).

    Args:
        path: Path to JSONL file
        encoding: File encoding

    Returns:
        List of parsed JSON objects
    """
    results = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_checkpoint_ids(path: Union[str, Path]) -> set[int]:
    """
    Load checkpoint and return set of already processed example IDs.

    Args:
        path: Path to JSONL result file

    Returns:
        Set of processed example IDs
    """
    path = Path(path)
    if not path.exists():
        return set()

    processed_ids = set()
    for item in load_jsonl(path):
        if "example_id" in item:
            processed_ids.add(item["example_id"])

    return processed_ids
