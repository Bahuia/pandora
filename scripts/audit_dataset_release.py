#!/usr/bin/env python3
"""Validate the companion Pandora dataset repository before publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SENSITIVE = {
    "private IPv4 address": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
    "local user path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "likely API key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
}
FORBIDDEN_FIELDS = {
    "prompt",
    "response",
    "instruction",
    "formatted_input",
    "seq_out",
    "struct_in",
    "code_annotation",
    "model_output",
    "prediction",
}
REQUIRED_NOTICES = {"spider-syn", "bird", "wikitq", "wikisql", "grailqa", "webqsp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from nested_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from nested_keys(nested)


def audit(root: Path) -> list[str]:
    blockers = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in REQUIRED_NOTICES:
        if not (root / "notices" / f"{name}.md").is_file():
            blockers.append(f"notice missing for {name}")

    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        if not path.is_file():
            blockers.append(f"artifact missing: {artifact['path']}")
            continue
        if path.stat().st_size != artifact["bytes"]:
            blockers.append(f"size mismatch: {artifact['path']}")
        if sha256(path) != artifact["sha256"]:
            blockers.append(f"checksum mismatch: {artifact['path']}")
        if artifact["format"] == "jsonl":
            records = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    records += 1
                    record = json.loads(line)
                    forbidden = FORBIDDEN_FIELDS.intersection(nested_keys(record))
                    if forbidden:
                        blockers.append(
                            f"forbidden fields in {artifact['path']}:{line_number}: "
                            + ", ".join(sorted(forbidden))
                        )
                        break
            if records != artifact["records"]:
                blockers.append(f"record count mismatch: {artifact['path']}")
        elif artifact["format"] == "json":
            value = json.loads(path.read_text(encoding="utf-8"))
            entries = len(value.get("qids", [])) if "qids" in value else len(value)
            if entries != artifact["entries"]:
                blockers.append(f"entry count mismatch: {artifact['path']}")
        elif artifact["format"] == "tar.zst":
            if artifact["bytes"] > 1024**3:
                blockers.append(f"archive exceeds 1 GiB: {artifact['path']}")
            if artifact.get("qids", 0) <= 0 or artifact.get("files", 0) <= 0:
                blockers.append(f"archive inventory is incomplete: {artifact['path']}")
        else:
            blockers.append(f"unsupported artifact format: {artifact['format']}")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {"results", "logs", "cache"} for part in path.parts):
            blockers.append(f"forbidden directory: {path.relative_to(root)}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SENSITIVE.items():
            if pattern.search(content):
                blockers.append(f"{label} found in {path.relative_to(root)}")
    return sorted(set(blockers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    blockers = audit(root)
    if blockers:
        print("Pandora dataset release audit: FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        return 1
    print("Pandora dataset release audit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
