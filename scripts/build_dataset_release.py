#!/usr/bin/env python3
"""Build the approved Pandora benchmark artifacts for the companion data repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pandora_data.cli import _manifest


PUBLISHABLE = ("spider-syn", "bird", "wikitq", "wikisql")
SOURCE_FILES = {
    "spider-syn": "spider-syn/spider-syn.test.json",
    "bird": "bird/bird.dev.json",
    "wikitq": "wikitq/wikitq.test.json",
    "wikisql": "wikisql/wikisql.test.json",
}
ARTIFACT_FILES = {
    "spider-syn": "processed/spider-syn/spider-syn.test.jsonl",
    "bird": "processed/bird/bird.dev.jsonl",
    "wikitq": "processed/wikitq/wikitq.test.jsonl",
    "wikisql": "processed/wikisql/wikisql.test.jsonl",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(name: str, record: dict) -> dict:
    """Retain only fields used by the public benchmark adapters."""
    if name == "spider-syn":
        return {
            key: record[key]
            for key in ("id", "question", "spider_question", "db_id", "label")
            if key in record
        }
    if name == "bird":
        return {
            key: record[key]
            for key in ("question_id", "question", "evidence", "db_id", "SQL", "difficulty")
            if key in record
        }
    if name == "wikitq":
        return {
            key: record[key]
            for key in ("id", "question", "table_id", "table", "answer_text")
            if key in record
        }
    if name == "wikisql":
        table = record.get("table", {})
        compact_table = {
            key: table[key]
            for key in ("id", "header", "rows", "types")
            if key in table
        }
        return {
            "id": record.get("id", compact_table.get("id", "")),
            "question": record.get("question", ""),
            "table": compact_table,
            "answer_text": record.get("answer_text", []),
            "sql": record.get("sql", {}),
        }
    raise ValueError(f"Dataset {name!r} is not approved for publication")


def build_one(name: str, input_root: Path, output_root: Path, expected: int) -> dict:
    source = input_root / SOURCE_FILES[name]
    with source.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if len(records) != expected:
        raise ValueError(f"{source}: expected {expected} records, found {len(records)}")

    target = output_root / ARTIFACT_FILES[name]
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with target.open("wb") as handle:
        for record in records:
            line = (
                json.dumps(transform(name, record), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            handle.write(line)
            digest.update(line)
            size += len(line)
    return {
        "dataset": name,
        "path": ARTIFACT_FILES[name],
        "target": _manifest()["datasets"][name]["annotation"],
        "format": "jsonl",
        "records": len(records),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "source_file": SOURCE_FILES[name],
        "source_sha256": file_sha256(source),
        "transformation": "pandora-benchmark-schema-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", choices=("all",) + PUBLISHABLE, default="all")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest = _manifest()
    names = PUBLISHABLE if args.dataset == "all" else (args.dataset,)
    artifacts = [
        build_one(name, input_root, output_root, manifest["datasets"][name]["expected_records"])
        for name in names
    ]
    release_manifest = {
        "schema_version": 1,
        "release": manifest["revision"],
        "artifacts": artifacts,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(release_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for artifact in artifacts:
        print(
            f"{artifact['dataset']}: {artifact['records']} records, "
            f"{artifact['bytes']} bytes, sha256={artifact['sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
