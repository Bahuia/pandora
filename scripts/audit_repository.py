#!/usr/bin/env python3
"""Audit the public code boundary separately from optional paper assets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

REQUIRED_FILES = {
    "LICENSE",
    "README.md",
    "DATASETS.md",
    "THIRD_PARTY_DATA.md",
    "pyproject.toml",
    "run.py",
    "configs/default.yaml",
    "prompts/tasks/nl2sql/code_reasoning.txt",
    "prompts/tasks/tableqa/code_reasoning.txt",
    "prompts/tasks/kbqa/code_reasoning.txt",
}
FORBIDDEN_ROOTS = {
    "data", "results", "bad_cases", "temp", "manuscript", "figure",
    "review", "response", "prompts/few_shot", ".idea",
}
FORBIDDEN_SUFFIXES = {
    ".sqlite", ".sqlite3", ".db", ".duckdb", ".csv", ".tsv", ".jsonl",
    ".parquet", ".npz", ".npy", ".pkl", ".pickle", ".pt", ".pth",
    ".ckpt", ".safetensors",
}
SENSITIVE_PATTERNS = {
    "private IPv4 address": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
    "macOS user path": re.compile(r"/Users/[^/\s]+/"),
    "API key fragment field": re.compile(r"api[_-]?key[_-]?(?:prefix|suffix)", re.I),
    "likely live API key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
}

EVALUATION_SPLITS = {
    "spider": (DATA / "spider" / "spider.dev.json", 1034),
    "spider-syn": (DATA / "spider-syn" / "spider-syn.test.json", 1034),
    "bird": (DATA / "bird" / "bird.dev.json", 1534),
    "wikitq": (DATA / "wikitq" / "wikitq.test.json", 4344),
    "wikisql": (DATA / "wikisql" / "wikisql.test.json", 15878),
    "grailqa": (DATA / "grailqa" / "grailqa.test.json", 6463),
    "webqsp": (DATA / "webqsp" / "webqsp.test.json", 1616),
}


def candidate_files() -> list[Path]:
    """Return tracked plus untracked, non-ignored release candidates."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def read_json_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return len(value)


def audit_code() -> dict:
    files = candidate_files()
    relative = [path.relative_to(ROOT).as_posix() for path in files]
    blockers: list[str] = []

    for required in sorted(REQUIRED_FILES):
        if required not in relative:
            blockers.append(f"required release file missing: {required}")

    total_bytes = sum(path.stat().st_size for path in files if path.is_file())
    oversized = [name for name, path in zip(relative, files) if path.stat().st_size > 5_000_000]
    if total_bytes > 5_000_000:
        blockers.append(f"candidate release is {total_bytes} bytes (limit: 5000000)")
    for name in oversized:
        blockers.append(f"file exceeds 5 MB: {name}")

    for name, path in zip(relative, files):
        if any(name == root or name.startswith(root + "/") for root in FORBIDDEN_ROOTS):
            blockers.append(f"forbidden release path: {name}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            blockers.append(f"forbidden generated/data file: {name}")
        if path.resolve() == Path(__file__).resolve():
            continue
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(content):
                blockers.append(f"{label} found in {name}")

    return {
        "ready": not blockers,
        "candidate_files": len(files),
        "total_bytes": total_bytes,
        "max_file_bytes": max((path.stat().st_size for path in files), default=0),
        "blockers": sorted(set(blockers)),
    }


def audit_assets() -> dict:
    splits = {}
    missing = []
    for dataset, (path, expected) in EVALUATION_SPLITS.items():
        actual = read_json_count(path)
        complete = actual == expected
        splits[dataset] = {"expected": expected, "actual": actual, "complete": complete}
        if not complete:
            missing.append(f"{dataset} evaluation split: expected {expected}, found {actual}")

    memory_files = list(DATA.glob("pandora.memory.*.json")) if DATA.exists() else []
    if not memory_files:
        missing.append("verified memory files are absent")
    cross_source = DATA / "cross_source" / "cross_source.test.json"
    if read_json_count(cross_source) != 23:
        missing.append("paper cross-source manifest is absent or incomplete")
    for dataset in ("grailqa", "webqsp"):
        if not (DATA / dataset / "box" / "box_schema.json").exists():
            missing.append(f"{dataset} KG BOX schemas are absent")

    return {"complete": not missing, "evaluation_splits": splits, "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["code-only", "reproducibility"], default="code-only")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    code = audit_code()
    assets = audit_assets()
    passed = code["ready"] and (args.mode == "code-only" or assets["complete"])
    report = {
        "mode": args.mode,
        "passed": passed,
        "code_health": code,
        "optional_paper_assets": assets,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Pandora {args.mode} audit: {'PASS' if passed else 'FAIL'}")
        for blocker in code["blockers"]:
            print(f"CODE BLOCKER: {blocker}")
        if not assets["complete"]:
            print("Optional paper assets are not included:")
            for item in assets["missing"]:
                print(f"- {item}")
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    sys.exit(main())
