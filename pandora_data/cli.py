#!/usr/bin/env python3
"""Prepare and validate benchmark assets used by Pandora."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

import requests
import zstandard


DATASETS = ("spider", "spider-syn", "bird", "wikitq", "wikisql", "grailqa", "webqsp")
ALIASES = {"spider_syn": "spider-syn", "wikitablequestions": "wikitq"}


def _manifest() -> dict:
    manifest_path = resources.files("pandora_data").joinpath("manifests/benchmarks.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _names(value: str) -> list[str]:
    if value == "all":
        return list(DATASETS)
    name = ALIASES.get(value, value)
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {value!r}; choose from {', '.join(DATASETS)}")
    return [name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    headers = {}
    mode = "wb"
    if partial.exists():
        headers["Range"] = f"bytes={partial.stat().st_size}-"
        mode = "ab"
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"

    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        if response.status_code == 200 and mode == "ab":
            mode = "wb"
        response.raise_for_status()
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if _sha256(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch while downloading {url}")
    partial.replace(target)


def _jsonl_to_json(source: Path, target: Path, expected_count: int) -> None:
    records = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
    if len(records) != expected_count:
        raise ValueError(
            f"Unexpected record count in {source}: {len(records)} (expected {expected_count})"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()

    def safe(member: str) -> bool:
        return (destination / member).resolve().is_relative_to(destination_resolved)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            if not all(safe(name) for name in handle.namelist()):
                raise ValueError(f"Unsafe path in archive: {archive}")
            handle.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            if not all(safe(member.name) for member in members):
                raise ValueError(f"Unsafe path in archive: {archive}")
            if any(member.issym() or member.islnk() for member in members):
                raise ValueError(f"Archive links are not supported: {archive}")
            handle.extractall(destination)
        return
    raise ValueError(f"Unsupported source archive: {archive}")


def _safe_extract_tar_zst(archive: Path, destination: Path) -> None:
    """Stream a zstd-compressed tar archive while enforcing safe member paths."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as handle:
                for member in handle:
                    target = (destination / member.name).resolve()
                    if not target.is_relative_to(destination_resolved):
                        raise ValueError(f"Unsafe path in archive: {archive}")
                    if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                        raise ValueError(f"Unsupported archive member: {member.name}")
                    handle.extract(member, destination)


def _source_tree(source: Path):
    if source.is_dir():
        return _NullContext(source)
    temporary = tempfile.TemporaryDirectory(prefix="pandora-data-")
    root = Path(temporary.name)
    _safe_extract(source, root)
    return _TemporaryContext(temporary, root)


class _NullContext:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *_args) -> None:
        return None


class _TemporaryContext(_NullContext):
    def __init__(self, temporary: tempfile.TemporaryDirectory, path: Path):
        super().__init__(path)
        self.temporary = temporary

    def __exit__(self, *_args) -> None:
        self.temporary.cleanup()


def _find_file(root: Path, names: Iterable[str]) -> Optional[Path]:
    wanted = set(names)
    matches = sorted(path for path in root.rglob("*") if path.is_file() and path.name in wanted)
    return matches[0] if matches else None


def _find_dir(root: Path, names: Iterable[str]) -> Optional[Path]:
    wanted = set(names)
    matches = sorted(path for path in root.rglob("*") if path.is_dir() and path.name in wanted)
    return matches[0] if matches else None


def _copy_file(source: Optional[Path], target: Path, label: str) -> None:
    if source is None:
        raise FileNotFoundError(f"Could not find {label} in the supplied official source")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Optional[Path], target: Path, label: str) -> None:
    if source is None:
        raise FileNotFoundError(f"Could not find {label} in the supplied official source")
    if target.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copytree(source, target)


def _import_official(name: str, source: Path, root: Path) -> None:
    with _source_tree(source) as tree:
        dataset_root = root / name
        if name == "spider":
            _copy_file(
                _find_file(tree, ("dev.json", "spider.dev.json")),
                dataset_root / "spider.dev.json",
                "Spider dev.json",
            )
            _copy_file(
                _find_file(tree, ("tables.json", "dev_tables.json", "spider.tables.dev.json")),
                dataset_root / "spider.tables.dev.json",
                "Spider tables.json",
            )
            _copy_tree(
                _find_dir(tree, ("dev_database", "database")),
                dataset_root / "dev_database",
                "Spider databases",
            )
        elif name == "bird":
            _copy_file(
                _find_file(tree, ("dev_tables.json", "bird.tables.dev.json")),
                dataset_root / "bird.tables.dev.json",
                "BIRD dev_tables.json",
            )
            _copy_tree(
                _find_dir(tree, ("dev_database", "dev_databases")),
                dataset_root / "dev_database",
                "BIRD dev databases",
            )
        elif name == "webqsp":
            source_file = _find_file(tree, ("WebQSP.test.json",))
            if source_file is None:
                raise FileNotFoundError("Could not find WebQSP.test.json in the official package")
            _materialize_webqsp(source_file, dataset_root)


def _materialize_webqsp(source: Path, dataset_root: Path) -> None:
    """Convert the official WebQSP test file into Pandora's minimal runtime schema."""
    subset_path = dataset_root / "subset.test.json"
    if not subset_path.is_file():
        raise FileNotFoundError("WebQSP subset metadata must be installed before annotation import")
    subset_data = json.loads(subset_path.read_text(encoding="utf-8"))
    selected_qids = set(subset_data["qids"])
    official = json.loads(source.read_text(encoding="utf-8"))
    questions = official.get("Questions", official) if isinstance(official, dict) else official

    records = []
    entity_links = {}
    for question in questions:
        qid = str(question.get("QuestionId", ""))
        if qid not in selected_qids:
            continue
        topics = {}
        links = {}
        compact_parses = []
        for parse in question.get("Parses", []):
            topic_name = str(parse.get("TopicEntityName", "")).strip()
            topic_mid = str(parse.get("TopicEntityMid", "")).strip()
            if topic_name and topic_mid:
                topics[topic_name] = topic_mid
                links[topic_name] = topic_mid
            answers = []
            for answer in parse.get("Answers", []):
                argument = str(answer.get("AnswerArgument", "")).strip()
                entity_name = str(answer.get("EntityName", "")).strip()
                if not argument:
                    continue
                answers.append({"AnswerArgument": argument, "EntityName": entity_name})
                if entity_name:
                    links[entity_name] = argument
            compact_parses.append(
                {
                    "TopicEntityName": topic_name,
                    "TopicEntityMid": topic_mid,
                    "Answers": answers,
                }
            )
        entity_prefix = " ".join(f"{name}: {mid}" for name, mid in topics.items())
        records.append(
            {
                "id": qid,
                "qid": qid,
                "schema": f"{entity_prefix} |" if entity_prefix else "",
                "raw_data": {
                    "QuestionId": qid,
                    "RawQuestion": question.get("RawQuestion", ""),
                    "ProcessedQuestion": question.get("ProcessedQuestion", ""),
                    "Parses": compact_parses,
                },
            }
        )
        entity_links[qid] = links

    if {record["id"] for record in records} != selected_qids:
        missing = sorted(selected_qids - {record["id"] for record in records})
        raise ValueError(f"Official WebQSP package is missing selected questions: {missing[:5]}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "webqsp.test.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    entity_link_path = dataset_root / "entity_link" / "webqsp.entity_link.test.json"
    entity_link_path.parent.mkdir(parents=True, exist_ok=True)
    entity_link_path.write_text(json.dumps(entity_links, ensure_ascii=False), encoding="utf-8")


def _artifact_url(repository: str, revision: str, artifact_path: str) -> str:
    return f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{artifact_path}"


def _install_processed(name: str, root: Path, cache: Path, manifest: dict) -> None:
    spec = manifest["datasets"][name]
    for artifact in spec.get("artifacts", []):
        cached = cache / artifact["path"]
        if not cached.exists() or _sha256(cached) != artifact["sha256"]:
            _download(
                _artifact_url(manifest["repository"], manifest["revision"], artifact["path"]),
                cached,
                artifact["sha256"],
            )
        target = root / artifact["target"]
        if artifact["format"] == "jsonl":
            _jsonl_to_json(cached, target, spec["expected_records"])
        elif artifact["format"] == "tar.zst":
            _safe_extract_tar_zst(cached, root)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, target)


def _count_json(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return len(value) if isinstance(value, list) else None
    except (OSError, json.JSONDecodeError):
        return None


def _sqlite_count(path: Path) -> int:
    return len(list(path.rglob("*.sqlite"))) if path.exists() else 0


def _json_object(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _verify_kg_assets(name: str, root: Path, spec: dict) -> list[str]:
    kg = spec["kg_assets"]
    expected = spec["expected_records"]
    issues = []
    subset = _json_object(root / kg["subset"])
    qids = subset.get("qids", []) if subset else []
    qid_set = {str(qid) for qid in qids}
    if len(qids) != expected or len(qid_set) != expected:
        issues.append(
            f"{root / kg['subset']}: expected {expected} unique qids, found {len(qid_set)}"
        )

    annotation_path = root / spec["annotation"]
    try:
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        annotations = []
    annotation_qids = {
        str(record.get(kg["id_field"], "")) for record in annotations if isinstance(record, dict)
    }
    if annotation_qids != qid_set:
        issues.append(f"{annotation_path}: question ids do not match the published subset")

    for label in ("schema", "entity_links"):
        value = _json_object(root / kg[label])
        if value is None or set(map(str, value)) != qid_set:
            issues.append(f"{root / kg[label]}: keys do not match the published subset")
    if _json_object(root / kg["entity_names"]) is None:
        issues.append(f"missing or invalid entity names: {root / kg['entity_names']}")

    box_root = root / kg["box_directory"]
    available = (
        {path.name for path in box_root.iterdir() if path.is_dir()} if box_root.is_dir() else set()
    )
    missing_boxes = qid_set - available
    if missing_boxes:
        issues.append(f"{box_root}: missing {len(missing_boxes)} question BOX directories")
    return issues


def verify_dataset(name: str, root: Path, manifest: dict) -> tuple[bool, list[str]]:
    spec = manifest["datasets"][name]
    issues = []
    annotation = root / spec["annotation"]
    actual = _count_json(annotation)
    if actual != spec["expected_records"]:
        issues.append(
            f"{annotation}: expected {spec['expected_records']} records, found {actual or 0}"
        )
    for required in spec.get("required_files", []):
        if not (root / required).is_file():
            issues.append(f"missing file: {root / required}")
    for directory, expected in spec.get("sqlite_directories", {}).items():
        actual_sqlite = _sqlite_count(root / directory)
        if actual_sqlite < expected:
            issues.append(
                f"{root / directory}: expected at least {expected} SQLite files, found {actual_sqlite}"
            )
    if spec.get("kg_assets"):
        issues.extend(_verify_kg_assets(name, root, spec))
    for dependency in spec.get("dependencies", []):
        dependency_ok, dependency_issues = verify_dataset(dependency, root, manifest)
        if not dependency_ok:
            issues.extend(f"dependency {dependency}: {issue}" for issue in dependency_issues)
    return not issues, issues


def command_list(manifest: dict) -> int:
    for name in DATASETS:
        spec = manifest["datasets"][name]
        print(
            f"{name:12} {spec['task']:8} {spec['stage']:4} {spec['expected_records']:6} records  {spec['license']}"
        )
    return 0


def command_status(names: list[str], root: Path, manifest: dict, verbose: bool) -> int:
    all_ready = True
    for name in names:
        ready, issues = verify_dataset(name, root, manifest)
        all_ready = all_ready and ready
        print(f"{name:12} {'ready' if ready else 'incomplete'}")
        if verbose:
            for issue in issues:
                print(f"  - {issue}")
    return 0 if all_ready else 1


def command_prepare(
    names: list[str], root: Path, cache: Path, source: Optional[Path], manifest: dict
) -> int:
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = manifest["datasets"][name]
        if spec.get("artifacts"):
            print(f"Preparing processed {name} assets")
            _install_processed(name, root, cache, manifest)
        if spec.get("official_assets"):
            if source is None:
                download_url = spec.get("official_download_url")
                if download_url:
                    official_source = cache / "official" / spec["official_download_filename"]
                    if (
                        not official_source.is_file()
                        or _sha256(official_source) != spec["official_download_sha256"]
                    ):
                        print(f"Downloading official {name} package")
                        _download(download_url, official_source, spec["official_download_sha256"])
                    print(f"Importing official {name} assets")
                    _import_official(name, official_source, root)
                else:
                    ready, _ = verify_dataset(name, root, manifest)
                    if not ready:
                        print(
                            f"{name}: obtain the official package from {spec['source_url']} and rerun "
                            f"with --source /path/to/package",
                            file=sys.stderr,
                        )
                        continue
            else:
                selected = source / name if len(names) > 1 and (source / name).exists() else source
                print(f"Importing official {name} assets from {selected}")
                _import_official(name, selected, root)
        ready, issues = verify_dataset(name, root, manifest)
        if not ready:
            for issue in issues:
                print(f"{name}: {issue}", file=sys.stderr)
    return command_status(names, root, manifest, verbose=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare benchmark data for Pandora")
    parser.add_argument("--version", action="version", version="pandora-data 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List supported benchmark datasets")
    for command in ("status", "verify"):
        child = subparsers.add_parser(command, help=f"{command.title()} prepared datasets")
        child.add_argument("--dataset", default="all")
        child.add_argument("--root", default=os.environ.get("PANDORA_DATA_ROOT", "./data"))
    prepare = subparsers.add_parser(
        "prepare", help="Download processed assets and import official data"
    )
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--root", default=os.environ.get("PANDORA_DATA_ROOT", "./data"))
    prepare.add_argument(
        "--cache-dir", default=os.environ.get("PANDORA_DATA_CACHE", "~/.cache/pandora")
    )
    prepare.add_argument(
        "--source", help="Official archive/directory, or a root containing dataset subdirectories"
    )
    return parser


def cli(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _manifest()
    if args.command == "list":
        return command_list(manifest)
    names = _names(args.dataset)
    root = Path(args.root).expanduser().resolve()
    if args.command in ("status", "verify"):
        return command_status(names, root, manifest, verbose=args.command == "verify")
    source = Path(args.source).expanduser().resolve() if args.source else None
    try:
        return command_prepare(
            names,
            root,
            Path(args.cache_dir).expanduser().resolve(),
            source,
            manifest,
        )
    except requests.RequestException as exc:
        sources = ", ".join(manifest["datasets"][name]["source_url"] for name in names)
        print(
            f"pandora-data: download failed: {exc}\n"
            "Configure HTTPS_PROXY/REQUESTS_CA_BUNDLE as needed, or download the "
            f"official package from {sources} and rerun with --source.",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError) as exc:
        print(f"pandora-data: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(cli())
