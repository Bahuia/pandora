#!/usr/bin/env python3
"""Build the approved Pandora benchmark artifacts for the companion data repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import zstandard

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pandora_data.cli import _manifest


PUBLISHABLE = ("spider-syn", "bird", "wikitq", "wikisql", "grailqa", "webqsp")
SOURCE_FILES = {
    "spider-syn": "spider-syn/spider-syn.test.json",
    "bird": "bird/bird.dev.json",
    "wikitq": "wikitq/wikitq.test.json",
    "wikisql": "wikisql/wikisql.test.json",
    "grailqa": "grailqa/grailqa.test.json",
    "webqsp": "webqsp/webqsp.test.json",
}
ARTIFACT_FILES = {
    "spider-syn": "processed/spider-syn/spider-syn.test.jsonl",
    "bird": "processed/bird/bird.dev.jsonl",
    "wikitq": "processed/wikitq/wikitq.test.jsonl",
    "wikisql": "processed/wikisql/wikisql.test.jsonl",
    "grailqa": "processed/grailqa/grailqa.test.jsonl",
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
            key: table[key] for key in ("id", "header", "rows", "types") if key in table
        }
        return {
            "id": record.get("id", compact_table.get("id", "")),
            "question": record.get("question", ""),
            "table": compact_table,
            "answer_text": record.get("answer_text", []),
            "sql": record.get("sql", {}),
        }
    if name == "grailqa":
        return {
            key: record[key]
            for key in (
                "qid",
                "question",
                "answer",
                "schema",
                "level",
                "s_expression",
                "sparql_query",
            )
            if key in record
        }
    raise ValueError(f"Dataset {name!r} is not approved for annotation publication")


def _artifact(file_path: Path, **metadata) -> dict:
    return {
        **metadata,
        "bytes": file_path.stat().st_size,
        "sha256": file_sha256(file_path),
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _has_gold(name: str, record: dict) -> bool:
    if name == "grailqa":
        answer = record.get("answer", {})
        return bool(answer.get("answer_argument") or answer.get("entity_name"))
    return any(
        answer.get("AnswerArgument")
        for parse in record.get("raw_data", {}).get("Parses", [])
        for answer in parse.get("Answers", [])
    )


def _kg_qids(name: str, input_root: Path, records: list[dict]) -> list[str]:
    dataset_root = input_root / name
    schemas = json.loads((dataset_root / "box" / "box_schema.json").read_text(encoding="utf-8"))
    id_field = "qid" if name == "grailqa" else "id"
    selected = []
    for record in records:
        qid = str(record.get(id_field, ""))
        if (
            qid in schemas
            and (dataset_root / "box" / "test" / qid).is_dir()
            and _has_gold(name, record)
        ):
            selected.append(qid)
    return selected


def _normalized_tar_info(path: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
    else:
        info.type = tarfile.REGTYPE
        info.mode = 0o644
        info.size = path.stat().st_size
    return info


def _write_box_archive(name: str, input_root: Path, output_root: Path, qids: list[str]) -> dict:
    source_root = input_root / name / "box" / "test"
    relative_path = f"processed/{name}/{name}.box.test.tar.zst"
    target = output_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    uncompressed_bytes = 0
    with target.open("wb") as raw:
        compressor = zstandard.ZstdCompressor(level=10, write_checksum=True)
        with compressor.stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for directory in (name, f"{name}/box", f"{name}/box/test"):
                    archive.addfile(_normalized_tar_info(source_root, directory))
                for qid in sorted(qids):
                    qid_root = source_root / qid
                    archive.addfile(_normalized_tar_info(qid_root, f"{name}/box/test/{qid}"))
                    for path in sorted(item for item in qid_root.rglob("*") if item.is_file()):
                        arcname = f"{name}/box/test/{qid}/{path.relative_to(qid_root).as_posix()}"
                        info = _normalized_tar_info(path, arcname)
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                        file_count += 1
                        uncompressed_bytes += info.size
    return _artifact(
        target,
        dataset=name,
        path=relative_path,
        target=".",
        format="tar.zst",
        qids=len(qids),
        files=file_count,
        uncompressed_bytes=uncompressed_bytes,
        transformation="pandora-kg-box-v1",
    )


def build_standard(name: str, input_root: Path, output_root: Path, expected: int) -> dict:
    source = input_root / SOURCE_FILES[name]
    records = json.loads(source.read_text(encoding="utf-8"))
    if len(records) != expected:
        raise ValueError(f"{source}: expected {expected} records, found {len(records)}")
    transformed = [transform(name, record) for record in records]
    relative_path = ARTIFACT_FILES[name]
    target = output_root / relative_path
    _write_jsonl(target, transformed)
    return _artifact(
        target,
        dataset=name,
        path=relative_path,
        target=_manifest()["datasets"][name]["annotation"],
        format="jsonl",
        records=len(records),
        source_file=SOURCE_FILES[name],
        source_sha256=file_sha256(source),
        transformation="pandora-benchmark-schema-v1",
    )


def build_kg(name: str, input_root: Path, output_root: Path, expected: int) -> list[dict]:
    dataset_root = input_root / name
    source = input_root / SOURCE_FILES[name]
    records = json.loads(source.read_text(encoding="utf-8"))
    qids = _kg_qids(name, input_root, records)
    if len(qids) != expected:
        raise ValueError(
            f"{name}: expected {expected} prepared/evaluable questions, found {len(qids)}"
        )
    qid_set = set(qids)
    artifacts = []

    subset_relative = f"processed/{name}/subset.test.json"
    subset_target = output_root / subset_relative
    _write_json(subset_target, {"dataset": name, "split": "test", "count": expected, "qids": qids})
    artifacts.append(
        _artifact(
            subset_target,
            dataset=name,
            path=subset_relative,
            target=f"{name}/subset.test.json",
            format="json",
            entries=expected,
            source_split="validation" if name == "grailqa" else "test",
            transformation="pandora-kg-subset-v1",
        )
    )

    schemas = json.loads((dataset_root / "box" / "box_schema.json").read_text(encoding="utf-8"))
    schema_relative = f"processed/{name}/box_schema.json"
    schema_target = output_root / schema_relative
    _write_json(schema_target, {qid: schemas[qid] for qid in qids})
    artifacts.append(
        _artifact(
            schema_target,
            dataset=name,
            path=schema_relative,
            target=f"{name}/box/box_schema.json",
            format="json",
            entries=expected,
            transformation="pandora-kg-schema-v1",
        )
    )

    entity_names_source = dataset_root / "entity_names.json"
    entity_names_relative = f"processed/{name}/entity_names.json"
    entity_names_target = output_root / entity_names_relative
    entity_names = json.loads(entity_names_source.read_text(encoding="utf-8"))
    _write_json(entity_names_target, entity_names)
    artifacts.append(
        _artifact(
            entity_names_target,
            dataset=name,
            path=entity_names_relative,
            target=f"{name}/entity_names.json",
            format="json",
            entries=len(entity_names),
            source_sha256=file_sha256(entity_names_source),
            transformation="pandora-kg-entity-names-v1",
        )
    )

    if name == "grailqa":
        selected_records = [
            transform(name, record) for record in records if str(record["qid"]) in qid_set
        ]
        annotation_relative = ARTIFACT_FILES[name]
        annotation_target = output_root / annotation_relative
        _write_jsonl(annotation_target, selected_records)
        artifacts.append(
            _artifact(
                annotation_target,
                dataset=name,
                path=annotation_relative,
                target=f"{name}/{name}.test.json",
                format="jsonl",
                records=expected,
                source_file=SOURCE_FILES[name],
                source_split="validation",
                source_sha256=file_sha256(source),
                transformation="pandora-grailqa-schema-v1",
            )
        )
        entity_links_source = dataset_root / "entity_link" / "grailqa.entity_link.test.json"
        entity_links = json.loads(entity_links_source.read_text(encoding="utf-8"))
        entity_links_relative = f"processed/{name}/grailqa.entity_link.test.json"
        entity_links_target = output_root / entity_links_relative
        _write_json(entity_links_target, {qid: entity_links[qid] for qid in qids})
        artifacts.append(
            _artifact(
                entity_links_target,
                dataset=name,
                path=entity_links_relative,
                target=f"{name}/entity_link/grailqa.entity_link.test.json",
                format="json",
                entries=expected,
                source_sha256=file_sha256(entity_links_source),
                transformation="pandora-kg-entity-link-v1",
            )
        )

    artifacts.append(_write_box_archive(name, input_root, output_root, qids))
    return artifacts


def _update_runtime_manifest(path: Path, artifacts: list[dict], release: str) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["revision"] = release
    for name in PUBLISHABLE:
        selected = [artifact for artifact in artifacts if artifact["dataset"] == name]
        manifest["datasets"][name]["artifacts"] = [
            {key: artifact[key] for key in ("path", "target", "format", "bytes", "sha256")}
            for artifact in selected
        ]
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", choices=("all",) + PUBLISHABLE, default="all")
    parser.add_argument("--release", default="v0.2.0-benchmark-preview")
    parser.add_argument("--runtime-manifest")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest = _manifest()
    names = PUBLISHABLE if args.dataset == "all" else (args.dataset,)
    artifacts = []
    for name in names:
        expected = manifest["datasets"][name]["expected_records"]
        if name in ("grailqa", "webqsp"):
            artifacts.extend(build_kg(name, input_root, output_root, expected))
        else:
            artifacts.append(build_standard(name, input_root, output_root, expected))

    release_manifest = {
        "schema_version": 2,
        "release": args.release,
        "artifacts": artifacts,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(release_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.runtime_manifest:
        _update_runtime_manifest(Path(args.runtime_manifest), artifacts, args.release)
    for artifact in artifacts:
        print(
            f"{artifact['dataset']}: {artifact['path']}, {artifact['bytes']} bytes, "
            f"sha256={artifact['sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
