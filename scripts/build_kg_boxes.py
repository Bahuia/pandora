#!/usr/bin/env python3
"""Build query-local KG BOXes using Algorithm 3 from the paper."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kg_box_builder import KGBoxBuilder, load_triples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--topic-entities", nargs="+", required=True)
    parser.add_argument("--max-hops", type=int, required=True)
    parser.add_argument("--relation-top-k", type=int, default=50)
    parser.add_argument("--relations-file", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    builder = KGBoxBuilder(load_triples(args.triples))
    if args.relations_file:
        relations = {
            line.strip() for line in args.relations_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    else:
        relations = builder.select_relations(args.question, args.relation_top_k)
    subgraph = builder.extract_subgraph(args.topic_entities, relations, args.max_hops)
    question_dir = args.output_root / "test" / args.qid
    schema, _ = builder.materialize(subgraph, question_dir)

    schema_path = args.output_root / "box_schema.json"
    schemas = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    schemas[str(args.qid)] = schema
    schema_path.write_text(json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {len(subgraph)} triples into {question_dir}")


if __name__ == "__main__":
    main()
