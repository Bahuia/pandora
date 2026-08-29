#!/usr/bin/env python3
"""Construct verified memory using the two-stage procedure from the paper."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent import PandoraAgent
from core.memory_builder import ProgressiveMemoryBuilder
from models.registry import ModelRegistry
from run import create_dataset
from utils.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["nl2sql", "tableqa", "kbqa"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", default="train")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None, help="Explicit OpenAI-compatible API base URL")
    parser.add_argument("--data-root", default=None, help="Benchmark data root")
    parser.add_argument("--config-dir", default=None, help="Configuration directory")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--phase", choices=["initialization", "adaptation"], required=True)
    parser.add_argument("--shot-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-memory",
        type=Path,
        nargs="+",
        help="Verified DB-memory files used during multi-task adaptation.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = Config(config_dir=args.config_dir, task_name=args.task)
    if args.data_root:
        config._config["paths"]["data_root"] = str(Path(args.data_root).expanduser().resolve())
    config._config["model"]["name"] = args.model
    config._config["model"]["temperature"] = args.temperature
    if args.phase == "initialization":
        config._config["retrieval"]["mode"] = "disabled"
        args.shot_k = 0
    else:
        config._config["retrieval"]["mode"] = "cross_task"
        if args.initial_memory:
            config._config["retrieval"]["memory_files"] = [
                str(path.resolve()) for path in args.initial_memory
            ]

    dataset = create_dataset(args.task, args.dataset, config)
    examples = dataset.load_examples(args.stage)
    model = ModelRegistry.create(
        args.model,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    agent = PandoraAgent(dataset=dataset, model=model, config=config.to_dict())
    records = ProgressiveMemoryBuilder(agent, dataset, args.output).build(
        examples=examples,
        target_count=args.target_count,
        shot_k=args.shot_k,
        resume=not args.no_resume,
    )
    print(f"Verified memory size: {len(records)} ({args.output})")


if __name__ == "__main__":
    main()
