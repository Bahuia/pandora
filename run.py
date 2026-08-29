#!/usr/bin/env python3
"""
Pandora Main Entry Point

Unified runner for NL2SQL, KBQA, and TableQA tasks.

Usage:
    # NL2SQL (BIRD)
    python run.py --task nl2sql --dataset bird --stage dev --model gpt-4o-mini

    # KBQA (GrailQA)
    python run.py --task kbqa --dataset grailqa --stage test --model deepseek-chat

    # TableQA (WikiTQ)
    python run.py --task tableqa --dataset wikitq --stage test --model gpt-4o-mini
"""

import argparse
import json
import time
import os
import sys
import logging
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.agent import PandoraAgent
from core.vanilla_agent import VanillaAgent
from models.registry import ModelRegistry
from utils.config import Config
from utils.logger import setup_logger

def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pandora: Unified Text-to-Code Framework")

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="pandora",
        choices=["pandora", "vanilla"],
        help="Agent mode: 'pandora' (code reasoning + execution feedback) or 'vanilla' (one-shot LLM)",
    )

    # Task settings
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["nl2sql", "kbqa", "tableqa", "cross_source"],
        help="Task type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., spider, bird, grailqa, webqsp, wikitq)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Data stage",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (e.g., gpt-4o-mini, deepseek-chat)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Explicit OpenAI-compatible API base URL (no proxy is configured by default)",
    )

    # Inference settings
    parser.add_argument(
        "--shot-k",
        type=int,
        default=10,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--n-votes",
        type=int,
        default=1,
        help="Number of votes for majority voting (1 = no voting)",
    )
    parser.add_argument(
        "--no-execution-guidance",
        action="store_true",
        help="Disable execution-guided code revision",
    )
    parser.add_argument(
        "--ablation",
        choices=[
            "full",
            "no_execution_feedback",
            "no_code_merge",
            "no_knowledge_transfer",
            "no_decomposition",
        ],
        default="full",
        help="Paper ablation setting.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["cross_task", "same_dataset", "disabled"],
        default=None,
        help="Override semantic demonstration retrieval scope.",
    )
    parser.add_argument(
        "--no-final-repair",
        action="store_true",
        help="Disable final full-data execution repair.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel sample workers (1 = sequential). "
             "Each sample can also run n-votes in parallel (double parallelism).",
    )

    # Sample settings
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to test (None = all)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index for sample slicing",
    )
    parser.add_argument(
        "--qids",
        type=str,
        nargs="+",
        help="Specific question IDs to test",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (auto-generated if not provided)",
    )

    parser.add_argument(
        "--data-root",
        default=None,
        help="Benchmark data root (or set PANDORA_DATA_ROOT)",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Configuration directory (defaults to the packaged configs)",
    )

    return parser.parse_args(argv)


def _sanitize_for_json(obj):
    """Recursively sanitize an object for JSON serialization."""
    import numpy as np
    import pandas as pd
    from datetime import datetime as dt

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (dt, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    return repr(obj)


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy/pandas/datetime types that are not natively JSON serializable."""

    def default(self, obj):
        import numpy as np
        import pandas as pd
        from datetime import datetime as dt

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, dt)):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


def create_dataset(task: str, dataset_name: str, config: Config):
    """Create dataset instance based on task and dataset name."""
    from datasets.nl2sql.spider import SpiderDataset
    from datasets.nl2sql.spider_syn import SpiderSynDataset
    from datasets.nl2sql.bird import BirdDataset
    from datasets.kbqa.grailqa import GrailQADataset
    from datasets.kbqa.webqsp import WebQSPDataset
    from datasets.tableqa.wikitq import WikiTQDataset
    from datasets.tableqa.wikisql import WikiSQLDataset
    from datasets.cross_source import CrossSourceDataset

    data_root = config.get("paths.data_root", "./data")

    if task == "nl2sql":
        if dataset_name == "spider":
            return SpiderDataset(data_root=data_root)
        elif dataset_name in ("spider-syn", "spider_syn"):
            return SpiderSynDataset(data_root=data_root)
        elif dataset_name == "bird":
            return BirdDataset(data_root=data_root)

    elif task == "kbqa":
        if dataset_name == "grailqa":
            return GrailQADataset(data_root=data_root)
        elif dataset_name == "webqsp":
            return WebQSPDataset(data_root=data_root)

    elif task == "tableqa":
        if dataset_name == "wikitq":
            return WikiTQDataset(data_root=data_root)
        elif dataset_name == "wikisql":
            return WikiSQLDataset(data_root=data_root)

    elif task == "cross_source" and dataset_name in ("cross_source", "cross-source"):
        return CrossSourceDataset(data_root=data_root)

    raise ValueError(f"Unknown dataset: {task}/{dataset_name}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration before resolving writable paths.
    config = Config(config_dir=args.config_dir, task_name=args.task)
    if args.data_root:
        config._config.setdefault("paths", {})["data_root"] = str(
            Path(args.data_root).expanduser().resolve()
        )
    if args.output_dir is None:
        args.output_dir = config.get("paths.result_root", str(Path.cwd() / "results"))

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_start_time = time.time()
    output_path = Path(args.output_dir) / f"{args.dataset}_{args.stage}_{timestamp}.json"

    # Setup logging
    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / f"{args.dataset}_{args.stage}_{timestamp}.log")

    logger = setup_logger("pandora", log_file=args.log_file)
    logger.info(f"Starting Pandora: task={args.task}, dataset={args.dataset}, model={args.model}")

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setFormatter(formatter)

    # Override with command line args
    config._config["model"]["name"] = args.model
    config._config["model"]["temperature"] = args.temperature
    config._config["inference"]["shot_k"] = args.shot_k
    config._config["inference"]["n_votes"] = args.n_votes
    config._config["inference"]["do_execution_guidance"] = not args.no_execution_guidance
    config._config["inference"]["do_final_execution_repair"] = not args.no_final_repair

    ablation_overrides = {
        "no_execution_feedback": {"do_execution_guidance": False},
        "no_code_merge": {"do_code_merge": False},
        "no_decomposition": {"do_decomposition": False},
    }
    config._config["inference"].update(ablation_overrides.get(args.ablation, {}))
    if args.ablation == "no_knowledge_transfer":
        config._config["retrieval"]["mode"] = "same_dataset"
    if args.retrieval_mode:
        config._config["retrieval"]["mode"] = args.retrieval_mode

    # Create model
    logger.info(f"Initializing model: {args.model}")
    model = ModelRegistry.create(
        model_name=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
    )

    # Create dataset
    logger.info(f"Initializing dataset: {args.dataset}")
    dataset = create_dataset(args.task, args.dataset, config)

    # Load examples
    logger.info(f"Loading {args.stage} examples")
    try:
        if args.qids:
            examples = dataset.load_examples(args.stage, qids=args.qids)
        else:
            all_examples = dataset.load_examples(args.stage)
            start = args.start_idx
            end = start + args.num_samples if args.num_samples else len(all_examples)
            examples = all_examples[start:end]
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("%s", exc)
        logger.error(
            "Benchmark assets are not distributed in this code-only release. "
            "See DATASETS.md and pass --data-root /path/to/data."
        )
        return 2

    logger.info(f"Testing on {len(examples)} examples")

    # Create agent
    if args.mode == "vanilla":
        logger.info("Using VanillaAgent (one-shot LLM baseline)")
        agent = VanillaAgent(
            dataset=dataset,
            model=model,
            config=config.to_dict(),
        )
    else:
        logger.info("Using PandoraAgent (code reasoning + execution feedback)")
        agent = PandoraAgent(
            dataset=dataset,
            model=model,
            config=config.to_dict(),
        )

    # ── Parallel inference ───────────────────────────────────────────────
    # Double parallelism:
    #   Outer: num_workers samples run concurrently
    #   Inner: each sample's n-votes run concurrently (via run_with_voting)
    num_workers = getattr(args, "num_workers", 1) or 1

    results = []
    results_lock = threading.Lock()

    def _process_single_sample(idx: int, example: dict) -> dict:
        """Run inference on a single example."""
        example_id = example.get("qid", example.get("id", example.get("question_id", "unknown")))
        question = example.get("question", "")[:80]
        logger.info(f"[{idx+1}/{len(examples)}] id={example_id} | {question}")

        sample_start = time.time()
        try:
            if args.n_votes > 1:
                result = agent.run_with_voting(example, n_votes=args.n_votes, shot_k=args.shot_k)
            else:
                result = agent.run(example, shot_k=args.shot_k)

            sample_time = time.time() - sample_start
            result["sample_time_sec"] = round(sample_time, 2)

            is_correct = result.get("metrics", {}).get("correct", False)
            # VanillaAgent doesn't have python_results/python_exception
            if args.mode == "vanilla":
                exec_ok = bool(result.get("answer")) or result.get("metrics", {}).get("em", 0) > 0
            else:
                exec_ok = result.get("python_results", {}).get("success", False) or not result.get("python_exception", True)
            logger.info(f"  -> correct={is_correct}, exec_ok={exec_ok}, time={sample_time:.1f}s")

            # Thread-safe append + incremental save
            with results_lock:
                results.append((idx, result, sample_time))
                _save_incremental_results()

            return result

        except Exception as e:
            sample_time = time.time() - sample_start
            error_result = {
                "example_id": str(example_id),
                "question": question,
                "metrics": {"em": 0.0, "f1": 0.0, "correct": False},
                "error": str(e),
                "sample_time_sec": round(sample_time, 2),
            }
            logger.error(f"  -> Error: {e}", exc_info=True)
            with results_lock:
                results.append((idx, error_result, sample_time))
                _save_incremental_results()
            return error_result

    def _save_incremental_results():
        """Save results incrementally. Must be called under results_lock."""
        # Sort by original index to keep order
        sorted_results = sorted(results, key=lambda x: x[0])
        live_total = len(sorted_results)
        if args.mode == "vanilla":
            # VanillaAgent: use answer presence or EM > 0
            live_exec_ok = sum(
                1 for _, r, _ in sorted_results
                if bool(r.get("answer")) or r.get("metrics", {}).get("em", 0) > 0
            )
        else:
            live_exec_ok = sum(
                1 for _, r, _ in sorted_results
                if r.get("python_results", {}).get("success", False) or not r.get("python_exception", True)
            )
        live_correct = sum(1 for _, r, _ in sorted_results if r.get("metrics", {}).get("correct", False))
        live_f1 = sum(float(r.get("metrics", {}).get("f1", 0.0)) for _, r, _ in sorted_results)
        live_hit_1 = sum(float(r.get("metrics", {}).get("hit_1", 0.0)) for _, r, _ in sorted_results)
        live_total_time = sum(t for _, _, t in sorted_results)
        live_metrics = {
            "total_samples": live_total,
            "successful_executions": live_exec_ok,
            "correct_answers": live_correct,
            "execution_success_rate": round(live_exec_ok / live_total, 4) if live_total > 0 else 0.0,
            "exact_match_accuracy": round(live_correct / live_total, 4) if live_total > 0 else 0.0,
            "average_f1": round(live_f1 / live_total, 4) if live_total > 0 else 0.0,
            "average_hit_1": round(live_hit_1 / live_total, 4) if live_total > 0 else 0.0,
            "average_execution_time_sec": round(live_total_time / live_total, 4) if live_total > 0 else 0.0,
        }
        output_data = {
            "test_config": {
                "task": args.task,
                "dataset": args.dataset,
                "model": args.model,
                "mode": args.mode,
                "ablation": args.ablation,
                "shot_k": args.shot_k,
                "retrieval_mode": config.get("retrieval.mode"),
                "num_samples": len(examples),
                "num_workers": num_workers,
                "timestamp": timestamp,
            },
            "accuracy_metrics": live_metrics,
            "total_time_sec": round(live_total_time, 2),
            "wall_clock_time_sec": round(time.time() - main_start_time, 2),
            "detailed_results": [_sanitize_for_json(r) for _, r, _ in sorted_results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        em = live_metrics["exact_match_accuracy"]
        exec_rate = live_metrics["execution_success_rate"]
        logger.info(f"Saved incremental results ({live_total}/{len(examples)}): EM={em:.1%}, ExecOK={exec_rate:.1%}")

    if num_workers > 1:
        logger.info(f"Running with {num_workers} parallel workers (each with n_votes={args.n_votes} inner parallelism)")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_process_single_sample, i, ex)
                for i, ex in enumerate(examples)
            ]
            for future in as_completed(futures):
                future.result()  # re-raise any unhandled exception
    else:
        # Sequential fallback (behaves exactly like before)
        for i, example in enumerate(examples):
            _process_single_sample(i, example)

    # Final Summary
    sorted_results = sorted(results, key=lambda x: x[0])
    wall_clock_sec = round(time.time() - main_start_time, 2)
    n = len(sorted_results)
    if args.mode == "vanilla":
        successful_executions = sum(
            1 for _, r, _ in sorted_results
            if bool(r.get("answer")) or r.get("metrics", {}).get("em", 0) > 0
        )
    else:
        successful_executions = sum(
            1 for _, r, _ in sorted_results
            if r.get("python_results", {}).get("success", False) or not r.get("python_exception", True)
        )
    correct_answers = sum(1 for _, r, _ in sorted_results if r.get("metrics", {}).get("correct", False))
    average_f1 = sum(float(r.get("metrics", {}).get("f1", 0.0)) for _, r, _ in sorted_results) / n if n else 0.0
    average_hit_1 = sum(float(r.get("metrics", {}).get("hit_1", 0.0)) for _, r, _ in sorted_results) / n if n else 0.0
    total_time = sum(t for _, _, t in sorted_results)
    exec_rate = successful_executions / n if n > 0 else 0
    em = correct_answers / n if n > 0 else 0

    mode_label = "Vanilla" if args.mode == "vanilla" else "Pandora"
    summary = (
        f"\n{'='*60}\n"
        f"{mode_label} {args.task.upper()} Test Complete\n"
        f"{'='*60}\n"
        f"Total samples: {n}\n"
        f"Workers: {num_workers} (outer) × {args.n_votes} (inner voting)\n"
        f"Successful executions: {successful_executions} ({exec_rate:.1%})\n"
        f"Correct answers: {correct_answers} (EM={em:.1%})\n"
        f"Average F1: {average_f1:.1%}\n"
        f"Average Hit@1: {average_hit_1:.1%}\n"
        f"Wall-Clock Time: {wall_clock_sec:.1f}s\n"
        f"Total time: {total_time:.1f}s\n"
        f"Results saved to: {output_path}"
    )
    logger.info(summary)
    print(summary)

    return {
        "total_samples": n,
        "successful_executions": successful_executions,
        "correct_answers": correct_answers,
        "execution_success_rate": round(exec_rate, 4),
        "exact_match_accuracy": round(em, 4),
        "average_f1": round(average_f1, 4),
        "average_hit_1": round(average_hit_1, 4),
        "average_execution_time_sec": round(total_time / n, 4) if n > 0 else 0.0,
        "wall_clock_time_sec": wall_clock_sec,
    }


def cli() -> int:
    """Console-script wrapper with conventional process exit codes."""
    result = main()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(cli())
