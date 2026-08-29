# Pandora

Official implementation of **Pandora: Unified Structured Knowledge Reasoning
with Executable Pandas BOXes**.

Pandora maps relational databases, tables, and query-local knowledge into a
shared DataFrame representation called BOX. The implementation includes
box-aware linking, step-wise decomposition, iterative code solving with
execution feedback, and final code merging.

## Benchmarks

| Task | Benchmarks |
|---|---|
| Text-to-SQL | Spider, Spider-Syn, BIRD |
| Table QA | WikiTableQuestions, WikiSQL |

## Installation

Python 3.9, 3.10, and 3.11 are tested.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pandora --help
pandora-data --help
```

## Prepare data

Pandora keeps benchmark data outside the source checkout. Choose a data root,
inspect the supported datasets, and prepare the desired benchmark:

```bash
export PANDORA_DATA_ROOT=/path/to/pandora-data

pandora-data list
pandora-data prepare --dataset wikitq
pandora-data verify --dataset wikitq
```

Spider and BIRD include publisher-provided database assets. Download the
official package and pass it to the preparation tool:

```bash
pandora-data prepare --dataset spider --source /path/to/spider-package
pandora-data prepare --dataset bird --source /path/to/bird-dev-package
```

The tool validates record counts, required schemas, database files, and
checksums. See [DATASETS.md](DATASETS.md) for official sources and exact setup
details.

## Configure a model

Official providers read their standard environment variables:

```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

For an OpenAI-compatible service, keep endpoint settings outside the repository:

```bash
export PANDORA_BASE_URL="http://your-endpoint.example/v1"
export PANDORA_API_KEY="..."
export PANDORA_MODEL="your-model-name"
export PANDORA_PROVIDER="openai-compatible"
```

## Run benchmarks

The example scripts use `PANDORA_DATA_ROOT` and the model environment variables
above. Additional `pandora` arguments can be appended to each command.

```bash
./examples/run_spider.sh --num-samples 5
./examples/run_spider_syn.sh --num-samples 5
./examples/run_bird.sh --num-samples 5
./examples/run_wikitq.sh --num-samples 5
./examples/run_wikisql.sh --num-samples 5
```

The equivalent direct command is:

```bash
pandora --task nl2sql --dataset spider --stage dev \
  --data-root "$PANDORA_DATA_ROOT" \
  --model "${PANDORA_MODEL:-gpt-4o-mini}" \
  --shot-k 0 --retrieval-mode disabled --num-samples 5
```

Outputs are written under `results/` by default. Set `PANDORA_OUTPUT_DIR` for
the example scripts or pass `--output-dir` directly.

## Method controls

Paper ablations are available through the main CLI:

```bash
pandora ... --ablation no_knowledge_transfer
pandora ... --ablation no_decomposition
pandora ... --ablation no_execution_feedback
pandora ... --ablation no_code_merge
```

## Verification

```bash
python -m pytest -m "not integration"
python -m compileall -q core datasets models pandora_data prompts utils scripts run.py
python -m build
python scripts/audit_repository.py --mode repository --strict
```

Generated code is AST-validated and run in an isolated subprocess with
configurable time, CPU, and memory limits. This research sandbox is not a
production security boundary.

## License

Pandora source code is licensed under Apache-2.0. Benchmark data retains its
upstream license and attribution requirements; see
[THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). Contributions and security reports
are covered by [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
