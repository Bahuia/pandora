# Pandora

Official code preview for **Pandora: Unified Structured Knowledge Reasoning
with Executable Pandas BOXes**.

Pandora maps relational databases, tables, and query-local knowledge graph
subgraphs into a shared DataFrame representation called BOX. The implementation
preserves the paper's box-aware linking, step-wise decomposition, iterative code
solving with execution feedback, and final code merging pipeline.

> **Release status:** `v0.1.0-code-preview` contains source code, task templates,
> configuration, preprocessing entry points, and offline tests. It intentionally
> contains no benchmark samples, generated memory, few-shot data, KG BOXes,
> experiment results, figures, or manuscript/review files. This release is not
> yet a complete reproduction package for the paper's reported numbers.

## Supported tasks

| Task | Benchmarks |
|---|---|
| Text-to-SQL | Spider, Spider-Syn, BIRD |
| Table QA | WikiTableQuestions, WikiSQL |
| Knowledge-base QA | GrailQA, WebQSP |
| Heterogeneous reasoning | Manifest-driven cross-source evaluation |

## Installation

Python 3.9, 3.10, and 3.11 are tested.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pandora --help
```

Set the API key for the selected official provider:

```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

No third-party endpoint is configured. For an explicitly chosen
OpenAI-compatible endpoint, pass `--base-url`.

## Data setup

Obtain each benchmark from its publisher and arrange it under one untracked
data root. Exact source links, licensing notes, and the expected layout are in
[DATASETS.md](DATASETS.md). Pandora does not download or redistribute the data.

```bash
export PANDORA_DATA_ROOT=/path/to/pandora-data

pandora --task nl2sql --dataset spider --stage dev \
  --model gpt-4o-mini --num-samples 5
```

The equivalent explicit options are `--data-root` and `--config-dir`. Missing
assets fail with a message pointing to the data guide.

## Method controls

Paper defaults are temperature 0, top-K 10, and at most three execution-repair
rounds. The paper ablations are exposed directly:

```bash
pandora ... --ablation no_knowledge_transfer
pandora ... --ablation no_decomposition
pandora ... --ablation no_execution_feedback
pandora ... --ablation no_code_merge
```

`no_knowledge_transfer` uses same-dataset verified memory. Use
`--retrieval-mode disabled` for a strict zero-memory diagnostic.

Verified memory and KG BOXes can be constructed after obtaining the required
source assets:

```bash
python scripts/build_memory.py --help
python scripts/build_kg_boxes.py --help
python scripts/preprocess_schema_offline.py --help
```

## Repository layout

```text
configs/        packaged runtime and task configuration
core/           Pandora agents, schema analysis, voting, memory construction
datasets/       benchmark and cross-source adapters
models/         official OpenAI, DeepSeek, and Qwen clients
prompts/tasks/  packaged task templates
scripts/        preprocessing and repository-audit entry points
tests/          synthetic offline unit tests and optional integration tests
utils/          execution, retrieval, schema, and KG BOX utilities
run.py          unified CLI implementation
```

## Verification

```bash
python -m pytest -m "not integration"
python -m compileall -q core datasets models prompts utils scripts run.py
python -m build
python scripts/audit_repository.py --mode code-only --strict
```

Generated code is AST-validated and run in an isolated subprocess with
configurable time, CPU, and memory limits. This research sandbox is not a
production security boundary.

## License and contributions

The Pandora source code in this repository is licensed under Apache-2.0. Data
and other third-party assets retain their own terms and are not covered by that
license; see [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). Contributions are
described in [CONTRIBUTING.md](CONTRIBUTING.md), and vulnerabilities should be
reported according to [SECURITY.md](SECURITY.md).
