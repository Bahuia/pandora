# Benchmark data

Pandora uses a separate data root selected with `PANDORA_DATA_ROOT` or
`--data-root`. `pandora-data` downloads Pandora-processed annotations from the
versioned [bahuia/pandora-data](https://huggingface.co/datasets/bahuia/pandora-data)
dataset repository and imports publisher assets where required.

## Sources and terms

| Dataset | Split | Publisher source | Data terms |
|---|---|---|---|
| Spider | dev | [Yale Spider](https://github.com/taoyds/spider) | Consult the current publisher terms |
| Spider-Syn | test | [Spider-Syn](https://github.com/ygan/Spider-Syn) | MIT |
| BIRD | dev | [BIRD](https://bird-bench.github.io/) | CC BY-SA 4.0 |
| WikiTableQuestions | test | [Stanford WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions) | CC BY-SA 4.0 |
| WikiSQL | test | [Salesforce WikiSQL](https://github.com/salesforce/WikiSQL) | BSD-3-Clause |

Always review the publisher's current terms before downloading or redistributing
benchmark content. Pandora's Apache-2.0 license applies only to source code.

## Setup

```bash
export PANDORA_DATA_ROOT=/path/to/pandora-data

pandora-data prepare --dataset wikitq
pandora-data prepare --dataset wikisql

pandora-data prepare --dataset spider --source /path/to/official-spider
pandora-data prepare --dataset spider-syn

pandora-data prepare --dataset bird --source /path/to/official-bird-dev

pandora-data verify --dataset all
```

`--source` accepts an extracted directory, `.zip`, or tar archive. For
`--dataset all`, it may also point to a directory containing `spider/` and
`bird/` subdirectories. Downloads use standard `HTTP_PROXY`, `HTTPS_PROXY`, and
`NO_PROXY` settings and resume from the local cache when the server supports
HTTP range requests.

## Materialized layout

```text
pandora-data/
├── spider/
│   ├── spider.dev.json
│   ├── spider.tables.dev.json
│   └── dev_database/<db_id>/<db_id>.sqlite
├── spider-syn/
│   └── spider-syn.test.json
├── bird/
│   ├── bird.dev.json
│   ├── bird.tables.dev.json
│   └── dev_database/<db_id>/<db_id>.sqlite
├── wikitq/
│   └── wikitq.test.json
└── wikisql/
    └── wikisql.test.json
```

Downloaded artifacts are pinned to the revision recorded in
`pandora_data/manifests/benchmarks.json`. `pandora-data verify` checks the
materialized layout before inference.
