# Dataset setup

This code-only release does not contain benchmark examples, databases, tables,
Freebase triples, generated memory, or query-local KG BOXes. Download each asset
from its publisher, review its terms, and keep it outside Git.

## Official sources

| Dataset | Publisher source |
|---|---|
| Spider | [Yale Spider repository](https://github.com/taoyds/spider) |
| Spider-Syn | [Spider-Syn repository](https://github.com/ygan/Spider-Syn) |
| BIRD | [BIRD benchmark](https://bird-bench.github.io/) |
| WikiTableQuestions | [Stanford release](https://github.com/ppasupat/WikiTableQuestions/releases) |
| WikiSQL | [Salesforce WikiSQL repository](https://github.com/salesforce/WikiSQL) |
| GrailQA | [GrailQA repository](https://github.com/dki-lab/GrailQA) |
| WebQSP | [WebQuestionsSP release](https://www.microsoft.com/en-us/download/details.aspx?id=52763) |

Use the publisher's current instructions. In particular, Spider asks users to
download the data again from its official site, and BIRD currently publishes
its data under CC BY-SA 4.0. Do not assume Pandora's Apache-2.0 license applies
to any benchmark.

## Expected layout

Pandora accepts the root through `PANDORA_DATA_ROOT` or `--data-root`:

```text
pandora-data/
├── spider/
│   ├── spider.dev.json
│   ├── dev_tables.json
│   └── dev_database/<db_id>/<db_id>.sqlite
├── spider-syn/
│   └── spider-syn.test.json
├── bird/
│   ├── bird.dev.json
│   └── dev_database/<db_id>/<db_id>.sqlite
├── wikitq/
│   ├── wikitq.test.json
│   └── csv/...
├── wikisql/
│   ├── wikisql.test.json
│   └── csv/...
├── grailqa/
│   ├── grailqa.test.json
│   ├── entity_link/grailqa.entity_link.test.json
│   └── box/{box_schema.json,test/<qid>/*.csv}
├── webqsp/
│   ├── webqsp.test.json
│   ├── entity_link/webqsp.entity_link.test.json
│   └── box/{box_schema.json,test/<qid>/*.csv}
└── cross_source/
    └── cross_source.test.json
```

The normalized `*.json` files shown above are adapter inputs produced from the
publisher formats. They are not distributed in this release. KG BOXes are
derived artifacts; build them with `scripts/build_kg_boxes.py` after obtaining
an appropriately licensed Freebase source. Verified memory is built with
`scripts/build_memory.py` and must remain under the ignored data root.

Run `python scripts/audit_repository.py --mode reproducibility` to report which
paper assets are still absent. Missing data in code-only mode is informational,
not a code-health failure.
