#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/_run_benchmark.sh" nl2sql spider dev "$@"
