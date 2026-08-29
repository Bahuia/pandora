#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/_run_benchmark.sh" tableqa wikitq test "$@"
