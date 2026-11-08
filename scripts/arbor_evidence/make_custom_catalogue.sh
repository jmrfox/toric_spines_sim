#!/usr/bin/env bash
# Build the frozen Arbor NMODL catalogue for the evidence scripts.
#   uv sync --group arbor-evidence && uv run --group arbor-evidence bash scripts/arbor_evidence/make_custom_catalogue.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/arbor_evidence/arbor_model/mechanisms"
uv run --group arbor-evidence arbor-build-catalogue custom my_catalogue
echo "Wrote $ROOT/scripts/arbor_evidence/arbor_model/mechanisms/custom-catalogue.so"
