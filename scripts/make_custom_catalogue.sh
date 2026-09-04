#!/usr/bin/env bash
# Build the custom NMODL catalogue used by TSModel (ampasyn, nmdasyn, hhnotemp, ...).
# Run from anywhere; requires uv. Rebuild after changing files in
# toric_spines_sim/mechanisms/my_catalogue/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/toric_spines_sim/mechanisms"
uv run arbor-build-catalogue custom my_catalogue
echo "Wrote $ROOT/toric_spines_sim/mechanisms/custom-catalogue.so"
