#!/usr/bin/env bash
set -euo pipefail
bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$bundle_root"
sha256sum -c SHA256SUMS
if [[ -e INSTALLATION.json ]]; then
  echo 'This directory is already installed. Extract a fresh bundle to reinstall.' >&2
  exit 1
fi
mkdir -p runtime analysis
tar -xzf packages/runtime.tar.gz -C runtime
tar -xzf packages/analysis.tar.gz -C analysis
export PATH="$bundle_root/runtime/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
runtime/bin/python runtime/bin/conda-unpack
analysis/bin/python analysis/bin/conda-unpack
tar -xzf packages/assets.tar.gz
mkdir -p tools/cuda tools/nsight
tar -xzf packages/cuda.tar.gz -C tools/cuda
tar -xzf packages/nsight.tar.gz -C tools/nsight
runtime/bin/python install.py
source env.sh
"$KV_EXP_PYTHON" verify_offline.py --cpu
echo "Installed at: $bundle_root"
echo "Next: source '$bundle_root/env.sh'"
echo 'GPU verification: "$KV_EXP_PYTHON" "$KV_OFFLINE_ROOT/verify_offline.py" --gpu-smoke'
