#!/usr/bin/env bash
set -euo pipefail
bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$bundle_root/env.sh"
if [[ -n "$(find "$KV_EXP_DATA/raw" -name metrics.json -print -quit)" ]]; then
  echo 'Existing experiment results found. Recalibrate in a fresh installation to avoid mixing cost models.' >&2
  exit 1
fi
cd "$KV_EXP_PROJECT"
"$KV_EXP_PYTHON" benchmarks/profile_costs.py --part prefill
"$KV_EXP_PYTHON" benchmarks/profile_costs.py --part transfer
"$KV_EXP_PYTHON" benchmarks/profile_costs.py --part merge
"$KV_EXP_PYTHON" - <<'PY'
import json, os
from pathlib import Path
import torch
p=Path(os.environ['KV_EXP_DATA'])/'cost_models/qwen3-1.7b-3090.json'
d=json.loads(p.read_text());d['gpu']=torch.cuda.get_device_name(0)
d['calibration_note']='Remeasured on the offline target; filename retained for existing benchmark drivers.'
p.write_text(json.dumps(d,indent=2))
print('TARGET_CALIBRATED',d['gpu'],p)
PY
"$KV_EXP_PYTHON" benchmarks/freeze_sources.py
