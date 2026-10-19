#!/usr/bin/env bash
# Source this file from a shell; uses the existing machine's pinned environment.
export KV_EXP_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KV_EXP_DATA=/mnt/ondeviceinfer/experiments/vllm_kv_scheduling
export KV_EXP_PYTHON="$KV_EXP_DATA/environment/bin/python"
export PYTHONPATH="$KV_EXP_DATA/vendor/optimized:$KV_EXP_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export PYTHONHASHSEED=17
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export OMP_NUM_THREADS=4
