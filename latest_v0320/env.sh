#!/usr/bin/env bash
# Keep the native 0.32.0 wheel isolated from the historical 0.11.2 overlay.
unset PYTHONPATH KV_SCHED_CONFIG VLLM_ENABLE_V1_MULTIPROCESSING
export KV032_DATA=/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320
export KV032_PYTHON="$KV032_DATA/environment/bin/python"
export UV_CACHE_DIR="$KV032_DATA/cache/uv"
export TMPDIR="$KV032_DATA/tmp"
export XDG_CACHE_HOME="$KV032_DATA/cache"
export TORCHINDUCTOR_CACHE_DIR="$KV032_DATA/cache/torchinductor"
export TRITON_CACHE_DIR="$KV032_DATA/cache/triton"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
