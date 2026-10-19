from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320')
PYTHON = DATA / 'environment/bin/python'
MODEL = '/mnt/ondeviceinfer/models/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
KV_BYTES = 114688

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def percentiles(xs):
    xs = sorted(xs)
    if not xs: return dict(count=0, mean=None, p50=None, p95=None, p99=None, max=None)
    def q(f):
        a = (len(xs)-1)*f; i = int(a)
        return xs[i] + (xs[min(i+1, len(xs)-1)]-xs[i])*(a-i)
    return dict(count=len(xs), mean=sum(xs)/len(xs), p50=q(.5), p95=q(.95), p99=q(.99), max=xs[-1])

def clean_env(candidate=False):
    env = os.environ.copy()
    for key in ['PYTHONPATH', 'KV_SCHED_CONFIG', 'VLLM_ENABLE_V1_MULTIPROCESSING']:
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', PYTHONHASHSEED='17',
        TMPDIR=str(DATA/'tmp'), XDG_CACHE_HOME=str(DATA/'cache'),
        TORCHINDUCTOR_CACHE_DIR=str(DATA/'cache/torchinductor'),
        TRITON_CACHE_DIR=str(DATA/'cache/triton'))
    if candidate: env['PYTHONPATH'] = str(ROOT)
    return env

def workload(kind, count, seed, rate):
    # Execute only the two frozen, pure trace functions, not old engine imports.
    path = ROOT/'benchmarks/run_trace.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'prompt', 'workload'}]
    scope = {'random':random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    rows = scope['workload'](kind, count, seed)
    for i, r in enumerate(rows):
        r['offset'] = 0 if kind=='F' else (i//16*16/rate)
    return rows

def prompt(seed, length):
    rng = random.Random(seed)
    return [rng.randrange(100, 30000) for _ in range(length)]
