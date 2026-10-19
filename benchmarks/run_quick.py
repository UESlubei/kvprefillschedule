"""Small paired validation; reuse calibrated costs and one engine per budget."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='v3_quick')
    args=parser.parse_args()
    specs = [
        ('gpu640', 640, 0, 24, [('A', 'G0', 'fcfs', 'default'),
                               ('A', 'G1', 'gpu', 'default'),
                               ('B', 'G0', 'fcfs', 'default'),
                               ('B', 'G4', 'fcfs', 'recompute')]),
        ('cpu', 1024, 128, 32, [('D', 'B1', 'fcfs', 'default'),
                               ('D', 'O3', 'tier', 'restore')]),
    ]
    status = []
    for section, blocks, cpu, count, policies in specs:
        dest = DATA / 'raw' / args.label / section
        dest.mkdir(parents=True, exist_ok=True)
        jobs = []
        for i, seed in enumerate([17, 29]):
            order = policies if i == 0 else list(reversed(policies))
            for work, policy, admission, preemption in order:
                ident = f'{work}-{policy}'
                if (dest / ident / f'seed-{seed}' / 'metrics.json').exists():
                    continue
                jobs.append(dict(id=ident, seed=seed, workload=work,
                                 admission=admission, preemption=preemption))
        if not jobs:
            continue
        suite = dest / 'suite.json'
        suite.write_text(json.dumps(jobs, indent=2))
        env = os.environ.copy()
        env.update(PYTHONPATH=f'{DATA}/vendor/optimized:{ROOT}',
                   HF_HUB_OFFLINE='1', PYTHONHASHSEED='17',
                   VLLM_ENABLE_V1_MULTIPROCESSING='0', OMP_NUM_THREADS='4')
        cmd = [sys.executable, str(ROOT / 'benchmarks/run_trace.py'),
               '--suite', str(suite), '--requests', str(count),
               '--blocks', str(blocks), '--cpu-blocks', str(cpu),
               '--cost-model', str(DATA / 'cost_models/qwen3-1.7b-3090.json'),
               '--output', str(dest), '--timeout', '180']
        print('QUICK_START', section, len(jobs), count, flush=True)
        started = time.time()
        with (dest / 'suite.log').open('a') as log:
            child = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        status.append(dict(section=section, runs=len(jobs), requests_per_run=count,
                           returncode=child.returncode, seconds=time.time()-started))
        (DATA / 'raw' / args.label / 'status.json').write_text(json.dumps(status, indent=2))
        print('QUICK_END', status[-1], flush=True)
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
