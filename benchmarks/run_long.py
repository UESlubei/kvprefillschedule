"""Independent 2000-request runs for the positive admission claim and stress."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    out=DATA/'raw/v3_long/gpu640';out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for i,seed in enumerate([307,331,347,367,389]):
        for policy in (['G0','G1'] if i%2==0 else ['G1','G0']):
            row=dict(id=f'A-{policy}',seed=seed,workload='A',admission='fcfs' if policy=='G0' else 'gpu',
                     preemption='default',arrival_rate=3.2)
            if not (out/row['id']/f'seed-{seed}/metrics.json').exists():jobs.append(row)
    if jobs:
        suite=out/'suite.json';suite.write_text(json.dumps(jobs,indent=2))
        cmd=[sys.executable,str(ROOT/'benchmarks/run_trace.py'),'--suite',str(suite),
             '--requests','2000','--blocks','640','--cost-model',str(DATA/'cost_models/qwen3-1.7b-3090.json'),
             '--output',str(out),'--timeout','1800']
        with (out/'suite.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    dest=DATA/'raw/v3_stress_long.log'
    with dest.open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'benchmarks/stress_abort.py'),'--cohorts','32',
                        '--label','v3_stress_long'],stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
