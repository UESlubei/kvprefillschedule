"""Serial single-GPU, paired Latin-order matrix. Resume only completed runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
SEEDS=[17,29,43,71,101]
POLICIES={'G0':('fcfs','default'),'G1':('gpu','default'),'G2':('fcfs','lcf'),
          'G3':('fcfs','pr_score'),'G4':('fcfs','recompute'),'G5':('gpu','recompute'),
          'B1':('fcfs','default'),'B2':('gpu','default'),'O1':('tier','default'),
          'O2':('fcfs','restore'),'O3':('tier','restore')}


def jobs(workloads,policies):
    result=[]
    for workload in workloads:
        for i,seed in enumerate(SEEDS):
            order=policies[i%len(policies):]+policies[:i%len(policies)]
            for policy in order:
                a,p=POLICIES[policy]
                result.append({'id':f'{workload}-{policy}','seed':seed,'workload':workload,
                               'admission':a,'preemption':p,
                               'arrival_rate':{'A':3.2,'B':1.1,'C':3.5,'E':3.2}.get(workload,0)})
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--requests',type=int,default=256)
    ap.add_argument('--label',default='formal')
    ap.add_argument('--sections',default='gpu640,gpu1024,cpu,controls')
    args=ap.parse_args()
    specs={
       'gpu640':(640,0,jobs(['A'],['G0','G1'])+jobs(['B'],['G0','G2','G3','G4'])),
       'gpu1024':(1024,0,jobs(['A'],['G0','G1'])+jobs(['B'],['G0','G2','G3','G4'])),
       'cpu':(1024,128,jobs(['C','D'],['B1','B2','O1','O2','O3'])),
       'controls':(4096,0,jobs(['E','F'],['G0','G5']))}
    status=[]
    for name in args.sections.split(','):
        blocks,cpu,rows=specs[name]
        dest=DATA/'raw'/args.label/name
        dest.mkdir(parents=True,exist_ok=True)
        rows=[r for r in rows if not (dest/r['id']/f"seed-{r['seed']}"/'metrics.json').exists()]
        if not rows: continue
        suite=dest/'suite.json'; suite.write_text(json.dumps(rows,indent=2))
        cmd=[sys.executable,str(ROOT/'benchmarks/run_trace.py'),'--suite',str(suite),
             '--requests',str(args.requests),'--blocks',str(blocks),'--cpu-blocks',str(cpu),
             '--cost-model',str(DATA/'cost_models/qwen3-1.7b-3090.json'),'--output',str(dest),'--timeout','900']
        env=os.environ.copy()
        env.update(PYTHONPATH=f'{DATA}/vendor/optimized:{ROOT}',HF_HUB_OFFLINE='1',
                   PYTHONHASHSEED='17',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='4')
        started=time.time()
        print('MATRIX_START',name,len(rows),flush=True)
        with (dest/'suite.log').open('a') as log:
            child=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.append({'section':name,'returncode':child.returncode,'elapsed':time.time()-started})
        (DATA/'raw'/args.label/'status.json').write_text(json.dumps(status,indent=2))
        print('MATRIX_END',name,status[-1],flush=True)
        if child.returncode: raise SystemExit(child.returncode)


if __name__=='__main__': main()
