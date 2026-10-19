"""Exclusive serial GPU pipeline. Every stage is resumable and keeps its log."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,allowed[:8])
    env=os.environ.copy()
    env.update(PYTHONPATH=f'{DATA}/vendor/optimized:{ROOT}',HF_HUB_OFFLINE='1',
               PYTHONHASHSEED='17',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='4')
    dest=DATA/'pipeline_v3';dest.mkdir(exist_ok=True)
    status_path=dest/'status.json'
    statuses=json.loads(status_path.read_text()) if status_path.exists() else {}
    stages=[
        ('followups',[str(ROOT/'benchmarks/run_followups.py')]),
        ('matrix',[str(ROOT/'benchmarks/run_matrix.py'),'--label','v3_full','--requests','256']),
        ('extended',[str(ROOT/'benchmarks/run_extended.py')]),
        ('long',[str(ROOT/'benchmarks/run_long.py')]),
        ('finalize',[str(ROOT/'benchmarks/finalize_full.py')]),
    ]
    for name,args in stages:
        if statuses.get(name,{}).get('returncode')==0:
            continue
        frozen=json.loads((DATA/'v3_full_source_snapshot.json').read_text())
        for rel,digest in frozen['hashes'].items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('frozen core changed: '+rel)
        for absolute,digest in frozen.get('vendor_hashes',{}).items():
            if hashlib.sha256(Path(absolute).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('frozen overlay changed: '+absolute)
        cmd=[sys.executable,*args]
        statuses[name]={'started_unix':time.time(),'command':cmd,'state':'running'}
        status_path.write_text(json.dumps(statuses,indent=2))
        print('STAGE_START',name,flush=True)
        with (dest/f'{name}.log').open('a') as log:
            child=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        statuses[name].update(ended_unix=time.time(),returncode=child.returncode,
                              state='completed' if child.returncode==0 else 'failed')
        status_path.write_text(json.dumps(statuses,indent=2))
        print('STAGE_END',name,child.returncode,flush=True)
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__=='__main__':main()
