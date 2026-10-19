"""Run validation/profiling after the exclusive single-GPU matrix finishes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-for-formal',action='store_true')
    args=ap.parse_args()
    status_path=DATA/'raw/v3_full/status.json'
    if args.wait_for_formal:
        while True:
            if status_path.exists():
                statuses=json.loads(status_path.read_text())
                if any(s['returncode'] for s in statuses):raise RuntimeError('formal matrix failed')
                if len(statuses)==4:break
            time.sleep(5)
    env=os.environ.copy()
    env.update(PYTHONPATH=f'{DATA}/vendor/optimized:{ROOT}',HF_HUB_OFFLINE='1',
               PYTHONHASHSEED='17',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='4')
    record_path=DATA/'pipeline_v3/followup_status.json'
    records=json.loads(record_path.read_text()) if record_path.exists() else []
    def run(name,cmd,custom_env=None):
        if any(r['name']==name and r['returncode']==0 for r in records):return
        log=DATA/'pipeline_v3'/f'{name}.log'
        if log.exists():
            shutil.copy2(log,log.with_name(f'{name}.previous-{int(log.stat().st_mtime)}.log'))
        print('FOLLOWUP_START',name,flush=True)
        start=time.time()
        with log.open('w') as f:
            child=subprocess.run(cmd,env=custom_env or env,stdout=f,stderr=subprocess.STDOUT)
        records.append({'name':name,'returncode':child.returncode,'elapsed':time.time()-start,'log':str(log)})
        record_path.write_text(json.dumps(records,indent=2))
        print('FOLLOWUP_END',records[-1],flush=True)
        if child.returncode:raise RuntimeError('followup failed '+name)
    for flavor in ['original_stock','instrumented_default']:
        local_env=env.copy()
        if flavor=='original_stock':local_env['PYTHONPATH']=str(ROOT)
        run('parity-'+flavor,[sys.executable,str(ROOT/'benchmarks/run_trace.py'),
            '--workload','B','--requests','8','--blocks','640','--seeds','17,29,43,71,101',
            '--window','0','--output',str(DATA/'raw/v3_parity'/flavor)],local_env)
    for mode in ['reference','baseline','combined']:
        run('text-'+mode,[sys.executable,str(ROOT/'benchmarks/validate_text.py'),'--mode',mode])
    run('teacher-forced',[sys.executable,str(ROOT/'benchmarks/validate_quality.py')])
    run('fault-injection',[sys.executable,str(ROOT/'benchmarks/stress_abort.py'),'--cohorts','4'])
    run('tests',[sys.executable,'-m','pytest','-q',str(ROOT/'tests')])
    run('model-manifest',[sys.executable,str(ROOT/'benchmarks/capture_model_manifest.py')])
    run('followup-audit',[sys.executable,str(ROOT/'benchmarks/validate_followups.py')])
    # HTTP uses independent EngineCore processes and captures true concurrent arrival times.
    run('http-validation',[sys.executable,str(ROOT/'benchmarks/http_benchmark.py'),'--requests','128','--rate','1.1'])
    (DATA/'profiles').mkdir(exist_ok=True)
    run('profile-combined',['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none',
        '--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
        '-o',str(DATA/'profiles/v3_combined'),sys.executable,str(ROOT/'benchmarks/profile_engine.py'),
        '--workload','C','--requests','24','--blocks','1024','--cpu-blocks','128',
        '--admission','tier','--preemption','restore','--cost-model',str(DATA/'cost_models/qwen3-1.7b-3090.json'),
        '--output',str(DATA/'raw/v3_profile-combined')])
    run('followup-audit',[sys.executable,str(ROOT/'benchmarks/validate_followups.py')])


if __name__=='__main__':main()
