"""Single-GPU resumable comparison; each run owns a fresh native server."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
from latest_v0320.common import DATA,ROOT,PYTHON,dump,clean_env,digest

def specs(stage):
    if stage=='correctness_default':
        return [dict(workload=w,blocks=b,seed=617,policy=p,track='default',requests=32,correctness=True)
            for w,b,ps in [('A',640,['N0','NG']),('C',1024,['N0','NT'])] for p in ps]
    if stage=='correctness':
        return [dict(workload='A',blocks=640,seed=607,policy=p,track='eager',requests=32,correctness=True)
            for p in ['N0','Noff','NG']]+[
            dict(workload='C',blocks=1024,seed=607,policy=p,track='eager',requests=32,correctness=True)
            for p in ['N0','NG','NT']]
    if stage=='observation':
        return [dict(workload=w,blocks=b,seed=s,policy=p,track='eager',requests=64)
            for w,b in [('A',640),('C',1024)] for s in [701,719]
            for p in (['N0','Nobs'] if s==701 else ['Nobs','N0'])]
    if stage in {'pilot','formal'}:
        cases=[('A',640,['N0','NG']),('C',1024,['N0','NT']),('E',4096,['N0','NG']),('F',4096,['N0','NG'])]
        if stage=='formal':cases.insert(1,('A',1024,['N0','NG']));cases[2]=('C',1024,['N0','NG','NT'])
        rows=[]
        for w,b,ps in cases:
            seeds=[1103,1129,1151,1171,1193] if stage=='formal' else ([811,823] if w in {'A','C'} else [811])
            for i,s in enumerate(seeds):
                order=ps[i%len(ps):]+ps[:i%len(ps)]
                for p in order:rows.append(dict(workload=w,blocks=b,seed=s,policy=p,track='eager',requests=256 if stage=='formal' else 64))
        return rows
    if stage in {'deployment_pilot','deployment_formal'}:
        rows=[]
        for w,b,p in [('A',640,'NG'),('C',1024,'NT'),('E',4096,'NG'),('F',4096,'NG')]:
            seeds=[811] if stage=='deployment_pilot' else [1103,1129,1151,1171,1193]
            for i,s in enumerate(seeds):
                for policy in (['N0',p] if i%2==0 else [p,'N0']):
                    rows.append(dict(workload=w,blocks=b,seed=s,policy=policy,track='default',
                        requests=64 if stage=='deployment_pilot' else 256))
        return rows
    raise ValueError(stage)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stages',default='correctness,observation,pilot,deployment_pilot');args=ap.parse_args()
    lock=(DATA/'performance.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    oldlock=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock').open('a')
    fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if gpu.strip():raise RuntimeError('GPU busy: '+gpu)
    state_path=DATA/'pipeline/status.json'
    status=json.loads(state_path.read_text()) if state_path.exists() else {}
    for stage in args.stages.split(','):
        if stage=='teacher':
            with (DATA/'logs/teacher-forced.log').open('a') as log:
                code=subprocess.call([str(PYTHON),'-m','latest_v0320.teacher_forced'],
                    cwd=ROOT,env=clean_env(True),stdout=log,stderr=subprocess.STDOUT)
            if code:raise RuntimeError('same-history probability diagnostics did not pass')
            dump(DATA/'pipeline/teacher-completed.json',{'completed':True,'ended_unix':time.time()})
            continue
        if stage!='correctness':
            frozen=DATA/'manifests/performance-source-snapshot.json'
            hashes={str(p):digest(p) for p in [ROOT/'latest_v0320/scheduler.py',
                ROOT/'latest_v0320/run_http.py',ROOT/'latest_v0320/common.py',
                ROOT/'benchmarks/run_trace.py',DATA/'cost_models/native.json']}
            if frozen.exists():assert json.loads(frozen.read_text())['hashes']==hashes,'frozen performance implementation changed'
            else:dump(frozen,{'hashes':hashes,'created_unix':time.time()})
        if stage in {'formal','deployment_formal'}:
            evidence=json.loads((ROOT/'reports/latest_v0320/quality-followup-assessment.json').read_text())
            assert evidence['supported_for_performance_experiment'],'quality followup has not supported formal execution'
        for job in specs(stage):
            key=f"{stage}/{job['workload']}-{job['blocks']}/{job['track']}-{job['policy']}/seed-{job['seed']}"
            dest=DATA/'raw/comparison'/key
            if (dest/'metrics.json').exists():
                m=json.loads((dest/'metrics.json').read_text())
                assert m['requests']==job['requests'] and m['errors']==0
                status.setdefault(key,{}).update(state='completed',returncode=0,recovered_from_artifact=True)
                dump(state_path,status);continue
            if stage!='correctness':
                frozen=json.loads((DATA/'manifests/performance-source-snapshot.json').read_text())
                assert all(digest(p)==h for p,h in frozen['hashes'].items()),'frozen source changed before a run'
            if (dest/'server.log').exists():
                failed=DATA/'raw/failed_attempts'/key/f'attempt-{int(time.time())}'
                failed.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(dest),str(failed))
            dest.mkdir(parents=True,exist_ok=True)
            rate=3.5 if job['workload']=='C' else 3.2
            cmd=[str(PYTHON),'-m','latest_v0320.run_http','--workload',job['workload'],
                '--blocks',str(job['blocks']),'--policy',job['policy'],'--seed',str(job['seed']),
                '--requests',str(job['requests']),'--track',job['track'],'--rate',str(rate),'--output',str(dest)]
            if job.get('correctness'):cmd+=['--correctness']
            runtime_env=clean_env(True)
            # Five-second arrivals must not race Uvicorn's five-second idle
            # connection expiry. Same native setting in all paired groups.
            runtime_env['VLLM_HTTP_TIMEOUT_KEEP_ALIVE']='120'
            dump(dest/'runtime-environment.json',{
                'VLLM_HTTP_TIMEOUT_KEEP_ALIVE':120,
                'transport_note':'server idle timeout exceeds client keepalive and 5-second bursts; identical across policies'})
            status[key]={'state':'running','started_unix':time.time(),'job':job};dump(state_path,status)
            print('START',key,flush=True)
            with (dest/'runner.log').open('a') as log:
                child=subprocess.run(cmd,cwd=ROOT,env=runtime_env,stdout=log,stderr=subprocess.STDOUT)
            status[key].update(state='completed' if child.returncode==0 else 'failed',
                returncode=child.returncode,ended_unix=time.time());dump(state_path,status)
            print('END',key,child.returncode,flush=True)
            if child.returncode:raise SystemExit(child.returncode)
        dump(DATA/f'pipeline/{stage}-completed.json',{'completed':True,'runs':len(specs(stage)),'ended_unix':time.time()})
        audit=subprocess.run([str(PYTHON),'-m','latest_v0320.audit_comparison','--stage',stage],
            cwd=ROOT,env=clean_env(True),capture_output=True,text=True)
        (DATA/f'pipeline/{stage}-audit.log').write_text(audit.stdout+audit.stderr)
        if audit.returncode:raise RuntimeError('stage artifact audit failed: '+stage)
        receipt=json.loads((ROOT/f'reports/latest_v0320/{stage}-audit.json').read_text())
        if stage.startswith('correctness') and receipt['unresolved_output_mismatches']:
            raise RuntimeError('output differences need same-history diagnostics before performance stages')
        subprocess.run([str(PYTHON),'-m','latest_v0320.summarize'],cwd=ROOT,
            env=clean_env(True),stdout=subprocess.DEVNULL,check=True)

if __name__=='__main__':main()
