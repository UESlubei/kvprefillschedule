"""Detached guardian: preserve an active run, resume interruptions, update reports."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from latest_v0320.common import DATA,ROOT,PYTHON,dump,clean_env

def alive(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        return state!='Z' and b'latest_v0320.pipeline' in cmd
    except (FileNotFoundError,ProcessLookupError):return False

def update_summary(log):
    # Keep report CPU work off the eight benchmark CPUs. Do not restrict the
    # guardian's affinity: any resumed benchmark inherits the original CPU set.
    cores=sorted(os.sched_getaffinity(0));analysis_cores=cores[8:16] or cores
    def affinity():os.sched_setaffinity(0,analysis_cores)
    subprocess.run([str(PYTHON),'-m','latest_v0320.summarize'],cwd=ROOT,env=clean_env(True),
        stdout=log,stderr=subprocess.STDOUT,preexec_fn=affinity,check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch-pid',type=int,required=True);args=ap.parse_args()
    lock=(DATA/'guardian.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=DATA/'pipeline/background-state.json'
    state={'guardian_pid':os.getpid(),'watch_pid':args.watch_pid,'state':'monitoring_existing_pipeline',
        'started_unix':time.time(),'report':str(ROOT/'reports/latest_v0320/latest_comparison_report.md')}
    with (DATA/'logs/background-guardian.log').open('a',buffering=1) as log:
        try:
            previous=-1
            while alive(args.watch_pid):
                count=len(list((DATA/'raw/comparison').glob('*/*/*/seed-*/metrics.json')))
                if count!=previous:update_summary(log);previous=count
                state.update(completed_runs=count,updated_unix=time.time());dump(state_path,state)
                time.sleep(20)
            completed=all((DATA/f'pipeline/{stage}-completed.json').exists() for stage in ['formal','deployment_formal'])
            if not completed:
                status=json.loads((DATA/'pipeline/status.json').read_text())
                failures=[k for k,v in status.items() if k.split('/')[0] in {'formal','deployment_formal'} and v.get('state')=='failed']
                if failures:raise RuntimeError('benchmark failure requires inspection: '+str(failures))
                state['state']='waiting_for_gpu_before_resume';dump(state_path,state)
                while True:
                    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
                    if not gpu.strip():break
                    time.sleep(20)
                state['state']='resuming_interrupted_pipeline';dump(state_path,state)
                cmd=[str(PYTHON),'-m','latest_v0320.pipeline','--stages','formal,deployment_formal']
                with (DATA/'logs/background-resume.log').open('a') as pipeline_log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=clean_env(True),stdout=pipeline_log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                    state['resumed_pipeline_pid']=child.pid
                    while child.poll() is None:
                        count=len(list((DATA/'raw/comparison').glob('*/*/*/seed-*/metrics.json')))
                        if count!=previous:update_summary(log);previous=count
                        state.update(completed_runs=count,updated_unix=time.time());dump(state_path,state)
                        time.sleep(20)
                    if child.returncode:raise RuntimeError('resumed pipeline exited '+str(child.returncode))
            state['state']='finalizing';dump(state_path,state)
            subprocess.run([str(PYTHON),'-m','latest_v0320.finalize_comparison'],cwd=ROOT,
                env=clean_env(True),stdout=log,stderr=subprocess.STDOUT,check=True)
            state.update(state='completed',ended_unix=time.time());dump(state_path,state)
        except Exception as exc:
            state.update(state='failed',error=repr(exc),ended_unix=time.time());dump(state_path,state)
            log.write(repr(exc)+'\n');raise

if __name__=='__main__':main()
