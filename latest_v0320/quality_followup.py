"""Native self-repeat controls and semantic cache checks after tight-FP16 cases."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from latest_v0320.common import DATA,ROOT,PYTHON,clean_env,dump

def execute(cmd,logfile):
    with logfile.open('a') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=clean_env(True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:code=124
        finally:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        if code:raise RuntimeError(f'quality followup failed: {logfile}, rc={code}')

def main():
    lock=(DATA/'performance.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock2=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock').open('a');fcntl.flock(lock2,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for group,policy in [('C-1024-eager-NT','NT'),('E-4096-eager-NG','NG')]:
        folder=DATA/'raw/teacher'/group
        for mode in ['N0',policy]:
            for rep in [1,2]:
                output=folder/f'{mode}-repeat-{rep}.json'
                if output.exists():continue
                print('REPEAT',group,mode,rep,flush=True)
                execute([str(PYTHON),'-m','latest_v0320.teacher_forced','--worker',str(folder/'spec.json'),
                    '--policy',mode,'--output',str(output)],folder/f'{mode}-repeat-{rep}.log')
    for track in ['eager','default']:
        for policy in ['N0','NT']:
            output=DATA/f'raw/semantic/{track}-{policy}.json';output.parent.mkdir(exist_ok=True,parents=True)
            if output.exists():continue
            print('SEMANTIC',track,policy,flush=True)
            execute([str(PYTHON),'-m','latest_v0320.semantic_cache','--policy',policy,'--track',track,
                '--output',str(output)],output.with_suffix('.log'))
    print('FOLLOWUP_COMPLETED',flush=True)

if __name__=='__main__':main()
