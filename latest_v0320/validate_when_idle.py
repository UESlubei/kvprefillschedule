"""Wait for existing serial experiments, then run isolated compatibility checks."""
from datetime import datetime,timezone
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320')
OLD=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    state_path=DATA/'manifests/validation_status.json'
    state={'state':'waiting_for_gpu','started_unix':time.time(),'checks':{}}
    if state_path.exists():state=json.loads(state_path.read_text())
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state,indent=2))
    own=(DATA/'validation.lock').open('a')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=(OLD/'pipeline_v3/background.lock').open('a')
    while True:
        try:fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            state['state']='waiting_for_existing_experiments';save();time.sleep(30);continue
        p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if p.stdout.strip():
            state['state']='waiting_for_gpu_processes';state['active_gpu_pids']=p.stdout.strip().splitlines();save()
            fcntl.flock(gate,fcntl.LOCK_UN);time.sleep(30);continue
        break
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:8])
    env=os.environ.copy()
    for k in ['PYTHONPATH','KV_SCHED_CONFIG','VLLM_ENABLE_V1_MULTIPROCESSING']:
        env.pop(k,None)
    env.update(HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONHASHSEED='17',
        TMPDIR=str(DATA/'tmp'),XDG_CACHE_HOME=str(DATA/'cache'),
        TORCHINDUCTOR_CACHE_DIR=str(DATA/'cache/torchinductor'),TRITON_CACHE_DIR=str(DATA/'cache/triton'))
    for mode in ['torch','eager','offload','default']:
        if state['checks'].get(mode,{}).get('returncode')==0:continue
        state['state']='running';state['current']=mode;save()
        cmd=[str(DATA/'environment/bin/python'),str(ROOT/'smoke_gpu.py'),'--mode',mode]
        started=time.time()
        with (DATA/f'logs/smoke-{mode}.log').open('a') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                code=124
            finally:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        state['checks'][mode]={'returncode':code,'seconds':time.time()-started}
        if code:
            state['state']='failed';save();raise SystemExit(code)
        save()
    state['state']='completed';state.pop('current',None);save()
    print('GPU_VALIDATION_COMPLETED',flush=True)

if __name__=='__main__':main()
