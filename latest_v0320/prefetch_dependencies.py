from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320')

def fetch(path):
    with (DATA/'logs'/f'prefetch-{path.stem}.log').open('a') as log:
        p=subprocess.run([sys.executable,str(ROOT/'download_wheel.py'),'--asset',str(path),'--workers','16'],stdout=log,stderr=subprocess.STDOUT)
    print('PREFETCH',path.stem,p.returncode,flush=True)
    return p.returncode

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as ex:
        codes=list(ex.map(fetch,sorted((DATA/'artifacts/dependency-assets').glob('*.json'))))
    raise SystemExit(any(codes))
