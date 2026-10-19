"""Independent HTTP/SSE open-loop validation, with real concurrent arrivals.

Inter-chunk intervals are intentionally not called token ITL.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from run_trace import DATA, MODEL, workload
from kv_sched.policy import Config,percentiles


async def measure(port,trace,rate):
    import aiohttp
    start=time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        async def one(i,row):
            due=start+i/rate
            await asyncio.sleep(max(0,due-time.perf_counter()))
            submitted=time.perf_counter()
            first=None; chunks=[]; usage=None; text=[]
            payload={'model':'qwen3-kv','prompt':row['tokens'],'max_tokens':row['output'],
                     'temperature':0,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
            async with session.post(f'http://127.0.0.1:{port}/v1/completions',json=payload) as response:
                if response.status!=200: raise RuntimeError(await response.text())
                async for line in response.content:
                    if not line.startswith(b'data: '): continue
                    content=line[6:].strip()
                    if content==b'[DONE]': break
                    d=json.loads(content)
                    if d.get('usage'): usage=d['usage']
                    for choice in d.get('choices',[]):
                        s=choice.get('text','')
                        if s:
                            stamp=time.perf_counter()
                            if first is None: first=stamp
                            chunks.append(stamp); text.append(s)
            ended=time.perf_counter()
            if first is None or usage is None: raise RuntimeError('missing nonempty content or usage')
            return {'id':row['id'],'output_tokens':usage['completion_tokens'],
                    'ttft_ms':(first-submitted)*1000,'e2e_ms':(ended-submitted)*1000,
                    'submission_lag_ms':(submitted-due)*1000,
                    'inter_chunk_ms':[(b-a)*1000 for a,b in zip(chunks,chunks[1:])],
                    'text':''.join(text),'status':200}
        rows=await asyncio.gather(*(one(i,r) for i,r in enumerate(trace)))
    wall=time.perf_counter()-start
    return rows,{'requests':len(rows),'output_tok_s':sum(r['output_tokens'] for r in rows)/wall,
                 'wall_seconds':wall,'ttft_ms':percentiles([r['ttft_ms'] for r in rows]),
                 'e2e_ms':percentiles([r['e2e_ms'] for r in rows]),
                 'inter_chunk_ms':percentiles([x for r in rows for x in r['inter_chunk_ms']]),
                 'submission_lag_ms':percentiles([r['submission_lag_ms'] for r in rows])}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--requests',type=int,default=128)
    ap.add_argument('--rate',type=float,default=2)
    ap.add_argument('--seeds',default='17,29,43,71,101')
    ap.add_argument('--label',default='v3_http')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    import urllib.request
    for j,seed in enumerate(map(int,args.seeds.split(','))):
        for policy in (['default','recompute'] if j%2==0 else ['recompute','default']):
            dest=DATA/'raw'/args.label/f'B-{policy}'/f'seed-{seed}'
            if (dest/'metrics.json').exists(): continue
            dest.mkdir(parents=True,exist_ok=True)
            cfg=Config(preemption=policy,cost_model=str(DATA/'cost_models/qwen3-1.7b-3090.json'))
            with socket.socket() as s:
                s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
            env=os.environ.copy()
            env.update(PYTHONPATH=f'{DATA}/vendor/optimized:{root}',HF_HUB_OFFLINE='1',
                       PYTHONHASHSEED='17',VLLM_ENABLE_V1_MULTIPROCESSING='1',OMP_NUM_THREADS='4',
                       KV_SCHED_CONFIG=json.dumps(asdict(cfg)),VLLM_SERVER_DEV_MODE='1')
            cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',MODEL,
                 '--served-model-name','qwen3-kv','--host','127.0.0.1','--port',str(port),
                 '--dtype','float16','--max-model-len','8192','--block-size','16',
                 '--num-gpu-blocks-override','640','--max-num-batched-tokens','1024',
                 '--max-num-seqs','16','--gpu-memory-utilization','0.8','--enable-prefix-caching',
                 '--enable-chunked-prefill','--enforce-eager','--disable-log-stats','--no-async-scheduling',
                 '--generation-config','vllm']
            log=(dest/'server.log').open('w')
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                ready=False
                for _ in range(240):
                    if child.poll() is not None: raise RuntimeError('HTTP server exited; see '+str(dest/'server.log'))
                    try:
                        urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1).close()
                        ready=True; break
                    except Exception: time.sleep(.5)
                if not ready: raise TimeoutError('server readiness timeout')
                trace=workload('B',args.requests,seed)
                # Warm all prompt lengths using unrelated tokens, then clear GPU cache.
                for length in [512,2048,4096,6144]:
                    from run_trace import prompt
                    payload=json.dumps({'model':'qwen3-kv','prompt':prompt(90000+length,length),
                                        'max_tokens':2,'ignore_eos':True,'temperature':0}).encode()
                    urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',
                         data=payload,headers={'Content-Type':'application/json'}),timeout=120).read()
                urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/reset_prefix_cache',method='POST'),timeout=10).read()
                rows,metrics=asyncio.run(measure(port,trace,args.rate))
                assert len(rows)==len(trace)
                expected={r['id']:r['output'] for r in trace}
                assert all(r['output_tokens']==expected[r['id']] for r in rows)
                (dest/'requests.json').write_text(json.dumps(rows))
                (dest/'trace.json').write_text(json.dumps(trace))
                result={'seed':seed,'policy':policy,'rate':args.rate,'metrics':metrics,
                        'trace_hash':hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest(),
                        'config':asdict(cfg),'command':cmd,'protocol':'HTTP SSE; inter-chunk latency, not token ITL',
                        'server_dev_mode':True,'bind_host':'127.0.0.1',
                        'source_snapshot':json.loads((DATA/'v3_full_source_snapshot.json').read_text())}
                (dest/'metrics.json').write_text(json.dumps(result,indent=2))
                print('HTTP_RESULT',str(dest),metrics,flush=True)
            finally:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait()
                log.close()


if __name__=='__main__': main()
