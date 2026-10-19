"""Fresh native HTTP server per run; paired traces, SSE timing and native metrics."""
from __future__ import annotations
import argparse
import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

import aiohttp
from prometheus_client.parser import text_string_to_metric_families
from latest_v0320.common import DATA,ROOT,PYTHON,MODEL,KV_BYTES,dump,digest,percentiles,clean_env,workload,prompt

def http(port,path,payload=None):
    data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,
        headers={'Content-Type':'application/json'},method='POST' if payload is not None else 'GET')
    with urllib.request.urlopen(req,timeout=120) as r:return r.read().decode()

def parse_metrics(text):
    return [{'name':s.name,'labels':s.labels,'value':s.value}
            for family in text_string_to_metric_families(text) for s in family.samples]

def metric(rows,name,labels=None):
    return sum(r['value'] for r in rows if r['name']==name and all(r['labels'].get(k)==v for k,v in (labels or {}).items()))

def delta_metrics(before,after):
    def count(name,labels=None):return metric(after,name,labels)-metric(before,name,labels)
    names=sorted({r['name'] for r in after})
    def find(stem):
        found=[n for n in names if n==stem or n==stem+'_total']
        return found[0] if found else None
    result={}
    for key,stem,labels in [
        ('gpu_cached_tokens','vllm:prompt_tokens_cached_by_source',{'source':'device'}),
        ('host_cached_tokens','vllm:prompt_tokens_cached_by_source',{'source':'host'}),
        ('h2d_bytes','vllm:kv_offload_total_bytes',{'transfer_type':'CPU_to_GPU'}),
        ('preemptions','vllm:num_preemptions',{}),
        ('prompt_tokens','vllm:prompt_tokens',{}),
        ('generation_tokens','vllm:generation_tokens',{})]:
        name=find(stem);result[key]=count(name,labels) if name else None
    return result

async def measure(port,trace,closed_loop=False):
    start=time.perf_counter(); sem=asyncio.Semaphore(1 if closed_loop else max(128,len(trace)))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900),
            connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one(row):
            due=start+row['offset'];await asyncio.sleep(max(0,due-time.perf_counter()))
            async with sem:
                submitted=time.perf_counter();first=None;stamps=[];text=[];tokens=[];usage=None
                record={'id':row['id'],'class':row['class'],'planned_offset_s':row['offset'],
                    'submit_offset_s':submitted-start,'submission_lag_ms':0 if closed_loop else (submitted-due)*1000}
                payload={'model':'qwen3-kv','prompt':row['tokens'],'max_tokens':row['output'],
                    'temperature':0,'ignore_eos':True,'stream':True,'return_token_ids':True,
                    'request_id':'measure-'+row['id'],'stream_options':{'include_usage':True}}
                try:
                    async with session.post(f'http://127.0.0.1:{port}/v1/completions',json=payload) as response:
                        record['status']=response.status
                        if response.status!=200:raise RuntimeError(await response.text())
                        async for line in response.content:
                            if not line.startswith(b'data: '):continue
                            raw=line[6:].strip()
                            if raw==b'[DONE]':break
                            chunk=json.loads(raw)
                            if chunk.get('error'):raise RuntimeError(str(chunk['error']))
                            if chunk.get('usage'):usage=chunk['usage']
                            for choice in chunk.get('choices',[]):
                                content=choice.get('text','')
                                tokens.extend(choice.get('token_ids') or [])
                                if content:
                                    now=time.perf_counter()
                                    if first is None:first=now
                                    stamps.append(now);text.append(content)
                    ended=time.perf_counter()
                    if first is None or usage is None:raise RuntimeError('missing content or usage')
                    if usage['completion_tokens']!=row['output']:raise RuntimeError('unexpected output token count')
                    record.update(ttft_ms=(first-submitted)*1000,e2e_ms=(ended-submitted)*1000,
                        output_tokens=usage['completion_tokens'],usage=usage,token_ids=tokens,
                        first_content_offset_s=first-start,end_offset_s=ended-start,
                        inter_chunk_ms=[(b-a)*1000 for a,b in zip(stamps,stamps[1:])],text=''.join(text),success=True)
                except Exception as exc:
                    record.update(success=False,error=repr(exc),end_offset_s=time.perf_counter()-start)
                return record
        rows=await asyncio.gather(*(one(row) for row in trace))
    wall=time.perf_counter()-start;ok=[r for r in rows if r['success']]
    result={'requests':len(rows),'completed':len(ok),'errors':len(rows)-len(ok),'completion_rate':len(ok)/len(rows),
        'wall_seconds':wall,'output_tokens':sum(r['output_tokens'] for r in ok),
        'output_tok_s':sum(r['output_tokens'] for r in ok)/wall,
        'ttft_ms':percentiles([r['ttft_ms'] for r in ok]),'e2e_ms':percentiles([r['e2e_ms'] for r in ok]),
        'submission_lag_ms':percentiles([r['submission_lag_ms'] for r in rows]),
        'by_class':{c:percentiles([r['ttft_ms'] for r in ok if r['class']==c]) for c in {r['class'] for r in rows}}}
    return rows,result

async def cancel_requests(port, cpu=False):
    """Disconnect during prefill/offload and ensure a later request can finish."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async def issue(i):
            ids=prompt(2000+i%4,2048)+prompt(81234+i,256) if cpu else prompt(81888+i,4096)
            payload={'model':'qwen3-kv','prompt':ids,'max_tokens':256,'temperature':0,
                'ignore_eos':True,'stream':True,'request_id':f'cancel-{i}'}
            async with session.post(f'http://127.0.0.1:{port}/v1/completions',json=payload) as response:
                async for _ in response.content:pass
        tasks=[asyncio.create_task(issue(i)) for i in range(8)]
        for i in range(4):
            await asyncio.sleep(.02)
            for task in tasks[2*i:2*i+2]:task.cancel()
        results=await asyncio.gather(*tasks,return_exceptions=True)
        return {'attempted':8,'cancelled':sum(isinstance(x,asyncio.CancelledError) for x in results)}

def stop(child):
    if child.poll() is None:
        # Graceful parent shutdown lets the scheduler export its diagnostics.
        child.send_signal(signal.SIGTERM)
        try:child.wait(timeout=25)
        except subprocess.TimeoutExpired:pass
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait()

def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    trace=workload(args.workload,args.requests,args.seed,args.rate)
    if args.correctness:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        questions=['What is 7 plus 5? Answer briefly.','请用一句话介绍 KV Cache 的作用。',
            'What is the capital of France?','写出数字 1 到 5。','Explain FIFO in one sentence.','计算 9 乘以 8。']
        for i,q in enumerate(questions):
            encoded=tok.apply_chat_template([{'role':'user','content':q}],
                tokenize=True,add_generation_prompt=True,enable_thinking=False)
            trace[i]['tokens']=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
            trace[i]['class']='natural';trace[i]['question']=q
        for row in trace:row['output']=16
    dump(dest/'trace.json',trace)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    candidate=args.policy!='N0'
    env=clean_env(candidate);env.update(VLLM_SERVER_DEV_MODE='1',VLLM_LOG_STATS_INTERVAL='1')
    cfg={'admission':{'NG':'gpu','NT':'tier','Nobs':'observe','Noff':'off'}.get(args.policy,'off'),
        'window':args.window,'threshold':args.threshold,'max_wait_ms':2000,
        'output':str(dest/'scheduler.json'),'cost_model':str(DATA/'cost_models/native.json')}
    env['KV032_POLICY']=json.dumps(cfg)
    cmd=[str(PYTHON),'-m','vllm.entrypoints.openai.api_server','--model',MODEL,
        '--served-model-name','qwen3-kv','--host','127.0.0.1','--port',str(port),
        '--dtype','float16','--max-model-len','8192','--block-size','16',
        '--num-gpu-blocks-override',str(args.blocks),'--max-num-batched-tokens','1024',
        '--max-num-seqs','16','--gpu-memory-utilization','0.75',
        '--enable-prefix-caching','--enable-chunked-prefill','--generation-config','vllm',
        '--disable-uvicorn-access-log']
    if args.track=='eager':cmd+=['--enforce-eager','--no-async-scheduling']
    if args.workload=='C':cmd+=['--kv-transfer-config',json.dumps({
        'kv_connector':'OffloadingConnector','kv_role':'kv_both','kv_connector_extra_config':{
            'cpu_bytes_to_use':32768*KV_BYTES,'block_size':256,'eviction_policy':'lru'}})]
    if candidate:
        name='SyncAdmissionScheduler' if args.track=='eager' else 'AsyncAdmissionScheduler'
        cmd+=['--scheduler-cls','latest_v0320.scheduler.'+name]
    manifest={'args':vars(args),'command':cmd,'trace_sha256':digest(dest/'trace.json'),
        'environment':json.loads((DATA/'manifests/environment.json').read_text()),
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'latest_v0320').glob('*.py'))},
        'cost_model_sha256':digest(DATA/'cost_models/native.json') if (DATA/'cost_models/native.json').exists() else None,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'started_unix':time.time(),
        'baseline_unmodified':not candidate,'f_concurrency':1 if args.workload=='F' else None}
    dump(dest/'manifest.json',manifest)
    child=None;metrics=None
    with (dest/'server.log').open('w') as log:
        try:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+600
            while True:
                if child.poll() is not None:raise RuntimeError('server exited: '+str(dest/'server.log'))
                try:http(port,'/health');break
                except Exception:
                    if time.monotonic()>deadline:raise TimeoutError('server startup timeout')
                    time.sleep(.4)
            manifest['startup_seconds']=time.time()-manifest['started_unix']
            server_info=json.loads(http(port,'/server_info?config_format=json'))
            dump(dest/'resolved_config.json',server_info)
            resolved=server_info['vllm_config']
            assert resolved['scheduler_config']['scheduler_reserve_full_isl'] is True
            assert resolved['scheduler_config']['watermark']==0
            assert resolved['scheduler_config']['async_scheduling']==(args.track=='default')
            def generate(ids,rid,n=1):
                obj=json.loads(http(port,'/v1/completions',{'model':'qwen3-kv','prompt':ids,
                    'max_tokens':n,'ignore_eos':True,'temperature':0,'return_token_ids':True,'request_id':rid}))
                assert obj['usage']['completion_tokens']==n
                return obj
            warmup=[]
            for length in [512,2048,4096,6144]:
                warmup.append(generate(prompt(99000+length,length),'warm-'+str(length),2))
            # Native 0.32.0 explicitly resets both GPU and connector caches.
            deadline=time.monotonic()+30
            while not json.loads(http(port,'/reset_prefix_cache?reset_external=true',{}))['success']:
                if time.monotonic()>deadline:raise RuntimeError('native two-level reset failed')
                time.sleep(.05)
            prepare=[]
            if args.workload=='C':
                for group in range(4):prepare.append(generate(prompt(2000+group,2048),'cpu-seed-'+str(group)))
                for group in range(4):prepare.append(generate(prompt(7000+group,4096),'evict-'+str(group)))
            if args.workload in {'A','C'}:
                for group in range(4):prepare.append(generate(prompt(1000+group,2048),'gpu-seed-'+str(group)))
            dump(dest/'preparation.json',{'warmup':warmup,'prepare':prepare,'native_reset_external':True})
            before_text=http(port,'/metrics');before=parse_metrics(before_text)
            (dest/'metrics-before.txt').write_text(before_text)
            rows,metrics=asyncio.run(measure(port,trace,args.workload=='F'))
            dump(dest/'requests.json',rows)
            time.sleep(.2)
            after_text=http(port,'/metrics');after=parse_metrics(after_text)
            (dest/'metrics-after.txt').write_text(after_text)
            metrics['native_delta']=delta_metrics(before,after)
            if metrics['errors']:raise RuntimeError(f"{metrics['errors']} failed requests")
            deadline=time.monotonic()+30
            while not json.loads(http(port,'/reset_prefix_cache?reset_external=true',{}))['success']:
                if time.monotonic()>deadline:raise RuntimeError('post-measurement native reset did not drain')
                time.sleep(.05)
            if args.correctness:
                if args.workload=='C':
                    for group in range(4):generate(prompt(2000+group,2048),'fault-seed-'+str(group))
                    for group in range(4):generate(prompt(7300+group,4096),'fault-evict-'+str(group))
                fault_before=parse_metrics(http(port,'/metrics'))
                fault=asyncio.run(cancel_requests(port,args.workload=='C'))
                proof=generate(prompt(100033,128),'after-cancel',8)
                deadline=time.monotonic()+20
                while True:
                    state=parse_metrics(http(port,'/metrics'))
                    active=metric(state,'vllm:num_requests_running')+metric(state,'vllm:num_requests_waiting')
                    if active==0:break
                    if time.monotonic()>deadline:raise RuntimeError('cancelled requests did not drain')
                    time.sleep(.1)
                fault['native_delta']=delta_metrics(fault_before,state)
                deadline=time.monotonic()+30
                while not json.loads(http(port,'/reset_prefix_cache?reset_external=true',{}))['success']:
                    if time.monotonic()>deadline:raise RuntimeError('fault recovery left protected KV/jobs')
                    time.sleep(.05)
                fault['native_reset_after_fault']=True
                fault['followup_output_tokens']=proof['usage']['completion_tokens']
                fault['active_after_drain']=active;dump(dest/'cancel-check.json',fault)
        finally:
            if child is not None:stop(child)
            manifest['ended_unix']=time.time();dump(dest/'manifest.json',manifest)
    if candidate:
        if not (dest/'scheduler.json').exists():raise RuntimeError('missing scheduler diagnostics')
        observation=json.loads((dest/'scheduler.json').read_text())
        metrics['project_counters']=observation['counters']
    dump(dest/'metrics.json',metrics)
    print('RUN_COMPLETE',args.policy,args.workload,args.seed,json.dumps(metrics),flush=True)
    return metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--policy',choices=['N0','Nobs','NG','NT','Noff'],default='N0')
    ap.add_argument('--workload',choices=['A','C','E','F'],default='A')
    ap.add_argument('--track',choices=['eager','default'],default='eager')
    ap.add_argument('--requests',type=int,default=64)
    ap.add_argument('--seed',type=int,default=811)
    ap.add_argument('--blocks',type=int,default=640)
    ap.add_argument('--rate',type=float,default=3.2)
    ap.add_argument('--window',type=int,default=8)
    ap.add_argument('--threshold',type=float,default=0.)
    ap.add_argument('--correctness',action='store_true')
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:8])
    run(args)

if __name__=='__main__':main()
