"""Run exact-token, open-loop traces through vLLM's synchronous V1 engine.

Times are local engine/client times, not HTTP serving latency. Independent process
per invocation; all trial repetitions get verified GPU/CPU cache reset at idle.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from kv_sched.policy import Config, CostModel, percentiles

DATA = Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
MODEL = '/mnt/ondeviceinfer/models/Qwen3-1.7B'


def prompt(seed, length):
    rng = random.Random(seed)
    return [rng.randrange(100, 30000) for _ in range(length)]


def workload(kind, count, seed, output=64, prefix_length=2048, suffix_length=256,
             sessions=16, output_relation='independent'):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        group = rng.randrange(4)
        if kind == 'A':
            warm = i%3 != 0
            ids = (prompt(1000+group, prefix_length)+prompt(seed*10000+i,suffix_length)) if warm else prompt(seed*10000+i,prefix_length+suffix_length)
            label = 'gpu_warm' if warm else 'cold'
        elif kind in {'C','D'}:
            label = ['gpu_warm']*3+['cpu_warm']*3+['cold']*4
            label = label[i%10] if kind=='C' else ['gpu_warm','cpu_warm','cold'][i%3]
            base = 1000+group if label=='gpu_warm' else 2000+group
            ids = (prompt(base,2048)+prompt(seed*10000+i,256)) if label!='cold' else prompt(seed*10000+i,2304)
            if kind == 'D':
                session=i%sessions
                round_idx=(i//sessions)%4
                cohort=i//(sessions*4)
                group=session%4
                label=['gpu_warm','cpu_warm','cold'][session%3]
                base=1000+group if label=='gpu_warm' else 2000+group
                session_seed=seed*1000+cohort*sessions+session
                prefix=prompt(base,2048) if label!='cold' else prompt(session_seed,2048)
                ids=prefix+prompt(session_seed+500000,(round_idx+1)*256)
        else:
            length = rng.choices([512,2048,4096,6144],[.4,.3,.2,.1])[0] if kind!='F' else 512
            ids, label = prompt(seed*10000+i,length), 'cold'
        nout = rng.choice([64,128,256]) if kind=='B' else output
        if kind=='B' and output_relation=='inverse':
            nout=256 if len(ids)<=512 else 128 if len(ids)<=2048 else 64
        # Bursts expose actual waiting-window decisions; offsets remain fixed.
        offset = i//16 * (2.0 if kind in {'D','F'} else .20)
        row={'id':f'{kind}-{seed}-{i:05d}','tokens':ids,'output':nout,'offset':offset,'class':label,'group':group}
        if kind=='D':
            # A fixed number of causal streams; each starts a fresh conversation
            # after four rounds, only when its previous conversation has ended.
            row.update(session_id=session,conversation_cycle=cohort,round=round_idx,
                       depends_on=f'{kind}-{seed}-{i-sessions:05d}' if i>=sessions else None,tool_wait_s=.1)
        rows.append(row)
    return rows


def stats(rows, wall):
    n = sum(r['output_tokens'] for r in rows)
    itls = [v for r in rows for v in r['itl_ms']]
    return {'completed':len(rows),'output_tokens':n,'wall_seconds':wall,'output_tok_s':n/wall,
            'request_s':len(rows)/wall,'ttft_ms':percentiles([r['ttft_ms'] for r in rows]),
            'e2e_ms':percentiles([r['e2e_ms'] for r in rows]),'itl_ms':percentiles(itls),
            'arrival_lag_ms':percentiles([r['arrival_lag_ms'] for r in rows]),
            'planned_ttft_ms':percentiles([r['ttft_ms']+r['arrival_lag_ms'] for r in rows]),
            'planned_e2e_ms':percentiles([r['e2e_ms']+r['arrival_lag_ms'] for r in rows]),
            'by_class':{c:percentiles([r['ttft_ms'] for r in rows if r['class']==c]) for c in {r['class'] for r in rows}}}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--admission',default='fcfs')
    ap.add_argument('--preemption',default='default')
    ap.add_argument('--workload',default='B',choices=list('ABCDEF'))
    ap.add_argument('--requests',type=int,default=256)
    ap.add_argument('--blocks',type=int,default=1024)
    ap.add_argument('--cpu-blocks',type=int,default=0)
    ap.add_argument('--cpu-block-size',type=int,default=256)
    ap.add_argument('--suite',type=Path)
    ap.add_argument('--seeds',default='17')
    ap.add_argument('--window',type=int,default=8)
    ap.add_argument('--threshold',type=float,default=0)
    ap.add_argument('--aging',type=float,default=.05)
    ap.add_argument('--max-wait-ms',type=float,default=2000)
    ap.add_argument('--prefix-length',type=int,default=2048)
    ap.add_argument('--suffix-length',type=int,default=256)
    ap.add_argument('--sessions',type=int,choices=[8,16],default=16)
    ap.add_argument('--output-relation',choices=['independent','inverse'],default='independent')
    ap.add_argument('--concurrency',type=int,default=0,
                    help='0=open loop; positive=closed-loop in-flight limit')
    ap.add_argument('--arrival-rate',type=float,default=0,
                    help='positive=open-loop requests/s in fixed 16-request bursts; 0=legacy offsets')
    ap.add_argument('--no-cpu-probe',action='store_true')
    ap.add_argument('--no-fairness',action='store_true')
    ap.add_argument('--no-resource-aware',action='store_true')
    ap.add_argument('--no-progress-aware',action='store_true')
    ap.add_argument('--cost-model')
    ap.add_argument('--trace',action='store_true')
    ap.add_argument('--stock',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=600)
    args=ap.parse_args()
    cfg=Config(admission=args.admission,preemption=args.preemption,window=args.window,
               threshold=args.threshold,aging=args.aging,max_wait_ms=args.max_wait_ms,
               cost_model=args.cost_model,trace=args.trace,probe_cpu=not args.no_cpu_probe,
               fairness=not args.no_fairness,resource_aware=not args.no_resource_aware,
               progress_aware=not args.no_progress_aware)
    if args.concurrency<0 or args.requests<1: raise ValueError('invalid request/concurrency count')
    os.environ['KV_SCHED_CONFIG']=json.dumps(asdict(cfg))
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt
    args.output.mkdir(parents=True,exist_ok=True)
    init=time.perf_counter()
    llm=LLM(model=MODEL,dtype='float16',kv_cache_dtype='auto',tensor_parallel_size=1,
            gpu_memory_utilization=.8,enforce_eager=True,max_model_len=8192,
            block_size=16,num_gpu_blocks_override=args.blocks,max_num_batched_tokens=1024,
            max_num_seqs=16,enable_prefix_caching=True,enable_chunked_prefill=True,
            async_scheduling=False,disable_log_stats=True,
            kv_transfer_config=KVTransferConfig(kv_connector='OffloadingConnector',kv_role='kv_both',
                 kv_connector_extra_config={'num_cpu_blocks':args.cpu_blocks,'block_size':args.cpu_block_size,'eviction_policy':'lru'}) if args.cpu_blocks else None)
    engine=llm.llm_engine
    core=engine.engine_core.engine_core
    sched=core.scheduler
    hooks=getattr(sched,'_kv',None)
    print('ENGINE_READY',time.perf_counter()-init,flush=True)
    def generate(ids,rid,n=1):
        engine.add_request(rid,TokensPrompt(prompt_token_ids=ids),SamplingParams(temperature=0,max_tokens=n,ignore_eos=True))
        while engine.has_unfinished_requests() or sched.has_requests():
            engine.step()
    def reset():
        # Drain connector acknowledgements before touching either cache.
        deadline=time.perf_counter()+30
        while sched.has_requests():
            engine.step()
            if time.perf_counter()>deadline: raise RuntimeError('idle drain timeout')
        assert not sched.running and not sched.waiting and not sched.requests
        assert sched.kv_cache_manager.reset_prefix_cache()
        if sched.connector:
            cs=sched.connector.connector_scheduler
            assert not cs._reqs_being_stored and not cs._reqs_being_loaded
            assert not cs._requests and not cs._reqs_to_load
            assert all(b.ref_cnt==0 for b in cs.manager.blocks.values())
            for b in list(cs.manager.blocks.values()): cs.manager.backend.free(b)
            cs.manager.blocks.clear()
            assert cs.manager.backend.get_num_free_blocks()==args.cpu_blocks
        if hooks: hooks.reset()
    # Shape warmup unrelated to measured prefix groups.
    for length in [512,2048,4096,6144]: generate(prompt(99000+length,length),f'warm-{length}',2)
    reset()
    seeds=[int(x) for x in args.seeds.split(',')]
    jobs=json.loads(args.suite.read_text()) if args.suite else [{'seed':s} for s in seeds]
    manifest={'args':vars(args)|{'output':str(args.output)},'config':asdict(cfg),
              'vllm_file':vllm.__file__,'vllm_version':vllm.__version__,
              'python':platform.python_version(),'torch':torch.__version__,
              'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
              'model_revision':Path(MODEL).resolve().name,'init_seconds':time.perf_counter()-init,
              'cpu_affinity':sorted(os.sched_getaffinity(0)),
              'measurement':'local synchronous V1 engine; A/B/C/E/F open-loop, D causal sessions with fixed histories; submission lag measured; not HTTP',
              'engine_config':repr(engine.vllm_config),
              'source_manifest':json.loads((DATA/'environment_manifest.json').read_text())}
    release=DATA/'v3_full_source_snapshot.json'
    if release.exists():manifest['release_snapshot']=json.loads(release.read_text())
    root=Path(__file__).resolve().parents[1]
    manifest['project_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [*sorted((root/'kv_sched').glob('*.py')),Path(__file__).resolve()]}
    if args.cost_model:
        manifest['cost_model_sha256']=hashlib.sha256(Path(args.cost_model).read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    for job in jobs:
        seed=job['seed']
        args.workload=job.get('workload',args.workload)
        cfg=Config(admission=job.get('admission',args.admission),preemption=job.get('preemption',args.preemption),
                   window=job.get('window',args.window),threshold=job.get('threshold',args.threshold),
                   aging=job.get('aging',args.aging),max_wait_ms=job.get('max_wait_ms',args.max_wait_ms),
                   cost_model=args.cost_model,trace=args.trace,
                   probe_cpu=job.get('probe_cpu',not args.no_cpu_probe),
                   fairness=job.get('fairness',not args.no_fairness),
                   resource_aware=job.get('resource_aware',not args.no_resource_aware),
                   progress_aware=job.get('progress_aware',not args.no_progress_aware))
        if hooks:
            hooks.config=cfg
            hooks.model=CostModel(cfg.cost_model)
        reset()
        prepare=[]
        if args.workload in {'A','C','D'}:
            # CPU-seed first, then overwrite GPU with unrelated cold requests.
            if args.workload in {'C','D'}:
                for g in range(4): generate(prompt(2000+g,2048),f'cpu-seed-{g}')
                for g in range(4): generate(prompt(7000+g,4096),f'evict-{g}')
            prefix_length=job.get('prefix_length',args.prefix_length) if args.workload=='A' else 2048
            for g in range(4): generate(prompt(1000+g,prefix_length),f'gpu-seed-{g}')
            from kv_sched.runtime import gpu_probe,cpu_probe
            from vllm.v1.request import Request
            for prefix in [1000,2000] if args.workload in {'C','D'} else [1000]:
                # Build real request hashes via the native engine preprocessing path.
                rid=f'check-{prefix}'
                check_length=prefix_length if prefix==1000 else 2048
                engine.add_request(rid,TokensPrompt(prompt_token_ids=prompt(prefix,check_length)+prompt(321,256)),
                                   SamplingParams(temperature=0,max_tokens=1,ignore_eos=True))
                r=sched.requests[rid]
                gp=gpu_probe(sched.kv_cache_manager,r)
                cp,nb,why=cpu_probe(sched.connector,r,gp)
                prepare.append({'prefix':prefix,'gpu':gp,'cpu_end':cp,'bytes':nb,'reason':why})
                engine.abort_request([rid])
                assert rid not in sched.requests, 'cache validation request must be aborted without execution'
            while sched.has_requests(): engine.step()
            if hooks: hooks.reset()
        trace=workload(args.workload,args.requests,seed,
                       prefix_length=job.get('prefix_length',args.prefix_length),
                       suffix_length=job.get('suffix_length',args.suffix_length),
                       sessions=job.get('sessions',args.sessions),
                       output_relation=job.get('output_relation',args.output_relation))
        concurrency=job.get('concurrency',args.concurrency)
        rate=job.get('arrival_rate',args.arrival_rate)
        if rate<0:raise ValueError('negative arrival rate')
        if rate:
            for k,row in enumerate(trace):row['offset']=k//16*16/rate
        if args.workload=='F' and not concurrency: concurrency=4
        if concurrency:
            for row in trace: row['offset']=0.0
        trace_hash=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest()
        run=(args.output/job['id']/f'seed-{seed}') if 'id' in job else args.output/f'seed-{seed}'
        run.mkdir(parents=True,exist_ok=True)
        (run/'trace.json').write_text(json.dumps(trace))
        (run/'manifest.json').write_text(json.dumps({**manifest,'config':asdict(cfg),'job':job,
            'effective_concurrency':concurrency,'arrival_mode':'closed_loop' if concurrency else 'open_loop',
            'arrival_rate':rate,'trace_hash':trace_hash},indent=2,default=str))
        rows,active={},{}
        start=time.perf_counter()
        i=0
        queued=list(range(len(trace)))
        completion_times={}
        steps=0
        while i<len(trace) or engine.has_unfinished_requests() or sched.has_requests():
            now=time.perf_counter()
            if now-start>args.timeout: raise TimeoutError(f'trace timeout seed={seed}, completed={len(rows)}')
            ready=[j for j in queued if trace[j]['offset']<=now-start and
                   (not trace[j].get('depends_on') or
                    trace[j]['depends_on'] in completion_times and
                    now-completion_times[trace[j]['depends_on']]>=trace[j]['tool_wait_s'])]
            if concurrency:
                ready=ready[:max(0,concurrency-(i-len(rows)))]
            for j in ready:
                row=trace[j]
                rid=row['id']
                ts=time.perf_counter()
                active[rid]={'submit':ts,'submission_epoch_s':time.time(),
                             'last':None,'first':None,'times':[], 'row':row}
                engine.add_request(rid,TokensPrompt(prompt_token_ids=row['tokens']),
                                   SamplingParams(temperature=0,max_tokens=row['output'],ignore_eos=True))
                active[rid]['engine_arrival_epoch_s']=sched.requests[rid].arrival_time
                i+=1
            if ready:
                submitted=set(ready)
                queued=[j for j in queued if j not in submitted]
            if engine.has_unfinished_requests() or sched.has_requests():
                outputs=engine.step()
                steps+=1
                stamp=time.perf_counter()
                for o in outputs:
                    if o.request_id not in active: continue
                    a=active[o.request_id]
                    n=len(o.outputs[0].token_ids)
                    previous=len(a['times'])
                    if n>previous:
                        a['times'].extend([stamp]*(n-previous))
                        if a['first'] is None: a['first']=stamp
                    if o.finished:
                        ts=a['times']
                        r=a['row']
                        admitted=hooks.admissions.get(o.request_id,{}) if hooks else {}
                        actual_class=('cpu_warm' if admitted.get('cpu',0)>0 else 'gpu_warm' if admitted.get('gpu',0)>0 else 'cold')
                        rows[o.request_id]={'id':o.request_id,'class':actual_class,'intended_class':r['class'],'output_tokens':n,
                           'planned_due_s':max(r['offset'],(completion_times.get(r.get('depends_on'),start)-start+r.get('tool_wait_s',0)) if r.get('depends_on') else r['offset']),
                           'submitted_s':a['submit']-start,'first_token_s':ts[0]-start,'completed_s':stamp-start,
                           'submission_epoch_s':a['submission_epoch_s'],'engine_arrival_epoch_s':a['engine_arrival_epoch_s'],
                           'ttft_ms':(ts[0]-a['submit'])*1000,'e2e_ms':(stamp-a['submit'])*1000,
                           'arrival_lag_ms':(a['submit']-max(start+r['offset'],
                               completion_times.get(r.get('depends_on'),start)+r.get('tool_wait_s',0) if r.get('depends_on') else start+r['offset']))*1000,
                           'itl_ms':[(b-a)*1000 for a,b in zip(ts,ts[1:])],
                           'tokens':list(o.outputs[0].token_ids)}
                        completion_times[o.request_id]=stamp
            else:
                next_due=[max(start+trace[j]['offset'],completion_times.get(trace[j].get('depends_on'),start)+trace[j].get('tool_wait_s',0))
                          for j in queued if not trace[j].get('depends_on') or trace[j]['depends_on'] in completion_times]
                if not next_due: raise RuntimeError('unsatisfied/cyclic session dependency')
                time.sleep(min(.001,max(0,min(next_due)-now)))
        wall=time.perf_counter()-start
        result={'seed':seed,'group':job.get('id'),'workload':args.workload,'gpu_blocks':args.blocks,
                'cpu_blocks':args.cpu_blocks,'cpu_block_size':args.cpu_block_size,
                'config':asdict(cfg),'trace_hash':trace_hash,'cache_preparation':prepare,
                'metrics':stats(list(rows.values()),wall),'scheduler':hooks.export() if hooks else None,
                'steps':steps,'gpu_peak_allocated':torch.cuda.max_memory_allocated(),
                'arrival_mode':'closed_loop' if concurrency else 'open_loop','concurrency':concurrency}
        result['arrival_rate']=rate
        (run/'requests.json').write_text(json.dumps(list(rows.values())))
        (run/'metrics.json').write_text(json.dumps(result,indent=2))
        print('RESULT',str(run),json.dumps(result['metrics']),flush=True)
    engine.engine_core.shutdown()


if __name__=='__main__': main()
