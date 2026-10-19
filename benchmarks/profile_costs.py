"""Measured prefill and native block-transfer cost calibration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from run_trace import prompt, DATA, MODEL


def transfers(torch):
    import numpy as np
    from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandler
    from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
    class Backend:
        @staticmethod
        def get_kv_cache_shape(num_blocks,block_size,num_kv_heads,head_size):
            return (2,num_blocks,block_size,num_kv_heads,head_size)
    caches={str(i):torch.zeros((2,512,16,8,128),device='cuda',dtype=torch.float16) for i in range(28)}
    h=CpuGpuOffloadingHandler(gpu_block_size=16,cpu_block_size=256,num_cpu_blocks=32,
                              gpu_caches=caches,attn_backends={str(i):Backend for i in range(28)})
    rows=[]; jid=0
    for chunks in [1,2,4,8,16,32]:
        spec=(CPULoadStoreSpec(list(range(chunks))),GPULoadStoreSpec(list(range(chunks*16))))
        samples=[]
        for rep in range(13):
            a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
            a.record(h.h2d_stream)
            ts=time.perf_counter()
            h.transfer_async(jid,spec)
            b.record(h.h2d_stream)
            submit=(time.perf_counter()-ts)*1000
            b.synchronize()
            wall=(time.perf_counter()-ts)*1000
            assert h.get_finished()==[(jid,True)]
            jid+=1
            if rep>=3: samples.append({'cuda_ms':a.elapsed_time(b),'job_ms':wall,'submit_ms':submit})
        rows.append({'chunks':chunks,'bytes':chunks*256*114688,'samples':samples,
                     'job_ms':statistics.median(s['job_ms'] for s in samples)})
    x=np.array([[1,r['bytes']] for r in rows]); y=np.array([r['job_ms'] for r in rows])
    fixed,slope=np.linalg.lstsq(x,y,rcond=None)[0]
    fit={'fixed_ms':max(0,float(fixed)),'bytes_per_ms':1/float(slope),'per_block_ms':0}
    continuous=[]
    for mib in [1,4,16,64,128,256]:
        src=torch.zeros(mib*2**20,dtype=torch.uint8,pin_memory=True)
        dst=torch.empty_like(src,device='cuda')
        vals=[]
        for rep in range(13):
            a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
            a.record(); dst.copy_(src,non_blocking=True); b.record(); b.synchronize()
            if rep>=3: vals.append(a.elapsed_time(b))
        continuous.append({'bytes':mib*2**20,'cuda_ms':statistics.median(vals)})
    return fit,{'native_handler':rows,'continuous_copy':continuous,
                'pinned':all(t.is_pinned() for t in h.cpu_tensors),'layout':'28 layers, FP16, 8 KV heads, dim 128'}


def prefill(torch):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm=LLM(model=MODEL,dtype='float16',enforce_eager=True,max_model_len=8192,
            gpu_memory_utilization=.8,num_gpu_blocks_override=4096,block_size=16,
            max_num_batched_tokens=1024,max_num_seqs=16,enable_prefix_caching=True,
            enable_chunked_prefill=True,async_scheduling=False,disable_log_stats=True)
    eng=llm.llm_engine; core=eng.engine_core.engine_core; sched=core.scheduler
    events=[]
    original=core.model_executor.execute_model
    def execute(output,**kwargs):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record()
        future=original(output,**kwargs)
        b.record()
        events.append((a,b,output.total_num_scheduled_tokens))
        return future
    core.model_executor.execute_model=execute
    serial=0
    def generate(ids):
        nonlocal serial
        for tokens in ids:
            serial+=1
            eng.add_request(f'profile-{serial}',TokensPrompt(prompt_token_ids=tokens),
                            SamplingParams(temperature=0,max_tokens=1,ignore_eos=True))
        while sched.has_requests() or eng.has_unfinished_requests(): eng.step()
        torch.cuda.synchronize()
    rows=[]
    grid=[(l,int(l*f)//16*16,b) for b in [1,4] for l in [512,1024,2048,4096,6144] for f in [0,.25,.5,.75]]
    validation=[(3072,768,1),(3072,1536,1),(5120,1280,1),(5120,2560,1)]
    for l,c,batch in grid+validation:
        values=[]
        for rep in range(12):
            assert sched.kv_cache_manager.reset_prefix_cache()
            ids=[prompt(700000+j,l) for j in range(batch)]
            if c: generate([x[:c] for x in ids])
            events.clear()
            ts=time.perf_counter(); generate(ids); wall=(time.perf_counter()-ts)*1000
            cuda=sum(a.elapsed_time(b) for a,b,_ in events)
            if rep>=2: values.append({'cuda_ms':cuda,'wall_ms':wall,'chunks':[n for _,_,n in events]})
        rows.append({'length':l,'cached':c,'batch':batch,'ms':statistics.median(x['cuda_ms'] for x in values),
                     'validation':(l,c,batch) in validation,'samples':values})
        print('PREFILL',l,c,batch,rows[-1]['ms'],flush=True)
    core.shutdown()
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--part',choices=['prefill','transfer','merge'],required=True)
    args=ap.parse_args()
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    out=DATA/'cost_models'; out.mkdir(exist_ok=True)
    if args.part=='merge':
        rows=json.loads((out/'prefill_raw.json').read_text())
        h=json.loads((out/'transfer_raw.json').read_text())
        model={'schema':1,'model_revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
               'vllm_commit':'275de34170654274616082721348b7edd9741d32','gpu':'NVIDIA GeForce RTX 3090',
               'kv_bytes_per_token':114688,'prefill':[ {k:r[k] for k in ['length','cached','batch','ms']}
                   for r in rows if not r['validation']], 'h2d':h['model']}
        path=out/'qwen3-1.7b-3090.json'; path.write_text(json.dumps(model,indent=2))
        from kv_sched.policy import CostModel
        cm=CostModel(str(path)); errors=[]
        for r in rows:
            if r['validation']:
                prediction=cm.prefill(r['length'],r['cached'],r['batch'])
                errors.append({**{k:r[k] for k in ['length','cached','batch','ms']},
                               'predicted_ms':prediction,'absolute_error_ms':abs(prediction-r['ms']),
                               'relative_error':abs(prediction-r['ms'])/max(r['ms'],.1)})
        (out/'validation.json').write_text(json.dumps(errors,indent=2))
        print(path,errors)
        return
    import torch
    torch.set_num_threads(4)
    if args.part=='prefill': (out/'prefill_raw.json').write_text(json.dumps(prefill(torch)))
    else:
        fit,raw=transfers(torch)
        (out/'transfer_raw.json').write_text(json.dumps({'model':fit,**raw},indent=2))
        print('TRANSFER',fit,flush=True)


if __name__=='__main__': main()
