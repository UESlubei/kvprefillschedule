"""Human-readable Chinese/English/code/math cache-recovery smoke test."""
import argparse
import json
import os
from pathlib import Path
import time
from dataclasses import asdict
from run_trace import DATA,MODEL,prompt
from kv_sched.policy import Config


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['reference','baseline','combined'],required=True)
    ap.add_argument('--label',default='v3_text_quality')
    args=ap.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    background='背景材料：推理服务把请求组织为批次，使用缓存复用共同上下文，并记录延迟、吞吐与资源占用。请直接回答最后的用户问题。'
    questions=[
       ['只输出 4+7 的结果。','只输出 8×9 的结果。','只输出 100-37 的结果。','只输出 144 除以 12 的结果。'],
       ['用 Python 写一个 add(a, b) 函数，返回两数之和。','Python 的 len([1,2,3]) 返回什么？','用一行 Python 把 [1,2,3] 中每个数乘以 2。','Python 中 dict 用于保存什么？一句话回答。'],
       ['把“缓存命中减少重复计算”翻译成英文。','用中文一句话解释先进先出。','What is the capital of France? Answer briefly.','用一句话说明吞吐和延迟的区别。']]
    rows=[]
    for group,qs in enumerate(questions):
        system=f'主题编号 {group}。'+background*32
        for i,q in enumerate(qs):
            ids=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':q}],
                                        tokenize=True,add_generation_prompt=True,enable_thinking=False)
            assert len(ids)+64<=8192
            rows.append({'id':f'text-{group}-{i}','group':group,'question':q,'tokens':ids})
    cfg=Config(admission='tier' if args.mode=='combined' else 'fcfs',
               preemption='restore' if args.mode=='combined' else 'default',
               cost_model=str(DATA/'cost_models/qwen3-1.7b-3090.json'))
    os.environ['KV_SCHED_CONFIG']=json.dumps(asdict(cfg))
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.config import KVTransferConfig
    llm=LLM(model=MODEL,dtype='float16',enforce_eager=True,max_model_len=8192,
            num_gpu_blocks_override=4096 if args.mode=='reference' else 1024,block_size=16,
            gpu_memory_utilization=.8,max_num_batched_tokens=1024,max_num_seqs=16,
            enable_prefix_caching=True,enable_chunked_prefill=True,async_scheduling=False,
            disable_log_stats=True,kv_transfer_config=KVTransferConfig(kv_connector='OffloadingConnector',kv_role='kv_both',
              kv_connector_extra_config={'num_cpu_blocks':128,'block_size':256,'eviction_policy':'lru'}) if args.mode!='reference' else None)
    eng=llm.llm_engine;s=eng.engine_core.engine_core.scheduler
    def add(rid,ids,n):eng.add_request(rid,TokensPrompt(prompt_token_ids=ids),SamplingParams(temperature=0,max_tokens=n,ignore_eos=False))
    def drain():
        outputs=[]
        deadline=time.perf_counter()+30
        while eng.has_unfinished_requests() or s.has_requests():
            outputs+=eng.step()
            if time.perf_counter()>deadline:
                from vllm.distributed.kv_transfer import get_kv_transfer_group
                cs=s.connector.connector_scheduler if s.connector else None
                cw=get_kv_transfer_group().connector_worker if s.connector else None
                state={'requests':{rid:{'status':r.status.name,'computed':r.num_computed_tokens,'tokens':r.num_tokens} for rid,r in s.requests.items()},
                       'finished_transfer_ids':list(s._finished_kv_transfer_req_ids),
                       'load':{k:len(v) for k,v in cs._reqs_being_loaded.items()} if cs else None,
                       'store':{k:len(v) for k,v in cs._reqs_being_stored.items()} if cs else None,
                       'worker_jobs':cw._jobs if cw else None,
                       'worker_store_jobs':{k:list(v) for k,v in cw._store_jobs.items()} if cw else None,
                       'worker_load_job':cw._load_job if cw else None,
                       'waiting_store_ack':list(cw._finished_reqs_waiting_for_store) if cw else None,
                       'unsubmitted':cw._unsubmitted_store_jobs if cw else None,
                       'counters':dict(s._kv.counters)}
                print('TEXT_DRAIN_TIMEOUT',json.dumps(state,default=str),flush=True)
                (DATA/'raw'/f'text_timeout_{time.time_ns()}.json').write_text(json.dumps(state,default=str,indent=2))
                raise TimeoutError('natural text drain; diagnostic state logged')
        return outputs
    def reset():
        assert not s.requests and not s.running and not s.waiting
        assert s.kv_cache_manager.reset_prefix_cache()
        if s.connector:
            cs=s.connector.connector_scheduler
            assert not cs._reqs_being_loaded and not cs._reqs_being_stored
            m=cs.manager
            assert all(b.ref_cnt==0 for b in m.blocks.values())
            for b in m.blocks.values():m.backend.free(b)
            m.blocks.clear()
    for length in [512,4096]:add(f'warm-{length}',prompt(90000+length,length),2);drain()
    single=[]
    for row in rows:
        reset()
        rid='single-'+row['id'];add(rid,row['tokens'],64)
        done={o.request_id:o for o in drain() if o.finished}
        o=done[rid].outputs[0]
        single.append({'id':row['id'],'tokens':list(o.token_ids),'answer':o.text})
    reset()
    for group in range(3):add(f'seed-{group}',rows[group*4]['tokens'],1);drain()
    for group in range(4):add(f'evict-{group}',prompt(70000+group,4096),1);drain()
    s._kv.reset()
    for r in rows:add(r['id'],r['tokens'],64)
    for i in range(4):add(f'pressure-{i}',prompt(88000+i,4096),16)
    output=drain()
    completed={o.request_id:o for o in output if o.finished}
    results=[]
    for r in rows:
        o=completed[r['id']].outputs[0]
        results.append({'id':r['id'],'question':r['question'],'input_tokens':len(r['tokens']),
                        'tokens':list(o.token_ids),'answer':o.text})
    out=DATA/'raw'/args.label;out.mkdir(exist_ok=True)
    payload={'mode':args.mode,'single':single,'requests':results,'scheduler':s._kv.export()}
    (out/f'{args.mode}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(json.dumps(payload,ensure_ascii=False),flush=True)
    eng.engine_core.shutdown()


if __name__=='__main__':main()
