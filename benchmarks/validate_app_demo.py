"""Eight real conversation streams; actual generated answers enter later turns.

Application demonstration only: input histories can diverge between policies,
so this run is excluded from paired performance claims.
"""
import argparse
from dataclasses import asdict
import json
import os
import re
import time

from run_trace import DATA,MODEL,prompt
from kv_sched.policy import Config
from kv_sched.runtime import gpu_probe,cpu_probe


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['baseline','combined'],required=True)
    args=parser.parse_args()
    cfg=Config(admission='tier' if args.mode=='combined' else 'fcfs',
               preemption='restore' if args.mode=='combined' else 'default',
               cost_model=str(DATA/'cost_models/qwen3-1.7b-3090.json'))
    os.environ['KV_SCHED_CONFIG']=json.dumps(asdict(cfg));os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.config import KVTransferConfig
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    llm=LLM(model=MODEL,dtype='float16',enforce_eager=True,max_model_len=8192,
        num_gpu_blocks_override=1024,block_size=16,gpu_memory_utilization=.8,
        max_num_batched_tokens=1024,max_num_seqs=16,enable_prefix_caching=True,
        enable_chunked_prefill=True,async_scheduling=False,disable_log_stats=True,
        kv_transfer_config=KVTransferConfig(kv_connector='OffloadingConnector',kv_role='kv_both',
            kv_connector_extra_config={'num_cpu_blocks':128,'block_size':256,'eviction_policy':'lru'}))
    engine=llm.llm_engine;s=engine.engine_core.engine_core.scheduler
    def add(rid,ids,n=64,ignore=False):
        engine.add_request(rid,TokensPrompt(prompt_token_ids=ids),
                           SamplingParams(temperature=0,max_tokens=n,ignore_eos=ignore))
    def drain():
        deadline=time.perf_counter()+60;done={}
        while engine.has_unfinished_requests() or s.has_requests():
            for output in engine.step():
                if output.finished:done[output.request_id]=output.outputs[0]
            if time.perf_counter()>deadline:raise TimeoutError('application demo drain')
        assert not s.requests
        return done
    histories=[[{'role':'system','content':f'会话编号 {i}。只根据本会话回答。'+
        '背景：推理服务使用缓存保存会话历史，在工具返回后继续处理请求；请记住用户给出的项目事实。'*32}]
        for i in range(8)]
    rounds=[];total_loaded=0
    for turn in range(4):
        if turn:
            # Reuse the same unrelated pressure prefixes: GPU eviction without
            # continually introducing fresh CPU entries that erase every session.
            for g in range(4):add(f'evict-{turn}-{g}',prompt(77000+g,4096),1,True);drain()
        s._kv.reset();inputs=[]
        for i,history in enumerate(histories):
            question=[f'请记住项目代号 CacheAgent-{i}，预算是 {100+i}。只回复已记录。',
                      '刚才记录的项目代号是什么？只回复代号。',
                      '工具返回：预算增加 7。请更新预算，只回复新预算整数。',
                      '请用一句话汇总项目代号和更新后的预算。'][turn]
            history.append({'role':'user','content':question})
            ids=tok.apply_chat_template(history,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            assert len(ids)+64<8192
            rid=f'app-{turn}-{i}';add(rid,ids)
            request=s.requests[rid];g=gpu_probe(s.kv_cache_manager,request)
            c,b,reason=cpu_probe(s.connector,request,g)
            inputs.append(dict(id=rid,session=i,question=question,input_tokens=ids,
                               preparation=dict(gpu=g,cpu_end=c,bytes=b,reason=reason)))
        for g in range(4):add(f'pressure-{turn}-{g}',prompt(77000+g,4096),16,True)
        done=drain();rows=[]
        for row in inputs:
            output=done[row['id']];i=row['session'];answer=output.text
            histories[i].append({'role':'assistant','content':answer})
            ok=None
            if turn==1:ok=f'CacheAgent-{i}'.lower() in answer.lower()
            if turn==2:
                numbers=re.findall(r'\d+',answer);ok=bool(numbers) and int(numbers[-1])==107+i
            if turn==3:ok=f'CacheAgent-{i}'.lower() in answer.lower() and str(107+i) in answer
            rows.append(dict(**row,answer=answer,output_tokens=list(output.token_ids),fact_check=ok))
        snapshot=s._kv.export();total_loaded+=snapshot['counters'].get('cpu_loaded_tokens',0)
        rounds.append(dict(turn=turn,requests=rows,scheduler=snapshot))
        print('APP_TURN',args.mode,turn,len(rows),snapshot['counters'].get('cpu_loaded_tokens',0),flush=True)
    assert total_loaded>0,'application demonstration did not exercise CPU recovery'
    cs=s.connector.connector_scheduler
    assert not cs._reqs_being_loaded and not cs._reqs_being_stored
    assert all(b.ref_cnt==0 for b in cs.manager.blocks.values())
    out=DATA/'raw/v3_app_demo';out.mkdir(exist_ok=True)
    checks=[r['fact_check'] for turn in rounds for r in turn['requests'] if r['fact_check'] is not None]
    payload=dict(mode=args.mode,completed=32,cpu_loaded_tokens=total_loaded,
        actual_generated_history=True,paired_performance=False,
        fact_checks=sum(checks),fact_checks_total=len(checks),rounds=rounds,histories=histories,
        source_snapshot=json.loads((DATA/'v3_full_source_snapshot.json').read_text()))
    (out/f'{args.mode}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print('APP_COMPLETE',args.mode,sum(checks),'/',len(checks),flush=True)
    engine.engine_core.shutdown()


if __name__=='__main__':main()
