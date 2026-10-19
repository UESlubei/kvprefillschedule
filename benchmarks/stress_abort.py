"""Real-engine cancellation under pressure with delayed transfer acknowledgements.

Only completion notifications are held; no new transfer protocol or kernels.
The native worker still executes all copies. Diagnostic runs are not benchmarks.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from run_trace import DATA,MODEL,prompt,workload
from kv_sched.policy import Config


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohorts',type=int,default=4)
    ap.add_argument('--label',default='v3_faults');args=ap.parse_args()
    cfg=Config(admission='tier',preemption='restore',cost_model=str(DATA/'cost_models/qwen3-1.7b-3090.json'))
    os.environ['KV_SCHED_CONFIG']=json.dumps(asdict(cfg))
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    llm=LLM(model=MODEL,dtype='float16',enforce_eager=True,max_model_len=8192,
            num_gpu_blocks_override=1024,block_size=16,gpu_memory_utilization=.8,
            max_num_batched_tokens=1024,max_num_seqs=16,enable_prefix_caching=True,
            enable_chunked_prefill=True,async_scheduling=False,disable_log_stats=True,
            kv_transfer_config=KVTransferConfig(kv_connector='OffloadingConnector',kv_role='kv_both',
                kv_connector_extra_config={'num_cpu_blocks':128,'block_size':256,'eviction_policy':'lru'}))
    eng=llm.llm_engine;s=eng.engine_core.engine_core.scheduler
    cs=s.connector.connector_scheduler
    worker_connector=get_kv_transfer_group();cw=worker_connector.connector_worker
    original_finished=worker_connector.get_finished
    hold_load=set();hold_store=set();buffer_load=set();buffer_store=set()
    def finished(ids):
        sending,receiving=original_finished(ids)
        buffer_load.update(receiving);buffer_store.update(sending)
        ready_load=buffer_load-hold_load;ready_store=buffer_store-hold_store
        buffer_load.difference_update(ready_load);buffer_store.difference_update(ready_store)
        return ready_store,ready_load
    worker_connector.get_finished=finished
    def add(rid,tokens,n=64):
        eng.add_request(rid,TokensPrompt(prompt_token_ids=tokens),
                        SamplingParams(temperature=0,max_tokens=n,ignore_eos=True))
    def busy():
        return bool(eng.has_unfinished_requests() or s.has_requests() or s.requests or
                    cs._reqs_being_loaded or cs._reqs_being_stored or buffer_load or buffer_store)
    def drain(limit=180):
        deadline=time.perf_counter()+limit
        while busy():
            eng.step()
            if time.perf_counter()>deadline:
                raise TimeoutError(f'drain: requests={list(s.requests)}, load={dict(cs._reqs_being_loaded)}, '
                                   f'store={dict(cs._reqs_being_stored)}, jobs={cw._jobs}, '
                                   f'buffer_load={buffer_load}, buffer_store={buffer_store}')
    def clean():
        assert not s.running and not s.waiting and not s.requests
        assert not s.finished_recving_kv_req_ids and not s.failed_recving_kv_req_ids
        assert not s._finished_kv_transfer_req_ids
        assert not cs._reqs_being_loaded and not cs._reqs_being_stored
        assert not cs._requests and not cs._reqs_to_load
        assert not cw._jobs and not cw._store_jobs and not cw._load_job
        assert not cw._unsubmitted_store_jobs and not cw._finished_reqs_waiting_for_store
        assert not buffer_load and not buffer_store
        blocks=s.kv_cache_manager.block_pool.blocks
        assert all(b.ref_cnt==0 for b in blocks if not b.is_null)
        assert all(b.ref_cnt==0 for b in cs.manager.blocks.values())
        return dict(gpu_free=s.kv_cache_manager.block_pool.get_num_free_blocks(),
                    gpu_ref_holders=0,cpu_ref_holders=0,pending_jobs=0)
    def reset():
        drain();clean()
        assert s.kv_cache_manager.reset_prefix_cache()
        for b in cs.manager.blocks.values():cs.manager.backend.free(b)
        cs.manager.blocks.clear()
        assert cs.manager.backend.get_num_free_blocks()==128
        s._kv.reset()
    out=DATA/'raw'/args.label;out.mkdir(parents=True,exist_ok=True)
    records=[]
    for cohort in range(args.cohorts):
        reset()
        for g in range(4):add(f'seed-{cohort}-{g}',prompt(2000+g,2048),1);drain()
        for g in range(4):add(f'evict-{cohort}-{g}',prompt(7000+g,4096),1);drain()
        for g in range(4):add(f'gpu-{cohort}-{g}',prompt(1000+g,2048),1);drain()
        s._kv.reset()
        events=[];cancelled=set();done={};started=time.perf_counter()
        rid=f'cancel-load-{cohort}'
        hold_load.add(rid)
        add(rid,prompt(2000,2048)+prompt(990000+cohort,256))
        for _ in range(100):
            eng.step()
            if cs._reqs_being_loaded.get(rid):break
        else:raise AssertionError('CPU load injection was not activated')
        owned=s.kv_cache_manager.get_block_ids(rid)
        assert owned[0]
        eng.abort_request([rid]);cancelled.add(rid)
        assert rid in s.requests and s.requests[rid].is_finished()
        assert s.kv_cache_manager.get_block_ids(rid)==owned
        events.append(dict(kind='abort_loading',id=rid,gpu_blocks_held=len(owned[0]),
                           note='native copies, delayed completion notification'))
        hold_load.remove(rid);drain()
        assert rid not in s.requests and rid not in s.finished_recving_kv_req_ids
        trace=workload('C',63,401+cohort)
        for i,row in enumerate(trace):
            row['id']=f'cohort-{cohort}-{i}'
            add(row['id'],row['tokens'],row['output'])
        waiting=trace[-1]['id'];eng.abort_request([waiting]);cancelled.add(waiting)
        events.append(dict(kind='abort_waiting',id=waiting))
        store_cancel=None;running_cancel=None;release_store_step=None;step=0
        while busy():
            outputs=eng.step();step+=1
            for output in outputs:
                if output.finished and output.request_id not in cancelled:
                    done[output.request_id]=list(output.outputs[0].token_ids)
            if store_cancel is None:
                candidates=[r for r in s.running if cs._reqs_being_stored.get(r.request_id)]
                if candidates:
                    r=candidates[-1];store_cancel=r.request_id;hold_store.add(store_cancel)
                    owned=s.kv_cache_manager.get_block_ids(store_cancel)
                    eng.abort_request([store_cancel]);cancelled.add(store_cancel)
                    assert store_cancel in s.requests
                    assert s.kv_cache_manager.get_block_ids(store_cancel)==owned
                    release_store_step=step+2
                    events.append(dict(kind='abort_storing',id=store_cancel,gpu_blocks_held=len(owned[0])))
            if running_cancel is None:
                candidates=[r for r in s.running if r.num_computed_tokens>0 and r.request_id not in cancelled]
                if candidates:
                    running_cancel=candidates[0].request_id
                    events.append(dict(kind='abort_running',id=running_cancel,
                                       computed=candidates[0].num_computed_tokens))
                    eng.abort_request([running_cancel]);cancelled.add(running_cancel)
            if release_store_step is not None and step>=release_store_step:
                hold_store.discard(store_cancel)
            if time.perf_counter()-started>180:raise TimeoutError('fault cohort timeout')
        assert store_cancel and running_cancel, 'required cancellation state not observed'
        assert len(cancelled)==4
        expected={r['id'] for r in trace}-cancelled
        assert set(done)==expected,(len(done),len(expected))
        assert all(len(tokens)==64 for tokens in done.values())
        idle=clean()
        row=dict(cohort=cohort,submitted=64,completed=len(done),cancelled=len(cancelled),
                 elapsed_seconds=time.perf_counter()-started,events=events,
                 idle=idle,scheduler=s._kv.export(),passed=True)
        (out/f'cohort-{cohort}.json').write_text(json.dumps(row,indent=2))
        records.append(row)
        print('FAULT_COHORT',cohort,len(done),len(cancelled),idle,flush=True)
    payload=dict(cohorts=args.cohorts,submitted=64*args.cohorts,
                 completed=sum(r['completed'] for r in records),cancelled=4*args.cohorts,
                 passed=all(r['passed'] for r in records),
                 source_snapshot=json.loads((DATA/'v3_full_source_snapshot.json').read_text()),
                 instrumentation='transfer acknowledgements delayed; excluded from performance results')
    (out/'summary.json').write_text(json.dumps(payload,indent=2))
    print('FAULT_PASS',payload,flush=True)
    eng.engine_core.shutdown()


if __name__=='__main__':main()
