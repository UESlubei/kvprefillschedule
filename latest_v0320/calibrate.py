"""Recalibrate one-token prefill service time and the actual native H2D handler."""
from dataclasses import asdict
import statistics
import time
import os
import numpy as np
from latest_v0320.common import DATA, MODEL, KV_BYTES, REVISION, dump, prompt

def transfer_on_worker(worker):
    import torch
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    from vllm.v1.kv_offload.base import GPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    conn=get_kv_transfer_group()
    h=conn.connector_worker.worker._load_handler
    assert not h._transfers
    rows=[]
    for chunks in [1,2,4,8,16]:
        samples=[]
        for rep in range(8):
            jid=1_000_000+chunks*100+rep
            a=time.perf_counter()
            assert h.transfer_async(jid,CPULoadStoreSpec(list(range(chunks))),GPULoadStoreSpec(list(range(chunks*16)),[chunks*16],[0]))
            submit=(time.perf_counter()-a)*1000
            h.wait({jid})
            wall=(time.perf_counter()-a)*1000
            result=h.get_finished()
            assert len(result)==1 and result[0].success
            assert result[0].transfer_size==chunks*256*KV_BYTES
            if rep>=2:
                samples.append({'wall_ms':wall,'submit_ms':submit,
                    'cuda_ms':result[0].transfer_time*1000})
        rows.append({'tokens':chunks*256,'bytes':chunks*256*KV_BYTES,
            'ms':statistics.median(s['wall_ms'] for s in samples),'samples':samples})
    return {'rows':rows,'pinned':all(t.is_pinned() for t in h.src_tensors),
        'handler':type(h).__name__,'swap_function':str(h._swap_blocks) if hasattr(h,'_swap_blocks') else None}

def main():
    # A trusted local callable inspects this benchmark's own worker only.
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    llm=LLM(model=MODEL,dtype='float16',enforce_eager=True,async_scheduling=False,
        max_model_len=8192,gpu_memory_utilization=.75,num_gpu_blocks_override=4096,
        block_size=16,max_num_batched_tokens=1024,max_num_seqs=16,
        enable_prefix_caching=True,enable_chunked_prefill=True,disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(kv_connector='OffloadingConnector',kv_role='kv_both',
            kv_connector_extra_config={'cpu_bytes_to_use':32768*KV_BYTES,'block_size':256,'eviction_policy':'lru'}))
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
    def generate(ids):return llm.generate([{'prompt_token_ids':x} for x in ids],params,use_tqdm=False)
    def reset():
        for _ in range(100):
            if llm.reset_prefix_cache(reset_connector=True):return
            time.sleep(.02)
        raise RuntimeError('calibration cache reset failed')
    raw=DATA/'cost_models/prefill-raw.json'
    rows=__import__('json').loads(raw.read_text()) if raw.exists() else []
    contexts=[(512,0),(1024,0),(2048,0),(2048,1024),(4096,0),(4096,1024),(4096,2048)]
    for batch in [1,4]:
        for length,cached in contexts:
            if any(r['length']==length and r['cached']==cached and r['batch']==batch for r in rows):continue
            ids=[prompt(600000+batch*10+j,length) for j in range(batch)]
            samples=[]
            for rep in range(5):
                reset()
                if cached:generate([x[:cached] for x in ids])
                ts=time.perf_counter(); outputs=generate(ids); ms=(time.perf_counter()-ts)*1000
                assert all(o.finished and len(o.outputs[0].token_ids)==1 for o in outputs)
                if rep>=2:samples.append(ms)
            row={'length':length,'cached':cached,'batch':batch,
                'ms':statistics.median(samples),'samples_ms':samples}
            rows.append(row);dump(raw,rows);print('PREFILL',row,flush=True)
    # Destructive H2D microbenchmark uses the now-idle, actual connector layout.
    # No further model output is consumed after overwriting these GPU pages.
    transfer=llm.collective_rpc(transfer_on_worker,timeout=120)[0]
    x=np.array([[1,r['bytes']] for r in transfer['rows']]); y=np.array([r['ms'] for r in transfer['rows']])
    fixed,slope=np.linalg.lstsq(x,y,rcond=None)[0]
    assert slope>0 and transfer['pinned']
    model={'vllm':'0.32.0+cu129','torch':'2.13.0+cu129','cuda':'12.9',
        'model_revision':REVISION,'kv_bytes_per_token':KV_BYTES,'prefill':rows,
        'prefill_scope':'native synchronous Eager one-output service wall time, includes dispatch/sampling; ranking approximation, not isolated GPU kernel time',
        'h2d':{'fixed_ms':max(0,float(fixed)),'bytes_per_ms':1/float(slope)},
        'transfer_raw':transfer}
    dump(DATA/'cost_models/native.json',model)
    print('CALIBRATED',model['h2d'],flush=True)
    llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
