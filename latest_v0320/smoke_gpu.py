"""Small real-GPU compatibility checks, separate from performance experiments."""
import argparse
from dataclasses import asdict,is_dataclass
import json
from pathlib import Path
import random
import time

DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320')
MODEL='/mnt/ondeviceinfer/models/Qwen3-1.7B'


def metrics(llm):
    if not hasattr(llm,'get_metrics'):
        return {'available':False}
    out=[]
    for m in llm.get_metrics():
        if is_dataclass(m):row=asdict(m)
        else:
            row={k:getattr(m,k) for k in ['name','value','values','labels','count','sum'] if hasattr(m,k)}
        out.append(row)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['torch','eager','offload','default'],required=True)
    args=ap.parse_args();started=time.time()
    import torch
    assert torch.version.cuda=='12.9',torch.version.cuda
    result=dict(mode=args.mode,started_unix=started,torch=torch.__version__,cuda_runtime=torch.version.cuda)
    if args.mode=='torch':
        torch.cuda.init()
        a=torch.ones((256,256),device='cuda',dtype=torch.float16)
        out=a@a;torch.cuda.synchronize()
        assert torch.equal(out,torch.full_like(out,256))
        host=torch.arange(1024*1024,dtype=torch.int32,pin_memory=True)
        device=host.to('cuda',non_blocking=True);torch.cuda.synchronize()
        assert torch.equal(device.cpu(),host)
        result.update(device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                      architecture_list=torch.cuda.get_arch_list(),matmul_passed=True,h2d_d2h_passed=True)
    else:
        from vllm import LLM,SamplingParams
        from transformers import AutoTokenizer
        kwargs=dict(model=MODEL,dtype='float16',max_model_len=8192,
            gpu_memory_utilization=.7,num_gpu_blocks_override=1024,block_size=16,
            max_num_batched_tokens=1024,max_num_seqs=4,enable_prefix_caching=True,
            enable_chunked_prefill=True,disable_log_stats=False)
        if args.mode!='default':kwargs.update(enforce_eager=True,async_scheduling=False)
        if args.mode=='offload':
            from vllm.config import KVTransferConfig
            kwargs['kv_transfer_config']=KVTransferConfig(kv_connector='OffloadingConnector',kv_role='kv_both',
                kv_connector_extra_config={'cpu_bytes_to_use':32768*114688,'block_size':256,'eviction_policy':'lru'})
        llm=LLM(**kwargs)
        config=getattr(llm.llm_engine,'vllm_config',None)
        result['requested_config']=kwargs
        result['resolved_config']=asdict(config) if config is not None and is_dataclass(config) else str(config)
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        texts=['请用一句话介绍 KV Cache 的作用。','What is 7 plus 5? Answer briefly.']
        prompts=[tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False) for q in texts]
        outputs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)
        assert len(outputs)==2 and all(o.finished and o.outputs[0].token_ids for o in outputs)
        result['natural_outputs']=[dict(question=q,text=o.outputs[0].text,tokens=list(o.outputs[0].token_ids)) for q,o in zip(texts,outputs)]
        rng=random.Random(8320);seed=[rng.randrange(100,20000) for _ in range(4096)]
        fixed=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        def generate(ids,params=fixed):
            output=llm.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)[0]
            assert output.finished and len(output.outputs[0].token_ids)==params.max_tokens
            return dict(output_tokens=list(output.outputs[0].token_ids),num_cached_tokens=getattr(output,'num_cached_tokens',None))
        result['seed']=generate(seed)
        result['before_pressure']=metrics(llm)
        if args.mode=='offload':
            result['evictions']=[generate([rng.randrange(100,20000) for _ in range(4096)]) for _ in range(4)]
        result['before_reuse']=metrics(llm)
        result['reuse']=generate(seed+[rng.randrange(100,20000) for _ in range(32)],SamplingParams(temperature=0,max_tokens=16,ignore_eos=True))
        result['after_reuse']=metrics(llm)
        if args.mode=='offload':
            def counter(rows,name,labels):
                return sum(m.get('value',0) for m in rows if m['name']==name
                    and all(m['labels'].get(k)==v for k,v in labels.items()))
            h2d=counter(result['after_reuse'],'vllm:kv_offload_total_bytes',{'transfer_type':'CPU_to_GPU'})-counter(result['before_reuse'],'vllm:kv_offload_total_bytes',{'transfer_type':'CPU_to_GPU'})
            host=counter(result['after_reuse'],'vllm:prompt_tokens_cached_by_source',{'source':'host'})-counter(result['before_reuse'],'vllm:prompt_tokens_cached_by_source',{'source':'host'})
            assert host==4096 and h2d==4096*114688,(host,h2d)
            result['cpu_recovery_evidence']={'host_tokens':host,'h2d_bytes':h2d}
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    result.update(passed=True,ended_unix=time.time(),purpose='compatibility smoke only; not a performance comparison')
    path=DATA/f'raw/smoke-{args.mode}.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('SMOKE_PASSED',args.mode,str(path),flush=True)

if __name__=='__main__':main()
