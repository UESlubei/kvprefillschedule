"""Actual V1 Scheduler remote-load resume state regression (no model execution)."""
from types import SimpleNamespace as NS
import pytest
import torch


def test_preempted_cpu_load_is_resumed_with_generated_history(monkeypatch):
    monkeypatch.setenv('KV_SCHED_CONFIG','{}')
    from vllm.engine.arg_utils import EngineArgs
    from vllm import SamplingParams
    from vllm.v1.request import Request,RequestStatus
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
    config=EngineArgs(model='/mnt/ondeviceinfer/models/Qwen3-1.7B',dtype='float16',
                     max_model_len=256,max_num_seqs=4,max_num_batched_tokens=128,
                     enforce_eager=True,enable_prefix_caching=True,enable_chunked_prefill=True).create_engine_config()
    config.cache_config.num_gpu_blocks=32
    config.cache_config.block_size=16
    spec=FullAttentionSpec(block_size=16,num_kv_heads=1,head_size=8,dtype=torch.float16)
    kv=KVCacheConfig(num_blocks=32,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(['layer'],spec)])
    s=Scheduler(config,kv,NS(),16)
    r=Request('resume',[42]*64,SamplingParams(temperature=0,max_tokens=8),None,None)
    r.append_output_token_ids([7,16])
    r.block_hashes=[bytes([i])*32 for i in range(1,5)]
    s.kv_cache_manager.allocate_slots(r,64,delay_cache_blocks=True)
    r.status=RequestStatus.WAITING_FOR_REMOTE_KVS
    r.num_preemptions=1
    s.requests[r.request_id]=r
    s.waiting.add_request(r)
    s.finished_recving_kv_req_ids.add(r.request_id)
    s.connector=NS(update_state_after_alloc=lambda *args:None,
                   build_connector_meta=lambda out:NS(reqs_to_store={},reqs_to_load={}))
    out=s.schedule()
    assert not out.scheduled_new_reqs
    assert out.scheduled_cached_reqs.resumed_req_ids=={'resume'}
    assert out.scheduled_cached_reqs.num_output_tokens==[2]
    assert out.scheduled_cached_reqs.all_token_ids['resume']==[42]*64+[7,16]
    assert list(r.output_token_ids)==[7,16]
    assert r.status==RequestStatus.RUNNING


@pytest.mark.parametrize('admission,first,remaining', [('fcfs','cold',96),('gpu','warm',128)])
def test_native_admission_prevents_cold_eviction(monkeypatch,admission,first,remaining):
    import json
    monkeypatch.setenv('KV_SCHED_CONFIG',json.dumps({'admission':admission,'window':2,'threshold':0}))
    from vllm.engine.arg_utils import EngineArgs
    from vllm import SamplingParams
    from vllm.v1.request import Request
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
    from kv_sched.runtime import gpu_probe
    config=EngineArgs(model='/mnt/ondeviceinfer/models/Qwen3-1.7B',dtype='float16',
                     max_model_len=192,max_num_seqs=4,max_num_batched_tokens=192,
                     enforce_eager=True,enable_prefix_caching=True,enable_chunked_prefill=True).create_engine_config()
    config.cache_config.num_gpu_blocks=17;config.cache_config.block_size=16
    spec=FullAttentionSpec(block_size=16,num_kv_heads=1,head_size=8,dtype=torch.float16)
    kv=KVCacheConfig(17,[],[KVCacheGroupSpec(['layer'],spec)])
    s=Scheduler(config,kv,NS(),16)
    def request(rid,n,hashes):
        r=Request(rid,[42]*n,SamplingParams(temperature=0,max_tokens=4),None,None)
        r.block_hashes=hashes
        return r
    hs=[bytes([i])*32 for i in range(1,10)]
    seed=request('seed',128,hs[:8]);s.kv_cache_manager.allocate_slots(seed,128)
    seed.num_computed_tokens=128;s.kv_cache_manager.free(seed)
    cold=request('cold',160,[bytes([i])*32 for i in range(20,30)])
    warm=request('warm',144,hs)
    assert gpu_probe(s.kv_cache_manager,warm)==128
    for r in [cold,warm]:s.requests[r.request_id]=r;s.waiting.add_request(r)
    output=s.schedule()
    assert output.scheduled_new_reqs[0].req_id==first
    assert gpu_probe(s.kv_cache_manager,warm)==remaining


@pytest.mark.parametrize('b_length,blocks,victim',[(128,18,'a'),(64,14,'b')])
def test_lcf_rolls_back_reserved_front_or_current_victim(monkeypatch,b_length,blocks,victim):
    monkeypatch.setenv('KV_SCHED_CONFIG','{"preemption":"lcf"}')
    from vllm.engine.arg_utils import EngineArgs
    from vllm import SamplingParams
    from vllm.v1.request import Request,RequestStatus
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
    config=EngineArgs(model='/mnt/ondeviceinfer/models/Qwen3-1.7B',dtype='float16',
                     max_model_len=256,max_num_seqs=4,max_num_batched_tokens=256,
                     enforce_eager=True,enable_prefix_caching=True,enable_chunked_prefill=True).create_engine_config()
    config.cache_config.num_gpu_blocks=blocks;config.cache_config.block_size=16
    spec=FullAttentionSpec(block_size=16,num_kv_heads=1,head_size=8,dtype=torch.float16)
    kv=KVCacheConfig(blocks,[],[KVCacheGroupSpec(['layer'],spec)])
    s=Scheduler(config,kv,NS(),16)
    rs=[]
    for index,(rid,n) in enumerate([('a',128),('b',b_length)]):
        r=Request(rid,[42]*n,SamplingParams(temperature=0,max_tokens=8),None,None)
        r.block_hashes=[bytes([index*32+j])*32 for j in range(1,n//16+2)]
        s.kv_cache_manager.allocate_slots(r,n);r.num_computed_tokens=n
        r.append_output_token_ids([7]);r.status=RequestStatus.RUNNING
        s.requests[rid]=r;s.running.append(r);rs.append(r)
    assert s.kv_cache_manager.block_pool.get_num_free_blocks()==1
    out=s.schedule()
    survivor='b' if victim=='a' else 'a'
    assert out.num_scheduled_tokens=={survivor:1}
    assert out.total_num_scheduled_tokens==1
    assert out.scheduled_cached_reqs.req_ids==[survivor]
    assert [r.request_id for r in s.running]==[survivor]
    assert [r.request_id for r in s.waiting]==[victim]
    r=s.requests[victim]
    assert r.status==RequestStatus.PREEMPTED and r.num_computed_tokens==0
    assert s.kv_cache_manager.get_block_ids(victim)==([],)
    assert s._kv.counters['reserved_victim_rollbacks']==(victim=='a')
    assert set(s._kv.execution)=={survivor}


def test_completed_load_holders_can_advance_past_memory_blocked_head(monkeypatch):
    monkeypatch.setenv('KV_SCHED_CONFIG','{}')
    from vllm.engine.arg_utils import EngineArgs
    from vllm import SamplingParams
    from vllm.v1.request import Request,RequestStatus
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
    config=EngineArgs(model='/mnt/ondeviceinfer/models/Qwen3-1.7B',dtype='float16',
        max_model_len=128,max_num_seqs=4,max_num_batched_tokens=64,enforce_eager=True,
        enable_prefix_caching=True,enable_chunked_prefill=True).create_engine_config()
    config.cache_config.num_gpu_blocks=10;config.cache_config.block_size=16
    spec=FullAttentionSpec(block_size=16,num_kv_heads=1,head_size=8,dtype=torch.float16)
    s=Scheduler(config,KVCacheConfig(10,[],[KVCacheGroupSpec(['layer'],spec)]),NS(),16)
    def request(rid,n,index):
        r=Request(rid,[42]*n,SamplingParams(temperature=0,max_tokens=8),None,None)
        r.block_hashes=[bytes([index*16+j])*32 for j in range(1,n//16+1)]
        return r
    front=request('blocked',64,1);s.requests[front.request_id]=front;s.waiting.add_request(front)
    for i in range(3):
        r=request(f'loaded-{i}',48,i+2)
        s.kv_cache_manager.allocate_slots(r,32,delay_cache_blocks=True)
        r.status=RequestStatus.WAITING_FOR_REMOTE_KVS
        s.requests[r.request_id]=r;s.waiting.add_request(r)
        s.finished_recving_kv_req_ids.add(r.request_id)
    assert s.kv_cache_manager.block_pool.get_num_free_blocks()==3
    s.connector=NS(get_num_new_matched_tokens=lambda *args:(0,False),
        update_state_after_alloc=lambda *args:None,
        build_connector_meta=lambda out:NS(reqs_to_store={},reqs_to_load={}))
    out=s.schedule()
    assert out.num_scheduled_tokens=={f'loaded-{i}':16 for i in range(3)}
    assert out.total_num_scheduled_tokens==48
    assert [r.request_id for r in s.waiting]==['blocked']
    assert s._kv.counters['native_ready_promotions']==3
