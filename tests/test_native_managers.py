"""Use real vLLM managers without model loading or a CUDA allocation."""
import copy
from types import SimpleNamespace as NS
import torch
from vllm import SamplingParams
from vllm.v1.request import Request
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from kv_sched.runtime import gpu_probe, cpu_probe, reclaimable, invalidate_unexecuted_cache


def request(rid,n,hashes):
    r=Request(rid,[42]*n,SamplingParams(temperature=0,max_tokens=4),None,None)
    r.block_hashes=hashes
    return r


def manager():
    spec=FullAttentionSpec(block_size=16,num_kv_heads=1,head_size=8,dtype=torch.float16)
    cfg=KVCacheConfig(num_blocks=32,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(['layer'],spec)])
    return KVCacheManager(cfg,max_model_len=256,enable_caching=True,log_stats=True,enable_kv_cache_events=True)


def snapshot(m):
    return ([(b.block_id,b.ref_cnt,b.block_hash) for b in m.block_pool.blocks],
            m.block_pool.get_num_free_blocks(),copy.deepcopy(vars(m.prefix_cache_stats)),
            copy.deepcopy(m.block_pool.kv_event_queue))


def test_native_gpu_probe_and_shared_refcounts():
    m=manager(); hs=[bytes([i])*32 for i in range(1,6)]
    a=request('a',64,hs[:4]); m.allocate_slots(a,64); a.num_computed_tokens=64
    m.free(a)
    b=request('b',80,hs)
    before=snapshot(m)
    for _ in range(5): assert gpu_probe(m,b)==64
    assert snapshot(m)==before
    blocks,g=m.get_computed_blocks(b)
    m.allocate_slots(b,16,g,blocks)
    b.num_computed_tokens=80
    assert reclaimable(m,b)==5
    c=request('c',80,hs)
    cb,cg=m.get_computed_blocks(c)
    m.allocate_slots(c,16,cg,cb)
    assert reclaimable(m,b)==1
    assert reclaimable(m,c)==1
    before_free=m.block_pool.get_num_free_blocks()
    m.free(b)
    assert m.block_pool.get_num_free_blocks()-before_free==1


def test_native_lru_ready_pending_and_probe_purity():
    h=[bytes([i])*32 for i in range(1,5)]
    backend=CPUBackend(block_size=16,num_blocks=8)
    m=LRUOffloadingManager(backend,enable_events=True)
    m.prepare_store(h)
    m.complete_store(h[:2])
    cs=NS(manager=m,offloaded_block_size=16,
          _get_block_hashes=lambda r,start_idx=0,end_idx=None: iter(r.block_hashes[start_idx:end_idx]))
    conn=NS(connector_scheduler=cs); r=request('r',65,h)
    before=(list(m.blocks),[b.ref_cnt for b in m.blocks.values()],copy.deepcopy(m.events))
    for _ in range(5): assert cpu_probe(conn,r,0)==(32,32*114688,'ready')
    assert before==(list(m.blocks),[b.ref_cnt for b in m.blocks.values()],m.events)
    assert cpu_probe(conn,r,32)==(32,0,'pending')
    m.complete_store(h[2:])
    assert cpu_probe(conn,r,32)==(64,32*114688,'ready')
    m.prepare_load(h)
    assert all(b.ref_cnt==1 for b in m.blocks.values())
    m.complete_load(h)
    assert all(b.ref_cnt==0 for b in m.blocks.values())


def test_rolled_back_reservation_cannot_publish_uncomputed_kv():
    m=manager(); hs=[bytes([i])*32 for i in range(1,5)]
    a=request('a',48,hs[:3]); m.allocate_slots(a,31); a.num_computed_tokens=31
    b=request('b',64,hs)
    assert gpu_probe(m,b)==16
    m.allocate_slots(a,1)
    assert gpu_probe(m,b)==32  # Optimistically registered before execution.
    assert invalidate_unexecuted_cache(m,a)==1
    m.free(a)
    assert gpu_probe(m,b)==16  # The unexecuted last token is never reused.


def test_cpu_probe_does_not_change_eviction_and_stale_prediction_is_a_miss():
    hs=[bytes([i])*32 for i in range(1,5)]
    managers=[LRUOffloadingManager(CPUBackend(block_size=16,num_blocks=2)) for _ in range(2)]
    for m in managers:
        m.prepare_store(hs[:2]);m.complete_store(hs[:2])
    m=managers[0]
    cs=NS(manager=m,offloaded_block_size=16,
          _get_block_hashes=lambda r,start_idx=0,end_idx=None:iter(r.block_hashes[start_idx:end_idx]))
    r=request('probe',33,hs[:2])
    for _ in range(100):assert cpu_probe(NS(connector_scheduler=cs),r,0)[0]==32
    for m in managers:
        m.prepare_store([hs[2]]);m.complete_store([hs[2]])
    assert list(managers[0].blocks)==list(managers[1].blocks)==hs[1:3]
    assert managers[0].lookup(hs[:2])==0  # Authoritative lookup sees eviction.
    managers[0].prepare_store([hs[3]])
    managers[0].complete_store([hs[3]],success=False)
    assert hs[3] not in managers[0].blocks
    assert all(b.ref_cnt==0 for b in managers[0].blocks.values())
