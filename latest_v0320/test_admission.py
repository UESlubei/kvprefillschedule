"""Native-manager purity and state-barrier regressions; no GPU allocation."""
import copy
from types import SimpleNamespace as NS
import unittest
import torch
from vllm import SamplingParams
from vllm.v1.request import Request
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.base import make_offload_key
from latest_v0320.scheduler import gpu_probe,cpu_probe,reorder_segments

def request(name,n,hashes):
    r=Request(name,[42]*n,SamplingParams(temperature=0,max_tokens=4),None)
    r.block_hashes=hashes
    return r

class TestAdmission(unittest.TestCase):
    def test_gpu_probe_preserves_native_resources_stats_events(self):
        spec=FullAttentionSpec(block_size=16,num_kv_heads=1,head_size=8,dtype=torch.float16)
        cfg=KVCacheConfig(num_blocks=32,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(['layer'],spec)])
        m=KVCacheManager(cfg,max_model_len=256,scheduler_block_size=16,hash_block_size=16,
            enable_caching=True,log_stats=True,enable_kv_cache_events=True)
        hashes=[bytes([i])*32 for i in range(1,6)]
        a=request('seed',64,hashes[:4]);m.allocate_slots(a,64);a.num_computed_tokens=64;m.free(a)
        r=request('probe',80,hashes)
        def snapshot():
            return ([(b.block_id,b.ref_cnt,b.block_hash) for b in m.block_pool.blocks],
                m.block_pool.get_num_free_blocks(),copy.deepcopy(vars(m.prefix_cache_stats)),
                copy.deepcopy(m.block_pool.kv_event_queue),
                [b.block_id for b in m.block_pool.free_block_queue.get_all_free_blocks()])
        before=snapshot()
        for _ in range(20):self.assertEqual(gpu_probe(m,r),64)
        self.assertEqual(snapshot(),before)
        self.assertEqual(gpu_probe(m,request('all-cached',64,hashes[:4])),48)

    def test_cpu_native_lru_probe_stops_at_pending_and_keeps_recency(self):
        m=CPUOffloadingManager(8)
        hashes=[bytes([i])*32 for i in range(1,49)]
        for i,h in enumerate(hashes[15::16]):
            c=ChunkStatus(i);c.ref_cnt= -1 if i==1 else 0
            m._policy.insert(make_offload_key(h,0),c)
        conn=NS(connector_scheduler=NS(manager=m,config=NS(kv_group_configs=[
            NS(tokens_per_chunk=256,hashes_per_chunk=16,group_idx=0,sliding_window_size_in_chunks=None)])))
        r=request('r',800,hashes)
        def snap():return (dict(m._policy._ranks),list(m._policy._heap),
            [(k,c.ref_cnt) for k,c in m._policy.chunks.items()])
        before=snap()
        for _ in range(20):self.assertEqual(cpu_probe(conn,r,0),(0,0,'pending'))
        self.assertEqual(cpu_probe(conn,r,256),(256,0,'pending'))
        self.assertEqual(cpu_probe(conn,r,512),(768,256*114688,'ready'))
        self.assertEqual(before,snap())
        pending_key=make_offload_key(hashes[31],0)
        m._policy.remove(pending_key)
        self.assertEqual(cpu_probe(conn,r,0),(256,256*114688,'ready'))
        conn.connector_scheduler._chunks_being_loaded={make_offload_key(hashes[15],0)}
        self.assertEqual(cpu_probe(conn,r,0),(0,0,'inflight'))

    def test_state_barriers_and_overdue_stable_order(self):
        rows=[request(str(i),16,[]) for i in range(6)]
        rows[2].num_computed_tokens=1
        rows[4].num_output_placeholders=1
        scores={str(i):i for i in range(6)};overdue={str(i):False for i in range(6)}
        out=reorder_segments(rows,scores,overdue)
        self.assertEqual([r.request_id for r in out],['1','0','2','3','4','5'])
        overdue['0']=True
        self.assertEqual(reorder_segments(rows,scores,overdue),rows)

if __name__=='__main__':unittest.main()
