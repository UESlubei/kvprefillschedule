from collections import OrderedDict
from types import SimpleNamespace as NS
from itertools import islice
import pytest

from kv_sched.runtime import cpu_probe, gpu_probe, reclaimable


class LRUOffloadingManager:
    def __init__(self, states):
        self.blocks=OrderedDict((h,NS(is_ready=state)) for h,state in states)
    def lookup(self, hashes):
        hits=0
        for h in hashes:
            b=self.blocks.get(h)
            if b is None or not b.is_ready: break
            hits+=1
        return hits


def connector(states):
    manager=LRUOffloadingManager(states)
    def hashes(r,start_idx=0,end_idx=None):
        return islice(r.block_hashes,16*start_idx+15,16*end_idx if end_idx else None,16)
    cs=NS(manager=manager,offloaded_block_size=256,_get_block_hashes=hashes)
    return NS(connector_scheduler=cs)


def test_cpu_probe_is_read_only_and_stops_at_pending():
    c=connector([(15,True),(31,False),(47,True)])
    r=NS(num_tokens=800,block_hashes=list(range(50)))
    before=list(c.connector_scheduler.manager.blocks.items())
    assert cpu_probe(c,r,0)==(256,256*114688,'ready')
    assert before==list(c.connector_scheduler.manager.blocks.items())
    assert cpu_probe(c,r,256)==(256,0,'pending')


def test_gpu_boundary_and_cpu_gap():
    c=connector([(15,True),(31,True),(47,True)])
    r=NS(num_tokens=800,block_hashes=list(range(50)))
    assert cpu_probe(c,r,80)==(768,688*114688,'ready')
    del c.connector_scheduler.manager.blocks[31]
    assert cpu_probe(c,r,80)==(80,0,'too_short')
    # GPU covers missing earlier chunks: no need to require CPU prefix from 0.
    assert cpu_probe(c,r,512)==(768,256*114688,'ready')


def test_shared_blocks_null_and_duplicates():
    blocks=[NS(block_id=0,is_null=True,ref_cnt=1),NS(block_id=1,is_null=False,ref_cnt=2),
            NS(block_id=2,is_null=False,ref_cnt=1)]
    manager=NS(coordinator=NS(get_blocks=lambda _: (blocks+[blocks[-1]],)))
    assert reclaimable(manager,NS(request_id='r'))==1


def test_gpu_probe_skips_without_calling_stats():
    class Coordinator:
        def find_longest_cache_hit(self,hashes,n):
            assert n==63
            return ([],),48
    m=NS(enable_caching=True,coordinator=Coordinator())
    r=NS(skip_reading_prefix_cache=False,num_tokens=64,block_hashes=['a'])
    assert gpu_probe(m,r)==48
    r.skip_reading_prefix_cache=True
    assert gpu_probe(m,r)==0
