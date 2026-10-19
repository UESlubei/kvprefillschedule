from types import SimpleNamespace as NS
from types import MethodType
import pytest
import torch


def test_aborted_cpu_load_holds_gpu_destination_until_ack():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnectorScheduler
    cs=OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    cs._requests={'cancel':object()};cs._request_block_ids={'cancel':[1]}
    cs._next_stored_block_idx={'cancel':0}
    cs._reqs_being_stored={};cs._reqs_being_loaded={'cancel':{b'hash'}}
    delayed,_=cs.request_finished(NS(request_id='cancel'),[1])
    assert delayed, 'H2D destination must not be freed before transfer acknowledgement'


@pytest.mark.parametrize('first',['load','store'])
def test_cancelled_request_waits_for_both_jobs_and_ignores_duplicate_ack(first):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.outputs import KVConnectorOutput
    cs=NS(_reqs_being_loaded={'r':{b'h'}},_reqs_being_stored={'r':{b'h'}})
    def update(output):
        for rid in output.finished_recving or []:cs._reqs_being_loaded.pop(rid,None)
        for rid in output.finished_sending or []:cs._reqs_being_stored.pop(rid,None)
    freed=[]
    req=NS(request_id='r',is_finished=lambda:True)
    s=NS(connector=NS(connector_scheduler=cs,update_connector_output=update),
         requests={'r':req},kv_cache_manager=NS(free=lambda r:freed.append(r.request_id)),
         finished_recving_kv_req_ids={'r'},failed_recving_kv_req_ids={'r'},
         finished_req_ids=set(),_finished_kv_transfer_req_ids={'r'})
    assert Scheduler.has_finished_requests(s), 'engine must keep polling deferred completions'
    s._free_blocks=MethodType(Scheduler._free_blocks,s)
    load=KVConnectorOutput(finished_recving={'r'})
    store=KVConnectorOutput(finished_sending={'r'})
    one,two=(load,store) if first=='load' else (store,load)
    Scheduler._update_from_kv_xfer_finished(s,one)
    assert 'r' in s.requests and not freed
    Scheduler._update_from_kv_xfer_finished(s,two)
    assert freed==['r'] and not s.requests
    assert not s.finished_recving_kv_req_ids and not s.failed_recving_kv_req_ids
    assert not Scheduler.has_finished_requests(s)
    Scheduler._update_from_kv_xfer_finished(s,KVConnectorOutput(finished_sending={'r'},finished_recving={'r'}))
    assert freed==['r']


def test_preemption_flushes_queued_stores_before_waiting():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnectorWorker
    obj=OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    calls=[]
    obj._unsubmitted_store_jobs=[(1,'spec-a'),(2,'spec-b')]
    obj._store_jobs={'a':{1},'b':{2}}
    obj.worker=NS(transfer_async=lambda j,s:(calls.append(('submit',j)) or True),
                  wait=lambda js:calls.append(('wait',js)))
    obj.handle_preemptions({'a'})
    assert calls==[('submit',1),('submit',2),('wait',{1})]
    assert obj._unsubmitted_store_jobs==[]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_native_partial_cpu_chunk_copy_and_fence():
    from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandler
    from vllm.v1.kv_offload.mediums import CPULoadStoreSpec,GPULoadStoreSpec
    class Backend:
        @staticmethod
        def get_kv_cache_shape(num_blocks,block_size,num_kv_heads,head_size):
            return (2,num_blocks,block_size,num_kv_heads,head_size)
    gpu=torch.zeros((2,64,16,2,8),device='cuda',dtype=torch.float16)
    handler=CpuGpuOffloadingHandler(16,256,4,{'layer':gpu},{'layer':Backend})
    cpu=handler.cpu_tensors[0]
    cpu.copy_(torch.arange(cpu.numel()).reshape(cpu.shape).remainder(1000).to(torch.float16))
    torch.cuda.synchronize()
    handler.transfer_async(1,(CPULoadStoreSpec([0,1]),GPULoadStoreSpec(list(range(20,39)))))
    handler.wait({1})
    assert handler.get_finished()==[(1,True)]
    assert torch.equal(gpu[:,20:39].cpu(),cpu[:,13:32])
    gpu.fill_(7)
    torch.cuda.synchronize()
    handler.transfer_async(2,(GPULoadStoreSpec(list(range(16))),CPULoadStoreSpec([1])))
    handler.wait({2})
    assert handler.get_finished()==[(2,True)]
    assert torch.equal(cpu[:,16:32],torch.full_like(cpu[:,16:32],7))
