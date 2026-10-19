"""Required fail-closed capability and cache-gating boundaries."""
from collections import deque
from types import SimpleNamespace as NS
import pytest

from kv_sched.policy import Config
from kv_sched.runtime import SchedulerHooks


@pytest.mark.parametrize('field,value',[('async_scheduling',True),('use_pp',True),
    ('num_spec_tokens',1),('is_encoder_decoder',True),('lora_config',object()),('tp',2)])
def test_unsupported_execution_modes_are_rejected(monkeypatch,field,value):
    monkeypatch.setenv('KV_SCHED_CONFIG','{}')
    s=NS(scheduler_config=NS(async_scheduling=False),use_pp=False,num_spec_tokens=0,
         is_encoder_decoder=False,lora_config=None,parallel_config=NS(tensor_parallel_size=1))
    if field=='async_scheduling':s.scheduler_config.async_scheduling=value
    elif field=='tp':s.parallel_config.tensor_parallel_size=value
    else:setattr(s,field,value)
    with pytest.raises(ValueError):SchedulerHooks(s)


@pytest.mark.parametrize('enabled,policy,usage,threshold,changes',[
    (False,'fcfs',1,.5,False),(True,'priority',1,.5,False),
    (True,'fcfs',.49,.5,False),(True,'fcfs',.5,.5,True),(True,'fcfs',.51,.5,True)])
def test_cache_and_threshold_gate_before_any_probe(enabled,policy,usage,threshold,changes):
    class Queue(deque):
        def pop_request(self):return self.popleft()
        def prepend_request(self,r):self.appendleft(r)
    def req(rid):return NS(request_id=rid,status=NS(name='WAITING'),num_computed_tokens=0,
        num_preemptions=0,skip_reading_prefix_cache=False,arrival_time=0)
    queue=Queue([req('cold'),req('warm')])
    hook=SchedulerHooks.__new__(SchedulerHooks)
    hook.s=NS(policy=NS(value=policy),kv_cache_manager=NS(enable_caching=enabled,usage=usage),waiting=queue)
    hook.config=Config(admission='gpu',window=2,threshold=threshold,max_wait_ms=0)
    hook.model=NS(calibrated=True)
    hook.reset()
    probes=[]
    hook.probe=lambda r,use_cpu:(probes.append(r.request_id) or (100 if r.request_id=='warm' else 0,0,0))
    hook.reorder()
    assert list(r.request_id for r in queue)==(['warm','cold'] if changes else ['cold','warm'])
    assert bool(probes)==changes


def test_invalid_transfer_cannot_commit_execution_statistics():
    hook=SchedulerHooks.__new__(SchedulerHooks)
    with pytest.raises(RuntimeError,match='invalid KV load'):
        hook.committed(None,NS(kv_connector_output=NS(invalid_block_ids={1})))
