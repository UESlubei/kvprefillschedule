from types import SimpleNamespace as NS
import random
import pytest

from kv_sched.policy import Config, IntervalLedger, Victim, choose_victim, reorder_segments


def req(name, status='WAITING', computed=0, preempted=0, skip=False):
    return NS(request_id=name,status=NS(name=status),num_computed_tokens=computed,
              num_preemptions=preempted,skip_reading_prefix_cache=skip)


@pytest.mark.parametrize('kwargs',[{'window':-1},{'threshold':1.1},{'admission':'typo'},
                                  {'preemption':'typo'},{'aging':-1},{'max_wait_ms':-1}])
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError): Config(**kwargs)


def test_recompute_interval_example():
    l=IntervalLedger()
    assert l.commit(0,512)==0
    assert l.commit(384,640)==128
    assert l.commit(0,256)==256
    assert l.intervals==[(0,640)]


def test_interval_ledger_against_independent_position_set():
    rng=random.Random(17)
    for _ in range(20):
        l=IntervalLedger(); seen=set()
        for _ in range(100):
            a=rng.randrange(200); b=a+rng.randrange(30)
            positions=set(range(a,b))
            assert l.commit(a,b)==len(positions & seen)
            seen |= positions


@pytest.mark.parametrize('special',[req('barrier','WAITING_FOR_REMOTE_KVS'),req('barrier',computed=1),
                                    req('barrier',preempted=1),req('barrier',skip=True)])
def test_barrier_is_not_crossed(special):
    rows=[req('a'),req('b'),special,req('c'),req('d')]
    scores={'a':0,'b':10,'c':1,'d':2}
    out=reorder_segments(rows,lambda r:scores[r.request_id],lambda r:False)
    assert [r.request_id for r in out]==['b','a','barrier','d','c']


def test_stability_and_fairness():
    rows=[req('old'),req('warm1'),req('warm2')]
    scores={'old':0,'warm1':10,'warm2':10}
    out=reorder_segments(rows,lambda r:scores[r.request_id],lambda r:r.request_id=='old')
    assert [r.request_id for r in out]==['old','warm1','warm2']


def test_sufficient_victim_and_recompute_cost():
    a=Victim('small',10,100,100,0,100,1)
    b=Victim('large',100,1000,1000,0,100,30)
    c=Victim('expensive',100,1000,1000,0,100,80)
    assert choose_victim([a,b,c],'recompute',80)=='large'
    assert choose_victim([a,b,c],'recompute',200)=='small'
    assert choose_victim([a,b,c],'lcf',80)=='small'
    assert choose_victim([a,b,c],'default',80)=='expensive'


def test_zero_reclaim_and_nearly_complete():
    a=Victim('shared',0,1,1,0,100,1)
    b=Victim('nearly_done',100,100,100,99,100,10)
    c=Victim('fresh',100,100,100,0,100,10)
    assert choose_victim([a,b,c],'restore',50)=='fresh'
    assert choose_victim([a],'restore',50) is None


def test_resource_and_progress_ablations_change_real_choices():
    a=Victim('small_cheap',4,100,100,0,100,1)
    b=Victim('sufficient',20,100,100,0,100,10)
    assert choose_victim([a,b],'restore',10)=='sufficient'
    assert choose_victim([a,b],'restore',10,resource_aware=False)=='small_cheap'
    near=Victim('cheap_near_done',20,100,100,99,100,1)
    fresh=Victim('fresh',20,100,100,0,100,10)
    assert choose_victim([near,fresh],'restore',10)=='fresh'
    assert choose_victim([near,fresh],'restore',10,progress_aware=False)=='cheap_near_done'


@pytest.mark.parametrize('policy',['default','lcf','pr_score','recompute','restore'])
def test_empty_candidates(policy):
    assert choose_victim([],policy,10) is None
