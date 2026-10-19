import importlib.util
from pathlib import Path
import sys

path=Path(__file__).resolve().parents[1]/'benchmarks/run_trace.py'
spec=importlib.util.spec_from_file_location('trace_generator',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_session_history_is_nested_and_causal():
    rows=module.workload('D',256,17)
    by_id={r['id']:r for r in rows}
    for r in rows:
        if r.get('depends_on'):
            previous=by_id[r['depends_on']]
            assert previous['session_id']==r['session_id']
            assert r['round']==(previous['round']+1)%4
            if r['round']:
                assert r['tokens'][:len(previous['tokens'])]==previous['tokens']
            else:
                assert r['conversation_cycle']==previous['conversation_cycle']+1
            assert r['class']==previous['class']
            assert r['tool_wait_s']>0
        assert len(r['tokens'])+r['output']<=8192


def test_group_pairing_is_deterministic_and_length_valid():
    for kind in 'ABCDEF':
        a=module.workload(kind,64,17)
        assert a==module.workload(kind,64,17)
        assert a!=module.workload(kind,64,29)
        assert all(len(r['tokens'])+r['output']<=8192 for r in a)


def test_eight_session_trace_and_inverse_output_budget():
    rows=module.workload('D',64,17,sessions=8)
    byid={r['id']:r for r in rows}
    assert len({r['session_id'] for r in rows})==8
    assert all(byid[r['depends_on']]['session_id']==r['session_id'] for r in rows if r['depends_on'])
    rows=module.workload('B',64,17,output_relation='inverse')
    assert all(r['output']==(256 if len(r['tokens'])<=512 else 128 if len(r['tokens'])<=2048 else 64) for r in rows)


def test_cpu_target_mix_is_30_30_40():
    rows=module.workload('C',100,17)
    assert {c:sum(r['class']==c for r in rows) for c in ['gpu_warm','cpu_warm','cold']}=={
        'gpu_warm':30,'cpu_warm':30,'cold':40}
