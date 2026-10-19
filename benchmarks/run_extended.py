"""V3 development scans and five-seed single-factor ablations; serial GPU."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from run_matrix import POLICIES

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
DEV=[211,223,239,251,263]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    out=DATA/'raw/v3_extended';out.mkdir(exist_ok=True)
    measured=DATA/'cost_models/qwen3-1.7b-3090.json'
    model=json.loads(measured.read_text())
    rate=statistics.median((p['length']-p['cached'])/p['ms'] for p in model['prefill'] if p['batch']==1 and p['cached']==0)
    proxy=json.loads(json.dumps(model));proxy['model_kind']='token_proxy_same_measured_h2d'
    proxy['prefill']=[{**p,'ms':(p['length']-p['cached'])/rate} for p in model['prefill']]
    proxy_path=DATA/'cost_models/v3_token_proxy.json';proxy_path.write_text(json.dumps(proxy,indent=2))
    groups=defaultdict(list)
    plan=[]
    def add(section,work,ident,policy='O3',seeds=DEV,gpu=1024,cpu=128,gran=256,
            count=64,cost=measured,baseline='C-O3',comparison_family=None,**options):
        a,p=POLICIES[policy]
        key=(section,gpu,cpu,gran,count,str(cost))
        for seed in seeds:
            row=dict(id=ident,seed=seed,workload=work,admission=a,preemption=p,**options)
            groups[key].append(row)
            plan.append(dict(section=section,group=ident,seed=seed,baseline=baseline,
                             run=str(out/section/ident/f'seed-{seed}'),
                             comparison_family=comparison_family or section,
                             gpu=gpu,cpu=cpu,cpu_block_size=gran,requests=count,cost=str(cost)))
    # Same CPU connector remains active for every metadata/cost ablation.
    for ident,opts in [('C-O3',{}),('C-NoCpu',dict(probe_cpu=False)),
                       ('C-NoFair',dict(fairness=False)),('C-NoResource',dict(resource_aware=False)),
                       ('C-NoProgress',dict(progress_aware=False))]:
        add('single_factor','C',ident,**opts)
    # Model changes need separate engines; alternate those engines per seed.
    for i,seed in enumerate(DEV):
        cases=[('calibrated',measured),('proxy',proxy_path)]
        if i%2:cases.reverse()
        for case,cost in cases:
            add(f'cost_pair_{seed}_{case}','C','C-Calibrated' if case=='calibrated' else 'C-Proxy',
                seeds=[seed],cost=cost,comparison_family='cost_model',
                baseline='C-Calibrated' if case=='calibrated' else f'cost_pair_{seed}_calibrated/C-Calibrated')
    # Gate and window scans use only the independent development seed.
    for k in [0,4,8,16,32]:
        add('window','A',f'A-K{k}',policy='G1',seeds=DEV[:1],cpu=0,window=k,baseline='A-K0')
    for t in [0,.5,.75,.9]:
        add('threshold','A',f'A-T{t}',policy='G1',seeds=DEV[:1],cpu=0,threshold=t,baseline='A-T0')
    for capacity in [32,64,128,256]:
        for pol in ['B1','O3']:
            add(f'capacity_{capacity}','C',f'C-{pol}',policy=pol,seeds=DEV[:1],cpu=capacity,baseline='C-B1')
    for budget in [640,1024,2048,4096]:
        for pol in ['G0','G1']:
            add(f'budget_{budget}','A',f'A-{pol}',policy=pol,seeds=DEV[:1],cpu=0,gpu=budget,baseline='A-G0')
    for con in [1,4,8,16,32]:
        for pol in ['G0','G5']:
            add(f'concurrency_{con}','B',f'B-{pol}',policy=pol,seeds=DEV[:1],cpu=0,
                concurrency=con,baseline='B-G0')
    for gran in [64,128,256]:
        for pol in ['B1','O3']:
            add(f'granularity_{gran}','C',f'C-{pol}',policy=pol,cpu=32768//gran,gran=gran,baseline='C-B1')
    for prefix in [1024,2048,4096]:
        for suffix in [128,512]:
            for pol in ['G0','G1']:
                add(f'prefix_{prefix}_suffix_{suffix}','A',f'A-{pol}',policy=pol,seeds=DEV[:1],cpu=0,
                    prefix_length=prefix,suffix_length=suffix,baseline='A-G0')
    for pol in ['G0','G4']:
        add('inverse_output','B',f'B-{pol}',policy=pol,cpu=0,gpu=640,output_relation='inverse',baseline='B-G0')
    add('inverse_output','B','B-NoProgress',policy='G4',cpu=0,gpu=640,
        output_relation='inverse',progress_aware=False,baseline='B-G4')
    for pol in ['B1','O3']:
        add('sessions_8','D',f'D-{pol}',policy=pol,sessions=8,baseline='D-B1')
    # Same-budget G0/B1 isolates Offload rather than confounding it with GPU size.
    add('offload_off','C','C-G0',policy='G0',cpu=0,baseline='offload_on/C-B1')
    add('offload_on','C','C-B1',policy='B1',baseline='C-B1')
    # Native capacity is ~1 request/s for B; fixed traces cover low/near/overload.
    for arrival in [.55,.88,1.1,2.2]:
        for pol in ['G0','G4']:
            add(f'arrival_{arrival}','B',f'B-{pol}',policy=pol,seeds=DEV[:1],cpu=0,gpu=640,
                arrival_rate=arrival,baseline='B-G0')
    for arrival in [1.6,2.56,3.2,6.4]:
        for pol in ['G0','G1']:
            add(f'admission_arrival_{arrival}','A',f'A-{pol}',policy=pol,seeds=DEV[:1],cpu=0,gpu=640,
                arrival_rate=arrival,baseline='A-G0')
    for con in [1,4]:
        for pol in ['G0','G5']:
            add(f'lowload_{con}','F',f'F-{pol}',policy=pol,seeds=DEV[:1],cpu=0,gpu=4096,
                concurrency=con,baseline='F-G0')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    if args.plan_only:
        print('EXTENDED_PLAN',len(plan),'runs',len(groups),'engine suites')
        return
    for mode in ['baseline','combined']:
        if not (DATA/f'raw/v3_app_demo/{mode}.json').exists():
            with (out/f'app-{mode}.log').open('a') as log:
                subprocess.run([sys.executable,str(ROOT/'benchmarks/validate_app_demo.py'),'--mode',mode],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
    statuses=[]
    for (section,gpu,cpu,gran,count,cost),rows in groups.items():
        dest=out/section;dest.mkdir(exist_ok=True)
        missing=[r for r in rows if not (dest/r['id']/f"seed-{r['seed']}"/'metrics.json').exists()]
        if not missing:continue
        # Rotate strategy order across seeds. Preserve the same trace in each pair.
        ordered=[]
        for i,seed in enumerate(DEV):
            rs=[r for r in missing if r['seed']==seed]
            if rs:
                rs=rs[i%len(rs):]+rs[:i%len(rs)]
                ordered.extend(rs)
        suite=dest/'suite.json';suite.write_text(json.dumps(ordered,indent=2))
        cmd=[sys.executable,str(ROOT/'benchmarks/run_trace.py'),'--suite',str(suite),
             '--requests',str(count),'--blocks',str(gpu),'--cpu-blocks',str(cpu),
             '--cpu-block-size',str(gran),'--cost-model',cost,'--output',str(dest),'--timeout','900']
        print('EXTENDED_START',section,len(ordered),flush=True)
        t=time.time()
        with (dest/'suite.log').open('a') as log:
            child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        statuses.append(dict(section=section,runs=len(ordered),returncode=child.returncode,seconds=time.time()-t))
        (out/'status.json').write_text(json.dumps(statuses,indent=2))
        print('EXTENDED_END',statuses[-1],flush=True)
        if child.returncode:raise SystemExit(child.returncode)


if __name__=='__main__':main()
