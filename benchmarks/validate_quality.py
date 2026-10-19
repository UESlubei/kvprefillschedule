"""Teacher-forced GPU reference vs CPU baseline vs combined scheduling."""
import json
import os
from pathlib import Path
import subprocess
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--label',default='v3_teacher_forced')
    args=ap.parse_args()
    targets=','.join(f'C-17-{i:05d}' for i in [0,1,2,4,15])
    results={}
    for name,blocks,cpu,a,p in [('gpu_reference',4096,0,'fcfs','default'),
                                ('gpu_pressure',1024,0,'fcfs','default'),
                                ('cpu_baseline',1024,128,'fcfs','default'),
                                ('combined',1024,128,'tier','restore')]:
        out=DATA/'raw'/args.label/name; out.mkdir(parents=True,exist_ok=True)
        if args.audit_only:
            continue
        capture=out/'logits.pt'
        env=os.environ.copy()
        env.update(PYTHONPATH=f'{DATA}/vendor/optimized:{ROOT}',HF_HUB_OFFLINE='1',PYTHONHASHSEED='17',
                   VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='4',KV_CAPTURE_IDS=targets,
                   KV_CAPTURE_POSITIONS='64',KV_LOGIT_CAPTURE=str(capture))
        if name!='gpu_reference':
            env['KV_FORCE_TOKENS']=str(DATA/'raw'/args.label/'gpu_reference/seed-17/requests.json')
        else:
            env.pop('KV_FORCE_TOKENS',None)
        cmd=[sys.executable,str(ROOT/'benchmarks/diagnose_quality.py'),'--workload','C','--requests','24',
             '--blocks',str(blocks),'--cpu-blocks',str(cpu),'--admission',a,'--preemption',p,
             '--cost-model',str(DATA/'cost_models/qwen3-1.7b-3090.json'),'--trace','--output',str(out)]
        with (out/'run.log').open('w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('TEACHER_FORCED_CAPTURE',name,flush=True)
    import torch
    torch.set_num_threads(4)
    def load(name):
        rows=torch.load(DATA/'raw'/args.label/name/'logits.pt',weights_only=False)
        return {(r['id'],r['position'],tuple(r['context'])):r['logits'].float() for r in rows}
    reference=load('gpu_reference')
    assert len(reference)==320
    for name in ['gpu_pressure','cpu_baseline','combined']:
        candidate=load(name)
        assert set(candidate)==set(reference), 'different teacher-forced histories'
        rows=[]
        for key,x in reference.items():
            y=candidate[key]; difference=(x-y).abs()
            logpx=x.double().log_softmax(0);logpy=y.double().log_softmax(0)
            kl=(logpx.exp()*(logpx-logpy)).sum().item()
            top=x.topk(2).values
            rows.append({'id':key[0],'position':key[1],'max_abs':difference.max().item(),
                         'rms':difference.square().mean().sqrt().item(),'kl':kl,
                         'max_probability_delta':(logpx.exp()-logpy.exp()).abs().max().item(),
                         'top1_equal':x.argmax().item()==y.argmax().item(),
                         'reference_margin':(top[0]-top[1]).item()})
        result={'positions':len(rows),'max_abs':max(r['max_abs'] for r in rows),
                'max_rms':max(r['rms'] for r in rows),'mean_kl':sum(r['kl'] for r in rows)/len(rows),
                'top1_agreement':sum(r['top1_equal'] for r in rows)/len(rows),
                'max_kl':max(r['kl'] for r in rows),
                'max_probability_delta':max(r['max_probability_delta'] for r in rows),
                'non_near_tie_top1_changes':sum(not r['top1_equal'] and r['reference_margin']>.05 for r in rows),
                'rows':rows}
        # Raw absolute logits are retained, but are not invariant to a constant
        # offset. GPU-only batch/pressure control also exceeds the initial raw
        # .1 tolerance. Use normalized probability and non-near-tie criteria.
        result['criteria']={'mean_kl_max':.0001,'max_kl_max':.001,'max_probability_delta_max':.01,
                            'non_near_tie_top1_changes_max':0,'near_tie_margin':.05}
        result['passed']=(result['mean_kl']<=.0001 and result['max_kl']<=.001 and
                          result['max_probability_delta']<=.01 and result['non_near_tie_top1_changes']==0)
        results[name]=result
        print('TEACHER_FORCED_RESULT',name,{k:v for k,v in result.items() if k!='rows'},flush=True)
    dest=ROOT/'reports'/f'{args.label}_quality.json'
    dest.write_text(json.dumps(results,indent=2))
    assert all(r['passed'] for r in results.values())


if __name__=='__main__': main()
