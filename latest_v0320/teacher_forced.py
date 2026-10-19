"""Same-history next-token probabilities at observed rollout divergences.

Diagnostic only: public LLM API, same prepared native cache paths, no logits patch.
This is a conditional-probability check, not full-vocabulary KL or proof of all
possible batched executions.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from latest_v0320.common import DATA,ROOT,PYTHON,MODEL,KV_BYTES,dump,prompt,clean_env

def make_llm(a, policy, output):
    from vllm import LLM
    from vllm.config import KVTransferConfig
    kwargs=dict(model=MODEL,dtype='float16',max_model_len=8192,gpu_memory_utilization=.75,
        num_gpu_blocks_override=a['blocks'],block_size=16,max_num_batched_tokens=1024,
        max_num_seqs=16,enable_prefix_caching=True,enable_chunked_prefill=True,disable_log_stats=False)
    if a['track']=='eager':kwargs.update(enforce_eager=True,async_scheduling=False)
    if a['workload']=='C':kwargs['kv_transfer_config']=KVTransferConfig(kv_connector='OffloadingConnector',
        kv_role='kv_both',kv_connector_extra_config={'cpu_bytes_to_use':32768*KV_BYTES,
            'block_size':256,'eviction_policy':'lru'})
    if policy!='N0':
        name='SyncAdmissionScheduler' if a['track']=='eager' else 'AsyncAdmissionScheduler'
        kwargs['scheduler_cls']='latest_v0320.scheduler.'+name
        os.environ['KV032_POLICY']=json.dumps({'admission':{'NG':'gpu','NT':'tier','Noff':'off'}[policy],
            'output':str(output)+'.scheduler.json','cost_model':str(DATA/'cost_models/native.json')})
    return LLM(**kwargs)

def worker(spec_path, policy, output):
    from vllm import SamplingParams
    spec=json.loads(Path(spec_path).read_text());a=spec['args']
    llm=make_llm(a,policy,output)
    def generate(ids,n=1,**extra):
        return llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=n,
            ignore_eos=True,**extra),use_tqdm=False)[0]
    def reset():
        deadline=time.monotonic()+30
        while not llm.reset_prefix_cache(reset_connector=True):
            if time.monotonic()>deadline:raise RuntimeError('teacher reset failed')
            time.sleep(.03)
    for n in [512,2048,4096,6144]:generate(prompt(99000+n,n),2)
    rows=[]
    for case in spec['cases']:
        reset()
        if a['workload']=='C':
            for g in range(4):generate(prompt(2000+g,2048))
            for g in range(4):generate(prompt(7000+g,4096))
        if a['workload'] in {'A','C'}:
            for g in range(4):generate(prompt(1000+g,2048))
        reference_ids=list(set([case['reference_token'],case['candidate_token']]))
        output_row=generate(case['history'],logprobs=len(reference_ids),logprob_token_ids=reference_ids)
        scores=output_row.outputs[0].logprobs[0]
        probs={str(token):float(value.logprob) for token,value in scores.items()}
        rows.append({'key':case['key'],'predicted_token':output_row.outputs[0].token_ids[0],
            'logprobs':probs,'finished':output_row.finished})
    dump(output,{'policy':policy,'args':a,'rows':rows,'scope':'same-history public-API one-step diagnostic'})
    llm.llm_engine.engine_core.shutdown()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stages',default='pilot,deployment_pilot,correctness_default')
    ap.add_argument('--worker');ap.add_argument('--policy');ap.add_argument('--output');args=ap.parse_args()
    if args.worker:return worker(args.worker,args.policy,args.output)
    groups=defaultdict(list)
    for stage in args.stages.split(','):
        base=DATA/'raw/comparison'/stage
        for p in sorted(base.glob('*/*/seed-*/metrics.json')):
            manifest=json.loads((p.parent/'manifest.json').read_text());a=manifest['args']
            if a['policy'] not in {'NG','NT','Noff'}:continue
            counterpart=p.parent.parent.parent/f"{a['track']}-N0"/p.parent.name
            if not (counterpart/'requests.json').exists():continue
            original={r['id']:r for r in json.loads((counterpart/'requests.json').read_text())}
            trace={r['id']:r for r in json.loads((counterpart/'trace.json').read_text())}
            for row in json.loads((p.parent/'requests.json').read_text()):
                b=original[row['id']]
                if row['token_ids']==b['token_ids']:continue
                pos=next(i for i,(x,y) in enumerate(zip(b['token_ids'],row['token_ids'])) if x!=y)
                key=f"{stage}:{a['workload']}:{a['seed']}:{row['id']}:{pos}"
                case={'key':key,'position':pos,'reference_token':b['token_ids'][pos],
                    'candidate_token':row['token_ids'][pos],
                    'history':trace[row['id']]['tokens']+b['token_ids'][:pos],
                    'class':row['class'],'baseline_run':str(counterpart),'candidate_run':str(p.parent)}
                group=(a['workload'],a['blocks'],a['track'],a['policy'])
                groups[group].append((a,case))
    summaries=[]
    for group,entries in groups.items():
        name='-'.join(map(str,group));dest=DATA/'raw/teacher'/name;dest.mkdir(parents=True,exist_ok=True)
        spec={'args':entries[0][0],'cases':[c for _,c in entries]};dump(dest/'spec.json',spec)
        for policy in ['N0',group[-1]]:
            out=dest/f'{policy}.json'
            if out.exists() and len(json.loads(out.read_text())['rows'])==len(entries):continue
            cmd=[str(PYTHON),'-m','latest_v0320.teacher_forced','--worker',str(dest/'spec.json'),
                '--policy',policy,'--output',str(out)]
            print('TEACHER_START',name,policy,len(entries),flush=True)
            with (dest/f'{policy}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=clean_env(True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=600)
                except subprocess.TimeoutExpired:code=124
                finally:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                if code:raise RuntimeError(f'teacher worker failed {name}/{policy}: {code}')
        b={r['key']:r for r in json.loads((dest/'N0.json').read_text())['rows']}
        c={r['key']:r for r in json.loads((dest/f'{group[-1]}.json').read_text())['rows']}
        for case in spec['cases']:
            br,cr=b[case['key']],c[case['key']]
            refs=[str(case['reference_token']),str(case['candidate_token'])]
            max_delta=max(abs(br['logprobs'][t]-cr['logprobs'][t]) for t in refs)
            gap=abs(br['logprobs'][refs[0]]-br['logprobs'][refs[1]])
            same_top1=br['predicted_token']==cr['predicted_token']
            near_tie=gap<=.05
            summaries.append({k:case[k] for k in ['key','position','class','reference_token','candidate_token']}|
                {'same_history_top1_equal':same_top1,'max_logprob_delta':max_delta,
                'reference_token_logprob_gap':gap,'near_tie_0_05':near_tie,
                'passed':same_top1 and max_delta<=.001 and near_tie,'evidence':str(dest)})
    result={'cases':summaries,'checked_divergences':len(summaries),'passed':all(r['passed'] for r in summaries),
        'scope':'one-step conditional probabilities at first divergence using baseline history; no full-vocabulary KL, not a proof of arbitrary batched bitwise equivalence'}
    dump(ROOT/'reports/latest_v0320/teacher-forced-audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if not result['passed']:raise SystemExit(2)

if __name__=='__main__':main()
