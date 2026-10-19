"""Default-off decision parity plus identical-history diagnostics for divergences."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    out=DATA/'raw/v3_default_diagnostic';out.mkdir(exist_ok=True)
    forced=[];targets=[];seeds=[];expected_positions=0
    for seed in [17,29,43,71,101]:
        original=json.loads((DATA/f'raw/v3_parity/original_stock/seed-{seed}/requests.json').read_text())
        candidate={r['id']:r['tokens'] for r in json.loads((DATA/f'raw/v3_parity/instrumented_default/seed-{seed}/requests.json').read_text())}
        changed=[r for r in original if r['tokens']!=candidate[r['id']]]
        if changed:
            seeds.append(seed);forced.extend(original)
            targets.extend(r['id'] for r in changed)
            expected_positions+=sum(len(r['tokens']) for r in changed)
    assert seeds
    token_path=out/'forced_tokens.json';token_path.write_text(json.dumps(forced))
    for flavor in ['stock','instrumented']:
        dest=out/flavor;dest.mkdir(exist_ok=True)
        if (dest/'logits.pt').exists() and (dest/'decisions.json').exists():continue
        env=os.environ.copy()
        env.update(PYTHONPATH=str(ROOT) if flavor=='stock' else f'{DATA}/vendor/optimized:{ROOT}',
            HF_HUB_OFFLINE='1',PYTHONHASHSEED='17',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='4',
            KV_CAPTURE_IDS=','.join(targets),KV_CAPTURE_POSITIONS='256',
            KV_FORCE_TOKENS=str(token_path),KV_LOGIT_CAPTURE=str(dest/'logits.pt'),
            KV_CAPTURE_DECISIONS=str(dest/'decisions.json'))
        cmd=[sys.executable,str(ROOT/'benchmarks/diagnose_quality.py'),'--workload','B','--requests','8',
            '--blocks','640','--seeds',','.join(map(str,seeds)),'--window','0','--output',str(dest)]
        with (dest/'run.log').open('w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    left=json.loads((out/'stock/decisions.json').read_text())
    right=json.loads((out/'instrumented/decisions.json').read_text())
    assert left==right,'default-off changed scheduling decisions with identical histories'
    import torch
    torch.set_num_threads(4)
    def load(flavor):
        rows=torch.load(out/flavor/'logits.pt',weights_only=False)
        return {(r['id'],r['position'],tuple(r['context'])):r['logits'].double() for r in rows}
    a=load('stock');b=load('instrumented')
    assert set(a)==set(b) and len(a)==expected_positions
    rows=[]
    for key,x in a.items():
        y=b[key];px=x.log_softmax(0);py=y.log_softmax(0)
        top=x.topk(2).values;margin=(top[0]-top[1]).item()
        rows.append(dict(id=key[0],position=key[1],max_abs=(x-y).abs().max().item(),
            kl=(px.exp()*(px-py)).sum().item(),max_probability_delta=(px.exp()-py.exp()).abs().max().item(),
            reference_margin=margin,top1_equal=x.argmax().item()==y.argmax().item()))
    result=dict(decision_steps=len(left),decisions_identical=True,positions=len(rows),
        mean_kl=sum(r['kl'] for r in rows)/len(rows),max_kl=max(r['kl'] for r in rows),
        max_probability_delta=max(r['max_probability_delta'] for r in rows),
        non_near_tie_top1_changes=sum(not r['top1_equal'] and r['reference_margin']>.05 for r in rows),rows=rows)
    result['passed']=(result['mean_kl']<=.0001 and result['max_kl']<=.001
        and result['max_probability_delta']<=.01 and result['non_near_tie_top1_changes']==0)
    (ROOT/'reports/v3_default_numerical_audit.json').write_text(json.dumps(result,indent=2))
    print('DEFAULT_NUMERICAL_AUDIT',{k:v for k,v in result.items() if k!='rows'},flush=True)
    assert result['passed']


if __name__=='__main__':main()
