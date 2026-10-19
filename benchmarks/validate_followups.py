"""Audit natural-text determinism and default-off parity without GPU work."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
ROOT=Path(__file__).resolve().parents[1]


def main():
    parity=[]
    for seed in [17,29,43,71,101]:
        base=DATA/'raw/v3_parity/original_stock'/f'seed-{seed}'
        opt=DATA/'raw/v3_parity/instrumented_default'/f'seed-{seed}'
        br={r['id']:r['tokens'] for r in json.loads((base/'requests.json').read_text())}
        rr={r['id']:r['tokens'] for r in json.loads((opt/'requests.json').read_text())}
        bm=json.loads((base/'metrics.json').read_text());om=json.loads((opt/'metrics.json').read_text())
        assert set(br)==set(rr) and all(len(br[k])==len(rr[k]) for k in br)
        assert bm['trace_hash']==om['trace_hash']
        parity.append(dict(seed=seed,requests=len(br),exact_requests=sum(br[k]==rr[k] for k in br),
                           changed_requests=[k for k in br if br[k]!=rr[k]],baseline_tok_s=bm['metrics']['output_tok_s'],
                           instrumented_tok_s=om['metrics']['output_tok_s'],
                           throughput_change_percent=(om['metrics']['output_tok_s']/bm['metrics']['output_tok_s']-1)*100))
    text={m:json.loads((DATA/f'raw/v3_text_quality/{m}.json').read_text()) for m in ['reference','baseline','combined']}
    result={}
    for mode in ['baseline','combined']:
        single={r['id']:r['tokens'] for r in text[mode]['single']}
        reference={r['id']:r['tokens'] for r in text['reference']['single']}
        assert single==reference,'single-request text determinism failed'
        batch={r['id']:r['tokens'] for r in text[mode]['requests']}
        batch_ref={r['id']:r['tokens'] for r in text['reference']['requests']}
        assert set(batch)==set(batch_ref) and len(batch)==12
        assert all(0<len(t)<=64 for t in batch.values())
        result[mode]=dict(single_exact=len(single),batch_exact=sum(batch[k]==batch_ref[k] for k in batch),
                          batch_total=12,answers=text[mode]['requests'])
    tokens_equal=all(r['exact_requests']==r['requests'] for r in parity)
    if not tokens_equal:
        diagnostic=ROOT/'reports/v3_default_numerical_audit.json'
        if not diagnostic.exists():
            subprocess.run([sys.executable,str(ROOT/'benchmarks/validate_default_numerics.py')],check=True)
        numeric=json.loads(diagnostic.read_text())
        assert numeric['passed'] and numeric['decisions_identical']
    payload=dict(parity=parity,default_off_tokens_passed=tokens_equal,
                 default_off_correctness_passed=True,
                 default_off_throughput_change_mean_percent=statistics.mean(r['throughput_change_percent'] for r in parity),
                 text=result,single_request_determinism_passed=True,
                 note='Natural-text batch equality is reported, not assumed. Quality is not a public-dataset score.')
    (ROOT/'reports/v3_followup_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print('FOLLOWUP_AUDIT_PASS',payload['default_off_throughput_change_mean_percent'])


if __name__=='__main__':main()
