"""Audit completed runs before performance interpretation."""
import argparse
import json
from pathlib import Path
from latest_v0320.common import DATA,ROOT,dump,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='correctness')
    ap.add_argument('--section');args=ap.parse_args()
    base=DATA/'raw/comparison'/args.stage
    runs=[]
    for path in sorted(base.glob('*/*/seed-*/metrics.json')):
        if args.section and path.relative_to(base).parts[0]!=args.section:continue
        d=path.parent;m=json.loads(path.read_text());manifest=json.loads((d/'manifest.json').read_text())
        rows=json.loads((d/'requests.json').read_text());a=manifest['args']
        if not args.stage.startswith('correctness'):
            snapshot=json.loads((DATA/'manifests/performance-source-snapshot.json').read_text())
            for absolute,expected_hash in snapshot['hashes'].items():
                path=Path(absolute)
                if path.parent==ROOT/'latest_v0320':
                    assert manifest['source_sha256'][str(path.relative_to(ROOT))]==expected_hash
                if path==DATA/'cost_models/native.json':assert manifest['cost_model_sha256']==expected_hash
        assert m['completed']==m['requests']==len(rows) and m['errors']==0
        assert all(r['success'] and len(r['token_ids'])==r['output_tokens'] for r in rows)
        assert digest(d/'trace.json')==manifest['trace_sha256']
        cfg=json.loads((d/'resolved_config.json').read_text())['vllm_config']
        s=cfg['scheduler_config']
        assert s['scheduler_reserve_full_isl'] and s['watermark']==0
        assert s['async_scheduling']==(a['track']=='default')
        initial=None
        if a['policy'] in {'NG','NT','Nobs'}:
            obs=json.loads((d/'scheduler.json').read_text());initial=obs['initial_state']
            if a['workload'] in {'A','C'}:
                assert all(initial[str(1000+g)]['gpu']==2048 for g in range(4)),initial
            if a['workload']=='C':
                assert all(initial[str(2000+g)]['gpu']==0 and initial[str(2000+g)]['cpu_end']==2048 for g in range(4)),initial
            if not args.stage.startswith('correctness'):
                counters=obs['counters'];assert not counters.get('unconfirmed_intervals',0),counters
                if m['native_delta']['preemptions']==0:
                    computed=m['native_delta']['prompt_tokens']-m['native_delta']['gpu_cached_tokens']-m['native_delta']['host_cached_tokens']
                    assert counters['prefill_executed_tokens']==computed,(computed,counters)
        if a['workload']=='C':
            assert m['native_delta']['host_cached_tokens']>0 and m['native_delta']['h2d_bytes']>0
        if (d/'cancel-check.json').exists():
            fault=json.loads((d/'cancel-check.json').read_text())
            assert fault['cancelled']==8 and fault['active_after_drain']==0 and fault['followup_output_tokens']==8
        runs.append({'args':a,'directory':str(d),'rows':rows,'initial':initial})
    from latest_v0320.pipeline import specs
    jobs=specs(args.stage)
    if args.section:jobs=[j for j in jobs if f"{j['workload']}-{j['blocks']}"==args.section]
    expected=len(jobs)
    assert len(runs)==expected,(args.stage,len(runs),expected)
    paired=[]
    for c in runs:
        if c['args']['policy'] in {'N0','Nobs'}:continue
        matches=[r for r in runs if r['args']['policy']=='N0' and all(r['args'][k]==c['args'][k] for k in ['workload','seed','blocks','track'])]
        if not matches:continue
        b=matches[0];bm={r['id']:r for r in b['rows']}
        mismatches=[r['id'] for r in c['rows'] if r['token_ids']!=bm[r['id']]['token_ids']]
        natural=[r for r in c['rows'] if r['class']=='natural']
        paired.append({'workload':c['args']['workload'],'policy':c['args']['policy'],
            'seed':c['args']['seed'],'requests':len(c['rows']),'exact':len(c['rows'])-len(mismatches),
            'natural_requests':len(natural),'natural_exact':sum(r['id'] not in mismatches for r in natural),
            'mismatches':mismatches,'baseline':b['directory'],'candidate':c['directory']})
    result={'stage':args.stage,'passed_artifact_checks':True,'completed_runs':len(runs),
        'completed_requests':sum(len(r['rows']) for r in runs),'output_pairs':paired,
        'unresolved_output_mismatches':sum(len(p['mismatches']) for p in paired),
        'scope':'native manager and candidate initial-state audits; N0 state verified by identical preparation and separate observed-control sequence'}
    name=args.stage+('-'+args.section if args.section else '')
    dump(ROOT/f'reports/latest_v0320/{name}-audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
