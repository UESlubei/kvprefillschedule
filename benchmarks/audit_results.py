"""Independent completeness, manifest, paired-trace and fairness audit."""
from collections import defaultdict
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
from kv_sched.policy import percentiles

DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',default='formal')
    parser.add_argument('--expected-runs',type=int,default=130)
    parser.add_argument('--pair-by-section',action='store_true')
    parser.add_argument('--source-snapshot',type=Path)
    args=parser.parse_args()
    frozen=json.loads(args.source_snapshot.read_text()) if args.source_snapshot else None
    errors=[];runs=[];hashes=defaultdict(set)
    for path in sorted((DATA/'raw'/args.label).rglob('metrics.json')):
        result=json.loads(path.read_text())
        if frozen:
            manifest_path=path.with_name('manifest.json')
            if not manifest_path.exists():errors.append(f'{path}: missing per-run manifest')
            else:
                manifest=json.loads(manifest_path.read_text())
                if manifest.get('release_snapshot',{}).get('sha256')!=frozen['sha256']:
                    errors.append(f'{path}: mixed frozen release')
                if manifest['config']!=result['config']:errors.append(f'{path}: config mismatch')
        trace=json.loads(path.with_name('trace.json').read_text())
        requests=json.loads(path.with_name('requests.json').read_text())
        byid={r['id']:r for r in requests}
        expected={r['id']:r for r in trace}
        if set(byid)!=set(expected):errors.append(f'{path}: incomplete request IDs')
        if result['metrics']['completed']!=len(trace):errors.append(f'{path}: completion count')
        for rid,r in byid.items():
            if len(r['tokens'])!=expected[rid]['output']:errors.append(f'{path}: output length {rid}')
        digest=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest()
        if digest!=result['trace_hash']:errors.append(f'{path}: trace hash mismatch')
        key=(result['gpu_blocks'],result['cpu_blocks'],result['workload'],result['seed'])
        if args.pair_by_section:key=(path.parents[2].name,*key)
        hashes[key].add(digest)
        counters=result['scheduler']['counters']
        if counters.get('fallback_uncalibrated_gpu',0):errors.append(f'{path}: cost model fallback')
        if counters.get('unconfirmed_executions',0):errors.append(f'{path}: unconfirmed execution')
        if counters.get('h2d_bytes',0)!=counters.get('h2d_submitted_bytes',0):errors.append(f'{path}: H2D count mismatch')
        if counters.get('recomputed_tokens',0)>counters.get('executed_tokens',0):errors.append(f'{path}: impossible recompute')
        admissions=result['scheduler']['admissions']
        fair={}
        for cls in sorted({r['intended_class'] for r in requests}):
            rows=[r for r in requests if r['intended_class']==cls]
            fair[cls]={'requests':len(rows),'ttft_ms':percentiles([r['ttft_ms'] for r in rows]),
                       'e2e_ms':percentiles([r['e2e_ms'] for r in rows]),
                       'admission_wait_ms':percentiles([admissions[r['id']]['wait_ms'] for r in rows])}
        ended=datetime.fromtimestamp(path.stat().st_mtime,tz=timezone.utc)
        audit={'run':str(path.parent),'group':result['group'],'seed':result['seed'],
               'source_end_utc_from_file_mtime':ended.isoformat(),
               'source_end_shanghai_from_file_mtime':ended.astimezone(timezone(timedelta(hours=8))).isoformat(),
               'timestamp_note':'file mtime follows metric serialization; not an exact runner clock timestamp',
               'fairness_by_fixed_trace_class':fair}
        path.with_name('audit.json').write_text(json.dumps(audit,indent=2))
        runs.append(audit)
    for key,values in hashes.items():
        if len(values)>1:errors.append(f'unpaired trace group {key}')
    payload={'label':args.label,'completed_runs':len(runs),'expected_runs':args.expected_runs,
             'errors':errors,'runs':runs,'data_checks_passed':not errors,
             'passed':not errors and len(runs)==args.expected_runs}
    (ROOT/'reports'/f'{args.label}_completeness_audit.json').write_text(json.dumps(payload,indent=2))
    print('AUDIT',len(runs),'runs',len(errors),'errors')
    if errors:raise RuntimeError(errors[:5])
    if len(runs)!=args.expected_runs:
        print('INCOMPLETE',len(runs),'/',args.expected_runs)
        raise SystemExit(2)


if __name__=='__main__':main()
