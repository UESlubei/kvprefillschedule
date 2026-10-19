"""Fail-closed completion audit, Nsight exports and reproducible final report."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')


def main():
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    validation=json.loads((DATA/'cost_models/validation.json').read_text())
    pairs=[(a,b) for i,a in enumerate(validation) for b in validation[i+1:]]
    ordering=sum((a['ms']-b['ms'])*(a['predicted_ms']-b['predicted_ms'])>0 for a,b in pairs)
    (reports/'v3_calibration_validation_analysis.json').write_text(json.dumps({
        'heldout_points':len(validation),'mean_relative_error':statistics.mean(r['relative_error'] for r in validation),
        'max_relative_error':max(r['relative_error'] for r in validation),
        'pairwise_correct_orderings':ordering,'pairwise_orderings':len(pairs),
        'out_of_range_rule':'proportional endpoint scaling; not independently time-validated for 1-16 token recovery',
        'rows':validation},indent=2))
    def run(script,*args):
        subprocess.run([sys.executable,str(ROOT/'benchmarks'/script),*map(str,args)],check=True)
    if not (DATA/'v3_model_manifest.json').exists():run('capture_model_manifest.py')
    plan=json.loads((DATA/'raw/v3_extended/plan.json').read_text())
    for label,count,extra in [('v3_full',130,[]),('v3_long',10,[]),
                              ('v3_extended',len(plan),['--pair-by-section'])]:
        run('audit_results.py','--label',label,'--expected-runs',count,
            '--source-snapshot',DATA/'v3_full_source_snapshot.json',*extra)
        audit=json.loads((reports/f'{label}_completeness_audit.json').read_text())
        assert audit['passed'],f'incomplete {label}: {audit["completed_runs"]}/{count}'
    for item in plan:
        path=Path(item['run'])/'metrics.json'
        result=json.loads(path.read_text())
        assert result['group']==item['group'] and result['seed']==item['seed']
        assert result['metrics']['completed']==item['requests']
        assert result['gpu_blocks']==item['gpu'] and result['cpu_blocks']==item['cpu']
        assert result['cpu_block_size']==item['cpu_block_size']
    for label in ['v3_full','v3_long']:run('summarize.py','--label',label)
    full=json.loads((reports/'v3_full_summary.json').read_text())
    controls=[]
    for c in full['comparisons']:
        if c['group'].startswith(('E-','F-')):
            delta=c['relative_delta_percent']
            throughput=delta['output_tok_s']['mean'];ttft=delta['ttft_p95_ms']['mean']
            controls.append({'group':c['group'],'throughput_delta_percent':throughput,
                             'ttft_p95_delta_percent':ttft,
                             'development_3percent_target_met':throughput>=-3 and ttft<=3})
    (reports/'v3_control_assessment.json').write_text(json.dumps(controls,indent=2))
    run('validate_followups.py')
    quality=json.loads((reports/'v3_teacher_forced_quality.json').read_text())
    assert set(quality)=={'gpu_pressure','cpu_baseline','combined'}
    assert all(r['passed'] and r['positions']==320 for r in quality.values())
    for label,count in [('v3_faults',256),('v3_stress_long',2048)]:
        result=json.loads((DATA/'raw'/label/'summary.json').read_text())
        assert result['passed'] and result['submitted']==count
        assert result['completed']+result['cancelled']==count
    for mode in ['baseline','combined']:
        app=json.loads((DATA/f'raw/v3_app_demo/{mode}.json').read_text())
        assert app['completed']==32 and app['actual_generated_history'] and app['cpu_loaded_tokens']>0
        assert app['source_snapshot']['sha256']==json.loads((DATA/'v3_full_source_snapshot.json').read_text())['sha256']
        assert len(app['rounds'])==4 and all(len(turn['requests'])==8 for turn in app['rounds'])
    http=list((DATA/'raw/v3_http').rglob('metrics.json'))
    assert len(http)==10
    for path in http:
        d=json.loads(path.read_text())
        assert d['metrics']['requests']==128
        assert d['source_snapshot']['sha256']==json.loads((DATA/'v3_full_source_snapshot.json').read_text())['sha256']
    profile=DATA/'profiles/v3_combined.nsys-rep'
    assert profile.exists() and profile.stat().st_size>5000
    cmd=['/usr/local/cuda/bin/nsys','stats','--report',
         'cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum,nvtx_sum',
         '--format','csv','--force-export','true',str(profile)]
    proc=subprocess.run(cmd,capture_output=True,text=True)
    (reports/'v3_nsys_stats.csv.txt').write_text(proc.stdout)
    (reports/'v3_nsys_stats.stderr.txt').write_text(proc.stderr)
    assert proc.returncode==0,proc.stderr
    assert 'does not contain CUDA kernel data' not in proc.stdout+proc.stderr
    run('analyze_profile.py')
    repro=reports/'v3_reproducibility';repro.mkdir(exist_ok=True)
    for src in [DATA/'environment_manifest.json',DATA/'requirements-lock.txt',
                DATA/'v3_machine_manifest.json',DATA/'v3_model_manifest.json',
                DATA/'v3_full_source_snapshot.json',DATA/'cost_models/qwen3-1.7b-3090.json',
                DATA/'cost_models/validation.json',DATA/'raw/v3_extended/plan.json']:
        shutil.copy2(src,repro/src.name)
    run('build_full_report.py')
    python=shutil.which('python')
    assert python
    subprocess.run([python,str(ROOT/'benchmarks/plot_full.py')],check=True)
    receipt={'passed':True,'completed_unix':time.time(),'formal_runs':130,
             'extended_runs':len(plan),'long_runs':10,'http_runs':10,
             'source_snapshot':json.loads((DATA/'v3_full_source_snapshot.json').read_text())['sha256'],
             'meaning':'mandatory experiments and data/correctness audits complete; does not assert a performance gain',
             'control_assessment':controls}
    (reports/'v3_completion_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('V3_COMPLETED',receipt,flush=True)


if __name__=='__main__':main()
