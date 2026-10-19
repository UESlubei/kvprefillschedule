"""Final artifacts after the two formal matrices; no additional GPU workloads."""
from datetime import datetime
import json
import subprocess
from pathlib import Path
from latest_v0320.common import DATA,ROOT,PYTHON,digest,dump,clean_env

def main():
    report=ROOT/'reports/latest_v0320'
    stages={'correctness':6,'correctness_default':4,'observation':8,'pilot':12,
        'deployment_pilot':8,'formal':55,'deployment_formal':40}
    receipts={}
    for stage,expected in stages.items():
        subprocess.run([str(PYTHON),'-m','latest_v0320.audit_comparison','--stage',stage],
            cwd=ROOT,env=clean_env(True),stdout=subprocess.DEVNULL,check=True)
        receipt=json.loads((report/f'{stage}-audit.json').read_text())
        assert receipt['completed_runs']==expected
        receipts[stage]={'runs':expected,'requests':receipt['completed_requests'],
            'output_mismatches':receipt['unresolved_output_mismatches']}
    frozen=json.loads((DATA/'manifests/performance-source-snapshot.json').read_text())
    assert all(digest(path)==h for path,h in frozen['hashes'].items())
    old=json.loads(Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/v3_full_source_snapshot.json').read_text())
    assert all(digest(ROOT/rel)==h for rel,h in old['hashes'].items())
    assert all(digest(path)==h for path,h in old.get('vendor_hashes',{}).items())
    subprocess.run([str(PYTHON),str(ROOT/'latest_v0320/audit_validation.py')],
        cwd=ROOT,env=clean_env(True),stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['/home/sdn/.local/bin/uv','pip','check','--python',str(PYTHON)],check=True)
    quality=json.loads((report/'quality-followup-assessment.json').read_text())
    assert quality['supported_for_performance_experiment']
    result={'completed_at':datetime.now().astimezone().isoformat(),'passed_artifact_audit':True,
        'stages':receipts,'performance_requests':sum(r['requests'] for k,r in receipts.items() if not k.startswith('correctness')),
        'correctness_requests':sum(r['requests'] for k,r in receipts.items() if k.startswith('correctness')),
        'original_strict_probability_check_passed':quality['original_strict_logprob_check_passed'],
        'supplementary_native_repeat_and_semantic_support':True,
        'quality_scope':'192 paired correctness outputs and 36 long-context semantic outputs; 8 pilot divergence locations checked, not all formal rollout differences',
        'not_run':['independent >=2000-request long runs','7B model extension','full-memory automatic KV budget comparison'],
        'frozen_hashes':frozen['hashes']}
    dump(report/'comparison-completion-receipt.json',result)
    subprocess.run([str(PYTHON),'-m','latest_v0320.summarize'],cwd=ROOT,
        env=clean_env(True),stdout=subprocess.DEVNULL,check=True)
    # Use the existing analysis interpreter, keeping inference dependencies locked.
    subprocess.run(['/home/sdn/anaconda3/envs/ljc/bin/python','-m','latest_v0320.plot_results'],
        cwd=ROOT,env=clean_env(True),check=True)
    progress=ROOT/'PROGRESS.md';text=progress.read_text()
    text=text.replace('现在串行运行固定 5 对种子的 55 Run 正式机制矩阵及 40 Run 默认部署矩阵；该阶段尚无完整正式结论。',
        '本轮固定 5 对种子的 55 Run 正式机制矩阵及 40 Run 默认部署矩阵均已完成，产物审计通过；最新数据见新版实测报告与完成记录。长测、7B 和满显存自动 KV 预算尚未执行。')
    if 'reports/latest_v0320/comparison-completion-receipt.json' not in text:
        text+='\n[新版本轮完成记录](reports/latest_v0320/comparison-completion-receipt.json)\n'
    progress.write_text(text)
    readme=ROOT/'README.md';text=readme.read_text()
    text=text.replace('GPU/Tier 准入已移植，正确性通过，性能对照执行中。',
        'GPU/Tier 准入已移植，本轮 55 Run 机制和 40 Run 默认部署正式对照完成，数据审计通过；有限质量证据及浮点分歧详见报告。')
    readme.write_text(text)
    plan=ROOT/'最新版vLLM对比实验方案_V2_CUDA129.md';text=plan.read_text()
    text=text.replace('独立默认路径正确性、同历史概率诊断与正式配对验证进行中。',
        '默认路径正确性、原生重复与语义缓存检查已完成，本轮 55 Run 机制和 40 Run 默认部署正式矩阵完成；长测、7B 和满显存自动 KV 预算仍未执行。')
    text=text.replace('| 55 Run 正式机制矩阵 | 已安排，先通过概率诊断 |',
        '| 55 Run 正式机制矩阵 | 已完成并通过数据审计，质量范围见报告 |')
    text=text.replace('| 默认部署、长测、可选 7B | 8 Run 默认 Pilot 已完成；正式待执行，长测/7B 未执行 |',
        '| 默认部署、长测、可选 7B | 默认 Pilot 及 40 Run 正式已完成；长测/7B 未执行 |')
    plan.write_text(text)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
