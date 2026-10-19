"""Paired per-run estimates; never count individual requests as hardware repeats."""
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import random
import statistics
from latest_v0320.common import DATA,ROOT,dump,percentiles

REPORT=ROOT/'reports/latest_v0320'

def ci(values):
    if len(values)<2:return [None,None]
    rng=random.Random(179239)
    boots=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
    return [boots[249],boots[9749]]

def main():
    lock=(REPORT/'summary.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    base=DATA/'raw/comparison';runs=[]
    for path in sorted(base.glob('*/*/*/seed-*/metrics.json')):
        manifest=json.loads((path.parent/'manifest.json').read_text());args=manifest['args']
        metrics=json.loads(path.read_text())
        # This native metric is initial prefill accounting, not a ledger of
        # repeated model execution after preemption. Keep the scopes separate.
        from latest_v0320.run_http import parse_metrics,metric
        before=parse_metrics((path.parent/'metrics-before.txt').read_text())
        after=parse_metrics((path.parent/'metrics-after.txt').read_text())
        stem='vllm:prompt_tokens_by_source_total';labels={'source':'local_compute'}
        if any(r['name']==stem for r in after):
            metrics['native_delta']['initial_local_compute_tokens']=metric(after,stem,labels)-metric(before,stem,labels)
        runs.append(dict(stage=path.relative_to(base).parts[0],args=args,metrics=metrics,
            trace_sha256=manifest['trace_sha256'],path=str(path.parent)))
    groups=defaultdict(list)
    for run in runs:
        a=run['args'];groups[(run['stage'],a['workload'],a['blocks'],a['track'])].append(run)
    comparisons=[]
    for key,rows in groups.items():
        if key[0].startswith('correctness'):continue
        policies=sorted({r['args']['policy'] for r in rows}-{'N0'})
        for policy in policies:
            pairs=[]
            for r in rows:
                if r['args']['policy']!=policy:continue
                match=[b for b in rows if b['args']['policy']=='N0' and b['args']['seed']==r['args']['seed']]
                if not match:continue
                b=match[0];assert b['trace_sha256']==r['trace_sha256']
                a,bm=r['metrics'],b['metrics']
                change={}
                for name,f in [('p95_ttft',lambda m:m['ttft_ms']['p95']),('p99_ttft',lambda m:m['ttft_ms']['p99']),
                        ('mean_e2e',lambda m:m['e2e_ms']['mean']),('throughput',lambda m:m['output_tok_s']),
                        ('initial_local_compute',lambda m:m['native_delta'].get('initial_local_compute_tokens')),
                        ('cold_mean',lambda m:m['by_class'].get('cold',{}).get('mean'))]:
                    av,bv=f(a),f(bm)
                    change[name]=(av/bv-1)*100 if av is not None and bv else None
                pairs.append({'seed':r['args']['seed'],'baseline':b['path'],'candidate':r['path'],'change_percent':change,
                    'baseline_p95_ms':bm['ttft_ms']['p95'],'candidate_p95_ms':a['ttft_ms']['p95']})
            if not pairs:continue
            summary={}
            for name in pairs[0]['change_percent']:
                values=[p['change_percent'][name] for p in pairs if p['change_percent'][name] is not None]
                summary[name]={'mean_percent':statistics.mean(values),'ci95_percent':ci(values)} if values else None
            comparisons.append(dict(stage=key[0],workload=key[1],blocks=key[2],track=key[3],
                candidate=policy,pairs=len(pairs),summary=summary,paired_runs=pairs))
        # C requires an additional NT/NG estimate, not just NT/N0.
        if key[1]=='C':
            ng={r['args']['seed']:r for r in rows if r['args']['policy']=='NG'}
            changes=[]
            for r in rows:
                if r['args']['policy']=='NT' and r['args']['seed'] in ng:
                    b=ng[r['args']['seed']];assert b['trace_sha256']==r['trace_sha256']
                    changes.append((r['metrics']['ttft_ms']['p95']/b['metrics']['ttft_ms']['p95']-1)*100)
            if changes:comparisons.append(dict(stage=key[0],workload='C',blocks=key[2],track=key[3],
                candidate='NT/NG',pairs=len(changes),summary={'p95_ttft':{'mean_percent':statistics.mean(changes),'ci95_percent':ci(changes)}}))
    dump(REPORT/'comparison_summary.json',{'runs':runs,'comparisons':comparisons})
    lines=['# vLLM 0.32.0 优化收益实测','',
        '数据来自独立官方 cu129 环境、同一 Qwen3-1.7B FP16。N0 为未修改原生代码；NG 为 GPU 准入，NT 为分层准入。每 Run 独立 HTTP 服务，外部 SSE 计时。',
        '',f'当前已归档 {len(runs)} Run；本表只使用已经完成且配对 Trace Hash 相同的结果。负数表示延迟下降，正数表示延迟增加；吞吐正数表示增长。',
        '', '| 阶段 | 场景/Blocks | 路径 | 对照 | 配对数 | P95 TTFT 变化 | 探索性95%区间 | 吞吐变化 | Cold均值变化 |',
        '| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |']
    finished=REPORT/'comparison-completion-receipt.json'
    if finished.exists():
        receipt=json.loads(finished.read_text())
        lines[2:2]=[f"本轮正式机制 55 Run 和默认部署 40 Run 已完成并通过产物审计。完成时间：{receipt['completed_at']}。长测、7B 和满显存自动 KV 预算不在本轮完成范围。",'']
    else:
        formal=sum(r['stage']=='formal' for r in runs)
        deployment=sum(r['stage']=='deployment_formal' for r in runs)
        lines[2:2]=[f'正式复核进行中：机制 {formal}/55 Run，默认部署 {deployment}/40 Run。已完成的 Pilot 与未完成的正式矩阵分行展示；表格自动更新，图暂为已完成 Pilot。','']
    def pct(x):return f'{x:+.2f}%' if x is not None else 'N/A'
    for c in comparisons:
        s=c['summary'];v=s['p95_ttft'];bounds=v['ci95_percent'];t=s.get('throughput')
        interval='N/A' if bounds[0] is None else '['+', '.join(pct(x) for x in bounds)+']'
        cold=s.get('cold_mean')
        lines.append(f"| {c['stage']} | {c['workload']}/{c['blocks']} | {c['track']} | {c['candidate']} | {c['pairs']} | {pct(v['mean_percent'])} | {interval} | {pct(t['mean_percent']) if t else 'N/A'} | {pct(cold['mean_percent']) if cold else 'N/A'} |")
    lines+=['','## 实际延迟与缓存机制','',
        '| 阶段/场景/路径 | 策略 | Run数 | 平均Run P95(ms) | GPU命中Tokens/Run | Host命中Tokens/Run | H2D MiB/Run | 原生初次本地Prefill Tokens/Run | 诊断实际Prefill Tokens/Run |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,rows in groups.items():
        if key[0].startswith('correctness'):continue
        for policy in sorted({r['args']['policy'] for r in rows}):
            rs=[r for r in rows if r['args']['policy']==policy]
            def avg(fn):
                vs=[fn(r['metrics']) for r in rs];vs=[v for v in vs if v is not None]
                return statistics.mean(vs) if vs else None
            prefill=avg(lambda m:m.get('project_counters',{}).get('prefill_executed_tokens'))
            def number(x):return f'{x:.1f}' if x is not None else 'N/A'
            vals=[avg(lambda m:m['ttft_ms']['p95']),avg(lambda m:m['native_delta']['gpu_cached_tokens']),
                avg(lambda m:m['native_delta']['host_cached_tokens']),
                avg(lambda m:m['native_delta']['h2d_bytes']/2**20 if m['native_delta']['h2d_bytes'] is not None else None),
                avg(lambda m:m['native_delta'].get('initial_local_compute_tokens')),prefill]
            lines.append(f"| {key[0]}/{key[1]}-{key[2]}/{key[3]} | {policy} | {len(rs)} | "+' | '.join(number(x) for x in vals)+' |')
    lines+=['','单对/短跑只作探索；正式结论需完成预定矩阵，并检查缓存命中、实际搬运、错误和 Cold 公平性。固定开放负载的吞吐变化不能替代最大服务容量。',
        '', '逐 Run 和逐请求数据：`/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/raw/comparison`。',
        '', '原生未直接观测的实际 Prefill/重复执行计数记为不可观测；详细机制只在相同观测条件下比较。',
        '', '原生 `prompt_tokens_by_source{source="local_compute"}` 为初次 Prefill 本地计算量，不含抢占后重复执行；不能当作总模型执行账本。诊断列才来自提交/完成区间计数。',
        '', 'HTTP 原生空闲连接超时在 Pilot 及后续基线/候选统一为 120 秒；此前观测阶段保留默认 5 秒且全部成功。边界断连的失败开发 Run 单独归档，未纳入性能统计。',
        '', '分层排序使用本次新环境同步 Eager 一输出服务成本表；默认执行路径沿用这一固定参考表，端到端收益实测，但不声称已验证默认编译路径的成本预测精度。',
        '', '参数、策略源码及成本表 Hash、Trace Hash、服务日志、原生指标前后快照全部保留。旧版 0.11.2 数据不混入此表。']
    qa=REPORT/'quality-followup-assessment.json'
    if qa.exists():
        q=json.loads(qa.read_text());maxp=max(c['max_probability_delta'] for c in q['cases'])
        lines+=['','## 输出与缓存正确性证据','',
            '同步/默认路径正确性 Run 合计 320 个请求，192 个基线/候选配对输出完全一致，包含 36 个自然问答对照。压力 Pilot 中另有 8 个合成请求出现逐 Token 分歧，未将它们静默当作完全一致。',
            '',f'首次分歧位置的同历史条件概率最大绝对差为 {maxp:.7f}。原先严格的 logprob 1e-3 / Top-1 一致检查未全部通过，原始失败记录保留；额外原生重复运行复现了对应 Top-1 翻转。补充概率尺度和原生波动对照支持继续性能实验，并非预注册准确率保证。',
            '', '独立长上下文自然文本检索：同步/默认 × 原生/分层候选 × 三种缓存路径，共 36/36 回答正确，各模式内逐 Token 一致。CPU 路径每次实际恢复 2048 Tokens / 224 MiB，避免仅验证 GPU 命中。',
            '', '这些是有限正确性证据，不宣称任意 Batch 位级等价，也不宣称完成全词表 KL 验证。',
            '', '[原严格检查](teacher-forced-audit.json) · [补充原生重复与语义检查](quality-followup-assessment.json)']
    lines+=['','[P95 对比图](latest_p95_comparison.png) · [配对数据与区间](comparison_summary.json)']
    (REPORT/'latest_comparison_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
