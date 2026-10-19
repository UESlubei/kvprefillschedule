"""Run-level summaries, paired bootstrap intervals, interaction and token audits."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import random
import statistics

DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
ROOT=Path(__file__).resolve().parents[1]


def interval(xs):
    if not xs: return None
    rng=random.Random(20261019)
    means=sorted(statistics.mean(rng.choices(xs,k=len(xs))) for _ in range(10000))
    return {'mean':statistics.mean(xs),'std':statistics.stdev(xs) if len(xs)>1 else None,
            'ci95':[means[249],means[9749]],'n':len(xs)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--label',default='formal')
    args=ap.parse_args()
    groups=defaultdict(list)
    rows=[]
    for p in sorted((DATA/'raw'/args.label).rglob('metrics.json')):
        d=json.loads(p.read_text()); m=d['metrics']; h=d['scheduler']; c=h['counters']
        row={'section':p.parents[2].name,'group':d['group'],'workload':d['workload'],
             'seed':d['seed'],'gpu_blocks':d['gpu_blocks'],'cpu_blocks':d['cpu_blocks'],
             'completed':m['completed'],'output_tokens':m['output_tokens'],
             'output_tok_s':m['output_tok_s'],'ttft_p95_ms':m['ttft_ms']['p95'],
             'ttft_p99_ms':m['ttft_ms']['p99'],'ttft_mean_ms':m['ttft_ms']['mean'],
             'itl_p99_ms':m['itl_ms']['p99'],'e2e_p95_ms':m['e2e_ms']['p95'],
             'arrival_lag_p95_ms':m['arrival_lag_ms']['p95'],
             'schedule_p95_ms':h['timings']['schedule_ms']['p95'],
             'recovery_mean_ms':h['timings'].get('recovery_ms',{}).get('mean'),
             **{k:c.get(k,0) for k in ['gpu_hit_tokens','cpu_matched_tokens','cpu_loaded_tokens',
                 'h2d_bytes','h2d_submitted_bytes','d2h_submitted_bytes','preemptions','recomputed_tokens',
                 'executed_tokens','prefill_executed_tokens','computed_at_preemption','reorder_calls','reorder_changes',
                 'non_tail_victims','reserved_victim_rollbacks','prefix_loss_tokens',
                 'cpu_prediction_mismatches','fallback_uncalibrated_gpu']},
             'trace_hash':d['trace_hash'],'source':str(p)}
        row['recompute_ratio']=row['recomputed_tokens']/max(1,row['executed_tokens'])
        rows.append(row)
        groups[(row['gpu_blocks'],row['cpu_blocks'],row['group'])].append(row)
    aggregate=[]
    fields=['output_tok_s','ttft_p95_ms','ttft_p99_ms','itl_p99_ms','e2e_p95_ms','schedule_p95_ms',
            'preemptions','recomputed_tokens','recompute_ratio','gpu_hit_tokens','cpu_loaded_tokens',
            'h2d_bytes','prefix_loss_tokens','arrival_lag_p95_ms','prefill_executed_tokens']
    for (gpu,cpu,group),rs in sorted(groups.items()):
        aggregate.append({'gpu_blocks':gpu,'cpu_blocks':cpu,'group':group,'runs':len(rs),
                          'metrics':{k:interval([r[k] for r in rs]) for k in fields}})
    comparisons=[]
    for key,rs in sorted(groups.items()):
        gpu,cpu,group=key
        work,policy=group.split('-')
        base='B1' if cpu else 'G0'
        if policy==base: continue
        baseline={r['seed']:r for r in groups.get((gpu,cpu,f'{work}-{base}'),[])}
        paired=[]
        for r in rs:
            b=baseline.get(r['seed'])
            if b:
                if r['trace_hash']!=b['trace_hash']: raise ValueError('unpaired trace hash')
                paired.append((b,r))
        diffs={k:interval([r[k]-b[k] for b,r in paired]) for k in fields}
        relative={k:interval([(r[k]-b[k])/b[k]*100 for b,r in paired if b[k]!=0]) for k in fields}
        # Compare actual tokens separately; tiny float/batch changes are not assumed equal.
        audit={'requests':0,'exact_requests':0,'token_positions':0,'equal_positions':0}
        for b,r in paired:
            bp=Path(b['source']).with_name('requests.json'); rp=Path(r['source']).with_name('requests.json')
            bd={x['id']:x['tokens'] for x in json.loads(bp.read_text())}
            for rr in json.loads(rp.read_text()):
                bt=bd[rr['id']]; rt=rr['tokens']
                audit['requests']+=1; audit['exact_requests']+=int(bt==rt)
                audit['token_positions']+=max(len(bt),len(rt))
                audit['equal_positions']+=sum(a==b for a,b in zip(bt,rt))
        comparisons.append({'gpu_blocks':gpu,'cpu_blocks':cpu,'baseline':f'{work}-{base}',
                            'group':group,'pairs':len(paired),'absolute_delta':diffs,
                            'relative_delta_percent':relative,'output_audit':audit})
    interactions=[]
    for work in ['C','D']:
        by={g:{r['seed']:r for r in groups.get((1024,128,f'{work}-{g}'),[])} for g in ['B1','O1','O2','O3']}
        seeds=set.intersection(*(set(x) for x in by.values()))
        for metric in ['output_tok_s','ttft_p95_ms','recomputed_tokens']:
            vals=[by['O3'][s][metric]-by['O1'][s][metric]-by['O2'][s][metric]+by['B1'][s][metric] for s in sorted(seeds)]
            interactions.append({'workload':work,'metric':metric,'interaction':interval(vals)})
    out=ROOT/'reports'; out.mkdir(exist_ok=True)
    payload={'runs':len(rows),'rows':rows,'groups':aggregate,'comparisons':comparisons,'interactions':interactions}
    (out/f'{args.label}_summary.json').write_text(json.dumps(payload,indent=2))
    md=['# 配对实验数据汇总','',f'已发现 {len(rows)} 个完成 Run。数字为 Run 级均值，尾延迟是各 Run 分位数的均值。','',
        '| GPU Blocks | CPU Blocks | 组别 | Runs | Output tok/s | P95 TTFT ms | P99 ITL ms | 重算 Tokens | 抢占次数 | Scheduler P95 ms |',
        '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in aggregate:
        m=r['metrics']
        md.append(f"| {r['gpu_blocks']} | {r['cpu_blocks']} | {r['group']} | {r['runs']} | "+
                  ' | '.join(f"{m[k]['mean']:.3f}" for k in ['output_tok_s','ttft_p95_ms','itl_p99_ms','recomputed_tokens','preemptions','schedule_p95_ms'])+' |')
    md+=['','## 相对同场景基线的配对变化','',
         '| GPU Blocks | 组别 | 配对数 | tok/s Δ % [95% CI] | P95 TTFT Δ % [95% CI] | 重算 Δ Tokens [95% CI] |',
         '| ---: | --- | ---: | --- | --- | --- |']
    def fmt(d):
        return 'N/A' if d is None else f"{d['mean']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"
    for r in comparisons:
        md.append(f"| {r['gpu_blocks']} | {r['group']} | {r['pairs']} | {fmt(r['relative_delta_percent']['output_tok_s'])} | {fmt(r['relative_delta_percent']['ttft_p95_ms'])} | {fmt(r['absolute_delta']['recomputed_tokens'])} |")
    md+=['','CI 为按配对 Run Bootstrap 的探索性区间；样本数见配对数列，少量重复不足以证明稳定收益。吞吐正向、延迟及重算负向通常表示改善。',
         '主矩阵为本地同步 V1 Engine Trace，不包含 HTTP；客户端提交延迟已单独保存。短 Run 的 P99 只用于探索。','']
    (out/f'{args.label}_summary.md').write_text('\n'.join(md))
    print('completed_runs',len(rows),'groups',len(groups))


if __name__=='__main__': main()
