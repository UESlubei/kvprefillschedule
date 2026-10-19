"""Build complete tables from audited runs; no fabricated completion or gains."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

from summarize import interval

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
OUT=ROOT/'reports'


def read(path):return json.loads(Path(path).read_text())


def fields(d):
    m=d['metrics'];s=d['scheduler'];c=s['counters']
    return dict(output_tok_s=m['output_tok_s'],ttft_p95_ms=m['ttft_ms']['p95'],
        ttft_p99_ms=m['ttft_ms']['p99'],itl_p99_ms=m['itl_ms']['p99'],e2e_p95_ms=m['e2e_ms']['p95'],
        planned_ttft_p95_ms=m['planned_ttft_ms']['p95'],arrival_lag_p95_ms=m['arrival_lag_ms']['p95'],
        schedule_p95_ms=s['timings']['schedule_ms']['p95'],
        **{k:c.get(k,0) for k in ['gpu_hit_tokens','cpu_loaded_tokens','h2d_bytes','recomputed_tokens',
             'prefill_executed_tokens','preemptions','native_ready_promotions','cpu_prediction_mismatches']})


def fmt(d):
    return 'N/A' if d is None else f"{d['mean']:+.2f}% [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"


def main():
    full=read(OUT/'v3_full_summary.json');long=read(OUT/'v3_long_summary.json')
    assert full['runs']==130 and long['runs']==10
    plan=read(DATA/'raw/v3_extended/plan.json')
    indexed={(r['section'],r['group'],r['seed']):r for r in plan}
    ext=[];grouped=defaultdict(list)
    for item in plan:
        p=Path(item['run'])/'metrics.json';d=read(p)
        row={**item,**fields(d),'trace_hash':d['trace_hash']}
        ext.append(row)
        base=item['baseline'].split('/')
        section,ident=(base if len(base)==2 else (item['section'],base[0]))
        if section==item['section'] and ident==item['group']:continue
        bitem=indexed[(section,ident,item['seed'])];b=read(Path(bitem['run'])/'metrics.json')
        assert d['trace_hash']==b['trace_hash'],'extended comparison has different input'
        bf=fields(b)
        family=item.get('comparison_family',item['section'])
        baseline_label=ident if family!=item['section'] else item['baseline']
        grouped[(family,item['group'],baseline_label)].append(
            {k:(row[k]-bf[k])/bf[k]*100 for k in row.keys() & bf.keys() if bf[k]})
    comparisons=[]
    for (section,group,baseline),rows in grouped.items():
        keys=set.intersection(*(set(x) for x in rows))
        comparisons.append(dict(section=section,group=group,baseline=baseline,pairs=len(rows),
            relative_delta_percent={k:interval([r[k] for r in rows]) for k in sorted(keys)}))
    (OUT/'v3_extended_summary.json').write_text(json.dumps(dict(runs=len(ext),rows=ext,comparisons=comparisons),indent=2))
    md=['# 开发 Trace 扫描与消融','',
        '种子 211/223/239/251/263 与正式、长测种子分开；范围扫描只用 211，不能据单次扫描作稳定收益结论。每 Run 64 请求。',
        '配对区间为探索性 Bootstrap，未作多重比较校正。', '',
        '| 扫描 | 策略 | 基线 | 配对数 | 吞吐变化 [95% CI] | P95 TTFT 变化 [95% CI] |',
        '| --- | --- | --- | ---: | --- | --- |']
    for c in comparisons:
        delta=c['relative_delta_percent']
        md.append(f"| {c['section']} | {c['group']} | {c['baseline']} | {c['pairs']} | {fmt(delta.get('output_tok_s'))} | {fmt(delta.get('ttft_p95_ms'))} |")
    (OUT/'v3_extended_summary.md').write_text('\n'.join(md)+'\n')
    fairness=[];composition=[];confusion=defaultdict(int);mechanisms=defaultdict(list)
    for source in full['rows']:
        path=Path(source['source']);d=read(path);requests=read(path.with_name('requests.json'))
        mechanisms[(d['gpu_blocks'],d['cpu_blocks'],d['group'])].append(d['scheduler']['counters'])
        for cls in sorted({r['intended_class'] for r in requests}):
            rs=[r for r in requests if r['intended_class']==cls]
            fairness.append(dict(group=d['group'],gpu=d['gpu_blocks'],cpu=d['cpu_blocks'],seed=d['seed'],
                intended_class=cls,requests=len(rs),ttft_mean_ms=statistics.mean(r['ttft_ms'] for r in rs),
                e2e_mean_ms=statistics.mean(r['e2e_ms'] for r in rs)))
        for r in requests:confusion[(d['group'],r['intended_class'],r['class'])]+=1
        ts=d['scheduler']['timings']
        totals={k:v['count']*(v['mean'] or 0) for k,v in ts.items()}
        composition.append(dict(group=d['group'],gpu=d['gpu_blocks'],cpu=d['cpu_blocks'],seed=d['seed'],
            schedule_ms=totals.get('schedule_ms',0),gpu_probe_ms=totals.get('probe_gpu_ms',0),
            cpu_probe_ms=totals.get('probe_cpu_ms',0),reorder_ms=totals.get('reorder_ms',0),
            victim_ms=totals.get('victim_ms',0)))
    (OUT/'v3_fairness_and_overhead.json').write_text(json.dumps(dict(fairness=fairness,composition=composition,
        intended_to_actual=[dict(group=k[0],intended=k[1],actual=k[2],requests=v) for k,v in confusion.items()]),indent=2))
    quality=read(OUT/'v3_teacher_forced_quality.json');follow=read(OUT/'v3_followup_audit.json')
    tests=(DATA/'pipeline_v3/tests.log').read_text().strip().splitlines()[-1]
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    report=['# vLLM KV 分层感知调度：完整实验报告','',f'生成时间：{now}。固定 RTX 3090 / Qwen3-1.7B FP16 / vLLM 0.11.2。', '',
        '## 1. 完成范围与结论口径','',
        f'完成 V3 的两个优化模块及同步单卡实验：130 个正式 Run（每 Run 256 请求），{len(ext)} 个开发扫描/消融 Run（每 Run 64 请求），10 个独立长测 Run（每 Run 2000 请求），10 个 HTTP Run（每 Run 128 请求），以及默认行为、自然文本、Teacher-forced 数值和取消压力核验。',
        'GPU/CPU 缓存淘汰仍使用原生 LRU；没有新增搬运 Kernel、主动预取、完整运行请求 Swap 协议、Async 或多卡支持。可选 Mistral 规模验证与 Window=64 扩展未列为必做，本轮未运行。',
        '全部性能结论由以下实际配对结果给出；吞吐正向、延迟/重算负向表示改善。区间跨零时不视为稳定改善。短 Run 的 P99 只作探索；正式 5 次和长测 5 次均为独立 Run，未将单个请求当作独立硬件重复。', '',
        '## 2. 正式矩阵','',
        '同一预算、Trace、种子、缓存准备和公共兼容修复下比较。表中绝对值是 Run 均值，尾延迟是各 Run 分位数的均值。', '',
        '| GPU / CPU Blocks | 组别 | n | GPU Hit Tokens | CPU Loaded Tokens | 实际重算 | P95 TTFT ms | P99 ITL ms | Output tok/s | Scheduler P95 ms |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    highlight=[]
    for label,payload,group,gpu in [('主矩阵共享前缀',full,'A-G1',640),
        ('独立 2000 请求长测',long,'A-G1',640),('CPU 联合 / C',full,'C-O3',1024),
        ('CPU 联合 / D',full,'D-O3',1024),('重算代价 / B',full,'B-G4',640)]:
        c=next(c for c in payload['comparisons'] if c['group']==group and c['gpu_blocks']==gpu)
        delta=c['relative_delta_percent']
        highlight.append(f"- {label}：吞吐 {fmt(delta['output_tok_s'])}；P95 TTFT {fmt(delta['ttft_p95_ms'])}。")
    report[4:4]=['### 核心实测结果','',*highlight,'',
        '以上均为同种子配对变化及探索性 95% CI；负向结果同样保留。先结合完成率、分类等待与实际执行量，再判断收益边界。','']
    for row in full['groups']:
        m=row['metrics'];vals=' | '.join(f"{m[k]['mean']:.3f}" for k in ['gpu_hit_tokens','cpu_loaded_tokens','recomputed_tokens','ttft_p95_ms','itl_p99_ms','output_tok_s','schedule_p95_ms'])
        report.append(f"| {row['gpu_blocks']} / {row['cpu_blocks']} | {row['group']} | {row['runs']} | {vals} |")
    report+=['','### 配对变化','', '| GPU / CPU | 对照 | n | 吞吐变化 [95% CI] | P95 TTFT 变化 [95% CI] | 重算变化 [95% CI] |',
             '| --- | --- | ---: | --- | --- | --- |']
    for c in full['comparisons']:
        delta=c['relative_delta_percent']
        report.append(f"| {c['gpu_blocks']} / {c['cpu_blocks']} | {c['group']} vs {c['baseline']} | {c['pairs']} | {fmt(delta['output_tok_s'])} | {fmt(delta['ttft_p95_ms'])} | {fmt(delta['recomputed_tokens'])} |")
    report+=['','组别：G0=FCFS/Default，G1=GPU Admission，G2=LCF，G3=PR-score，G4=Recompute，G5=GPU Admission+Recompute；B1=CPU Offload+FCFS/Default，B2=GPU Admission，O1=Tier Admission，O2=Restore，O3=Tier+Restore。CPU 组始终保留 Offload。', '',
        '主矩阵 A/B/C/E 为开放到达的固定突发 Trace；每 16 个请求一组，A/E 为 3.2 req/s、B 为 1.1 req/s、C 为 3.5 req/s。D 为固定 16 条因果请求流、每 4 轮重新开始会话周期，前轮完成后等待 0.1 秒；F 为真正闭环并发 4。低于处理能力时吞吐受提供负载限制，不能把近似相同吞吐解释为最大服务容量相同。', '',
        '## 3. 独立长测、交互与公平性','',
        '长测使用全新种子 307/331/347/367/389；A 的 G0/G1 每 Run 2000 请求、固定 3.2 req/s，GPU 640 Blocks。', '',
        '| 对照 | 配对数 | 吞吐变化 [95% CI] | P95 TTFT 变化 [95% CI] | P99 TTFT 变化 [95% CI] |',
        '| --- | ---: | --- | --- | --- |']
    for c in long['comparisons']:
        delta=c['relative_delta_percent']
        report.append(f"| {c['group']} vs {c['baseline']} | {c['pairs']} | {fmt(delta['output_tok_s'])} | {fmt(delta['ttft_p95_ms'])} | {fmt(delta['ttft_p99_ms'])} |")
    report+=['','2×2 交互采用绝对量：O3−O1−O2+B1。不能相加百分比。完整交互区间见 [主汇总 JSON](v3_full_summary.json)，分类等待和调度组成见 [公平性与开销 JSON](v3_fairness_and_overhead.json)。', '',
        '| GPU / CPU | 策略 | 固定输入类别 | 平均 TTFT ms | 平均 E2E ms |',
        '| --- | --- | --- | ---: | ---: |']
    fairgroups=defaultdict(list)
    for r in fairness:fairgroups[(r['gpu'],r['cpu'],r['group'],r['intended_class'])].append(r)
    for (gpu,cpu,group,cls),rs in sorted(fairgroups.items()):
        if group not in ['A-G0','A-G1','C-B1','C-B2','C-O1','C-O2','C-O3','D-B1','D-O3']:continue
        report.append(f"| {gpu} / {cpu} | {group} | {cls} | {statistics.mean(r['ttft_mean_ms'] for r in rs):.2f} | {statistics.mean(r['e2e_mean_ms'] for r in rs):.2f} |")
    report+=['','分类以固定 Trace 的目标类别为主，另保存目标→实际缓存路径映射，避免策略改变类别成员后造成假公平。老化与 2000 ms 阈值约束准入顺序，不能保证 TTFT 小于 2 秒。GPU Hit 含重复准入的复用，不是唯一请求首次命中率。Baseline 不执行优化 Probe，不能把其缺失的 PrefixLoss 当作 0 与优化组比较。原生资源冲突测试已证明 128→96 Tokens 的破坏与 128 Tokens 的保护。', '',
        '## 4. 扫描、消融与 CPU Offload 因素','',
        '完整结果见 [开发扫描与消融](v3_extended_summary.md) / [JSON](v3_extended_summary.json)。覆盖 GPU 640/1024/2048/4096，客户端并发 1/4/8/16/32，Window 0/4/8/16/32，Threshold 0/0.5/0.75/0.9，CPU Blocks 32/64/128/256，公平性开关，CPU Probe 开关，测量成本/线性 Token 成本，资源满足性与进度因子，CPU Block 64/128/256 Tokens（总 CPU Tokens 固定 32768），A 的 1K/2K/4K 前缀与 128/512 后缀，B 的反相关输出预算，D 的 8 条请求流，以及到达速率范围。',
        '关键单因子、CPU 粒度、反相关输出和 Session 对照用 5 个开发种子；范围扫描只用 1 个开发种子，不能据其选出的最好一行宣称稳定收益。Offload On/Off 在相同 GPU 1024、同一 C Trace 下配对，单独保存比较。CPU Probe 关闭时仅禁止优化器读 CPU 元数据，真实 Offload 仍执行。公平性关闭同时关闭 Aging 与超时优先，未将 aging=0 错写成完全关闭公平性。', '',
        '## 5. 正确性、兼容性与故障核验','',f'测试记录：`{tests}`。',
        '公共兼容修复包括部分 CPU Chunk 映射与抢占 Store Flush 的上游适配，以及本地的生成历史恢复、取消时持有 DMA 源/目标直至完成、双向/重复回调、空闲完成轮询、无 Running 可释放内存时推进已完成加载的持有者。基线与优化共同使用，不把这些修复算作调度收益。失败复现和通过测试均保留；详见 `configs/local_compatibility_fixes.json`。',
        '固定版本取消 API 为 `abort_request([rid])`。本轮初态校验确保请求立即移除且未执行；旧版 quick/formal 的错误参数可能改变缓存初态，旧数据全部排除。所有非队尾预留回滚同时撤销尚未执行的 KV Hash 登记，并恢复 Token Budget 和索引。', '',
        '| 路径 | 同历史位置数 | Top-1 一致率 | 最大概率差 | 最大 KL | 非近似并列 Top-1 变化 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in quality.items():
        report.append(f"| {name} | {r['positions']} | {r['top1_agreement']:.4%} | {r['max_probability_delta']:.6g} | {r['max_kl']:.6g} | {r['non_near_tie_top1_changes']} |")
    report+=['','上述门槛在本轮运行前固定：最大概率差≤0.01，平均 KL≤1e-4，最大 KL≤1e-3，参考 Margin>0.05 时 Top-1 不变。原始 Logits、RMS 和失败过的旧阈值记录保留，不声称所有原始 Logits 逐位一致。',
        f"默认关闭策略与原生 GPU-only 对照的自然输出 {sum(r['exact_requests'] for r in follow['parity'])}/{sum(r['requests'] for r in follow['parity'])} 个请求 Token 完全一致；小样本吞吐变化均值 {follow['default_off_throughput_change_mean_percent']:+.2f}%。若自然输出有分歧，额外逐 Step 比较相同历史的调度决策和 Logits，不把分歧直接归因于浮点误差。证据见 [默认/自然文本审计](v3_followup_audit.json) 和 `v3_default_numerical_audit.json`。", '',
        '中文、英文、代码和数学固定集先逐请求验证确定性，再做 CPU/压力混合 Batch。答案原文保留，64 Token 上限会截断部分代码；不当作公共数据集质量分数或语义正确率。', '',
        '| 自然文本路径 | 单请求完全一致 | Batch 完全一致 | Batch 总数 |',
        '| --- | ---: | ---: | ---: |']
    for name,r in follow['text'].items():report.append(f"| {name} | {r['single_exact']} | {r['batch_exact']} | {r['batch_total']} |")
    report+=['', '### 实际生成历史的多轮应用示例', '',
        '每条路径运行 8 个独立会话、每会话 4 轮。上一轮真实生成的答案写回下一轮上下文，中间插入无关前缀以触发 GPU 淘汰和 CPU 恢复。事实检查只针对本例的项目代号和预算，不是公共质量基准。策略间输入历史可能因答案不同而分歧，因此本例不进入配对性能统计。', '',
        '| 路径 | 完成请求 | 事实检查通过 / 总数 | CPU Loaded Tokens |',
        '| --- | ---: | ---: | ---: |']
    app_rows=[]
    for mode in ['baseline','combined']:
        app=read(DATA/f'raw/v3_app_demo/{mode}.json')
        report.append(f"| {mode} | {app['completed']} | {app['fact_checks']} / {app['fact_checks_total']} | {app['cpu_loaded_tokens']} |")
        app_rows.append({k:app[k] for k in ['mode','completed','fact_checks','fact_checks_total',
                                          'cpu_loaded_tokens','actual_generated_history','paired_performance']})
    (OUT/'v3_application_demo_summary.json').write_text(json.dumps(app_rows,indent=2))
    report+=['','取消核验包括 4 Cohorts / 256 请求和 32 Cohorts / 2048 请求；每 Cohort 60 完成、4 在等待/运行/加载/存储状态取消。故障注入只延迟原生完成通知，不改写搬运 Kernel；结束检查 GPU/CPU 引用、Pending Jobs、旧完成集合均归零。该诊断不进入吞吐统计。原生 Worker 不支持故障 DMA 继续运行，非法加载直接报错并拒绝生成性能结果；不能将其写成故障自动恢复。', '',
        '## 6. HTTP、Profile 与成本模型','',
        'HTTP/SSE 使用独立 API Server 与并发异步客户端，B Trace 每 Run 128 请求、5 对种子、1.1 req/s。首个非空内容记 TTFT，SSE 间隔只称 Inter-chunk Latency，不称逐 Token ITL。服务性能与本地同步 Engine 数据分开，客户端发送滞后单独报告。',
        'Nsight Systems 与 cProfile 在独立短 Trace 上采集；范围包含 Seed 与正式诊断，不进入主性能统计。CPU/GPU/搬运数据见 [Nsight CSV 导出](v3_nsys_stats.csv.txt)、数据目录 `profiles/v3_combined.nsys-rep` 与 `profiles/v3_combined_cpu.json`。Scheduler 聚合计时覆盖主主体、排除末尾进度更新；完整函数时间另由 cProfile 保留。Probe 属于 Reorder 子时间，不能重复相加。恢复耗时包含排队至首次恢复计算，不等同于纯 H2D 时间。',
        'Prefill 40 个训练点、4 个留出点，每点 10 次采样；留出误差约 1%～14%。原生多层 H2D 有效斜率约 6.06 GB/s、固定开销约 0.50 ms，连续 Copy 单独记录。成本表不会主动选择理想 min(重算,恢复)，而是估计 Connector 实际会走的路径。',
        '保守 Victim 估计假定 GPU 前缀可能不保留；真实 BlockPool 经常仍能复用。短恢复不在校准网格内，线性外推及多请求竞争也有限制。这些可解释策略收益边界，尚未将修正这些估计的未来工作写成已实现收益。', '',
        '## 7. 复现、来源与边界','',
        '模型 Revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；vLLM 基础 Commit `275de34170654274616082721348b7edd9741d32`；PyTorch 2.9.0+cu128；FP16、TP=PP=1、Full Attention、同步调度、Eager、max_model_len=8192、GPU Block=16、Batch Tokens=1024、max_num_seqs=16。Window=8、Threshold=0、Aging=0.05、max_wait=2000 ms。固定 8 个可用 CPU Core、OMP=4，不锁 GPU 时钟。',
        '所有模型权重/Tokenizer 文件 SHA256、实际 Overlay 文件、补丁、环境锁、Trace 和逐 Run Manifest 保存在 [复现资料](v3_reproducibility/) 与原始数据目录。方案基于现有二进制做私有源码 Overlay，没有修改原安装。',
        '参考 [Admission #54625](https://github.com/vllm-project/vllm/pull/54625)、[PR-score #56897](https://github.com/vllm-project/vllm/pull/56897)、[LCF #53723](https://github.com/vllm-project/vllm/pull/53723)、[重算观测 #57698](https://github.com/vllm-project/vllm/pull/57698)，固定 Head SHA 见配置。PR-score 的 Token 为 `Request.num_tokens`（Prompt+已生成），进度项不截断；项目 Recompute/Restore 的进度下限为 0.05。均为固定旧版本的参考策略适配，不声称完整 Cherry-pick 复现，也不移用其他硬件 PR 的性能数字。',
        '单命令恢复：在项目目录 `source env.sh` 后执行 `"$KV_EXP_PYTHON" benchmarks/full_pipeline.py`。阶段间校验冻结哈希，已完成项跳过；核心变化必须使用独立发布目录。原始根目录为 `/mnt/ondeviceinfer/experiments/vllm_kv_scheduling`，各阶段见 `pipeline_v3/status.json`。',
        '结果为合成 Token、固定输出预算和单张 3090 上的实验，不能直接外推真实业务、其他模型、公共语义质量、Async 或多卡。主结果、负向结果和单次探索都保留；不以设计目标代替实测。', '']
    http=defaultdict(list)
    for path in (DATA/'raw/v3_http').rglob('metrics.json'):
        d=read(path);m=d['metrics']
        http[d['policy']].append(dict(seed=d['seed'],output_tok_s=m['output_tok_s'],
            ttft_p95_ms=m['ttft_ms']['p95'],e2e_p95_ms=m['e2e_ms']['p95'],
            inter_chunk_p99_ms=m['inter_chunk_ms']['p99'],submission_lag_p95_ms=m['submission_lag_ms']['p95'],
            trace_hash=d['trace_hash']))
    assert all(len(http[p])==5 for p in ['default','recompute'])
    paired=[];base={r['seed']:r for r in http['default']}
    for r in http['recompute']:
        b=base[r['seed']];assert r['trace_hash']==b['trace_hash']
        paired.append({k:(r[k]/b[k]-1)*100 for k in ['output_tok_s','ttft_p95_ms','e2e_p95_ms']})
    http_summary=dict(rows=dict(http),paired_relative_delta_percent={k:interval([r[k] for r in paired]) for k in paired[0]})
    (OUT/'v3_http_summary.json').write_text(json.dumps(http_summary,indent=2))
    report+=['## 8. HTTP 实测表与导出图','',
        '| 策略 | n | Output tok/s | P95 TTFT ms | P95 E2E ms | P99 Inter-chunk ms | P95 发送滞后 ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for policy,rows in http.items():
        report.append(f'| {policy} | {len(rows)} | '+' | '.join(f'{statistics.mean(r[k] for r in rows):.3f}' for k in
            ['output_tok_s','ttft_p95_ms','e2e_p95_ms','inter_chunk_p99_ms','submission_lag_p95_ms'])+' |')
    for k,v in http_summary['paired_relative_delta_percent'].items():report.append(f'- HTTP {k} 配对变化：{fmt(v)}。')
    report+=['','同一端口只绑定本机 127.0.0.1；启用本机开发路由用于预热后重置缓存，配置和每个服务器命令均保留。','']
    report+=['### 无复用和低负载对照', '',
        '开发目标为吞吐均值退化不超过 3%、P95 TTFT 均值增加不超过 3%。该目标单独判断，数据和正确性审计通过不代表性能目标通过；同时保留上文的配对区间。', '',
        '| 对照 | 吞吐变化 | P95 TTFT 变化 | 3% 开发目标 |',
        '| --- | ---: | ---: | --- |']
    for c in read(OUT/'v3_control_assessment.json'):
        target='达到' if c['development_3percent_target_met'] else '未达到'
        report.append(f"| {c['group']} | {c['throughput_delta_percent']:+.2f}% | {c['ttft_p95_delta_percent']:+.2f}% | {target} |")
    calibration=read(OUT/'v3_calibration_validation_analysis.json')
    report+=['',
        f"成本模型 4 个留出点的平均相对误差 {calibration['mean_relative_error']:.2%}、最大相对误差 {calibration['max_relative_error']:.2%}；点间耗时排序 {calibration['pairwise_correct_orderings']}/{calibration['pairwise_orderings']} 对正确。该排序检查仅针对留出点，不能代表所有 Victim 恢复排序的准确率。",'']
    report+=['### 机制与公共兼容路径触发量','',
        '以下为每 Run 均值。公共 Ready-holder 推进由基线和优化共享，不能归入优化器重排次数。', '',
        '| GPU / CPU | 组别 | Reorder Calls | Reorder Changes | Non-tail Victims | 预留回滚 | 公共 Ready 推进 | CPU 预测失配 | H2D MiB |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (gpu,cpu,group),rows in sorted(mechanisms.items()):
        values=[statistics.mean(r.get(k,0) for r in rows) for k in ['reorder_calls','reorder_changes','non_tail_victims',
            'reserved_victim_rollbacks','native_ready_promotions','cpu_prediction_mismatches']]
        values.append(statistics.mean(r.get('h2d_bytes',0) for r in rows)/2**20)
        report.append(f'| {gpu} / {cpu} | {group} | '+' | '.join(f'{x:.2f}' for x in values)+' |')
    report.append('')
    for name,title in [('v3_main_comparison','正式主结果'),('v3_long_comparison','独立长测'),
        ('v3_budget_sweep','开发预算扫描，n=1'),('v3_interaction','两模块的 2×2 对照'),
        ('v3_scheduler_composition','Scheduler 主体 CPU 组成'),('v3_cold_warm_cdf','固定类别的 TTFT CDF'),
        ('v3_victim_recovery','诊断 Victim 释放量与恢复时间'),('v3_calibration_validation','成本模型留出点')]:
        report.extend([f'### {title}','',f'![{title}](figures/{name}.png)',''])
    (OUT/'完整实验报告_V3.md').write_text('\n'.join(report))
    print('REPORT_BUILT',len(ext),'extended runs')


if __name__=='__main__':main()
