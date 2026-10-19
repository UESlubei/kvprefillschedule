"""Standalone PNG/PDF experiment figures. Run using Python with matplotlib."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling')
OUT=ROOT/'reports/figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def read(path):return json.loads(Path(path).read_text())


def save(fig,name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(name+'.png'),dpi=180)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)


def bars(ax,groups,key,title,ylabel):
    labels=[g['group'] for g in groups];means=[g['metrics'][key]['mean'] for g in groups]
    error=[[g['metrics'][key]['mean']-g['metrics'][key]['ci95'][0] for g in groups],
           [g['metrics'][key]['ci95'][1]-g['metrics'][key]['mean'] for g in groups]]
    ax.bar(labels,means,yerr=error,capsize=3,color=['#94a3b8']+['#2563eb']*(len(groups)-1))
    ax.set(title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.2)


def main():
    full=read(ROOT/'reports/v3_full_summary.json');long=read(ROOT/'reports/v3_long_summary.json')
    ext=read(ROOT/'reports/v3_extended_summary.json')
    def groups(work,gpu=1024,cpu=128):
        return [g for g in full['groups'] if g['group'].startswith(work+'-') and g['gpu_blocks']==gpu and g['cpu_blocks']==cpu]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    bars(axes[0,0],groups('A',640,0),'output_tok_s','A, GPU=640, CPU off','Output tokens/s')
    bars(axes[0,1],groups('A',640,0),'ttft_p95_ms','A, GPU=640, CPU off','P95 TTFT (ms)')
    bars(axes[1,0],groups('C'),'output_tok_s','C, GPU=1024, CPU=128','Output tokens/s')
    bars(axes[1,1],groups('D'),'ttft_p95_ms','D, GPU=1024, CPU=128','P95 TTFT (ms)')
    fig.suptitle('Formal paired runs: 5 seeds, 256 requests/run; bootstrap 95% CI')
    save(fig,'v3_main_comparison')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for policy,color in [('G0','#64748b'),('G1','#2563eb')]:
        rs=sorted((r for r in ext['rows'] if r['section'].startswith('budget_') and r['group']==f'A-{policy}'),key=lambda r:r['gpu'])
        for ax,key in zip(axes,['output_tok_s','ttft_p95_ms']):
            ax.plot([r['gpu'] for r in rs],[r[key] for r in rs],'o-',label=policy,color=color)
            ax.set(xlabel='GPU blocks (16 tokens/block)');ax.legend();ax.grid(alpha=.2)
    axes[0].set(ylabel='Output tokens/s');axes[1].set(ylabel='P95 TTFT (ms)')
    fig.suptitle('Budget development scan: seed 211 only, 64 requests/run; exploratory')
    save(fig,'v3_budget_sweep')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,label in zip(axes,['ttft_p95_ms','ttft_p99_ms'],['P95 TTFT (ms)','P99 TTFT (ms)']):
        bars(ax,long['groups'],key,'A, GPU=640, arrival=3.2 req/s',label)
    fig.suptitle('Independent long runs: 5 paired seeds, 2000 requests/run')
    save(fig,'v3_long_comparison')
    fair=read(ROOT/'reports/v3_fairness_and_overhead.json')
    comp=defaultdict(list)
    for r in fair['composition']:
        if r['group'].startswith('C-'):comp[r['group']].append(r)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    labels=sorted(comp);bottom=[0.]*len(labels)
    components=[('GPU probes',lambda r:r['gpu_probe_ms']),('CPU probes',lambda r:r['cpu_probe_ms']),
                ('Reorder exclusive',lambda r:max(0,r['reorder_ms']-r['gpu_probe_ms']-r['cpu_probe_ms'])),
                ('Victim selection',lambda r:r['victim_ms']),
                ('Other schedule body',lambda r:max(0,r['schedule_ms']-r['reorder_ms']-r['victim_ms']))]
    for name,fn in components:
        values=[statistics.mean(fn(r) for r in comp[label]) for label in labels]
        ax.bar(labels,values,bottom=bottom,label=name)
        bottom=[a+b for a,b in zip(bottom,values)]
    ax.set(ylabel='CPU milliseconds/run',title='C: scheduler body composition, mean of 5 runs')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    save(fig,'v3_scheduler_composition')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for policy in ['G0','G1']:
        samples=[]
        for r in full['rows']:
            if r['group']==f'A-{policy}' and r['gpu_blocks']==640:
                samples.extend(read(Path(r['source']).with_name('requests.json')))
        for ax,cls in zip(axes,['cold','gpu_warm']):
            xs=sorted(r['ttft_ms'] for r in samples if r['intended_class']==cls)
            if xs:ax.plot(xs,[(i+1)/len(xs) for i in range(len(xs))],label=policy)
            ax.set(title=cls,xlabel='TTFT (ms)',ylabel='Empirical CDF');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('A, GPU=640: pooled display from 5 runs; no request-level confidence claims')
    save(fig,'v3_cold_warm_cdf')
    profile=read(ROOT/'reports/v3_profile_analysis.json')
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    points=profile['victim_recoveries']
    if points:ax.scatter([r['blocks'] for r in points],[r['recovery_ms'] for r in points],alpha=.8)
    ax.set(xlabel='Exclusive blocks before preemption',ylabel='Queue + first resumed compute (ms)',
           title=f'Independent diagnostic: {len(points)} matched victim/recovery events')
    ax.grid(alpha=.2);save(fig,'v3_victim_recovery')
    validation=read(DATA/'cost_models/validation.json')
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    xs=[r['ms'] for r in validation];ys=[r['predicted_ms'] for r in validation]
    ax.scatter(xs,ys);bound=max(xs+ys)*1.1;ax.plot([0,bound],[0,bound],'--',color='#64748b')
    ax.set(xlabel='Measured CUDA interval (ms)',ylabel='Predicted interval (ms)',title='4 held-out prefill combinations')
    ax.grid(alpha=.2);save(fig,'v3_calibration_validation')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,work in zip(axes,['C','D']):
        rs=[g for g in groups(work) if g['group'].split('-')[1] in ['B1','O1','O2','O3']]
        bars(ax,rs,'output_tok_s',f'{work}: B1 / O1 / O2 / O3','Output tokens/s')
    fig.suptitle('2 x 2 factors; absolute interactions and paired intervals in summary JSON')
    save(fig,'v3_interaction')
    print('FIGURES_EXPORTED',OUT)


if __name__=='__main__':main()
