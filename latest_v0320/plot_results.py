"""Render scientific result artifacts using the existing plotting environment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from latest_v0320.common import ROOT

def main():
    report=ROOT/'reports/latest_v0320'
    comparisons=json.loads((report/'comparison_summary.json').read_text())['comparisons']
    stages=[('formal','Synchronous Eager'),('deployment_formal','Native Async + CUDA Graph')]
    if not any(c['stage']=='formal' for c in comparisons):
        stages=[('pilot','Synchronous Eager — pilot'),('deployment_pilot','Native Async + CUDA Graph — pilot')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,(stage,title) in zip(axes,stages):
        rows=[c for c in comparisons if c['stage']==stage and c['candidate'] in {'NG','NT'}]
        rows.sort(key=lambda r:(r['workload'],r['blocks'],r['candidate']))
        labels=[]
        for i,c in enumerate(rows):
            p=c['summary']['p95_ttft'];y=p['mean_percent'];ci=p['ci95_percent']
            ax.bar(i,y,width=.65,color='#2878b5' if y<0 else '#c44e52')
            if ci[0] is not None:ax.errorbar(i,y,yerr=[[y-ci[0]],[ci[1]-y]],fmt='none',color='#222',capsize=4)
            ax.annotate(f'{y:+.1f}%',(i,y),xytext=(0,-15 if y<0 else 7),textcoords='offset points',ha='center')
            labels.append(f"{c['workload']} / {c['blocks']}\n{c['candidate']}, n={c['pairs']}")
        ax.axhline(0,color='#444',linewidth=.8)
        ax.set_xticks(range(len(labels)),labels,rotation=0)
        ax.set_title(title)
        ax.set_ylabel('P95 TTFT change vs unmodified vLLM 0.32.0 (%)')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        if rows:
            low=min(c['summary']['p95_ttft']['mean_percent'] for c in rows)
            high=max(c['summary']['p95_ttft']['mean_percent'] for c in rows)
            ax.set_ylim(min(low-15,-10),max(high+15,10))
    selected=[c for c in comparisons if c['stage'] in {s[0] for s in stages} and c['candidate'] in {'NG','NT'}]
    if selected:
        low=min(c['summary']['p95_ttft']['mean_percent'] for c in selected)
        high=max(c['summary']['p95_ttft']['mean_percent'] for c in selected)
        for ax in axes:ax.set_ylim(min(low-15,-10),max(high+15,10))
    fig.suptitle('RTX 3090 / Qwen3-1.7B FP16 / controlled KV budgets\nNegative = lower latency; intervals bootstrap paired runs, not individual requests',fontsize=12)
    fig.savefig(report/'latest_p95_comparison.png',dpi=180)
    fig.savefig(report/'latest_p95_comparison.pdf')
    plt.close(fig)

if __name__=='__main__':main()
